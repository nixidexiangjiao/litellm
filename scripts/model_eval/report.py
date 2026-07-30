"""Turn evaluation results into a JSONL trace and a two-sheet Excel report.

Only the workbook writer needs ``openpyxl``, so it is imported there; the
statistics and the column definitions stay importable without it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TextIO, TypeVar

from .client import CallFailed, CallSucceeded, TokenUsage
from .pricing import CostBreakdown, ModelPrice
from .runner import EvalResult

if TYPE_CHECKING:
    from openpyxl import Workbook

CellValue = str | float | int | None

EXCEL_CELL_LIMIT = 32_767

_MS = "0.0"
_MONEY = "0.000000"
_RATIO = "0.000"

_RowT = TypeVar("_RowT")


@dataclass(frozen=True, slots=True)
class Distribution:
    mean: float
    p50: float
    p90: float
    p99: float


@dataclass(frozen=True, slots=True)
class ModelSummary:
    price: ModelPrice
    requests: int
    succeeded: int
    failed: int
    ttft_ms: Distribution | None
    tpot_ms: Distribution | None
    total_ms: Distribution | None
    output_tps: Distribution | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    cost_total: float

    @property
    def cache_hit_rate(self) -> float | None:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else None

    @property
    def cost_per_request(self) -> float | None:
        return self.cost_total / self.succeeded if self.succeeded else None

    @property
    def cost_per_1m_output_tokens(self) -> float | None:
        return self.cost_total * 1_000_000 / self.completion_tokens if self.completion_tokens else None


@dataclass(frozen=True, slots=True)
class Column(Generic[_RowT]):
    header: str
    number_format: str | None
    value: Callable[[_RowT], CellValue]


def summarise(results: Sequence[EvalResult]) -> tuple[ModelSummary, ...]:
    models = tuple(dict.fromkeys(result.price.model for result in results))
    return tuple(
        _summarise_model(tuple(result for result in results if result.price.model == model)) for model in models
    )


def distribution(values: Sequence[float]) -> Distribution | None:
    if not values:
        return None
    ordered = tuple(sorted(values))
    return Distribution(
        mean=sum(ordered) / len(ordered),
        p50=_percentile(ordered, 50),
        p90=_percentile(ordered, 90),
        p99=_percentile(ordered, 99),
    )


def tee_jsonl(results: Iterator[EvalResult], handle: TextIO) -> Iterator[EvalResult]:
    """Write every result to disk as it lands, so a long run survives a crash."""
    for result in results:
        handle.write(json.dumps(as_json_record(result), ensure_ascii=False) + "\n")
        handle.flush()
        yield result


def as_json_record(result: EvalResult) -> dict[str, CellValue]:
    succeeded = _succeeded(result)
    return {column.header: column.value(result) for column in REQUEST_COLUMNS} | {
        "tool_calls": succeeded.tool_calls if succeeded is not None else None,
        "response": succeeded.content if succeeded is not None else None,
        "reasoning": succeeded.reasoning if succeeded is not None else None,
    }


def write_report(path: Path, results: Sequence[EvalResult], summaries: Sequence[ModelSummary]) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.worksheets[0])
    _write_sheet(workbook, "summary", SUMMARY_COLUMNS, summaries)
    _write_sheet(workbook, "requests", REQUEST_COLUMNS, results)
    workbook.save(path)


def format_console_table(summaries: Sequence[ModelSummary]) -> str:
    headers = ("model", "ok/total", "ttft p50", "tpot p50", "total p50", "out tok/s", "cost", "cost/req")
    rows = tuple(
        (
            summary.price.label,
            f"{summary.succeeded}/{summary.requests}",
            _fmt(_stat(summary.ttft_ms, lambda d: d.p50)),
            _fmt(_stat(summary.tpot_ms, lambda d: d.p50)),
            _fmt(_stat(summary.total_ms, lambda d: d.p50)),
            _fmt(_stat(summary.output_tps, lambda d: d.mean)),
            f"{summary.cost_total:.4f} {summary.price.currency}",
            _fmt(summary.cost_per_request, digits=6),
        )
        for summary in summaries
    )
    widths = tuple(max(len(cell) for cell in column) for column in zip(headers, *rows, strict=True))
    return "\n".join(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in (headers, *rows)
    )


def _summarise_model(results: Sequence[EvalResult]) -> ModelSummary:
    succeeded = tuple(result for result in results if isinstance(result.outcome, CallSucceeded))
    tokens = tuple(usage for usage in (_usage(result) for result in succeeded) if usage is not None)
    return ModelSummary(
        price=results[0].price,
        requests=len(results),
        succeeded=len(succeeded),
        failed=len(results) - len(succeeded),
        ttft_ms=distribution(_present(_ttft(result) for result in succeeded)),
        tpot_ms=distribution(_present(_tpot(result) for result in succeeded)),
        total_ms=distribution(_present(result.outcome.total_ms for result in succeeded)),
        output_tps=distribution(_present(_output_tps(result) for result in succeeded)),
        prompt_tokens=sum(usage.prompt_tokens for usage in tokens),
        completion_tokens=sum(usage.completion_tokens for usage in tokens),
        cached_tokens=sum(usage.cached_tokens for usage in tokens),
        cache_creation_tokens=sum(usage.cache_creation_tokens for usage in tokens),
        cost_total=sum(result.cost.total for result in results if result.cost is not None),
    )


def _present(values: Iterable[float | None]) -> tuple[float, ...]:
    return tuple(value for value in values if value is not None)


def _percentile(ordered: Sequence[float], percent: float) -> float:
    rank = max(math.ceil(percent / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def _stat(dist: Distribution | None, project: Callable[[Distribution], float]) -> float | None:
    return project(dist) if dist is not None else None


def _succeeded(result: EvalResult) -> CallSucceeded | None:
    return result.outcome if isinstance(result.outcome, CallSucceeded) else None


def _failed(result: EvalResult) -> CallFailed | None:
    return result.outcome if isinstance(result.outcome, CallFailed) else None


def _usage(result: EvalResult) -> TokenUsage | None:
    succeeded = _succeeded(result)
    return succeeded.usage if succeeded is not None else None


def _ttft(result: EvalResult) -> float | None:
    succeeded = _succeeded(result)
    return succeeded.ttft_ms if succeeded is not None else None


def _tpot(result: EvalResult) -> float | None:
    succeeded = _succeeded(result)
    return succeeded.tpot_ms if succeeded is not None else None


def _output_tps(result: EvalResult) -> float | None:
    succeeded = _succeeded(result)
    if succeeded is None or succeeded.usage is None or succeeded.total_ms <= 0:
        return None
    return succeeded.usage.completion_tokens / (succeeded.total_ms / 1000)


def _truncate_cell(text: str) -> str:
    return text if len(text) <= EXCEL_CELL_LIMIT else f"{text[: EXCEL_CELL_LIMIT - 3]}..."


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


REQUEST_COLUMNS: tuple[Column[EvalResult], ...] = (
    Column("model", None, lambda r: r.price.model),
    Column("label", None, lambda r: r.price.label),
    Column("repetition", None, lambda r: r.repetition),
    Column("sequence", None, lambda r: r.sequence),
    Column("user_id", None, lambda r: r.row.user_id),
    Column("timestamp", None, lambda r: r.row.timestamp.isoformat()),
    Column("sheet_row", None, lambda r: r.row.sheet_row),
    Column("recorded_model", None, lambda r: r.row.recorded_model),
    Column("status", None, lambda r: "ok" if _succeeded(r) is not None else "failed"),
    Column("streamed", None, lambda r: _map_ok(r, lambda ok: ok.streamed)),
    Column("error_kind", None, lambda r: _map_failed(r, lambda err: err.kind)),
    Column("http_status", None, lambda r: _map_failed(r, lambda err: err.status_code)),
    Column("error_detail", None, lambda r: _map_failed(r, lambda err: err.detail)),
    Column("ttft_ms", _MS, _ttft),
    Column("tpot_ms", _MS, _tpot),
    Column("total_ms", _MS, lambda r: r.outcome.total_ms),
    Column("output_tokens_per_s", _RATIO, _output_tps),
    Column("prompt_tokens", None, lambda r: _map_usage(r, lambda u: u.prompt_tokens)),
    Column("completion_tokens", None, lambda r: _map_usage(r, lambda u: u.completion_tokens)),
    Column("total_tokens", None, lambda r: _map_usage(r, lambda u: u.total_tokens)),
    Column("cached_tokens", None, lambda r: _map_usage(r, lambda u: u.cached_tokens)),
    Column("cache_creation_tokens", None, lambda r: _map_usage(r, lambda u: u.cache_creation_tokens)),
    Column("reasoning_tokens", None, lambda r: _map_usage(r, lambda u: u.reasoning_tokens)),
    Column("cost_total", _MONEY, lambda r: _map_cost(r, lambda c: c.total)),
    Column("cost_uncached_input", _MONEY, lambda r: _map_cost(r, lambda c: c.uncached_input)),
    Column("cost_cached_input", _MONEY, lambda r: _map_cost(r, lambda c: c.cached_input)),
    Column("cost_cache_write", _MONEY, lambda r: _map_cost(r, lambda c: c.cache_write)),
    Column("cost_output", _MONEY, lambda r: _map_cost(r, lambda c: c.output)),
    Column("currency", None, lambda r: r.price.currency),
    Column("proxy_reported_cost", _MONEY, lambda r: _map_ok(r, lambda ok: ok.proxy_reported_cost)),
    Column("finish_reason", None, lambda r: _map_ok(r, lambda ok: ok.finish_reason)),
    Column("chunk_count", None, lambda r: _map_ok(r, lambda ok: ok.chunk_count)),
    Column("response_model", None, lambda r: _map_ok(r, lambda ok: ok.response_model)),
    Column("tool_calls", None, lambda r: _map_ok(r, lambda ok: _truncate_cell(ok.tool_calls))),
    Column("response", None, lambda r: _map_ok(r, lambda ok: _truncate_cell(ok.content))),
    Column("reasoning", None, lambda r: _map_ok(r, lambda ok: _truncate_cell(ok.reasoning))),
)


SUMMARY_COLUMNS: tuple[Column[ModelSummary], ...] = (
    Column("model", None, lambda s: s.price.model),
    Column("label", None, lambda s: s.price.label),
    Column("requests", None, lambda s: s.requests),
    Column("succeeded", None, lambda s: s.succeeded),
    Column("failed", None, lambda s: s.failed),
    Column("ttft_ms_mean", _MS, lambda s: _stat(s.ttft_ms, lambda d: d.mean)),
    Column("ttft_ms_p50", _MS, lambda s: _stat(s.ttft_ms, lambda d: d.p50)),
    Column("ttft_ms_p90", _MS, lambda s: _stat(s.ttft_ms, lambda d: d.p90)),
    Column("ttft_ms_p99", _MS, lambda s: _stat(s.ttft_ms, lambda d: d.p99)),
    Column("tpot_ms_mean", _MS, lambda s: _stat(s.tpot_ms, lambda d: d.mean)),
    Column("tpot_ms_p50", _MS, lambda s: _stat(s.tpot_ms, lambda d: d.p50)),
    Column("tpot_ms_p90", _MS, lambda s: _stat(s.tpot_ms, lambda d: d.p90)),
    Column("total_ms_mean", _MS, lambda s: _stat(s.total_ms, lambda d: d.mean)),
    Column("total_ms_p50", _MS, lambda s: _stat(s.total_ms, lambda d: d.p50)),
    Column("total_ms_p90", _MS, lambda s: _stat(s.total_ms, lambda d: d.p90)),
    Column("total_ms_p99", _MS, lambda s: _stat(s.total_ms, lambda d: d.p99)),
    Column("output_tokens_per_s_mean", _RATIO, lambda s: _stat(s.output_tps, lambda d: d.mean)),
    Column("prompt_tokens", None, lambda s: s.prompt_tokens),
    Column("completion_tokens", None, lambda s: s.completion_tokens),
    Column("cached_tokens", None, lambda s: s.cached_tokens),
    Column("cache_creation_tokens", None, lambda s: s.cache_creation_tokens),
    Column("cache_hit_rate", _RATIO, lambda s: s.cache_hit_rate),
    Column("cost_total", _MONEY, lambda s: s.cost_total),
    Column("cost_per_request", _MONEY, lambda s: s.cost_per_request),
    Column("cost_per_1m_output_tokens", _MONEY, lambda s: s.cost_per_1m_output_tokens),
    Column("currency", None, lambda s: s.price.currency),
)


def _map_ok(result: EvalResult, project: Callable[[CallSucceeded], CellValue]) -> CellValue:
    succeeded = _succeeded(result)
    return project(succeeded) if succeeded is not None else None


def _map_failed(result: EvalResult, project: Callable[[CallFailed], CellValue]) -> CellValue:
    failed = _failed(result)
    return project(failed) if failed is not None else None


def _map_usage(result: EvalResult, project: Callable[[TokenUsage], CellValue]) -> CellValue:
    usage = _usage(result)
    return project(usage) if usage is not None else None


def _map_cost(result: EvalResult, project: Callable[[CostBreakdown], CellValue]) -> CellValue:
    return project(result.cost) if result.cost is not None else None


def _write_sheet(
    workbook: "Workbook",
    title: str,
    columns: Sequence[Column[_RowT]],
    rows: Sequence[_RowT],
) -> None:
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet(title)
    sheet.append([column.header for column in columns])
    for row in rows:
        sheet.append([column.value(row) for column in columns])
    for index, column in enumerate(columns, start=1):
        letter = get_column_letter(index)
        sheet.column_dimensions[letter].width = min(max(len(column.header) + 2, 10), 40)
        if column.number_format is not None:
            for cell in sheet[letter][1:]:
                cell.number_format = column.number_format
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
