"""Command line entry point for the LiteLLM model evaluation replay."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import httpx

from .client import CallFailed, CallSucceeded, ProxyTarget, StreamMode
from .pricing import ModelPrice, PriceTable, load_price_table
from .report import format_console_table, summarise, tee_jsonl, write_report
from .runner import EvalResult, run_evaluation
from .templates import write_pricing_template, write_workload_template
from .workload import Workload, WorkloadRow, load_workload, order_rows

_API_KEY_ENV_VARS = ("LITELLM_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True, slots=True)
class Options:
    workload: Path
    pricing: Path
    workload_sheet: str | None
    pricing_sheet: str | None
    base_url: str
    api_key: str
    models: tuple[str, ...]
    output: Path
    jsonl: Path
    stream_mode: StreamMode
    timeout_s: float
    repeat: int
    limit: int | None
    sleep_s: float
    dry_run: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)

    if namespace.write_templates is not None:
        return _write_templates(Path(namespace.write_templates))
    if namespace.workload is None or namespace.pricing is None:
        parser.error("--workload and --pricing are required (or use --write-templates)")

    options = _to_options(namespace)
    workload = load_workload(options.workload, options.workload_sheet)
    price_table = load_price_table(options.pricing, options.pricing_sheet)
    _report_input_problems(workload, price_table)

    prices = _select_prices(price_table, options.models)
    if not prices:
        _say("no model to evaluate: the price sheet is empty or --model matched nothing")
        return 2

    rows = order_rows(workload.rows)[: options.limit]
    if not rows:
        _say("no usable request row in the workload sheet")
        return 2

    planned = len(prices) * options.repeat * len(rows)
    _say(
        f"replaying {len(rows)} request(s) x {options.repeat} repetition(s) "
        f"across {len(prices)} model(s) = {planned} call(s) against {options.base_url}"
    )
    if options.dry_run:
        for price in prices:
            _say(f"  would call {price.model} ({price.label}) at {price.input_per_1m}/{price.output_per_1m} per 1M")
        return 0

    results = _execute(options, prices, rows, planned)
    if not results:
        _say("no result recorded")
        return 1

    summaries = summarise(results)
    write_report(options.output, results, summaries)
    _say("")
    _say(format_console_table(summaries))
    _say(f"\nreport: {options.output}\ntrace:  {options.jsonl}")
    return 0


def _execute(
    options: Options,
    prices: Sequence[ModelPrice],
    rows: Sequence[WorkloadRow],
    planned: int,
) -> tuple[EvalResult, ...]:
    target = ProxyTarget(base_url=options.base_url, api_key=options.api_key, timeout_s=options.timeout_s)
    options.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=options.timeout_s) as client, options.jsonl.open("w", encoding="utf-8") as trace:
        stream = run_evaluation(
            client=client,
            target=target,
            prices=prices,
            rows=rows,
            stream_mode=options.stream_mode,
            repeat=options.repeat,
            sleep_s=options.sleep_s,
        )
        return tuple(_until_interrupted(_with_progress(tee_jsonl(stream, trace), planned)))


def _with_progress(results: Iterator[EvalResult], planned: int) -> Iterator[EvalResult]:
    for index, result in enumerate(results, start=1):
        _say(f"[{index}/{planned}] {result.price.model} {_describe(result)}")
        yield result


def _until_interrupted(results: Iterator[EvalResult]) -> Iterator[EvalResult]:
    """Keep whatever finished when the operator gives up on a long run."""
    try:
        yield from results
    except KeyboardInterrupt:
        _say("\ninterrupted; reporting on the calls that completed")


def _describe(result: EvalResult) -> str:
    location = f"user={result.row.user_id} row={result.row.sheet_row}"
    match result.outcome:
        case CallFailed(kind=kind, status_code=status, detail=detail):
            return f"{location} FAILED {kind}{f' {status}' if status is not None else ''}: {detail[:120]}"
        case CallSucceeded(ttft_ms=ttft, total_ms=total, usage=usage):
            tokens = f"{usage.completion_tokens} out tok" if usage is not None else "usage missing"
            ttft_text = f"ttft {ttft:.0f}ms " if ttft is not None else ""
            cost = f" cost {result.cost.total:.6f} {result.price.currency}" if result.cost is not None else ""
            return f"{location} ok {ttft_text}total {total:.0f}ms {tokens}{cost}"


def _select_prices(price_table: PriceTable, models: tuple[str, ...]) -> tuple[ModelPrice, ...]:
    if not models:
        return price_table.prices
    selected = tuple(price for model in models for price in price_table.prices if price.model == model)
    for missing in (model for model in models if price_table.get(model) is None):
        _say(f"warning: --model {missing} has no row in the price sheet; skipping it")
    return selected


def _report_input_problems(workload: Workload, price_table: PriceTable) -> None:
    for problem in workload.problems:
        _say(f"warning: workload row {problem.sheet_row} skipped: {problem.reason}")
    for problem in price_table.problems:
        _say(f"warning: price row {problem.sheet_row} skipped: {problem.reason}")


def _write_templates(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    workload_path = directory / "workload_template.xlsx"
    pricing_path = directory / "pricing_template.xlsx"
    write_workload_template(workload_path)
    write_pricing_template(pricing_path)
    _say(f"wrote {workload_path}\nwrote {pricing_path}")
    return 0


def _to_options(namespace: argparse.Namespace) -> Options:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = Path(namespace.output) if namespace.output else Path(f"model_eval_{stamp}.xlsx")
    jsonl = Path(namespace.jsonl) if namespace.jsonl else output.with_suffix(".jsonl")
    return Options(
        workload=Path(namespace.workload),
        pricing=Path(namespace.pricing),
        workload_sheet=namespace.workload_sheet,
        pricing_sheet=namespace.pricing_sheet,
        base_url=namespace.base_url,
        api_key=namespace.api_key if namespace.api_key is not None else _api_key_from_env(),
        models=tuple(namespace.model or ()),
        output=output,
        jsonl=jsonl,
        stream_mode=namespace.stream_mode,
        timeout_s=namespace.timeout,
        repeat=namespace.repeat,
        limit=namespace.limit,
        sleep_s=namespace.sleep,
        dry_run=namespace.dry_run,
    )


def _say(message: str) -> None:
    """This is a CLI; stdout is the product."""
    print(message, flush=True)  # noqa: T201  # see docstring


def _api_key_from_env() -> str:
    return next((os.environ[name] for name in _API_KEY_ENV_VARS if os.environ.get(name)), "")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m model_eval",
        description="Replay recorded OpenAI-format requests through a LiteLLM proxy and compare models on "
        "latency, tokens and cost.",
    )
    parser.add_argument("--workload", help="Excel file: column A userId, column B timestamp, column C request body")
    parser.add_argument("--pricing", help="Excel price sheet; it also decides which models are evaluated")
    parser.add_argument("--workload-sheet", default=None, help="worksheet name (default: the first sheet)")
    parser.add_argument("--pricing-sheet", default=None, help="worksheet name (default: the first sheet)")
    parser.add_argument("--base-url", default="http://localhost:4000", help="LiteLLM proxy base URL")
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"proxy key; defaults to the first of {', '.join(_API_KEY_ENV_VARS)}, then to no auth header",
    )
    parser.add_argument("--model", action="append", help="evaluate only this model (repeatable)")
    parser.add_argument("--output", default=None, help="xlsx report path")
    parser.add_argument("--jsonl", default=None, help="raw per-request trace path (default: alongside --output)")
    parser.add_argument(
        "--stream-mode",
        choices=get_args(StreamMode),
        default="as_recorded",
        help="replay streaming as recorded, or force every request one way (TTFT/TPOT need streaming)",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    parser.add_argument("--repeat", type=int, default=1, help="replay the whole workload N times per model")
    parser.add_argument("--limit", type=int, default=None, help="only replay the first N ordered requests")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between requests")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without calling anything")
    parser.add_argument("--write-templates", default=None, help="write starter workload/pricing workbooks here")
    return parser
