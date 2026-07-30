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
from .pricing import ModelPrice, load_price_table
from .report import export_workbook, format_console_table
from .runner import EvalResult, run_evaluation
from .storage import EvalStore, RunMetadata, new_run_id, open_store
from .templates import write_pricing_template, write_workload_template
from .workload import WorkloadRow, load_workload, order_rows

_API_KEY_ENV_VARS = ("LITELLM_API_KEY", "OPENAI_API_KEY")
_DEFAULT_DB = "model_eval.duckdb"


@dataclass(frozen=True, slots=True)
class RunOptions:
    workload: Path
    workload_sheet: str | None
    base_url: str
    api_key: str
    models: tuple[str, ...]
    stream_mode: StreamMode
    timeout_s: float
    repeat: int
    limit: int | None
    sleep_s: float
    note: str
    xlsx: Path | None
    dry_run: bool


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    if namespace.command == "templates":
        return _write_templates(Path(namespace.directory))

    with open_store(Path(namespace.db)) as store:
        match namespace.command:
            case "run":
                return _run(store, _to_run_options(namespace))
            case "import-prices":
                return _import_prices(store, Path(namespace.file), namespace.sheet)
            case "list-prices":
                return _list_prices(store)
            case "sql":
                return _sql(store, namespace.query)
            case "report":
                return _report(store, namespace.run_id, Path(namespace.xlsx) if namespace.xlsx else None)
            case _:
                return 2


def _run(store: EvalStore, options: RunOptions) -> int:
    workload = load_workload(options.workload, options.workload_sheet)
    for problem in workload.problems:
        _say(f"warning: workload row {problem.sheet_row} skipped: {problem.reason}")

    prices = _select_prices(store, options.models)
    if not prices:
        _say("no model to evaluate; import a price sheet first (see: import-prices)")
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

    run = RunMetadata(
        run_id=new_run_id(),
        started_at=datetime.now(tz=timezone.utc),
        base_url=options.base_url,
        workload=str(options.workload),
        stream_mode=options.stream_mode,
        repeat_count=options.repeat,
        note=options.note,
    )
    store.start_run(run)
    _execute(store, run.run_id, options, prices, rows, planned)
    store.finish_run(run.run_id)

    _say(f"\nrun_id: {run.run_id}")
    return _report(store, run.run_id, options.xlsx)


def _execute(
    store: EvalStore,
    run_id: str,
    options: RunOptions,
    prices: Sequence[ModelPrice],
    rows: Sequence[WorkloadRow],
    planned: int,
) -> None:
    target = ProxyTarget(base_url=options.base_url, api_key=options.api_key, timeout_s=options.timeout_s)
    with httpx.Client(timeout=options.timeout_s) as client:
        stream = run_evaluation(
            client=client,
            target=target,
            prices=prices,
            rows=rows,
            stream_mode=options.stream_mode,
            repeat=options.repeat,
            sleep_s=options.sleep_s,
        )
        for _ in _until_interrupted(_with_progress(_tee_to_store(stream, store, run_id), planned)):
            pass


def _tee_to_store(results: Iterator[EvalResult], store: EvalStore, run_id: str) -> Iterator[EvalResult]:
    """Persist each call as it lands, so an interrupted run still reports."""
    for result in results:
        store.record(run_id, result)
        yield result


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
            return f"{location} ok {ttft_text}total {total:.0f}ms {tokens}"


def _report(store: EvalStore, run_id: str | None, xlsx: Path | None) -> int:
    resolved = run_id or store.latest_run_id()
    if resolved is None:
        _say("no run recorded yet")
        return 1

    summaries = store.summary(resolved)
    if not summaries:
        _say(f"run {resolved} has no request recorded")
        return 1

    _say("")
    _say(format_console_table(summaries))
    if xlsx is not None:
        export_workbook(store, xlsx, resolved)
        _say(f"\nworkbook: {xlsx}")
    return 0


def _sql(store: EvalStore, query: str) -> int:
    """Ad-hoc reporting; the views are the point of keeping results in DuckDB."""
    store.connection.sql(query).show()
    return 0


def _import_prices(store: EvalStore, path: Path, sheet: str | None) -> int:
    table = load_price_table(path, sheet)
    for problem in table.problems:
        _say(f"warning: price row {problem.sheet_row} skipped: {problem.reason}")
    if not table.prices:
        _say("nothing imported")
        return 2
    _say(f"imported {store.replace_prices(table.prices)} price row(s) from {path}")
    return _list_prices(store)


def _list_prices(store: EvalStore) -> int:
    prices = store.prices()
    if not prices:
        _say("model_prices is empty")
        return 0
    for price in prices:
        cache_read = price.cache_read_per_1m if price.cache_read_per_1m is not None else price.input_per_1m
        _say(
            f"  {price.model:<32} in {price.input_per_1m:>9} / out {price.output_per_1m:>9} / "
            f"cache-read {cache_read:>9} {price.currency}  ({price.label})"
        )
    return 0


def _select_prices(store: EvalStore, models: tuple[str, ...]) -> tuple[ModelPrice, ...]:
    selected = store.prices(models)
    known = {price.model for price in selected}
    for missing in (model for model in models if model not in known):
        _say(f"warning: --model {missing} has no enabled row in model_prices; skipping it")
    return selected


def _write_templates(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    write_workload_template(directory / "workload_template.xlsx")
    write_pricing_template(directory / "pricing_template.xlsx")
    _say(f"wrote {directory / 'workload_template.xlsx'}\nwrote {directory / 'pricing_template.xlsx'}")
    return 0


def _to_run_options(namespace: argparse.Namespace) -> RunOptions:
    return RunOptions(
        workload=Path(namespace.workload),
        workload_sheet=namespace.workload_sheet,
        base_url=namespace.base_url,
        api_key=namespace.api_key if namespace.api_key is not None else _api_key_from_env(),
        models=tuple(namespace.model or ()),
        stream_mode=namespace.stream_mode,
        timeout_s=namespace.timeout,
        repeat=namespace.repeat,
        limit=namespace.limit,
        sleep_s=namespace.sleep,
        note=namespace.note or "",
        xlsx=Path(namespace.xlsx) if namespace.xlsx else None,
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
        "latency, tokens and cost. Results and prices live in a DuckDB file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    database = argparse.ArgumentParser(add_help=False)
    database.add_argument("--db", default=_DEFAULT_DB, help=f"DuckDB file (default: {_DEFAULT_DB})")

    run = subparsers.add_parser("run", parents=[database], help="replay the workload against every enabled model")
    run.add_argument("--workload", required=True, help="Excel: column A userId, column B time, column C request body")
    run.add_argument("--workload-sheet", default=None, help="worksheet name (default: the first sheet)")
    run.add_argument("--base-url", default="http://localhost:4000", help="LiteLLM proxy base URL")
    run.add_argument(
        "--api-key",
        default=None,
        help=f"proxy key; defaults to the first of {', '.join(_API_KEY_ENV_VARS)}, then to no auth header",
    )
    run.add_argument("--model", action="append", help="evaluate only this model (repeatable)")
    run.add_argument(
        "--stream-mode",
        choices=get_args(StreamMode),
        default="as_recorded",
        help="replay streaming as recorded, or force every request one way (TTFT/TPOT need streaming)",
    )
    run.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    run.add_argument("--repeat", type=int, default=1, help="replay the whole workload N times per model")
    run.add_argument("--limit", type=int, default=None, help="only replay the first N ordered requests")
    run.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between requests")
    run.add_argument("--note", default=None, help="free text stored on the run, e.g. what you were testing")
    run.add_argument("--xlsx", default=None, help="also export this run to a workbook")
    run.add_argument("--dry-run", action="store_true", help="print the plan without calling anything")

    prices = subparsers.add_parser(
        "import-prices", parents=[database], help="replace model_prices from an Excel price sheet"
    )
    prices.add_argument("file", help="xlsx with model/input_per_1m/output_per_1m columns")
    prices.add_argument("--sheet", default=None, help="worksheet name (default: the first sheet)")

    subparsers.add_parser("list-prices", parents=[database], help="show the price table currently in the database")

    report = subparsers.add_parser("report", parents=[database], help="print (and optionally export) a run's summary")
    report.add_argument("--run-id", default=None, help="defaults to the most recent run")
    report.add_argument("--xlsx", default=None, help="export summary and per-request rows to this workbook")

    query = subparsers.add_parser("sql", parents=[database], help="run a query against the results")
    query.add_argument(
        "query",
        nargs="?",
        default="SELECT * FROM eval_summary ORDER BY run_id DESC, model",
        help="defaults to the whole eval_summary view",
    )

    templates = subparsers.add_parser("templates", help="write starter workload/pricing workbooks")
    templates.add_argument("directory", help="where to write them")

    return parser
