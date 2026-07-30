"""DuckDB is the store: prices in, raw per-request facts in, reporting out in SQL.

Cost is deliberately *not* written to disk. It is derived by ``eval_request_costs``
from the tokens the run measured and whatever ``model_prices`` says today, so
correcting a price re-values every historical run instead of stranding it.

Timestamps are stored naive in UTC. Everything upstream is already normalised
to UTC, and keeping the columns tz-free means reading them back needs no extra
timezone library.

``duckdb`` is imported inside the connect helper, matching how ``openpyxl`` is
handled elsewhere in this package: the schema and the row mapping stay
importable without the driver installed.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from pydantic import BaseModel, ConfigDict

from .client import CallFailed, CallSucceeded
from .pricing import ModelPrice
from .runner import EvalResult

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_prices (
    model              VARCHAR PRIMARY KEY,
    label              VARCHAR NOT NULL,
    input_per_1m       DOUBLE  NOT NULL,
    output_per_1m      DOUBLE  NOT NULL,
    cache_read_per_1m  DOUBLE,
    cache_write_per_1m DOUBLE,
    currency           VARCHAR NOT NULL DEFAULT 'USD',
    enabled            BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id       VARCHAR PRIMARY KEY,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    base_url     VARCHAR NOT NULL,
    workload     VARCHAR NOT NULL,
    stream_mode  VARCHAR NOT NULL,
    repeat_count INTEGER NOT NULL,
    note         VARCHAR
);

CREATE TABLE IF NOT EXISTS eval_requests (
    run_id                VARCHAR NOT NULL,
    model                 VARCHAR NOT NULL,
    repetition            INTEGER NOT NULL,
    sequence              INTEGER NOT NULL,
    called_at             TIMESTAMP NOT NULL,
    user_id               VARCHAR NOT NULL,
    request_ts            TIMESTAMP NOT NULL,
    sheet_row             INTEGER NOT NULL,
    recorded_model        VARCHAR,
    status                VARCHAR NOT NULL,
    streamed              BOOLEAN,
    error_kind            VARCHAR,
    http_status           INTEGER,
    error_detail          VARCHAR,
    ttft_ms               DOUBLE,
    tpot_ms               DOUBLE,
    total_ms              DOUBLE NOT NULL,
    prompt_tokens         INTEGER,
    completion_tokens     INTEGER,
    total_tokens          INTEGER,
    cached_tokens         INTEGER,
    cache_creation_tokens INTEGER,
    reasoning_tokens      INTEGER,
    chunk_count           INTEGER,
    finish_reason         VARCHAR,
    response_model        VARCHAR,
    proxy_reported_cost   DOUBLE,
    tool_calls            VARCHAR,
    response              VARCHAR,
    reasoning             VARCHAR,
    PRIMARY KEY (run_id, model, repetition, sequence)
);

CREATE OR REPLACE VIEW eval_request_costs AS
WITH priced AS (
    SELECT
        r.*,
        p.label,
        p.currency,
        p.input_per_1m,
        p.output_per_1m,
        COALESCE(p.cache_read_per_1m, p.input_per_1m)  AS cache_read_per_1m,
        COALESCE(p.cache_write_per_1m, p.input_per_1m) AS cache_write_per_1m,
        LEAST(COALESCE(r.cached_tokens, 0), COALESCE(r.prompt_tokens, 0)) AS billed_cache_read
    FROM eval_requests AS r
    LEFT JOIN model_prices AS p ON p.model = r.model
), split AS (
    SELECT
        priced.*,
        LEAST(
            COALESCE(cache_creation_tokens, 0),
            COALESCE(prompt_tokens, 0) - billed_cache_read
        ) AS billed_cache_write
    FROM priced
)
SELECT
    split.* EXCLUDE (input_per_1m, output_per_1m, cache_read_per_1m, cache_write_per_1m),
    CASE WHEN total_ms > 0 THEN completion_tokens / (total_ms / 1000) END AS output_tokens_per_s,
    (COALESCE(prompt_tokens, 0) - billed_cache_read - billed_cache_write)
        * input_per_1m / 1e6 AS cost_uncached_input,
    billed_cache_read  * cache_read_per_1m  / 1e6 AS cost_cached_input,
    billed_cache_write * cache_write_per_1m / 1e6 AS cost_cache_write,
    completion_tokens  * output_per_1m      / 1e6 AS cost_output,
    (COALESCE(prompt_tokens, 0) - billed_cache_read - billed_cache_write) * input_per_1m / 1e6
        + billed_cache_read  * cache_read_per_1m  / 1e6
        + billed_cache_write * cache_write_per_1m / 1e6
        + completion_tokens  * output_per_1m      / 1e6 AS cost_total
FROM split;

CREATE OR REPLACE VIEW eval_summary AS
SELECT
    run_id,
    model,
    any_value(label)    AS label,
    any_value(currency) AS currency,
    count(*)                                       AS requests,
    count(*) FILTER (WHERE status = 'ok')          AS succeeded,
    count(*) FILTER (WHERE status = 'failed')      AS failed,
    avg(ttft_ms)                    AS ttft_ms_mean,
    quantile_disc(ttft_ms,  0.50)   AS ttft_ms_p50,
    quantile_disc(ttft_ms,  0.90)   AS ttft_ms_p90,
    quantile_disc(ttft_ms,  0.99)   AS ttft_ms_p99,
    avg(tpot_ms)                    AS tpot_ms_mean,
    quantile_disc(tpot_ms,  0.50)   AS tpot_ms_p50,
    quantile_disc(tpot_ms,  0.90)   AS tpot_ms_p90,
    avg(total_ms)                   AS total_ms_mean,
    quantile_disc(total_ms, 0.50)   AS total_ms_p50,
    quantile_disc(total_ms, 0.90)   AS total_ms_p90,
    quantile_disc(total_ms, 0.99)   AS total_ms_p99,
    avg(output_tokens_per_s)        AS output_tokens_per_s_mean,
    sum(prompt_tokens)              AS prompt_tokens,
    sum(completion_tokens)          AS completion_tokens,
    sum(cached_tokens)              AS cached_tokens,
    sum(cache_creation_tokens)      AS cache_creation_tokens,
    sum(cached_tokens) / nullif(sum(prompt_tokens), 0)      AS cache_hit_rate,
    COALESCE(sum(cost_total), 0)                            AS cost_total,
    sum(cost_total) / nullif(count(*) FILTER (WHERE status = 'ok'), 0) AS cost_per_request,
    sum(cost_total) * 1e6 / nullif(sum(completion_tokens), 0)          AS cost_per_1m_output_tokens
FROM eval_request_costs
GROUP BY run_id, model;
"""

_PRICE_COLUMNS = (
    "model",
    "label",
    "input_per_1m",
    "output_per_1m",
    "cache_read_per_1m",
    "cache_write_per_1m",
    "currency",
)

_REQUEST_COLUMNS = (
    "run_id",
    "model",
    "repetition",
    "sequence",
    "called_at",
    "user_id",
    "request_ts",
    "sheet_row",
    "recorded_model",
    "status",
    "streamed",
    "error_kind",
    "http_status",
    "error_detail",
    "ttft_ms",
    "tpot_ms",
    "total_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "chunk_count",
    "finish_reason",
    "response_model",
    "proxy_reported_cost",
    "tool_calls",
    "response",
    "reasoning",
)

SUMMARY_COLUMNS = (
    "run_id",
    "model",
    "label",
    "currency",
    "requests",
    "succeeded",
    "failed",
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p90",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "tpot_ms_p50",
    "tpot_ms_p90",
    "total_ms_mean",
    "total_ms_p50",
    "total_ms_p90",
    "total_ms_p99",
    "output_tokens_per_s_mean",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_creation_tokens",
    "cache_hit_rate",
    "cost_total",
    "cost_per_request",
    "cost_per_1m_output_tokens",
)


class SummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str
    model: str
    label: str | None
    currency: str | None
    requests: int
    succeeded: int
    failed: int
    ttft_ms_mean: float | None
    ttft_ms_p50: float | None
    ttft_ms_p90: float | None
    ttft_ms_p99: float | None
    tpot_ms_mean: float | None
    tpot_ms_p50: float | None
    tpot_ms_p90: float | None
    total_ms_mean: float | None
    total_ms_p50: float | None
    total_ms_p90: float | None
    total_ms_p99: float | None
    output_tokens_per_s_mean: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cache_creation_tokens: int | None
    cache_hit_rate: float | None
    cost_total: float
    cost_per_request: float | None
    cost_per_1m_output_tokens: float | None


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    started_at: datetime
    base_url: str
    workload: str
    stream_mode: str
    repeat_count: int
    note: str


@dataclass(frozen=True, slots=True)
class EvalStore:
    connection: DuckDBPyConnection

    def replace_prices(self, prices: Sequence[ModelPrice]) -> int:
        self.connection.execute("DELETE FROM model_prices")
        self.connection.executemany(
            f"INSERT INTO model_prices ({', '.join(_PRICE_COLUMNS)}) VALUES ({', '.join('?' * len(_PRICE_COLUMNS))})",
            [
                (
                    price.model,
                    price.label,
                    price.input_per_1m,
                    price.output_per_1m,
                    price.cache_read_per_1m,
                    price.cache_write_per_1m,
                    price.currency,
                )
                for price in prices
            ],
        )
        return len(prices)

    def prices(self, only: Sequence[str] = ()) -> tuple[ModelPrice, ...]:
        rows = self.connection.execute(
            f"SELECT {', '.join(_PRICE_COLUMNS)} FROM model_prices WHERE enabled ORDER BY model"
        ).fetchall()
        selected = tuple(_to_price(row) for row in rows)
        if not only:
            return selected
        by_model = {price.model: price for price in selected}
        return tuple(by_model[model] for model in only if model in by_model)

    def start_run(self, run: RunMetadata) -> None:
        self.connection.execute(
            "INSERT INTO eval_runs (run_id, started_at, base_url, workload, stream_mode, repeat_count, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run.run_id,
                _utc_naive(run.started_at),
                run.base_url,
                run.workload,
                run.stream_mode,
                run.repeat_count,
                run.note,
            ],
        )

    def finish_run(self, run_id: str) -> None:
        self.connection.execute(
            "UPDATE eval_runs SET finished_at = ? WHERE run_id = ?",
            [_utc_naive(datetime.now(tz=timezone.utc)), run_id],
        )

    def record(self, run_id: str, result: EvalResult) -> None:
        self.connection.execute(
            f"INSERT INTO eval_requests ({', '.join(_REQUEST_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_REQUEST_COLUMNS))})",
            list(_to_request_row(run_id, result)),
        )

    def summary(self, run_id: str) -> tuple[SummaryRow, ...]:
        rows = self.connection.execute(
            f"SELECT {', '.join(SUMMARY_COLUMNS)} FROM eval_summary WHERE run_id = ? ORDER BY model",
            [run_id],
        ).fetchall()
        return tuple(SummaryRow.model_validate(dict(zip(SUMMARY_COLUMNS, row, strict=True))) for row in rows)

    def latest_run_id(self) -> str | None:
        row = self.connection.execute("SELECT run_id FROM eval_runs ORDER BY started_at DESC LIMIT 1").fetchone()
        return str(row[0]) if row is not None else None


def new_run_id() -> str:
    return f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@contextmanager
def open_store(path: Path) -> Iterator[EvalStore]:
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.execute(SCHEMA)
        yield EvalStore(connection=connection)
    finally:
        connection.close()


def _utc_naive(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def _to_price(row: Sequence[object]) -> ModelPrice:
    return ModelPrice(
        model=str(row[0]),
        label=str(row[1]),
        input_per_1m=float(row[2]),  # pyright: ignore[reportArgumentType]  # NOT NULL in the schema
        output_per_1m=float(row[3]),  # pyright: ignore[reportArgumentType]  # NOT NULL in the schema
        cache_read_per_1m=float(row[4]) if row[4] is not None else None,  # pyright: ignore[reportArgumentType]
        cache_write_per_1m=float(row[5]) if row[5] is not None else None,  # pyright: ignore[reportArgumentType]
        currency=str(row[6]),
    )


def _to_request_row(run_id: str, result: EvalResult) -> tuple[object, ...]:
    outcome = result.outcome
    succeeded = outcome if isinstance(outcome, CallSucceeded) else None
    failed = outcome if isinstance(outcome, CallFailed) else None
    usage = succeeded.usage if succeeded is not None else None
    return (
        run_id,
        result.price.model,
        result.repetition,
        result.sequence,
        _utc_naive(result.called_at),
        result.row.user_id,
        _utc_naive(result.row.timestamp),
        result.row.sheet_row,
        result.row.recorded_model,
        "ok" if succeeded is not None else "failed",
        succeeded.streamed if succeeded is not None else None,
        failed.kind if failed is not None else None,
        failed.status_code if failed is not None else None,
        failed.detail if failed is not None else None,
        succeeded.ttft_ms if succeeded is not None else None,
        succeeded.tpot_ms if succeeded is not None else None,
        outcome.total_ms,
        usage.prompt_tokens if usage is not None else None,
        usage.completion_tokens if usage is not None else None,
        usage.total_tokens if usage is not None else None,
        usage.cached_tokens if usage is not None else None,
        usage.cache_creation_tokens if usage is not None else None,
        usage.reasoning_tokens if usage is not None else None,
        succeeded.chunk_count if succeeded is not None else None,
        succeeded.finish_reason if succeeded is not None else None,
        succeeded.response_model if succeeded is not None else None,
        succeeded.proxy_reported_cost if succeeded is not None else None,
        succeeded.tool_calls if succeeded is not None else None,
        succeeded.content if succeeded is not None else None,
        succeeded.reasoning if succeeded is not None else None,
    )
