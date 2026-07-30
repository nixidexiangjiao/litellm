"""Replay the workload against every priced model, one request at a time.

The replay is deliberately sequential: prompt-cache hit rates depend on
requests arriving in the recorded order, and firing them concurrently would
make the cached-token and cost columns meaningless.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .client import CallOutcome, ProxyTarget, StreamMode, call_model
from .pricing import ModelPrice
from .workload import WorkloadRow


@dataclass(frozen=True, slots=True)
class EvalResult:
    price: ModelPrice
    repetition: int
    sequence: int
    called_at: datetime
    row: WorkloadRow
    outcome: CallOutcome


def run_evaluation(
    client: httpx.Client,
    target: ProxyTarget,
    prices: Sequence[ModelPrice],
    rows: Sequence[WorkloadRow],
    stream_mode: StreamMode,
    repeat: int = 1,
    sleep_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[EvalResult]:
    for price in prices:
        for repetition in range(1, repeat + 1):
            for sequence, row in enumerate(rows, start=1):
                if sleep_s > 0 and not (repetition == 1 and sequence == 1):
                    sleep(sleep_s)
                yield _run_one(client, target, price, repetition, sequence, row, stream_mode)


def _run_one(
    client: httpx.Client,
    target: ProxyTarget,
    price: ModelPrice,
    repetition: int,
    sequence: int,
    row: WorkloadRow,
    stream_mode: StreamMode,
) -> EvalResult:
    called_at = datetime.now(tz=timezone.utc)
    return EvalResult(
        price=price,
        repetition=repetition,
        sequence=sequence,
        called_at=called_at,
        row=row,
        outcome=call_model(client, target, price.model, row.body, stream_mode),
    )
