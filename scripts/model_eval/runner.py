"""Replay the workload against every priced model, one request at a time.

The replay is deliberately sequential: prompt-cache hit rates depend on
requests arriving in the recorded order, and firing them concurrently would
make the cached-token and cost columns meaningless.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import httpx

from .client import CallFailed, CallOutcome, CallSucceeded, ProxyTarget, StreamMode, call_model
from .pricing import CostBreakdown, ModelPrice, TokenCounts, compute_cost
from .workload import WorkloadRow


@dataclass(frozen=True, slots=True)
class EvalResult:
    price: ModelPrice
    repetition: int
    sequence: int
    row: WorkloadRow
    outcome: CallOutcome
    cost: CostBreakdown | None


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
    outcome = call_model(client, target, price.model, row.body, stream_mode)
    return EvalResult(
        price=price,
        repetition=repetition,
        sequence=sequence,
        row=row,
        outcome=outcome,
        cost=_cost_of(price, outcome),
    )


def _cost_of(price: ModelPrice, outcome: CallOutcome) -> CostBreakdown | None:
    match outcome:
        case CallFailed():
            return None
        case CallSucceeded(usage=None):
            return None
        case CallSucceeded(usage=usage):
            return compute_cost(
                price,
                TokenCounts(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    cache_creation_tokens=usage.cache_creation_tokens,
                ),
            )
