"""Replay the workload against every priced model.

Two replay modes:

- Sequential (default): one request at a time, in recorded order. Prompt-cache
  hit rates depend on requests arriving in the recorded order.
- Concurrent (--concurrency N > 1): requests are grouped into per-user chains;
  chains race each other while each user's own requests stay strictly ordered.
  Prompt-cache affinity lives per conversation, so cache metrics stay meaningful
  while wall-clock time drops by roughly the number of parallel chains.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    concurrency: int = 1,
) -> Iterator[EvalResult]:
    for price in prices:
        for repetition in range(1, repeat + 1):
            if concurrency > 1:
                yield from _run_repetition_concurrent(
                    client, target, price, repetition, rows, stream_mode, concurrency, sleep_s, sleep
                )
                continue
            for sequence, row in enumerate(rows, start=1):
                if sleep_s > 0 and not (repetition == 1 and sequence == 1):
                    sleep(sleep_s)
                yield _run_one(client, target, price, repetition, sequence, row, stream_mode)


def _run_repetition_concurrent(
    client: httpx.Client,
    target: ProxyTarget,
    price: ModelPrice,
    repetition: int,
    rows: Sequence[WorkloadRow],
    stream_mode: StreamMode,
    concurrency: int,
    sleep_s: float,
    sleep: Callable[[float], None],
) -> Iterator[EvalResult]:
    """Race per-user chains against each other; each chain stays ordered.

    Results stream back through a single consumer queue, so callers persist them
    from one thread even though the calls themselves run on a worker pool.
    httpx.Client is documented as thread-safe, so sharing it is fine.
    """
    chains: dict[str, list[tuple[int, WorkloadRow]]] = {}
    for sequence, row in enumerate(rows, start=1):
        chains.setdefault(row.user_id, []).append((sequence, row))

    mailbox: queue.Queue[EvalResult | BaseException] = queue.Queue()

    def worker(chain: list[tuple[int, WorkloadRow]]) -> None:
        try:
            for index, (sequence, row) in enumerate(chain):
                if sleep_s > 0 and index > 0:
                    sleep(sleep_s)
                mailbox.put(_run_one(client, target, price, repetition, sequence, row, stream_mode))
        except BaseException as exc:  # surface worker failures on the consumer thread
            mailbox.put(exc)

    total = sum(len(chain) for chain in chains.values())
    consumed = 0
    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(chains)))) as pool:
        for chain in chains.values():
            pool.submit(worker, chain)
        while consumed < total:
            item = mailbox.get()
            if isinstance(item, BaseException):
                raise item
            yield item
            consumed += 1


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
