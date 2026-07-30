"""Load the hand-maintained price table and turn token usage into money.

The price sheet is keyed on the model name as configured in the LiteLLM proxy,
so the same upstream model served by two vendors is two rows. It doubles as the
list of models to evaluate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("model", "input_per_1m", "output_per_1m")
OPTIONAL_COLUMNS = ("label", "cache_read_per_1m", "cache_write_per_1m", "currency")

_TOKENS_PER_UNIT = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    model: str
    label: str
    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float
    cache_write_per_1m: float
    currency: str


@dataclass(frozen=True, slots=True)
class PriceProblem:
    sheet_row: int
    reason: str


@dataclass(frozen=True, slots=True)
class PriceTable:
    prices: tuple[ModelPrice, ...]
    problems: tuple[PriceProblem, ...]

    def get(self, model: str) -> ModelPrice | None:
        return next((price for price in self.prices if price.model == model), None)


@dataclass(frozen=True, slots=True)
class TokenCounts:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cache_creation_tokens: int


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    uncached_input: float
    cached_input: float
    cache_write: float
    output: float
    currency: str

    @property
    def total(self) -> float:
        return self.uncached_input + self.cached_input + self.cache_write + self.output


def load_price_table(path: Path, sheet_name: str | None = None) -> PriceTable:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet_name] if sheet_name is not None else workbook.worksheets[0]
        rows = tuple(worksheet.iter_rows(values_only=True))
    finally:
        workbook.close()

    if not rows:
        return PriceTable((), (PriceProblem(0, "price sheet is empty"),))

    header = tuple(_normalise_header(cell) for cell in rows[0])
    missing = tuple(column for column in REQUIRED_COLUMNS if column not in header)
    if missing:
        return PriceTable(
            (),
            (PriceProblem(1, f"missing required column(s): {', '.join(missing)}; found: {', '.join(header)}"),),
        )

    parsed = tuple(
        _parse_price_row(index, header, cells)
        for index, cells in enumerate(rows[1:], start=2)
        if any(cell is not None and str(cell).strip() != "" for cell in cells)
    )
    return PriceTable(
        prices=tuple(item for item in parsed if isinstance(item, ModelPrice)),
        problems=tuple(item for item in parsed if isinstance(item, PriceProblem)),
    )


def compute_cost(price: ModelPrice, tokens: TokenCounts) -> CostBreakdown:
    """Price one request.

    ``cached_tokens`` and ``cache_creation_tokens`` are subsets of
    ``prompt_tokens`` (that is how LiteLLM normalises every provider), so they
    are billed at their own rate and subtracted from the full-price input.
    """
    billed_cache_read = min(tokens.cached_tokens, tokens.prompt_tokens)
    billed_cache_write = min(tokens.cache_creation_tokens, tokens.prompt_tokens - billed_cache_read)
    uncached = max(tokens.prompt_tokens - billed_cache_read - billed_cache_write, 0)
    return CostBreakdown(
        uncached_input=uncached * price.input_per_1m / _TOKENS_PER_UNIT,
        cached_input=billed_cache_read * price.cache_read_per_1m / _TOKENS_PER_UNIT,
        cache_write=billed_cache_write * price.cache_write_per_1m / _TOKENS_PER_UNIT,
        output=tokens.completion_tokens * price.output_per_1m / _TOKENS_PER_UNIT,
        currency=price.currency,
    )


def _parse_price_row(sheet_row: int, header: Sequence[str], cells: Sequence[object]) -> ModelPrice | PriceProblem:
    values = {name: cells[index] for index, name in enumerate(header) if index < len(cells) and name}

    model = str(values.get("model") or "").strip()
    if not model:
        return PriceProblem(sheet_row, "'model' is empty")

    input_per_1m = _parse_money(values.get("input_per_1m"))
    output_per_1m = _parse_money(values.get("output_per_1m"))
    if input_per_1m is None or output_per_1m is None:
        return PriceProblem(sheet_row, f"'{model}' has a non-numeric input_per_1m/output_per_1m")

    cache_read = _parse_money(values.get("cache_read_per_1m"))
    cache_write = _parse_money(values.get("cache_write_per_1m"))
    label = str(values.get("label") or "").strip() or model
    currency = str(values.get("currency") or "").strip().upper() or "USD"

    return ModelPrice(
        model=model,
        label=label,
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
        cache_read_per_1m=cache_read if cache_read is not None else input_per_1m,
        cache_write_per_1m=cache_write if cache_write is not None else input_per_1m,
        currency=currency,
    )


def _normalise_header(cell: object) -> str:
    return str(cell or "").strip().lower().replace(" ", "_")


def _parse_money(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lstrip("$￥¥€£").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
