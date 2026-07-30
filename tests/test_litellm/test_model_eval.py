"""Tests for `scripts/model_eval`, the recorded-traffic replay used to compare models.

The numbers this tool prints are the whole point of it, so the cases below pin
down the measurement (TTFT, TPOT), the request rewriting (model swap, forced
`include_usage`), the replay ordering, and the DuckDB views that turn measured
tokens into money. A regression in any of those silently produces a
plausible-looking but wrong comparison.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_eval.client import (  # noqa: E402
    CallFailed,
    CallSucceeded,
    ProxyTarget,
    StreamChunk,
    TimedChunk,
    call_model,
    first_token_ms,
    prepare_body,
    summarise_stream,
    tpot_ms,
)
from model_eval.pricing import ModelPrice, load_price_table  # noqa: E402
from model_eval.report import export_workbook, format_console_table  # noqa: E402
from model_eval.runner import EvalResult, run_evaluation  # noqa: E402
from model_eval.storage import EvalStore, RunMetadata, open_store  # noqa: E402
from model_eval.templates import write_pricing_template, write_workload_template  # noqa: E402
from model_eval.workload import WorkloadRow, load_workload, order_rows  # noqa: E402

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("duckdb")

TARGET = ProxyTarget(base_url="http://localhost:4000", api_key="sk-test", timeout_s=30.0)


def _row(user_id: str, timestamp: str, sheet_row: int = 1, stream: bool = True) -> WorkloadRow:
    return WorkloadRow(
        sheet_row=sheet_row,
        user_id=user_id,
        timestamp=datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc),
        body={"model": "recorded", "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
    )


def _price(model: str = "vendor-a", **overrides: float | str | None) -> ModelPrice:
    defaults: dict[str, float | str | None] = {
        "label": model,
        "input_per_1m": 10.0,
        "output_per_1m": 30.0,
        "cache_read_per_1m": 1.0,
        "cache_write_per_1m": 12.5,
        "currency": "USD",
    }
    return ModelPrice(model=model, **{**defaults, **overrides})  # type: ignore[arg-type]  # checked by ModelPrice


def _chunk(elapsed_s: float, payload: dict[str, object]) -> TimedChunk:
    return TimedChunk(elapsed_s=elapsed_s, chunk=StreamChunk.model_validate(payload))


def _sse(*payloads: dict[str, object]) -> bytes:
    body = "".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads)
    return (body + "data: [DONE]\n\n").encode()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[EvalStore]:
    with open_store(tmp_path / "eval.duckdb") as opened:
        yield opened


def _record_run(
    store: EvalStore,
    prices: tuple[ModelPrice, ...],
    handler,
    rows: tuple[WorkloadRow, ...] = (),
    repeat: int = 1,
    stream_mode: str = "as_recorded",
) -> str:
    store.replace_prices(prices)
    run = RunMetadata(
        run_id="run-1",
        started_at=datetime.now(tz=timezone.utc),
        base_url="http://localhost:4000",
        workload="workload.xlsx",
        stream_mode=stream_mode,
        repeat_count=repeat,
        note="",
    )
    store.start_run(run)
    with _client(handler) as client:
        for result in run_evaluation(
            client=client,
            target=TARGET,
            prices=prices,
            rows=rows or (_row("u", "2026-01-01T10:00:00"),),
            stream_mode=stream_mode,  # pyright: ignore[reportArgumentType]  # constrained by the callers
            repeat=repeat,
        ):
            store.record(run.run_id, result)
    store.finish_run(run.run_id)
    return run.run_id


def _streaming_handler(prompt_tokens: int = 100, completion_tokens: int = 10, cached: int = 0, created: int = 0):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached, "cache_creation_tokens": created},
                    },
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    return handler


class TestOrdering:
    def test_orders_by_user_then_timestamp_not_by_sheet_position(self):
        rows = (
            _row("bob", "2026-01-01T10:00:00", sheet_row=1),
            _row("alice", "2026-01-01T12:00:00", sheet_row=2),
            _row("bob", "2026-01-01T09:00:00", sheet_row=3),
            _row("alice", "2026-01-01T08:00:00", sheet_row=4),
        )

        ordered = order_rows(rows)

        assert [(row.user_id, row.sheet_row) for row in ordered] == [
            ("alice", 4),
            ("alice", 2),
            ("bob", 3),
            ("bob", 1),
        ]

    def test_equal_timestamps_keep_the_recorded_sheet_order(self):
        rows = (
            _row("alice", "2026-01-01T10:00:00", sheet_row=7),
            _row("alice", "2026-01-01T10:00:00", sheet_row=3),
        )

        assert [row.sheet_row for row in order_rows(rows)] == [3, 7]


class TestWorkloadLoading:
    def _write(self, path: Path, rows: list[list[object]]) -> Path:
        workbook = openpyxl.Workbook()
        sheet = workbook.worksheets[0]
        for row in rows:
            sheet.append(row)
        workbook.save(path)
        return path

    def test_reads_positional_columns_skips_header_and_flags_bad_rows(self, tmp_path: Path):
        body = json.dumps({"model": "recorded", "messages": [{"role": "user", "content": "hi"}]})
        path = self._write(
            tmp_path / "workload.xlsx",
            [
                ["userId", "time", "body"],
                ["u-1", "2026-01-01 10:00:00", body],
                ["u-2", "2026-01-01 11:00:00", "not json"],
                ["u-3", "nonsense-date", body],
                [None, None, None],
                ["u-4", 1767261600, body],
            ],
        )

        workload = load_workload(path)

        assert [row.user_id for row in workload.rows] == ["u-1", "u-4"]
        assert workload.rows[0].timestamp == datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        assert workload.rows[1].timestamp == datetime.fromtimestamp(1767261600, tz=timezone.utc)
        assert [problem.sheet_row for problem in workload.problems] == [3, 4]

    def test_keeps_the_first_row_when_it_is_already_data(self, tmp_path: Path):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
        path = self._write(tmp_path / "workload.xlsx", [["u-1", "2026-01-01 10:00:00", body]])

        assert len(load_workload(path).rows) == 1

    def test_generated_templates_are_loadable_and_demo_ready(self, tmp_path: Path):
        write_workload_template(tmp_path / "w.xlsx")
        write_pricing_template(tmp_path / "p.xlsx")

        workload = load_workload(tmp_path / "w.xlsx")
        prices = load_price_table(tmp_path / "p.xlsx").prices

        assert len({row.user_id for row in workload.rows}) > 1
        assert "mock-fast" in {price.model for price in prices}


class TestBodyRewrite:
    def test_swaps_the_model_and_forces_usage_without_touching_other_fields(self):
        body = {
            "model": "recorded-model",
            "stream": True,
            "stream_options": {"chunk_size": 4},
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "vendor_specific_flag": True,
        }

        prepared = prepare_body(body, "vendor-b", "as_recorded")

        assert prepared["model"] == "vendor-b"
        assert prepared["stream_options"] == {"chunk_size": 4, "include_usage": True}
        assert prepared["temperature"] == 0.2
        assert prepared["vendor_specific_flag"] is True
        assert body["model"] == "recorded-model"

    def test_non_streaming_replay_carries_no_stream_keys(self):
        body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "messages": []}

        prepared = prepare_body(body, "vendor-b", "force_non_stream")

        assert "stream" not in prepared
        assert "stream_options" not in prepared

    def test_as_recorded_leaves_a_blocking_request_blocking(self):
        prepared = prepare_body({"model": "m", "messages": []}, "vendor-b", "as_recorded")

        assert "stream" not in prepared

    def test_force_stream_turns_a_blocking_recording_into_a_stream(self):
        prepared = prepare_body({"model": "m", "messages": []}, "vendor-b", "force_stream")

        assert prepared["stream"] is True
        assert prepared["stream_options"] == {"include_usage": True}


class TestStreamMetrics:
    def test_ttft_ignores_the_role_only_opening_chunk(self):
        chunks = (
            _chunk(0.10, {"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            _chunk(0.50, {"choices": [{"delta": {"content": "he"}}]}),
            _chunk(0.60, {"choices": [{"delta": {"content": "llo"}}]}),
        )

        assert first_token_ms(chunks) == pytest.approx(500.0)

    def test_ttft_counts_reasoning_and_tool_call_deltas_as_output(self):
        reasoning = (_chunk(0.2, {"choices": [{"delta": {"reasoning_content": "hmm"}}]}),)
        tool_call = (_chunk(0.3, {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1"}]}}]}),)

        assert first_token_ms(reasoning) == pytest.approx(200.0)
        assert first_token_ms(tool_call) == pytest.approx(300.0)

    def test_tpot_divides_the_post_first_token_time_by_the_remaining_tokens(self):
        result = summarise_stream(
            (_chunk(0.2, {"choices": [{"delta": {"content": "x"}}], "usage": {"completion_tokens": 5}}),),
            total_ms=1000.0,
        )
        assert isinstance(result, CallSucceeded)

        assert result.ttft_ms == pytest.approx(200.0)
        assert result.tpot_ms == pytest.approx(200.0)

    def test_tpot_is_unavailable_for_a_single_token_answer(self):
        result = summarise_stream(
            (_chunk(0.2, {"choices": [{"delta": {"content": "x"}}], "usage": {"completion_tokens": 1}}),),
            total_ms=1000.0,
        )
        assert isinstance(result, CallSucceeded)

        assert result.tpot_ms is None

    def test_tpot_needs_both_a_first_token_and_a_usage_block(self):
        assert tpot_ms(1000.0, None, None) is None

    def test_buffers_content_usage_cache_tokens_and_finish_reason(self):
        chunks = (
            _chunk(0.1, {"model": "vendor-a", "choices": [{"delta": {"role": "assistant"}}]}),
            _chunk(0.2, {"choices": [{"delta": {"content": "Hello "}}]}),
            _chunk(0.3, {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}),
            _chunk(
                0.4,
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 2,
                        "total_tokens": 1002,
                        "prompt_tokens_details": {"cached_tokens": 800, "cache_creation_tokens": 100},
                        "completion_tokens_details": {"reasoning_tokens": 7},
                    },
                },
            ),
        )

        result = summarise_stream(chunks, total_ms=400.0)
        assert isinstance(result, CallSucceeded)

        assert result.content == "Hello world"
        assert result.finish_reason == "stop"
        assert result.response_model == "vendor-a"
        assert result.chunk_count == 4
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (1000, 2)
        assert (result.usage.cached_tokens, result.usage.cache_creation_tokens) == (800, 100)
        assert result.usage.reasoning_tokens == 7

    def test_streamed_tool_call_fragments_are_reassembled(self):
        chunks = (
            _chunk(
                0.1, {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f"}}]}}]}
            ),
            _chunk(0.2, {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}}]}),
            _chunk(0.3, {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]}),
        )

        result = summarise_stream(chunks, total_ms=300.0)
        assert isinstance(result, CallSucceeded)

        assert json.loads(result.tool_calls) == [{"id": "c1", "name": "f", "arguments": '{"a":1}'}]

    def test_an_empty_stream_is_a_failure_not_a_zero_token_success(self):
        result = summarise_stream((), total_ms=120.0)

        assert isinstance(result, CallFailed)
        assert result.kind == "malformed_response"


class TestHttpCalls:
    def test_streaming_call_measures_a_real_response(self):
        sent: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return _streaming_handler(prompt_tokens=12, completion_tokens=3)(request)

        with _client(handler) as client:
            result = call_model(client, TARGET, "vendor-a", _row("u", "2026-01-01T00:00:00").body, "as_recorded")

        assert isinstance(result, CallSucceeded)
        assert result.streamed is True
        assert result.content == "ok"
        assert result.usage is not None and result.usage.completion_tokens == 3
        assert sent[0]["model"] == "vendor-a"
        assert sent[0]["stream_options"] == {"include_usage": True}

    def test_blocking_call_reads_usage_and_the_proxy_cost_header(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "vendor-a",
                    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                },
                headers={"x-litellm-response-cost": "0.000123"},
            )

        with _client(handler) as client:
            result = call_model(client, TARGET, "vendor-a", {"messages": []}, "force_non_stream")

        assert isinstance(result, CallSucceeded)
        assert result.streamed is False
        assert result.ttft_ms is None
        assert result.content == "hello"
        assert result.proxy_reported_cost == pytest.approx(0.000123)

    def test_content_blocks_are_flattened_into_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": [{"type": "text", "text": "block"}]}}], "usage": {}},
            )

        with _client(handler) as client:
            result = call_model(client, TARGET, "vendor-a", {"messages": []}, "force_non_stream")

        assert isinstance(result, CallSucceeded)
        assert result.content == "block"

    def test_a_rate_limit_is_recorded_as_a_failure_rather_than_raised(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "slow down"}})

        with _client(handler) as client:
            result = call_model(client, TARGET, "vendor-a", {"messages": []}, "force_non_stream")

        assert isinstance(result, CallFailed)
        assert (result.kind, result.status_code) == ("http_status", 429)
        assert "slow down" in result.detail

    def test_a_streaming_error_status_is_recorded_with_its_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "upstream exploded"})

        with _client(handler) as client:
            result = call_model(client, TARGET, "vendor-a", {"messages": [], "stream": True}, "as_recorded")

        assert isinstance(result, CallFailed)
        assert (result.kind, result.status_code) == ("http_status", 500)
        assert "upstream exploded" in result.detail

    def test_a_transport_error_is_recorded_as_a_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with _client(handler) as client:
            result = call_model(client, TARGET, "vendor-a", {"messages": []}, "force_non_stream")

        assert isinstance(result, CallFailed)
        assert result.kind == "transport"

    def test_no_api_key_means_no_authorization_header(self):
        seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("authorization"))
            return httpx.Response(200, json={"choices": [], "usage": {}})

        anonymous = ProxyTarget(base_url="http://localhost:4000", api_key="", timeout_s=5.0)
        with _client(handler) as client:
            call_model(client, anonymous, "vendor-a", {"messages": []}, "force_non_stream")

        assert seen == [None]


class TestReplayOrder:
    def test_every_model_replays_every_request_in_order(self):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(json.loads(request.content)["model"]))
            return _streaming_handler()(request)

        rows = (_row("bob", "2026-01-01T10:00:00", 1), _row("alice", "2026-01-01T09:00:00", 2))
        with _client(handler) as client:
            results = tuple(
                run_evaluation(
                    client=client,
                    target=TARGET,
                    prices=(_price("vendor-a"), _price("vendor-b")),
                    rows=order_rows(rows),
                    stream_mode="as_recorded",
                    repeat=2,
                )
            )

        assert calls == ["vendor-a"] * 4 + ["vendor-b"] * 4
        assert [(r.price.model, r.repetition, r.row.user_id) for r in results[:4]] == [
            ("vendor-a", 1, "alice"),
            ("vendor-a", 1, "bob"),
            ("vendor-a", 2, "alice"),
            ("vendor-a", 2, "bob"),
        ]


class TestPriceStorage:
    def test_prices_round_trip_including_absent_cache_columns(self, store: EvalStore):
        store.replace_prices((_price("vendor-a"), _price("vendor-b", cache_read_per_1m=None, currency="CNY")))

        prices = store.prices()

        assert [price.model for price in prices] == ["vendor-a", "vendor-b"]
        assert prices[0].cache_read_per_1m == 1.0
        assert prices[1].cache_read_per_1m is None
        assert prices[1].currency == "CNY"

    def test_importing_again_replaces_the_table_rather_than_appending(self, store: EvalStore):
        store.replace_prices((_price("vendor-a"), _price("vendor-b")))
        store.replace_prices((_price("vendor-c"),))

        assert [price.model for price in store.prices()] == ["vendor-c"]

    def test_selecting_models_keeps_the_requested_order_and_drops_unknown_ones(self, store: EvalStore):
        store.replace_prices((_price("vendor-a"), _price("vendor-b")))

        assert [price.model for price in store.prices(("vendor-b", "nope", "vendor-a"))] == ["vendor-b", "vendor-a"]

    def test_a_disabled_row_is_never_evaluated(self, store: EvalStore):
        store.replace_prices((_price("vendor-a"), _price("vendor-b")))
        store.connection.execute("UPDATE model_prices SET enabled = FALSE WHERE model = 'vendor-b'")

        assert [price.model for price in store.prices()] == ["vendor-a"]

    def test_excel_import_feeds_the_table(self, store: EvalStore, tmp_path: Path):
        workbook = openpyxl.Workbook()
        sheet = workbook.worksheets[0]
        sheet.append(["model", "input_per_1m", "output_per_1m", "label", "currency"])
        sheet.append(["vendor-a", 1, 2, None, None])
        sheet.append(["vendor-b", 1, 2, "vendor B", "cny"])
        workbook.save(tmp_path / "p.xlsx")

        store.replace_prices(load_price_table(tmp_path / "p.xlsx").prices)
        prices = store.prices()

        assert (prices[0].label, prices[0].currency) == ("vendor-a", "USD")
        assert (prices[1].label, prices[1].currency) == ("vendor B", "CNY")
        assert prices[0].cache_read_per_1m is None

    def test_a_price_sheet_without_the_required_columns_is_rejected(self, tmp_path: Path):
        workbook = openpyxl.Workbook()
        workbook.worksheets[0].append(["model", "price"])
        workbook.save(tmp_path / "p.xlsx")

        table = load_price_table(tmp_path / "p.xlsx")

        assert table.prices == ()
        assert "input_per_1m" in table.problems[0].reason


class TestCostView:
    def _cost(self, store: EvalStore, run_id: str) -> dict[str, float]:
        row = store.connection.execute(
            "SELECT cost_uncached_input, cost_cached_input, cost_cache_write, cost_output, cost_total "
            "FROM eval_request_costs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        assert row is not None
        return dict(zip(("uncached", "cached", "write", "output", "total"), row, strict=True))

    def test_cached_and_cache_write_tokens_are_billed_at_their_own_rate(self, store: EvalStore):
        run_id = _record_run(
            store,
            (_price(),),
            _streaming_handler(prompt_tokens=1000, completion_tokens=50, cached=800, created=100),
        )

        cost = self._cost(store, run_id)

        assert cost["uncached"] == pytest.approx(100 * 10.0 / 1e6)
        assert cost["cached"] == pytest.approx(800 * 1.0 / 1e6)
        assert cost["write"] == pytest.approx(100 * 12.5 / 1e6)
        assert cost["output"] == pytest.approx(50 * 30.0 / 1e6)
        assert cost["total"] == pytest.approx(0.00455)

    def test_a_full_cache_hit_never_double_charges_the_prompt(self, store: EvalStore):
        run_id = _record_run(
            store,
            (_price(),),
            _streaming_handler(prompt_tokens=500, completion_tokens=0, cached=500),
        )

        cost = self._cost(store, run_id)

        assert cost["uncached"] == 0.0
        assert cost["total"] == pytest.approx(500 * 1.0 / 1e6)

    def test_usage_claiming_more_cached_tokens_than_prompt_tokens_cannot_go_negative(self, store: EvalStore):
        run_id = _record_run(
            store,
            (_price(),),
            _streaming_handler(prompt_tokens=100, completion_tokens=0, cached=500),
        )

        cost = self._cost(store, run_id)

        assert cost["uncached"] == 0.0
        assert cost["total"] == pytest.approx(100 * 1.0 / 1e6)

    def test_a_blank_cache_price_falls_back_to_the_input_price(self, store: EvalStore):
        run_id = _record_run(
            store,
            (_price(cache_read_per_1m=None),),
            _streaming_handler(prompt_tokens=1000, completion_tokens=0, cached=400),
        )

        assert self._cost(store, run_id)["total"] == pytest.approx(1000 * 10.0 / 1e6)

    def test_correcting_a_price_revalues_a_run_that_already_finished(self, store: EvalStore):
        run_id = _record_run(store, (_price(output_per_1m=30.0),), _streaming_handler(completion_tokens=10))
        before = store.summary(run_id)[0].cost_total

        store.connection.execute("UPDATE model_prices SET output_per_1m = 60.0 WHERE model = 'vendor-a'")
        after = store.summary(run_id)[0].cost_total

        assert after == pytest.approx(before + 10 * 30.0 / 1e6)

    def test_a_failed_call_has_no_cost_and_no_invented_tokens(self, store: EvalStore):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "unavailable"})

        run_id = _record_run(store, (_price(),), handler, stream_mode="force_non_stream")

        row = store.connection.execute(
            "SELECT status, error_kind, http_status, prompt_tokens, cost_total FROM eval_request_costs WHERE run_id = ?",
            [run_id],
        ).fetchone()

        assert row == ("failed", "http_status", 503, None, None)


class TestSummary:
    def test_reports_per_model_totals_and_unit_cost(self, store: EvalStore):
        run_id = _record_run(
            store,
            (_price("vendor-a"), _price("vendor-b", output_per_1m=60.0)),
            _streaming_handler(prompt_tokens=100, completion_tokens=10),
            repeat=3,
        )

        summaries = store.summary(run_id)

        assert [summary.model for summary in summaries] == ["vendor-a", "vendor-b"]
        assert [summary.requests for summary in summaries] == [3, 3]
        assert summaries[0].completion_tokens == 30
        assert summaries[1].cost_total == pytest.approx(summaries[0].cost_total + 30 * 30.0 / 1e6)
        assert summaries[0].cost_per_request == pytest.approx(summaries[0].cost_total / 3)
        assert summaries[0].cost_per_1m_output_tokens == pytest.approx(summaries[0].cost_total * 1e6 / 30)

    def test_cache_hit_rate_is_the_cached_share_of_the_prompt(self, store: EvalStore):
        run_id = _record_run(
            store,
            (_price(),),
            _streaming_handler(prompt_tokens=200, completion_tokens=5, cached=150),
        )

        assert store.summary(run_id)[0].cache_hit_rate == pytest.approx(0.75)

    def test_a_failed_run_counts_the_failure_and_totals_nothing(self, store: EvalStore):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "unavailable"})

        run_id = _record_run(store, (_price(),), handler, stream_mode="force_non_stream")
        summary = store.summary(run_id)[0]

        assert (summary.succeeded, summary.failed) == (0, 1)
        assert summary.cost_total == 0.0
        assert summary.ttft_ms_p50 is None
        assert summary.cost_per_request is None

    def test_percentiles_are_nearest_rank_so_small_samples_stay_observed_values(self, store: EvalStore):
        store.replace_prices((_price(),))
        store.start_run(
            RunMetadata(
                run_id="run-p",
                started_at=datetime.now(tz=timezone.utc),
                base_url="",
                workload="",
                stream_mode="as_recorded",
                repeat_count=1,
                note="",
            )
        )
        for sequence, latency in enumerate((10.0, 20.0, 30.0, 40.0), start=1):
            store.record("run-p", _synthetic_result(sequence, latency))

        summary = store.summary("run-p")[0]

        assert (summary.total_ms_p50, summary.total_ms_p90, summary.total_ms_p99) == (20.0, 40.0, 40.0)
        assert summary.total_ms_mean == pytest.approx(25.0)

    def test_the_latest_run_is_the_one_reported_by_default(self, store: EvalStore):
        _record_run(store, (_price(),), _streaming_handler())

        assert store.latest_run_id() == "run-1"


class TestOutput:
    def test_workbook_exports_the_summary_and_the_priced_requests(self, store: EvalStore, tmp_path: Path):
        run_id = _record_run(
            store,
            (_price(),),
            _streaming_handler(prompt_tokens=100, completion_tokens=10),
            repeat=2,
        )

        export_workbook(store, tmp_path / "report.xlsx", run_id)
        workbook = openpyxl.load_workbook(tmp_path / "report.xlsx")

        assert workbook.sheetnames == ["summary", "requests"]
        requests_sheet = workbook["requests"]
        headers = [cell.value for cell in requests_sheet[1]]
        assert requests_sheet.max_row == 3
        assert {"ttft_ms", "tpot_ms", "total_ms", "cached_tokens", "cost_total", "response"} <= set(headers)
        first_row = dict(zip(headers, [cell.value for cell in requests_sheet[2]], strict=True))
        assert first_row["status"] == "ok"
        assert first_row["response"] == "ok"
        assert first_row["completion_tokens"] == 10
        assert first_row["cost_total"] == pytest.approx(100 * 10.0 / 1e6 + 10 * 30.0 / 1e6)

    def test_console_table_shows_one_line_per_model_plus_a_header(self, store: EvalStore):
        run_id = _record_run(store, (_price("vendor-a"), _price("vendor-b")), _streaming_handler())

        lines = format_console_table(store.summary(run_id)).splitlines()

        assert len(lines) == 3
        assert "model" in lines[0]
        assert "vendor-a" in lines[1] and "vendor-b" in lines[2]


def _synthetic_result(sequence: int, total_ms: float) -> EvalResult:
    return EvalResult(
        price=_price(),
        repetition=1,
        sequence=sequence,
        called_at=datetime.now(tz=timezone.utc),
        row=_row("u", "2026-01-01T10:00:00", sheet_row=sequence),
        outcome=CallSucceeded(
            streamed=True,
            total_ms=total_ms,
            ttft_ms=total_ms / 2,
            tpot_ms=None,
            usage=None,
            content="",
            reasoning="",
            tool_calls="",
            finish_reason="stop",
            chunk_count=1,
            response_model="vendor-a",
            proxy_reported_cost=None,
        ),
    )
