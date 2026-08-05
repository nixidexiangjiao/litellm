"""Issue one recorded request against the LiteLLM proxy and time it.

Bodies are replayed verbatim apart from the model swap, so this talks raw HTTP
instead of going through an SDK that would normalise or drop unknown fields.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar, assert_never, cast

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from .workload import JSONObject

StreamMode = Literal["as_recorded", "force_stream", "force_non_stream"]
FailureKind = Literal["http_status", "transport", "malformed_response"]

RESPONSE_COST_HEADER = "x-litellm-response-cost"


@dataclass(frozen=True, slots=True)
class ProxyTarget:
    base_url: str
    api_key: str
    timeout_s: float
    retries: int = 0
    retry_delay_s: float = 3.0
    retry_sleep: Callable[[float], None] = time.sleep

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def headers(self) -> dict[str, str]:
        """An empty key means the proxy runs without auth, so send no bearer at all."""
        auth = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return {"Content-Type": "application/json", **auth}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True, slots=True)
class CallSucceeded:
    streamed: bool
    total_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    usage: TokenUsage | None
    content: str
    reasoning: str
    tool_calls: str
    finish_reason: str
    chunk_count: int
    response_model: str
    proxy_reported_cost: float | None


@dataclass(frozen=True, slots=True)
class CallFailed:
    kind: FailureKind
    total_ms: float
    status_code: int | None
    detail: str
    retry_after_s: float | None = None


CallOutcome = CallSucceeded | CallFailed


class WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FunctionCall(WireModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(WireModel):
    index: int | None = None
    id: str | None = None
    function: FunctionCall | None = None


class MessageContent(WireModel):
    """The parts of a ``delta`` / ``message`` object this evaluation reads."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None

    @field_validator("content", "reasoning_content", mode="before")
    @classmethod
    def _flatten_content_blocks(cls, value: object) -> object:
        """Some providers answer with content blocks rather than a bare string."""
        if not isinstance(value, list):
            return value
        blocks = cast("list[object]", value)
        return "".join(str(block["text"]) for block in blocks if isinstance(block, Mapping) and "text" in block)

    @property
    def carries_output(self) -> bool:
        return bool(self.content) or bool(self.reasoning_content) or bool(self.tool_calls)


class PromptTokensDetails(WireModel):
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None


class CompletionTokensDetails(WireModel):
    reasoning_tokens: int | None = None


class Usage(WireModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class StreamChoice(WireModel):
    delta: MessageContent | None = None
    finish_reason: str | None = None


class StreamChunk(WireModel):
    model: str | None = None
    choices: tuple[StreamChoice, ...] = ()
    usage: Usage | None = None


class Choice(WireModel):
    message: MessageContent | None = None
    finish_reason: str | None = None


class Completion(WireModel):
    model: str | None = None
    choices: tuple[Choice, ...] = ()
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class TimedChunk:
    elapsed_s: float
    chunk: StreamChunk


def prepare_body(body: JSONObject, model: str, stream_mode: StreamMode) -> JSONObject:
    """Swap the model and settle the streaming flag, leaving everything else alone."""
    streaming = _resolve_streaming(body.get("stream") is True, stream_mode)
    swapped: JSONObject = {**body, "model": model}
    if not streaming:
        return {key: value for key, value in swapped.items() if key not in ("stream", "stream_options")}
    recorded_options = swapped.get("stream_options")
    options = recorded_options if isinstance(recorded_options, dict) else {}
    return {**swapped, "stream": True, "stream_options": {**options, "include_usage": True}}


def call_model(
    client: httpx.Client,
    target: ProxyTarget,
    model: str,
    body: JSONObject,
    stream_mode: StreamMode,
    clock: Callable[[], float] = time.perf_counter,
) -> CallOutcome:
    payload = prepare_body(body, model, stream_mode)
    for attempt in range(max(1, target.retries + 1)):
        started = clock()
        if payload.get("stream") is True:
            outcome = _call_streaming(client, target, payload, started, clock)
        else:
            outcome = _call_blocking(client, target, payload, started, clock)
        if not _retryable(outcome) or attempt >= target.retries:
            return outcome
        target.retry_sleep(_backoff_seconds(outcome, attempt, target.retry_delay_s))
    return outcome  # unreachable: the loop always returns


_RATE_LIMIT_HINTS = ("rate_limit", "rate limit", "ratelimit", "429", "tpm", "throttl")


def _retryable(outcome: CallOutcome) -> bool:
    """Provider throttling is worth re-trying; other failures replay the same way.

    Some proxies wrap upstream rate limits in a 500 (the detail text then
    carries the provider's rate-limit marker), so sniff those too.
    """
    if not isinstance(outcome, CallFailed):
        return False
    if outcome.status_code in (429, 503):
        return True
    if outcome.status_code == 500:
        detail = (outcome.detail or "").lower()
        return any(hint in detail for hint in _RATE_LIMIT_HINTS)
    return False


def _backoff_seconds(outcome: CallFailed, attempt: int, base_delay: float) -> float:
    """Exponential backoff with jitter, honouring Retry-After when the provider sets it."""
    delay = base_delay * (2**attempt) * random.uniform(0.8, 1.2)
    if outcome.retry_after_s is not None:
        delay = max(delay, outcome.retry_after_s)
    return delay


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _call_blocking(
    client: httpx.Client,
    target: ProxyTarget,
    payload: JSONObject,
    started: float,
    clock: Callable[[], float],
) -> CallOutcome:
    try:
        response = client.post(
            target.chat_completions_url,
            json=payload,
            headers=target.headers,
            timeout=target.timeout_s,
        )
    except httpx.HTTPError as error:
        return CallFailed("transport", _ms(clock() - started), None, f"{type(error).__name__}: {error}")

    total_ms = _ms(clock() - started)
    if response.status_code != httpx.codes.OK:
        return CallFailed("http_status", total_ms, response.status_code, _truncate_error(response.text), retry_after_s=_retry_after(response))

    completion = _validate(Completion, response.text)
    if completion is None:
        return CallFailed("malformed_response", total_ms, response.status_code, _truncate_error(response.text))

    choice = completion.choices[0] if completion.choices else None
    message = choice.message if choice is not None and choice.message is not None else MessageContent()
    return CallSucceeded(
        streamed=False,
        total_ms=total_ms,
        ttft_ms=None,
        tpot_ms=None,
        usage=to_token_usage(completion.usage),
        content=message.content or "",
        reasoning=message.reasoning_content or "",
        tool_calls=render_tool_calls(message.tool_calls),
        finish_reason=(choice.finish_reason or "") if choice is not None else "",
        chunk_count=0,
        response_model=completion.model or "",
        proxy_reported_cost=_reported_cost(response.headers),
    )


def _call_streaming(
    client: httpx.Client,
    target: ProxyTarget,
    payload: JSONObject,
    started: float,
    clock: Callable[[], float],
) -> CallOutcome:
    try:
        with client.stream(
            "POST",
            target.chat_completions_url,
            json=payload,
            headers=target.headers,
            timeout=target.timeout_s,
        ) as response:
            if response.status_code != httpx.codes.OK:
                detail = _truncate_error(response.read().decode("utf-8", errors="replace"))
                return CallFailed("http_status", _ms(clock() - started), response.status_code, detail, retry_after_s=_retry_after(response))
            chunks = tuple(_timed_chunks(response.iter_lines(), started, clock))
            reported_cost = _reported_cost(response.headers)
    except httpx.HTTPError as error:
        return CallFailed("transport", _ms(clock() - started), None, f"{type(error).__name__}: {error}")

    return summarise_stream(chunks, total_ms=_ms(clock() - started), proxy_reported_cost=reported_cost)


def summarise_stream(
    chunks: tuple[TimedChunk, ...],
    total_ms: float,
    proxy_reported_cost: float | None = None,
) -> CallOutcome:
    if not chunks:
        return CallFailed("malformed_response", total_ms, None, "stream closed without any data chunk")

    deltas = tuple(choice.delta for timed in chunks for choice in timed.chunk.choices if choice.delta is not None)
    usage = to_token_usage(next((timed.chunk.usage for timed in reversed(chunks) if timed.chunk.usage), None))
    ttft_ms = first_token_ms(chunks)
    return CallSucceeded(
        streamed=True,
        total_ms=total_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms(total_ms, ttft_ms, usage),
        usage=usage,
        content="".join(delta.content or "" for delta in deltas),
        reasoning="".join(delta.reasoning_content or "" for delta in deltas),
        tool_calls=render_tool_calls(tuple(call for delta in deltas for call in delta.tool_calls or ())),
        finish_reason=next(
            (
                choice.finish_reason
                for timed in reversed(chunks)
                for choice in timed.chunk.choices
                if choice.finish_reason
            ),
            "",
        ),
        chunk_count=len(chunks),
        response_model=next((timed.chunk.model for timed in chunks if timed.chunk.model), ""),
        proxy_reported_cost=proxy_reported_cost,
    )


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload and payload != "[DONE]":
            yield payload


def first_token_ms(chunks: tuple[TimedChunk, ...]) -> float | None:
    """Time to the first chunk carrying generated output.

    The leading ``{"role": "assistant"}`` chunk is not a token, so counting it
    would flatter every provider that sends one.
    """
    return next(
        (
            _ms(timed.elapsed_s)
            for timed in chunks
            if any(choice.delta is not None and choice.delta.carries_output for choice in timed.chunk.choices)
        ),
        None,
    )


def tpot_ms(total_ms: float, ttft_ms: float | None, usage: TokenUsage | None) -> float | None:
    """Mean time per output token after the first one."""
    if ttft_ms is None or usage is None or usage.completion_tokens < 2:
        return None
    return (total_ms - ttft_ms) / (usage.completion_tokens - 1)


def to_token_usage(usage: Usage | None) -> TokenUsage | None:
    if usage is None:
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.total_tokens or (prompt_tokens + completion_tokens),
        cached_tokens=(prompt_details.cached_tokens or 0) if prompt_details is not None else 0,
        cache_creation_tokens=(prompt_details.cache_creation_tokens or 0) if prompt_details is not None else 0,
        reasoning_tokens=(completion_details.reasoning_tokens or 0) if completion_details is not None else 0,
    )


def render_tool_calls(tool_calls: tuple[ToolCall, ...] | None) -> str:
    if not tool_calls:
        return ""
    merged = tuple(
        _merge_tool_call(tuple(call for call in tool_calls if _tool_call_index(call) == index))
        for index in sorted({_tool_call_index(call) for call in tool_calls})
    )
    return json.dumps(merged, ensure_ascii=False)


def _tool_call_index(call: ToolCall) -> int:
    return call.index if call.index is not None else 0


def _merge_tool_call(fragments: tuple[ToolCall, ...]) -> dict[str, str]:
    return {
        "id": next((fragment.id for fragment in fragments if fragment.id), ""),
        "name": next(
            (
                fragment.function.name
                for fragment in fragments
                if fragment.function is not None and fragment.function.name
            ),
            "",
        ),
        "arguments": "".join(
            fragment.function.arguments or "" for fragment in fragments if fragment.function is not None
        ),
    }


def _timed_chunks(lines: Iterable[str], started: float, clock: Callable[[], float]) -> Iterator[TimedChunk]:
    for payload in iter_sse_payloads(lines):
        elapsed = clock() - started
        chunk = _validate(StreamChunk, payload)
        if chunk is not None:
            yield TimedChunk(elapsed_s=elapsed, chunk=chunk)


def _resolve_streaming(recorded: bool, stream_mode: StreamMode) -> bool:
    match stream_mode:
        case "as_recorded":
            return recorded
        case "force_stream":
            return True
        case "force_non_stream":
            return False
        case _:
            assert_never(stream_mode)


def _reported_cost(headers: httpx.Headers) -> float | None:
    raw = headers.get(RESPONSE_COST_HEADER)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


_WireModelT = TypeVar("_WireModelT", bound=WireModel)


def _validate(model: type[_WireModelT], payload: str) -> _WireModelT | None:
    try:
        return model.model_validate_json(payload)
    except ValueError:
        return None


def _truncate_error(text: str, limit: int = 500) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}..."


def _ms(seconds: float) -> float:
    return seconds * 1000.0
