"""A tiny OpenAI-compatible provider with configurable latency.

It exists so the whole chain (proxy config, replay, DuckDB, report) can be
verified in a fresh sandbox before a single vendor key is added, and so the
timing columns can be checked against numbers we chose. It is a test fixture,
not a load generator: single purpose, standard library only.

    python -m model_eval.mock_provider --port 4123
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

_ANSWER = "The quick brown fox jumps over the lazy dog".split()


class _Handler(BaseHTTPRequestHandler):
    ttft_s: ClassVar[float] = 0.2
    tpot_s: ClassVar[float] = 0.02
    cached_tokens: ClassVar[int] = 64

    def log_message(self, format: str, *args: object) -> None:
        """Keep the sandbox transcript readable; the replay prints its own progress."""

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        model = str(request.get("model") or "mock-model")
        words = tuple(f"{word} " for word in _ANSWER)
        if request.get("stream") is True:
            self._stream(model, words)
        else:
            self._blocking(model, words)

    def _usage(self, words: Sequence[str]) -> dict[str, object]:
        return {
            "prompt_tokens": 120,
            "completion_tokens": len(words),
            "total_tokens": 120 + len(words),
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        }

    def _blocking(self, model: str, words: Sequence[str]) -> None:
        time.sleep(self.ttft_s + self.tpot_s * len(words))
        payload = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(words).strip()},
                        "finish_reason": "stop",
                    }
                ],
                "usage": self._usage(words),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self, model: str, words: Sequence[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._chunk({"model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        time.sleep(self.ttft_s)
        for word in words:
            self._chunk({"model": model, "choices": [{"index": 0, "delta": {"content": word}}]})
            time.sleep(self.tpot_s)
        self._chunk({"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self._chunk({"model": model, "choices": [], "usage": self._usage(words)})
        self._write(b"data: [DONE]\n\n")

    def _chunk(self, payload: dict[str, object]) -> None:
        self._write(f"data: {json.dumps(payload)}\n\n".encode())

    def _write(self, body: bytes) -> None:
        """HTTP/1.0 close-delimited: the connection ending is the end of the stream."""
        self.wfile.write(body)
        self.wfile.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock provider with configurable latency")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4123)
    parser.add_argument("--ttft", type=float, default=0.2, help="seconds before the first content chunk")
    parser.add_argument("--tpot", type=float, default=0.02, help="seconds between content chunks")
    namespace = parser.parse_args(argv)

    _Handler.ttft_s = namespace.ttft
    _Handler.tpot_s = namespace.tpot
    ThreadingHTTPServer((namespace.host, namespace.port), _Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
