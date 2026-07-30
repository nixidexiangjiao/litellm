# model_eval

Replays recorded chat-completions traffic through a running LiteLLM proxy, once per model you
want to compare, and reports latency, tokens and money side by side. The point is price/performance:
run the same real workload against the same model at two vendors, or against two different models,
and see what each one costs and how fast it answers

Every request is sent verbatim except for the `model` field, so tools, response formats, sampling
parameters and vendor-specific extras in the recording are preserved

## Install

The only dependency beyond what LiteLLM already needs is `openpyxl`

```bash
uv pip install openpyxl==3.1.5
```

## Input: the workload sheet

Positional columns, one request per row, header row optional (a first row whose third cell is not
JSON is treated as a header and skipped)

| column | meaning |
| --- | --- |
| A | user id |
| B | request timestamp |
| C | the recorded request body, JSON, OpenAI `/v1/chat/completions` shape |

Timestamps may be real Excel date cells, `2026-01-30 09:15:00`, ISO 8601, or a unix epoch in
seconds or milliseconds. Rows are replayed sorted by user id, then timestamp, then sheet position,
so a user's conversation is replayed in the order it happened; that is what makes the prompt-cache
columns mean anything

Rows that cannot be parsed are reported as warnings and skipped rather than aborting the run

## Input: the price sheet

Header-driven, one row per proxy model. It doubles as the list of models to evaluate, so a model
with no row here is never called (unless you pass `--model`, which selects a subset of the sheet)

| column | required | meaning |
| --- | --- | --- |
| `model` | yes | model name as configured in the proxy, e.g. `vendor-a-deepseek-v3` |
| `input_per_1m` | yes | price per 1M uncached input tokens |
| `output_per_1m` | yes | price per 1M output tokens |
| `label` | no | display name in the report; defaults to `model` |
| `cache_read_per_1m` | no | price per 1M cache-hit input tokens; defaults to `input_per_1m` |
| `cache_write_per_1m` | no | price per 1M cache-write tokens; defaults to `input_per_1m` |
| `currency` | no | label only, no conversion is done; defaults to `USD` |

Costs are never mixed across currencies; each model is totalled in its own

Both sheets have a starter workbook:

```bash
python -m model_eval --write-templates ./eval-inputs
```

## Running

From the `scripts/` directory, against a proxy you already have configured with each vendor's key:

```bash
python -m model_eval \
  --workload ./eval-inputs/workload.xlsx \
  --pricing ./eval-inputs/pricing.xlsx \
  --base-url http://localhost:4000 \
  --api-key sk-1234 \
  --output ./eval-2026-01-30.xlsx
```

Useful flags:

- `--model NAME` (repeatable) evaluate only these models out of the price sheet
- `--limit N` replay only the first N ordered requests, for a smoke run
- `--repeat N` replay the whole workload N times per model, to get a distribution worth quoting
- `--stream-mode {as_recorded,force_stream,force_non_stream}`; TTFT and TPOT only exist for
  streamed requests, so force streaming if the recordings are blocking and you care about them
- `--sleep S` pause between requests when a vendor rate-limits you
- `--dry-run` print the plan and exit

Requests are issued one at a time on purpose. Firing them concurrently would destroy the
prompt-cache behaviour the cached-token and cost columns are measuring

## Output

Three things:

1. a console table, one line per model
2. `--output` xlsx with a `summary` sheet (one row per model) and a `requests` sheet (one row per
   call, including the response text, truncated to Excel's cell limit)
3. `--jsonl` trace, written and flushed as each call returns, holding the untruncated response.
   A run killed with Ctrl-C still writes a report for the calls that finished

## What the numbers mean

- **ttft_ms**: time from sending the request to the first chunk carrying generated output.
  The `{"role": "assistant"}` opener that most providers send first is not counted as a token
- **tpot_ms**: `(total_ms - ttft_ms) / (completion_tokens - 1)`, the mean time per output token
  after the first. Undefined for answers of one token
- **total_ms**: request start to the stream closing, or to the response arriving for blocking calls
- **output_tokens_per_s**: `completion_tokens / total_ms`, end to end, so it includes the wait
  for the first token; `tpot_ms` is the number to use for pure generation speed
- **cached_tokens** / **cache_creation_tokens**: subsets of `prompt_tokens`, as LiteLLM normalises
  them across providers. Cost is therefore
  `(prompt - cached - cache_creation) * input + cached * cache_read + cache_creation * cache_write
  + completion * output`, all per 1M
- **proxy_reported_cost**: the proxy's own `x-litellm-response-cost` header, for a cross-check
  against the price sheet. It reflects LiteLLM's built-in price map, which is why the sheet exists

Streaming requests get `stream_options.include_usage` forced on, otherwise providers return no
usage block and every token and cost column would be empty

Failed calls are recorded, not raised: the row keeps its `error_kind`, `http_status` and
`error_detail`, and is excluded from the latency and cost aggregates so one vendor's 429s cannot
make it look cheap
