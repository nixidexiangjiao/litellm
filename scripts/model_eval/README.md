# model_eval

Replays recorded chat-completions traffic through a running LiteLLM proxy, once per model you
want to compare, and stores latency, tokens and cost in DuckDB. The point is price/performance:
run the same real workload against the same model at two vendors, or against two different
models, and see what each one costs and how fast it answers

Every request is sent verbatim except for the `model` field, so tools, response formats, sampling
parameters and vendor-specific extras in the recording are preserved

## One click, in a sandbox

```bash
scripts/model_eval/quickstart.sh demo
```

That creates a workspace (default `./model-eval`, override with `MODEL_EVAL_HOME`), installs
`litellm[proxy]`, `duckdb` and `openpyxl` into a venv inside it, starts a bundled mock provider
plus the proxy, replays a sample workload and prints the report. No vendor key needed, so you can
confirm the whole chain works before spending anything

Then point it at real vendors:

```bash
scripts/model_eval/quickstart.sh down          # stop the mock
$EDITOR model-eval/.env                        # vendor keys
$EDITOR model-eval/config.yaml                 # which vendor serves which model_name
$EDITOR model-eval/prices.xlsx                 # your prices, one row per model_name
scripts/model_eval/quickstart.sh up            # proxy only, no mock
scripts/model_eval/quickstart.sh run --repeat 5
```

Other subcommands: `report` (reprint the last run, `--xlsx out.xlsx` to export), `sql "SELECT ..."`,
`prices`, `status`, `down`

Under the hood it is just the CLI, which you can run directly:

```bash
PYTHONPATH=scripts python -m model_eval run --db eval.duckdb --workload workload.xlsx
```

## Configuring vendors

`config.example.yaml` is copied into the workspace on first run. The pattern that matters:
`model_name` is the label the evaluation calls and the key in the price table, while
`litellm_params.model` decides who actually serves it. Give the same upstream model one
`model_name` per vendor:

```yaml
model_list:
  - model_name: deepseek-official
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY

  - model_name: deepseek-volcengine
    litellm_params:
      model: volcengine/deepseek-v3-2-251201
      api_key: os.environ/VOLCENGINE_API_KEY
      api_base: https://ark.cn-beijing.volces.com/api/v3

  - model_name: deepseek-siliconflow
    litellm_params:
      # any vendor without a dedicated LiteLLM provider is just OpenAI-compatible
      model: openai/deepseek-ai/DeepSeek-V3
      api_key: os.environ/SILICONFLOW_API_KEY
      api_base: https://api.siliconflow.cn/v1
```

The example file also covers OpenAI, Anthropic, Gemini, Azure, Bedrock, DashScope, Moonshot and a
self-hosted vLLM endpoint

## Input: the workload sheet

Positional columns, one request per row, header row optional (a first row whose third cell is not
JSON is treated as a header and skipped)

| column | meaning |
| --- | --- |
| A | request id |
| B | StartTime |
| C | UserId |
| D | Input (the recorded request body, JSON, OpenAI `/v1/chat/completions` shape) |

Timestamps may be real Excel date cells, `2026-01-30 09:15:00`, ISO 8601, or a unix epoch in
seconds or milliseconds. Rows are replayed sorted by user id, then timestamp, then sheet position,
so a user's conversation is replayed in the order it happened; that is what makes the prompt-cache
columns mean anything

Rows that cannot be parsed are reported as warnings and skipped rather than aborting the run

## Input: the price table

There is no init step. Every command opens the DuckDB file given by `--db` (default
`model_eval.duckdb`), creating it and running the schema if it is not there yet, so the first
thing you do can be the import:

```bash
python -m model_eval templates ./inputs                 # starter workbooks
python -m model_eval import-prices ./inputs/pricing_template.xlsx --db eval.duckdb
```

Prices live in the `model_prices` table, one row per proxy model. It doubles as the list of models
to evaluate, so a model with no enabled row is never called (`--model` selects a subset of it)

| column | required | meaning |
| --- | --- | --- |
| `model` | yes | model name as configured in the proxy, e.g. `deepseek-volcengine` |
| `input_per_1m` | yes | price per 1M uncached input tokens |
| `output_per_1m` | yes | price per 1M output tokens |
| `label` | no | display name in reports; defaults to `model` |
| `cache_read_per_1m` | no | price per 1M cache-hit input tokens; `NULL` bills them as input |
| `cache_write_per_1m` | no | price per 1M cache-write tokens; `NULL` bills them as input |
| `currency` | no | label only, no conversion is done; defaults to `USD` |
| `enabled` | no | set `FALSE` to park a vendor without deleting its history |

Maintain it whichever way suits you. `import-prices` replaces the whole table, which is what you
want when the spreadsheet is the master copy. For a single vendor, or a single corrected number,
go straight at the table:

```bash
python -m model_eval sql --db eval.duckdb "
  INSERT INTO model_prices (model, label, input_per_1m, output_per_1m, cache_read_per_1m, currency)
  VALUES ('deepseek-volcengine', 'Volcengine / deepseek-v3', 2.0, 8.0, 0.4, 'CNY')"

python -m model_eval sql --db eval.duckdb "
  UPDATE model_prices SET output_per_1m = 9.0 WHERE model = 'deepseek-volcengine'"

python -m model_eval sql --db eval.duckdb "
  UPDATE model_prices SET enabled = FALSE WHERE model = 'mock-fast'"

python -m model_eval list-prices --db eval.duckdb
```

`quickstart.sh` does the import for you on every `up` and `demo`, from `prices.xlsx` in the
workspace; edit that spreadsheet and run `quickstart.sh up` again to reload it

Costs are never mixed across currencies; each model is totalled in its own

## Output: the database

Two tables of raw facts and two views that price them:

- `eval_runs` - one row per replay: when, against what, with which settings, plus your `--note`
- `eval_requests` - one row per call: latency, tokens, finish reason, the buffered response, and
  the error detail when it failed. Written and committed as each call returns, so an interrupted
  run keeps everything that finished
- `eval_request_costs` - `eval_requests` joined to `model_prices`, with the cost split out
- `eval_summary` - per run and model: percentiles, token totals, cache hit rate, unit costs

**Cost is not stored, it is derived.** Correcting a price re-values every historical run instead of
stranding it, which is what you want when a vendor changes their rates mid-comparison or you
mistyped a zero. The end-of-run console table is a query against `eval_summary`, so the terminal,
the spreadsheet export and anything you write yourself can never disagree

Reporting is whatever SQL you like:

```bash
python -m model_eval sql "
  SELECT label, requests, ttft_ms_p50, tpot_ms_p50, cost_per_1m_output_tokens, currency
  FROM eval_summary WHERE run_id = '20260730-174126-c08c34' ORDER BY cost_per_1m_output_tokens"
```

The file is a plain DuckDB database, so BI tools, `pandas.read_sql`, or the `duckdb` CLI all work
against it directly. `python -m model_eval report --xlsx out.xlsx` exports one run to a workbook
when someone wants a spreadsheet

## Running the replay

```bash
python -m model_eval run \
  --db eval.duckdb \
  --workload ./inputs/workload.xlsx \
  --base-url http://localhost:4000 \
  --api-key sk-1234 \
  --repeat 5
```

Useful flags:

- `--model NAME` (repeatable) evaluate only these models out of the price table
- `--limit N` replay only the first N ordered requests, for a smoke run
- `--repeat N` replay the whole workload N times per model, to get a distribution worth quoting.
  Worth doing: the very first call through a freshly started proxy carries its warm-up
- `--stream-mode {as_recorded,force_stream,force_non_stream}`; TTFT and TPOT only exist for
  streamed requests, so force streaming if the recordings are blocking and you care about them
- `--sleep S` pause between requests when a vendor rate-limits you
- `--note TEXT` stored on the run, so you can tell two runs apart six weeks later
- `--dry-run` print the plan and exit

Requests are issued one at a time on purpose. Firing them concurrently would destroy the
prompt-cache behaviour the cached-token and cost columns are measuring

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
  against your price table. It reflects LiteLLM's built-in price map, which is why your own table
  exists

Streaming requests get `stream_options.include_usage` forced on, otherwise providers return no
usage block and every token and cost column would be empty

Failed calls are recorded, not raised: the row keeps its `error_kind`, `http_status` and
`error_detail`, has no tokens and no cost, and is excluded from the latency aggregates so one
vendor's 429s cannot make it look cheap
