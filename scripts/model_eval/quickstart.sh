#!/usr/bin/env bash
#
# One-click setup for the model evaluation replay in a fresh sandbox.
#
#   ./quickstart.sh demo     install, start a mock provider + the proxy, replay, report
#   ./quickstart.sh up       install and start the proxy against your own config.yaml
#   ./quickstart.sh run ...  replay again; every extra argument goes to `model_eval run`
#   ./quickstart.sh report   reprint the last run (add --xlsx out.xlsx to export)
#   ./quickstart.sh sql ["SELECT ..."]  query the results (defaults to eval_summary)
#   ./quickstart.sh status   what is running
#   ./quickstart.sh down     stop everything this script started
#
# Everything lands in one workspace directory (default ./model-eval, override with
# MODEL_EVAL_HOME) so removing it removes the whole experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${MODEL_EVAL_HOME:-$PWD/model-eval}"
VENV="$WORKSPACE/venv"
RUN_DIR="$WORKSPACE/run"
LOG_DIR="$WORKSPACE/logs"
DB="$WORKSPACE/model_eval.duckdb"

LITELLM_VERSION="${LITELLM_VERSION:-1.94.0}"
DUCKDB_VERSION="1.4.4"
OPENPYXL_VERSION="3.1.5"

PROXY_PORT="${PROXY_PORT:-4000}"
MOCK_PORT="${MOCK_PORT:-4123}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

ensure_workspace() {
    mkdir -p "$WORKSPACE" "$RUN_DIR" "$LOG_DIR"
    [ -f "$WORKSPACE/config.yaml" ] || cp "$SCRIPT_DIR/config.example.yaml" "$WORKSPACE/config.yaml"
    [ -f "$WORKSPACE/.env" ] || cp "$SCRIPT_DIR/.env.example" "$WORKSPACE/.env"
}

ensure_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        say "creating a virtualenv in $VENV"
        python3 -m venv "$VENV"
        "$VENV/bin/python" -m pip install --quiet --upgrade pip
    fi
    if [ ! -x "$VENV/bin/litellm" ]; then
        say "installing litellm[proxy]==$LITELLM_VERSION, duckdb==$DUCKDB_VERSION, openpyxl==$OPENPYXL_VERSION"
        "$VENV/bin/python" -m pip install --quiet \
            "litellm[proxy]==$LITELLM_VERSION" \
            "duckdb==$DUCKDB_VERSION" \
            "openpyxl==$OPENPYXL_VERSION"
    fi
}

ensure_inputs() {
    if [ ! -f "$WORKSPACE/workload.csv" ] || [ ! -f "$WORKSPACE/prices.xlsx" ]; then
        say "writing starter workload.csv and prices.xlsx"
        eval_cli templates "$WORKSPACE" >/dev/null
        [ -f "$WORKSPACE/workload.csv" ] || mv "$WORKSPACE/workload_template.csv" "$WORKSPACE/workload.csv"
        [ -f "$WORKSPACE/prices.xlsx" ] || mv "$WORKSPACE/pricing_template.xlsx" "$WORKSPACE/prices.xlsx"
        rm -f "$WORKSPACE/workload_template.csv" "$WORKSPACE/workload_template.xlsx" "$WORKSPACE/pricing_template.xlsx"
    fi
}

# Runs the CLI from this checkout, against the workspace venv's dependencies
eval_cli() {
    PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}" "$VENV/bin/python" -m model_eval "$@"
}

start_background() {
    local name="$1"; shift
    if is_running "$name"; then
        say "$name already running (pid $(cat "$RUN_DIR/$name.pid"))"
        return 0
    fi
    say "starting $name; log: $LOG_DIR/$name.log"
    ( "$@" >"$LOG_DIR/$name.log" 2>&1 & echo $! >"$RUN_DIR/$name.pid" )
}

is_running() {
    local pidfile="$RUN_DIR/$1.pid"
    [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

ensure_port_free() {
    local port="$1" name="$2"
    local pidfile="$RUN_DIR/$name.pid"

    if [ -f "$pidfile" ]; then
        local pid
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            say "stopping existing $name (pid $pid) on port $port"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser -k "$port/tcp" >/dev/null 2>&1 || true
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti tcp:"$port" | xargs -r kill -9 >/dev/null 2>&1 || true
    fi

    if python3 - "$port" <<'PY' >/dev/null 2>&1
import socket, sys
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
    raise SystemExit(0)
PY
    then
        return 0
    fi

    die "$name needs port $port but it is already in use"
}

wait_for() {
    local url="$1" name="$2" waited=0
    while [ "$waited" -lt "$READY_TIMEOUT" ]; do
        if curl -fs -o /dev/null "$url" 2>/dev/null; then
            say "$name is ready"
            return 0
        fi
        is_running "$name" || die "$name exited early; see $LOG_DIR/$name.log"
        sleep 2
        waited=$((waited + 2))
    done
    die "$name did not become ready within ${READY_TIMEOUT}s; see $LOG_DIR/$name.log"
}

start_mock() {
    ensure_port_free "$MOCK_PORT" mock
    start_background mock env PYTHONPATH="$SCRIPT_DIR/.." "$VENV/bin/python" -m model_eval.mock_provider \
        --port "$MOCK_PORT" --ttft 0.25 --tpot 0.03
    sleep 1
    is_running mock || die "the mock provider exited; see $LOG_DIR/mock.log"
    say "mock provider on http://127.0.0.1:$MOCK_PORT/v1"
}

start_proxy() {
    ensure_port_free "$PROXY_PORT" proxy
    set -a
    # shellcheck disable=SC1091  # generated from .env.example at first run
    . "$WORKSPACE/.env"
    set +a
    start_background proxy "$VENV/bin/litellm" --config "$WORKSPACE/config.yaml" --port "$PROXY_PORT"
    wait_for "http://127.0.0.1:$PROXY_PORT/health/readiness" proxy
}

load_prices() {
    say "loading prices from $WORKSPACE/prices.xlsx into $DB"
    eval_cli import-prices "$WORKSPACE/prices.xlsx" --db "$DB"
}

reload_prices() {
    say "reloading prices from $WORKSPACE/prices.xlsx into $DB"
    eval_cli import-prices "$WORKSPACE/prices.xlsx" --db "$DB"
}

replay() {
    # shellcheck disable=SC1091  # generated from .env.example at first run
    . "$WORKSPACE/.env"
    eval_cli run \
        --db "$DB" \
        --workload "$WORKSPACE/workload.csv" \
        --base-url "http://127.0.0.1:$PROXY_PORT" \
        --api-key "${LITELLM_API_KEY:-}" \
        "$@"
}

cmd_up() {
    ensure_workspace
    ensure_venv
    ensure_inputs
    load_prices
    start_proxy
    say "proxy on http://127.0.0.1:$PROXY_PORT; edit $WORKSPACE/config.yaml and $WORKSPACE/.env to add vendors"
}

cmd_demo() {
    ensure_workspace
    ensure_venv
    ensure_inputs
    load_prices
    start_mock
    start_proxy
    replay --model mock-fast --repeat 2 --note "quickstart demo" --xlsx "$WORKSPACE/demo-report.xlsx"
    say "done; results in $DB, workbook in $WORKSPACE/demo-report.xlsx"
}

cmd_down() {
    for name in proxy mock; do
        if is_running "$name"; then
            say "stopping $name"
            kill "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null || true
        fi
        rm -f "$RUN_DIR/$name.pid"
    done
}

cmd_status() {
    for name in proxy mock; do
        if is_running "$name"; then
            printf '%-6s running (pid %s)\n' "$name" "$(cat "$RUN_DIR/$name.pid")"
        else
            printf '%-6s stopped\n' "$name"
        fi
    done
    printf 'workspace %s\n' "$WORKSPACE"
}

cmd_restart() {
    ensure_venv
    cmd_down
    cmd_up
}

case "${1:-demo}" in
    up)     cmd_up ;;
    demo)   cmd_demo ;;
    run)    shift; ensure_venv; replay "$@" ;;
    reload-prices) ensure_venv; reload_prices ;;
    restart) ensure_venv; cmd_restart ;;
    report) shift; ensure_venv; eval_cli report --db "$DB" "$@" ;;
    prices) shift; ensure_venv; eval_cli list-prices --db "$DB" "$@" ;;
    sql)    shift; ensure_venv; eval_cli sql --db "$DB" "$@" ;;
    status) cmd_status ;;
    down)   cmd_down ;;
    *)      die "unknown command '${1}'; try: demo | up | run | reload-prices | restart | report | prices | sql | status | down" ;;
esac
