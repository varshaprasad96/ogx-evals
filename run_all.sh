#!/usr/bin/env bash
# Run all experiments for the multitenant RAG security evaluation.
#
# Usage:
#   ./run_all.sh                    # Run all 4 configs (A, B, C, D)
#   ./run_all.sh --config D         # Run a single config
#   ./run_all.sh --analysis-only    # Regenerate figures from existing results
#
# Prerequisites:
#   - Python 3.12+, uv (https://docs.astral.sh/uv/)
#   - OPENAI_API_KEY environment variable set
#   - No other services on ports 8321 (OGX) or 9999 (auth server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
CONFIGS="A B C D"
ANALYSIS_ONLY=false
SERVER_URL="http://localhost:8321"
AUTH_PORT=9999
LOG_DIR="logs"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIGS="$2"; shift 2 ;;
        --analysis-only) ANALYSIS_ONLY=true; shift ;;
        --server-url) SERVER_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Cleanup function to kill background processes on exit
PIDS=()
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Cleaning up background processes..."
    if ((${#PIDS[@]})); then
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT

log_step() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_cmd() {
    log_step "→ $*"
    "$@"
}

# -------------------------------------------------------------------
# Analysis-only path: regenerate figures from pre-computed results
# -------------------------------------------------------------------
if $ANALYSIS_ONLY; then
    log_step "=== Analysis only ==="
    echo ""
    log_step "--- Regenerating figures from pre-computed results (Experiments 1-3) ---"
    run_cmd uv run python scripts/analyze_results.py
    echo ""
    log_step "--- Running Experiment 4: Synthetic retrieval benchmarks (80 tests) ---"
    run_cmd uv run pytest tests/multitenant/ -v
    echo ""
    log_step "--- Running Experiment 6: Predicate pushdown scaling ---"
    run_cmd uv run python scripts/bench_predicate_pushdown.py
    echo ""
    log_step "Done. Figures in figures/. Experiment 5 (GPU latency) results in data/results/e2e_latency_gpu.csv."
    exit 0
fi

# -------------------------------------------------------------------
# Full experiment path
# -------------------------------------------------------------------

# Check prerequisites
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed. See https://docs.astral.sh/uv/"
    exit 1
fi

log_step "=== Installing dependencies ==="
run_cmd uv sync --frozen

echo ""
log_step "=== Generating synthetic data ==="
run_cmd uv run python scripts/generate_data.py
mkdir -p "$LOG_DIR"

# Determine which configs need auth server and which need server-side orchestration
UNGATED_CONFIGS="A C"
GATED_CONFIGS="B D"
CLIENT_CONFIGS="A B"
SERVER_CONFIGS="C D"

for CONFIG in $CONFIGS; do
    CONFIG_START=$(date +%s)
    echo ""
    echo "============================================================"
    log_step "  CONFIG $CONFIG"
    echo "============================================================"

    CONFIG_LOWER=$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')
    NEEDS_AUTH=false
    CONFIG_FILE=""

    case $CONFIG in
        A) CONFIG_FILE="configs/config_a_ungated_client.yaml" ;;
        B) CONFIG_FILE="configs/config_b_gated_client.yaml"; NEEDS_AUTH=true ;;
        C) CONFIG_FILE="configs/config_c_ungated_server.yaml" ;;
        D) CONFIG_FILE="configs/config_d_gated_server.yaml"; NEEDS_AUTH=true ;;
        *) echo "Unknown config: $CONFIG"; exit 1 ;;
    esac

    # Start auth server if needed
    if $NEEDS_AUTH; then
        log_step "  Starting auth server on port $AUTH_PORT..."
        AUTH_LOG="$LOG_DIR/config_${CONFIG}_auth.log"
        log_step "  Auth server logs: $AUTH_LOG"
        uv run python scripts/auth_server.py --port "$AUTH_PORT" >"$AUTH_LOG" 2>&1 &
        PIDS+=($!)
        sleep 2
    fi

    # Start OGX server
    log_step "  Starting OGX server..."
    OGX_LOG="$LOG_DIR/config_${CONFIG}_ogx.log"
    log_step "  OGX server logs: $OGX_LOG"
    uv run llama stack run "$CONFIG_FILE" --port 8321 >"$OGX_LOG" 2>&1 &
    PIDS+=($!)

    # Wait for server to be ready
    log_step "  Waiting for server to be ready..."
    for i in $(seq 1 30); do
        if curl -s "$SERVER_URL/v1/health" >/dev/null 2>&1; then
            log_step "  Server ready."
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_step "  ERROR: Server did not start within 5 minutes."
            exit 1
        fi
        log_step "  Still waiting for server... ($i/30)"
        sleep 10
    done

    # Ingest documents
    log_step "  Ingesting documents..."
    run_cmd uv run python scripts/ingest_data.py --config "$CONFIG" --server-url "$SERVER_URL"

    # Run main experiment
    log_step "  Running experiment..."
    run_cmd uv run python scripts/run_experiment.py --config "$CONFIG" --server-url "$SERVER_URL" --progress-every 10

    # Run injection probes
    log_step "  Running injection probes..."
    run_cmd uv run python scripts/run_injection_probes.py --config "$CONFIG" --server-url "$SERVER_URL" --progress-every 10

    # Stop servers for this config
    log_step "  Stopping servers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    PIDS=()
    CONFIG_ELAPSED=$(($(date +%s) - CONFIG_START))
    log_step "  Config $CONFIG complete in ${CONFIG_ELAPSED}s."
    sleep 2
done

# -------------------------------------------------------------------
# Experiment 4: Synthetic retrieval benchmarks (no API key needed)
# -------------------------------------------------------------------
echo ""
echo "============================================================"
log_step "  EXPERIMENT 4: Synthetic Retrieval Benchmarks"
echo "============================================================"
run_cmd uv run pytest tests/multitenant/ -v

# -------------------------------------------------------------------
# Experiment 6: Predicate pushdown scaling (no API key needed)
# -------------------------------------------------------------------
echo ""
echo "============================================================"
log_step "  EXPERIMENT 6: Predicate Pushdown Scaling"
echo "============================================================"
run_cmd uv run python scripts/bench_predicate_pushdown.py

# -------------------------------------------------------------------
# Generate figures and summary
# -------------------------------------------------------------------
echo ""
echo "============================================================"
log_step "  GENERATING FIGURES AND SUMMARY"
echo "============================================================"
run_cmd uv run python scripts/analyze_results.py

echo ""
echo "============================================================"
log_step "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Results:  data/results/"
echo "Figures:  figures/"
echo "Summary:  data/results/summary.json"
echo ""
echo "Note: Experiment 5 (GPU latency) requires OpenShift + vLLM."
echo "Pre-computed results are in data/results/e2e_latency_gpu.csv."
