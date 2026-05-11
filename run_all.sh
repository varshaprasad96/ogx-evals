#!/usr/bin/env bash
# Run all experiments for the multitenant RAG security evaluation.
#
# Usage:
#   ./run_all.sh                    # Run all 4 configs (A, B, C, D)
#   ./run_all.sh --config D         # Run a single config
#   ./run_all.sh --analysis-only    # Regenerate figures from existing results
#   ./run_all.sh --fresh            # Wipe stale state before each config run
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
FRESH=false
SERVER_URL="http://localhost:8321"
AUTH_PORT=9999

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIGS="$2"; shift 2 ;;
        --analysis-only) ANALYSIS_ONLY=true; shift ;;
        --fresh) FRESH=true; shift ;;
        --server-url) SERVER_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Cleanup function to kill background processes on exit
PIDS=()
cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

run_cmd() {
    echo "  → $*"
    "$@"
}

# -------------------------------------------------------------------
# Analysis-only path: regenerate figures from pre-computed results
# -------------------------------------------------------------------
if $ANALYSIS_ONLY; then
    echo "=== Analysis only ==="
    echo ""
    echo "--- Regenerating figures from pre-computed results (Experiments 1-3) ---"
    run_cmd uv run python scripts/analyze_results.py
    echo ""
    echo "--- Running Experiment 4: Synthetic retrieval benchmarks (80 tests) ---"
    run_cmd uv run pytest tests/multitenant/ -v
    echo ""
    echo "--- Running Experiment 6: Predicate pushdown scaling ---"
    run_cmd uv run python scripts/bench_predicate_pushdown.py
    echo ""
    echo "Done. Figures in figures/. Experiment 5 (GPU latency) results in data/results/e2e_latency_gpu.csv."
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

echo "=== Installing dependencies ==="
run_cmd uv sync --frozen

echo ""
echo "=== Generating synthetic data ==="
run_cmd uv run python scripts/generate_data.py

# Determine which configs need auth server and which need server-side orchestration
UNGATED_CONFIGS="A C"
GATED_CONFIGS="B D"
CLIENT_CONFIGS="A B"
SERVER_CONFIGS="C D"

for CONFIG in $CONFIGS; do
    echo ""
    echo "============================================================"
    echo "  CONFIG $CONFIG"
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

    # Wipe stale state if --fresh
    if $FRESH; then
        DIST_DIR="$HOME/.llama/distributions/experiment-${CONFIG_LOWER}"
        if [[ -d "$DIST_DIR" ]]; then
            log_step "  Wiping stale state: $DIST_DIR"
            rm -rf "$DIST_DIR"
        fi
    fi

    # Start auth server if needed
    if $NEEDS_AUTH; then
        echo "  Starting auth server on port $AUTH_PORT..."
        uv run python scripts/auth_server.py --port "$AUTH_PORT" &
        PIDS+=($!)
        sleep 2
    fi

    # Start OGX server
    echo "  Starting OGX server..."
    uv run llama stack run "$CONFIG_FILE" --port 8321 &
    PIDS+=($!)

    # Wait for server to be ready
    echo "  Waiting for server to be ready..."
    for i in $(seq 1 30); do
        if curl -s "$SERVER_URL/v1/health" >/dev/null 2>&1; then
            echo "  Server ready."
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo "  ERROR: Server did not start within 5 minutes."
            exit 1
        fi
        sleep 10
    done

    # Ingest documents
    echo "  Ingesting documents..."
    run_cmd uv run python scripts/ingest_data.py --config "$CONFIG" --server-url "$SERVER_URL"

    # Run main experiment
    echo "  Running experiment..."
    run_cmd uv run python scripts/run_experiment.py --config "$CONFIG" --server-url "$SERVER_URL"

    # Run injection probes
    echo "  Running injection probes..."
    run_cmd uv run python scripts/run_injection_probes.py --config "$CONFIG" --server-url "$SERVER_URL"

    # Stop servers for this config
    echo "  Stopping servers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    PIDS=()
    sleep 2
done

# -------------------------------------------------------------------
# Experiment 4: Synthetic retrieval benchmarks (no API key needed)
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  EXPERIMENT 4: Synthetic Retrieval Benchmarks"
echo "============================================================"
run_cmd uv run pytest tests/multitenant/ -v

# -------------------------------------------------------------------
# Experiment 6: Predicate pushdown scaling (no API key needed)
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  EXPERIMENT 6: Predicate Pushdown Scaling"
echo "============================================================"
run_cmd uv run python scripts/bench_predicate_pushdown.py

# -------------------------------------------------------------------
# Generate figures and summary
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  GENERATING FIGURES AND SUMMARY"
echo "============================================================"
run_cmd uv run python scripts/analyze_results.py

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Results:  data/results/"
echo "Figures:  figures/"
echo "Summary:  data/results/summary.json"
echo ""
echo "Note: Experiment 5 (GPU latency) requires OpenShift + vLLM."
echo "Pre-computed results are in data/results/e2e_latency_gpu.csv."
