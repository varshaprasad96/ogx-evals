# Reproducing the Evaluation Results

This document maps each claim in the paper to the exact command needed to verify or reproduce it.

## Step-by-step quick verification (5 minutes, no API key, no GPU)

### Step 1: Provision the environment

```bash
git clone https://github.com/varshaprasad96/ogx-evals.git
cd ogx-evals
uv sync --frozen
```

**Expected output**: `Installed 107 packages in ...` (takes ~10 seconds)

### Step 2: Run all locally-reproducible experiments

```bash
./run_all.sh --analysis-only
```

**Expected output**:
- Security metrics table (CTLR=100%/0%/98.3%/0% for configs A/B/C/D)
- `Saved figures/security_metrics.pdf` (+ 3 more PDFs)
- `80 passed in ~6s` (Experiment 4 pytest results)
- Predicate pushdown scaling table with 16 rows (Experiment 6, ~2 minutes)

### Step 3 (alternative): Docker verification

```bash
docker build -t ogx-evals .
docker run --rm -v $(pwd)/figures:/eval/figures ogx-evals
```

**Expected output**: Same metrics table as Step 2. Generated PDFs appear in `figures/`.

---

## Paper claims → reproduction commands

### Security (Section 5.2)

| Claim | How to verify |
|-------|---------------|
| CTLR=100% ungated, 0% gated | `uv run python scripts/analyze_results.py` — prints security metrics table |
| CTLR=0% under gating for both client and server orchestration | Same command, check Configs B and D |
| 52% leakage with synthetic embeddings | `uv run pytest tests/multitenant/test_cross_tenant_leakage.py -v` |
| 0% false positive rate (48-case ABAC matrix) | `uv run pytest tests/multitenant/test_resource_access_control.py -v` |
| Prompt injection: 0% leak under gating | `uv run python scripts/analyze_results.py` — prints injection table |

### Retrieval quality (Section 5.4)

| Claim | How to verify |
|-------|---------------|
| Gating improves Precision@5 by 2.2x | `uv run pytest tests/multitenant/test_retrieval_quality.py -v` |
| MRR improves from 0.700 to 1.000 | Same command |
| All 4 adversarial attack patterns blocked | `uv run pytest tests/multitenant/test_adversarial_scenarios.py -v` |

### Performance (Section 5.3)

| Claim | How to verify |
|-------|---------------|
| ~19ms ABAC overhead (API-based) | `uv run python scripts/analyze_results.py` — prints ABAC overhead |
| Throughput scales linearly, no gating bottleneck | Same command — prints throughput table |
| ABAC evaluation is sub-millisecond | `uv run pytest tests/multitenant/test_latency_overhead.py -v` |
| Filter overhead < 5ms at all corpus sizes | `uv run pytest tests/multitenant/test_latency_overhead.py -v` |

### Predicate pushdown (Section 5.4)

| Claim | How to verify |
|-------|---------------|
| Recall@5 = 1.000 at 100 chunks, 0.002 at 50K | `uv run python scripts/bench_predicate_pushdown.py` |
| Filter overhead 0.7-3ms at 5x multiplier | Same command |
| Latency overhead small regardless of corpus size | Same command |

### GPU infrastructure overhead (Section 5.3)

| Claim | How to verify |
|-------|---------------|
| 4.7ms routing overhead (1.0%) | Pre-computed: `cat data/results/e2e_latency_gpu.csv` |
| 5.5ms filter overhead (1.9%) | Same file |

See [Reproducing Experiment 5](#reproducing-experiment-5-gpu-infrastructure) below for full reproduction.

---

## Full reproduction of Experiments 1-3 (~2 hours, ~$5-10)

Requires an OpenAI API key. Estimated cost: ~$5-10 for gpt-4o-mini inference + text-embedding-3-small embeddings across 4 configs × (300 authorized + 300 cross-tenant + 90 injection) queries.

```bash
export OPENAI_API_KEY=sk-...
./run_all.sh
```

This runs all 4 configurations (A, B, C, D) sequentially, managing server lifecycle automatically. Results are written to `data/results/` and figures to `figures/`.

To run a single config:
```bash
./run_all.sh --config D
```

---

## Using alternative models

The default configs use OpenAI's `gpt-4o-mini` for inference and `text-embedding-3-small` (1536 dimensions) for embeddings. All model references are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OGX_MODEL_ID` | `openai/gpt-4o-mini` | Chat/inference model |
| `OGX_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `OGX_EMBEDDING_DIM` | `1536` | Embedding vector dimension |

### Ready-to-use: Ollama (no API key, no GPU required)

```bash
# 1. Install and start Ollama
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve

# 2. Set model env vars
export OGX_MODEL_ID=ollama/llama3.1:8b
export OGX_EMBEDDING_MODEL=ollama/nomic-embed-text
export OGX_EMBEDDING_DIM=768

# 3. Run with Ollama config
./run_all.sh --config D --config-file configs/config_d_gated_server_ollama.yaml
```

Config file: `configs/config_d_gated_server_ollama.yaml` (pre-configured, no edits needed).

### Ready-to-use: vLLM (requires GPU)

```bash
# 1. Start vLLM
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 2. Set model env vars
export OGX_MODEL_ID=vllm/meta-llama/Llama-3.1-8B-Instruct
export OGX_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export OGX_EMBEDDING_DIM=384

# 3. Run with vLLM config
./run_all.sh --config D --config-file configs/config_d_gated_server_vllm.yaml
```

Config file: `configs/config_d_gated_server_vllm.yaml` (uses sentence-transformers for embeddings on CPU, vLLM for chat on GPU).

### Custom models

For any other provider, set the three env vars and use the appropriate config YAML. The security claims (CTLR, AVR, injection leak rates) are model-independent — they depend on ABAC gating, not inference output.

### Clear stale state between runs

OGX registers models and vector stores in a SQLite kvstore under `~/.llama/distributions/`. If you re-run with different models, clear stale state:

```bash
./run_all.sh --config D --fresh
# or manually: rm -rf ~/.llama/distributions/experiment-*
```

---

## Reproducing Experiment 5 (GPU infrastructure)

Experiment 5 measures OGX's routing and filtering overhead on self-hosted GPU infrastructure. The pre-computed results are in `data/results/e2e_latency_gpu.csv`.

### Verify pre-computed results

```bash
python3 -c "
import csv
with open('data/results/e2e_latency_gpu.csv') as f:
    for row in csv.DictReader(f):
        p95, p99 = float(row['p95']), float(row['p99'])
        assert p99 >= p95, f'P99 < P95 for {row[\"label\"]}'
        print(f'{row[\"label\"]:30} median={float(row[\"median\"]):.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms')
print('All percentile invariants hold.')
"
```

### Reproduce on any machine with a GPU

The benchmark script works with any vLLM instance and OGX deployment, not just OpenShift. You need:

1. A GPU that can serve a small LLM (any NVIDIA GPU with >= 4GB VRAM)
2. A HuggingFace account with access to `meta-llama/Llama-3.2-1B-Instruct`

```bash
# Terminal 1: Start vLLM
pip install vllm
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --max-model-len 2048 --port 8000

# Terminal 2: Start OGX
pip install llama-stack
llama stack run configs/config_e2e_vllm_gpu.yaml --port 8321

# Terminal 3: Run benchmark
python scripts/bench_e2e_latency.py \
    --vllm-url http://localhost:8000 \
    --llama-stack-url http://localhost:8321 \
    --num-requests 50 \
    --output-csv data/results/e2e_latency_gpu_reproduced.csv
```

### Original infrastructure

The results in the paper were collected on:

| Component | Specification |
|-----------|--------------|
| Platform | Red Hat OpenShift 4.21 on AWS |
| Instance | g4dn.2xlarge |
| GPU | NVIDIA T4 (16GB VRAM, Turing) |
| Inference | vLLM serving meta-llama/Llama-3.2-1B-Instruct |
| Embeddings | nomic-ai/nomic-embed-text-v1.5 via inline::sentence-transformers |
| Vector store | inline::sqlite-vec |
| OGX | v0.7.1 (distribution-starter image) |

The OGX server config used is in `configs/config_e2e_vllm_gpu.yaml`.

---

## Experiment-to-file mapping

| Experiment | Paper section | Script | Results file | Reproducible locally? |
|-----------|--------------|--------|-------------|----------------------|
| 1. Cross-tenant leakage | 5.2 | `scripts/run_experiment.py` | `data/results/config_*_results.json` | Yes (needs OPENAI_API_KEY) |
| 2. Throughput scaling | 5.3 | `scripts/run_experiment.py` | `data/results/config_*_throughput.json` | Yes (needs OPENAI_API_KEY) |
| 3. Prompt injection | 5.2 | `scripts/run_injection_probes.py` | `data/results/config_*_injection_results.json` | Yes (needs OPENAI_API_KEY) |
| 4. Synthetic retrieval | 5.4 | `tests/multitenant/test_*.py` | pytest output (80 tests) | Yes (free, ~6s) |
| 5. GPU latency | 5.3 | `scripts/bench_e2e_latency.py` | `data/results/e2e_latency_gpu.csv` | Needs GPU + vLLM |
| 6. Predicate pushdown | 5.4 | `scripts/bench_predicate_pushdown.py` | `data/results/predicate_pushdown_scaling.csv` | Yes (free, ~2min) |

---

## Estimated costs

| Path | Time | Cost |
|------|------|------|
| Verify pre-computed results | ~5 minutes | Free |
| Experiments 4 + 6 (local) | ~3 minutes | Free |
| Experiments 1-3 (full) | ~2 hours | ~$5-10 (OpenAI API) |
| Experiment 5 (GPU) | ~30 minutes setup + 5 min run | GPU instance cost |

---

## Expected results and hardware sensitivity

### Results that are hardware-independent (deterministic)

These produce identical results on any machine:
- **Security metrics** (CTLR, AVR): 0% or 100% — binary outcomes determined by ABAC policy, not hardware
- **ABAC correctness** (48-case matrix): 100% accuracy, 0% false positives — policy evaluation logic
- **Adversarial scenarios** (4 attack patterns): all blocked under gating — deterministic filter behavior
- **Retrieval quality** (Recall@5, Precision@5, MRR): synthetic embeddings produce identical rankings
- **Predicate pushdown recall** (Recall@5 at each corpus size): determined by embedding geometry, not speed

### Results that vary with hardware

- **Latency numbers** (Experiments 1-3): Dominated by OpenAI API response time. Expect ±30% variation depending on network conditions and API load. The ABAC overhead delta (~19ms) should be consistent.
- **GPU overhead** (Experiment 5): Absolute latency depends on GPU model. The overhead percentage (routing: ~1%, filtering: ~2%) should be consistent since it is a fixed cost independent of inference speed.
- **Predicate pushdown timing** (Experiment 6): Filter overhead in milliseconds scales with CPU speed, but the shape of the curve (flat overhead, declining recall) is hardware-independent.
- **Throughput** (Experiment 2): QPS depends on network and CPU. The relative pattern (gated ≈ ungated, client ~2x server) should hold across hardware.
