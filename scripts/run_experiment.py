"""
Main experiment runner for the 2x2 comparative evaluation.

Runs authorized queries, cross-tenant probes, and throughput tests
against a running OGX server for a given configuration.

Configurations:
  A: Client-side orchestration + ungated retrieval
  B: Client-side orchestration + gated retrieval
  C: Server-side orchestration + ungated retrieval
  D: Server-side orchestration + gated retrieval

Usage:
    python scripts/run_experiment.py --config A|B|C|D [--server-url http://localhost:8321]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from openai import OpenAI

# Add scripts dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_orchestration import client_side_rag, search_vector_store


TENANTS = ["finance", "engineering", "legal"]
MODEL_ID = "openai/gpt-4o-mini"
NUM_RUNS = 3  # Repeat each query for statistical rigor

# Config properties
CLIENT_SIDE_CONFIGS = {"A", "B"}
SERVER_SIDE_CONFIGS = {"C", "D"}
GATED_CONFIGS = {"B", "D"}
UNGATED_CONFIGS = {"A", "C"}


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def print_progress(label: str, completed: int, total: int, start: float, errors: int = 0) -> None:
    """Print progress with elapsed time, estimated time remaining, and error count."""
    elapsed = time.perf_counter() - start
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = (total - completed) / rate if rate > 0 else 0
    pct = completed / total * 100 if total else 100
    print(
        f"  [{label}] {completed}/{total} ({pct:5.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)} "
        f"rate={rate * 60:.1f}/min errors={errors}",
        flush=True,
    )


def get_client(server_url: str, tenant: str | None = None, user_idx: int = 0) -> OpenAI:
    if tenant:
        api_key = f"token-{tenant}-{user_idx}"
    else:
        api_key = "no-auth"
    return OpenAI(base_url=f"{server_url}/v1", api_key=api_key, timeout=120.0)


def load_queries(data_dir: str, query_type: str) -> list[dict]:
    filename = {
        "authorized": "authorized_queries.json",
        "cross_tenant_probe": "cross_tenant_probes.json",
        "prompt_injection": "injection_probes.json",
    }[query_type]
    path = os.path.join(data_dir, "queries", filename)
    with open(path) as f:
        return json.load(f)


def load_store_map(data_dir: str, config: str) -> dict:
    path = os.path.join(data_dir, "results", f"store_map_{config}.json")
    with open(path) as f:
        return json.load(f)


def get_vector_store_id(store_map: dict, tenant: str, config: str) -> str:
    """Get the appropriate vector store ID for a query."""
    if config in UNGATED_CONFIGS:
        return store_map["shared"]
    else:
        return store_map[tenant]


def get_target_vector_store_id(store_map: dict, target_tenant: str, config: str) -> str:
    """Get vector store ID for cross-tenant probes."""
    if config in UNGATED_CONFIGS:
        return store_map["shared"]
    else:
        return store_map[target_tenant]


def extract_tenant_from_chunks(chunks: list[dict]) -> list[str]:
    """Extract tenant IDs from retrieved chunk attributes."""
    tenants = set()
    for chunk in chunks:
        attrs = chunk.get("attributes", {})
        if isinstance(attrs, dict):
            tid = attrs.get("tenant_id")
            if tid:
                tenants.add(tid)
    return list(tenants)


def run_client_side_query(
    server_url: str,
    query: dict,
    store_map: dict,
    config: str,
) -> dict:
    """Run a single query using client-side orchestration (Configs A, B)."""
    tenant = query["tenant"]
    user_idx = query.get("user_idx", 0)
    query_text = query["query_text"]

    # For probes, use the target tenant's store (or shared)
    if query["query_type"] == "cross_tenant_probe":
        vs_id = get_target_vector_store_id(store_map, query["target_tenant"], config)
    else:
        vs_id = get_vector_store_id(store_map, tenant, config)

    client = get_client(server_url, tenant=tenant if config in GATED_CONFIGS else None, user_idx=user_idx)

    result = client_side_rag(client, query_text, vs_id, MODEL_ID)

    retrieved_tenants = extract_tenant_from_chunks(result.retrieved_chunks)

    return {
        "query_id": query["query_id"],
        "tenant": tenant,
        "target_tenant": query.get("target_tenant"),
        "query_type": query["query_type"],
        "query_text": query_text,
        "config": config,
        "orchestration": "client",
        "retrieved_chunk_count": len(result.retrieved_chunks),
        "retrieved_tenants": retrieved_tenants,
        "search_latency_ms": result.search_latency_ms,
        "inference_latency_ms": result.inference_latency_ms,
        "total_latency_ms": result.total_latency_ms,
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_server_side_query(
    server_url: str,
    query: dict,
    store_map: dict,
    config: str,
) -> dict:
    """Run a single query using server-side orchestration via Responses API (Configs C, D)."""
    tenant = query["tenant"]
    user_idx = query.get("user_idx", 0)
    query_text = query["query_text"]

    # For probes, use the target tenant's store
    if query["query_type"] == "cross_tenant_probe":
        vs_id = get_target_vector_store_id(store_map, query["target_tenant"], config)
    else:
        vs_id = get_vector_store_id(store_map, tenant, config)

    client = get_client(server_url, tenant=tenant if config in GATED_CONFIGS else None, user_idx=user_idx)

    start = time.perf_counter()
    error = None
    retrieved_tenants = []
    retrieved_chunk_count = 0
    response_text = ""

    try:
        response = client.responses.create(
            model=MODEL_ID,
            input=query_text,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vs_id],
                "max_num_results": 5,
            }],
        )
        total_ms = (time.perf_counter() - start) * 1000

        # Parse response output to extract retrieved chunks info
        if hasattr(response, "output") and response.output:
            for item in response.output:
                # Look for file_search_call results
                if hasattr(item, "type"):
                    if item.type == "file_search_call" and hasattr(item, "results"):
                        results = item.results or []
                        retrieved_chunk_count = len(results)
                        for r in results:
                            attrs = getattr(r, "attributes", {}) or {}
                            if isinstance(attrs, dict) and "tenant_id" in attrs:
                                retrieved_tenants.append(attrs["tenant_id"])
                    elif item.type == "message" and hasattr(item, "content"):
                        for c in item.content:
                            if hasattr(c, "text"):
                                response_text += c.text

        retrieved_tenants = list(set(retrieved_tenants))

    except Exception as e:
        total_ms = (time.perf_counter() - start) * 1000
        error = str(e)

    return {
        "query_id": query["query_id"],
        "tenant": tenant,
        "target_tenant": query.get("target_tenant"),
        "query_type": query["query_type"],
        "query_text": query_text,
        "config": config,
        "orchestration": "server",
        "retrieved_chunk_count": retrieved_chunk_count,
        "retrieved_tenants": retrieved_tenants,
        "search_latency_ms": None,  # Not separately measurable in server-side
        "inference_latency_ms": None,
        "total_latency_ms": total_ms,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_single_query(server_url: str, query: dict, store_map: dict, config: str) -> dict:
    """Route to client-side or server-side based on config."""
    if config in CLIENT_SIDE_CONFIGS:
        return run_client_side_query(server_url, query, store_map, config)
    else:
        return run_server_side_query(server_url, query, store_map, config)


def run_query_workload(
    server_url: str,
    queries: list[dict],
    store_map: dict,
    config: str,
    label: str,
    num_runs: int = NUM_RUNS,
    progress_every: int = 10,
) -> list[dict]:
    """Run a batch of queries, repeating each num_runs times for statistical rigor."""
    all_results = []
    total = len(queries) * num_runs
    completed = 0
    errors = 0
    start = time.perf_counter()
    progress_every = max(1, progress_every)

    print(
        f"  [{label}] Starting {total} queries "
        f"({len(queries)} queries x {num_runs} run{'s' if num_runs != 1 else ''})",
        flush=True,
    )

    for run_idx in range(num_runs):
        print(f"  [{label}] Run {run_idx + 1}/{num_runs} starting...", flush=True)
        for query in queries:
            result = run_single_query(server_url, query, store_map, config)
            result["run_idx"] = run_idx
            all_results.append(result)
            completed += 1
            if result.get("error") is not None:
                errors += 1

            if completed == 1 or completed % progress_every == 0 or completed == total:
                print_progress(label, completed, total, start, errors)

    print(f"  [{label}] All {total} queries completed in {format_duration(time.perf_counter() - start)}.")
    return all_results


def run_throughput_test(
    server_url: str,
    queries: list[dict],
    store_map: dict,
    config: str,
    concurrency_levels: list[int] = [1, 5, 10, 25],
    progress_every: int = 5,
) -> list[dict]:
    """Measure throughput at various concurrency levels."""
    # Use a subset of authorized queries for throughput testing
    test_queries = queries[:50]
    results = []

    for concurrency in concurrency_levels:
        print(f"  [Throughput] Testing concurrency={concurrency}...", flush=True)

        # Warm-up: run 5 queries
        warmup_start = time.perf_counter()
        warmup_errors = 0
        for i, q in enumerate(test_queries[:5], start=1):
            result = run_single_query(server_url, q, store_map, config)
            if result.get("error") is not None:
                warmup_errors += 1
            print_progress(f"Throughput c={concurrency} warmup", i, 5, warmup_start, warmup_errors)

        # Timed run
        batch = test_queries[:concurrency * 2]  # enough queries to keep threads busy

        print(f"  [Throughput] Timed run: {len(batch)} queries at concurrency={concurrency}", flush=True)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(run_single_query, server_url, q, store_map, config)
                for q in batch
            ]
            query_results = []
            errors = 0
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                query_results.append(result)
                if result.get("error") is not None:
                    errors += 1
                if i == 1 or i % max(1, progress_every) == 0 or i == len(batch):
                    print_progress(f"Throughput c={concurrency}", i, len(batch), start, errors)
        wall_clock = time.perf_counter() - start

        qps = len(batch) / wall_clock
        latencies = [r["total_latency_ms"] for r in query_results if r["error"] is None]

        results.append({
            "config": config,
            "concurrency": concurrency,
            "num_queries": len(batch),
            "wall_clock_seconds": wall_clock,
            "qps": qps,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "errors": sum(1 for r in query_results if r["error"] is not None),
        })

        print(f"    QPS={qps:.1f}, mean_latency={results[-1]['mean_latency_ms']:.0f}ms, errors={results[-1]['errors']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiment for a given config")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--server-url", type=str, default="http://localhost:8321")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--skip-authorized", action="store_true", help="Skip authorized queries")
    parser.add_argument("--skip-probes", action="store_true", help="Skip cross-tenant probes")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput test")
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS, help="Runs per query")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed queries")
    args = parser.parse_args()

    config = args.config
    store_map = load_store_map(args.data_dir, config)

    print(f"=== Running Experiment Config {config} ===")
    print(f"  Orchestration: {'client-side' if config in CLIENT_SIDE_CONFIGS else 'server-side'}")
    print(f"  Retrieval: {'gated' if config in GATED_CONFIGS else 'ungated'}")
    print(f"  Runs per query: {args.num_runs}")
    print(f"  Progress interval: {args.progress_every} queries")
    print(f"  Store map: {store_map}")
    print()

    all_results = []
    results_dir = os.path.join(args.data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Authorized queries
    if not args.skip_authorized:
        print("--- Running authorized queries ---")
        auth_queries = load_queries(args.data_dir, "authorized")
        auth_results = run_query_workload(
            args.server_url, auth_queries, store_map, config,
            "Authorized", args.num_runs, args.progress_every,
        )
        all_results.extend(auth_results)

    # 2. Cross-tenant probes
    if not args.skip_probes:
        print("\n--- Running cross-tenant probes ---")
        probe_queries = load_queries(args.data_dir, "cross_tenant_probe")
        probe_results = run_query_workload(
            args.server_url, probe_queries, store_map, config,
            "Probes", args.num_runs, args.progress_every,
        )
        all_results.extend(probe_results)

    # 3. Throughput test
    throughput_results = None
    if not args.skip_throughput:
        print("\n--- Running throughput test ---")
        auth_queries = load_queries(args.data_dir, "authorized")
        throughput_results = run_throughput_test(
            args.server_url, auth_queries, store_map, config, progress_every=max(1, args.progress_every // 2),
        )

    # Save results (only if we ran queries, to avoid overwriting existing data)
    if all_results:
        results_path = os.path.join(results_dir, f"config_{config}_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nQuery results saved to {results_path} ({len(all_results)} records)")

    if throughput_results:
        throughput_path = os.path.join(results_dir, f"config_{config}_throughput.json")
        with open(throughput_path, "w") as f:
            json.dump(throughput_results, f, indent=2)
        print(f"Throughput results saved to {throughput_path}")

    # Quick summary
    print(f"\n=== Config {config} Summary ===")
    probes = [r for r in all_results if r["query_type"] == "cross_tenant_probe"]
    if probes:
        leaked = [r for r in probes if r.get("target_tenant") in r.get("retrieved_tenants", [])]
        denied = [r for r in probes if r.get("error") is not None]
        print(f"  Cross-tenant probes: {len(probes)} total")
        print(f"  Leaked: {len(leaked)} ({100*len(leaked)/len(probes):.1f}%)")
        print(f"  Access denied: {len(denied)} ({100*len(denied)/len(probes):.1f}%)")

    auth_results_only = [r for r in all_results if r["query_type"] == "authorized" and r["error"] is None]
    if auth_results_only:
        latencies = [r["total_latency_ms"] for r in auth_results_only]
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"  Authorized query latency: p50={p50:.0f}ms, p99={p99:.0f}ms")


if __name__ == "__main__":
    main()
