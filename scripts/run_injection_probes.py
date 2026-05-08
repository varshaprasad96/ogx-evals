"""
Prompt injection adversarial testing.

Runs injection probe queries against server-side and client-side configs
to test whether the system can be manipulated into bypassing access controls.

Results are reported separately from the main experiment.

Usage:
    python scripts/run_injection_probes.py --config A|B|C|D [--server-url http://localhost:8321]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_orchestration import client_side_rag, search_vector_store

TENANTS = ["finance", "engineering", "legal"]
MODEL_ID = "openai/gpt-4o-mini"
CLIENT_SIDE_CONFIGS = {"A", "B"}
GATED_CONFIGS = {"B", "D"}
UNGATED_CONFIGS = {"A", "C"}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def print_progress(label: str, completed: int, total: int, start: float, errors: int = 0) -> None:
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
    return OpenAI(base_url=f"{server_url}/v1", api_key=api_key)


def load_injection_probes(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, "queries", "injection_probes.json")
    with open(path) as f:
        return json.load(f)


def load_store_map(data_dir: str, config: str) -> dict:
    path = os.path.join(data_dir, "results", f"store_map_{config}.json")
    with open(path) as f:
        return json.load(f)


def get_all_store_ids(store_map: dict) -> list[str]:
    """Get all vector store IDs."""
    return list(store_map.values())


def run_injection_client_side(
    server_url: str,
    probe: dict,
    store_map: dict,
    config: str,
) -> dict:
    """Run an injection probe via client-side orchestration."""
    tenant = probe["tenant"]
    query_text = probe["query_text"]
    user_idx = probe.get("user_idx", 0)

    client = get_client(server_url, tenant=tenant if config in GATED_CONFIGS else None, user_idx=user_idx)

    # For injection probes, we try to search the user's OWN vector store
    # The injection is in the query text, attempting to manipulate the LLM
    if config in UNGATED_CONFIGS:
        vs_id = store_map["shared"]
    else:
        vs_id = store_map[tenant]

    start = time.perf_counter()
    result = client_side_rag(client, query_text, vs_id, MODEL_ID)
    total_ms = (time.perf_counter() - start) * 1000

    # Check what tenants appear in retrieved chunks
    retrieved_tenants = set()
    for chunk in result.retrieved_chunks:
        attrs = chunk.get("attributes", {})
        if isinstance(attrs, dict) and "tenant_id" in attrs:
            retrieved_tenants.add(attrs["tenant_id"])

    # Determine if cross-tenant data leaked
    other_tenants = retrieved_tenants - {tenant}
    leaked = len(other_tenants) > 0

    return {
        "query_id": probe["query_id"],
        "tenant": tenant,
        "query_type": "prompt_injection",
        "query_text": query_text,
        "config": config,
        "orchestration": "client",
        "retrieved_chunk_count": len(result.retrieved_chunks),
        "retrieved_tenants": list(retrieved_tenants),
        "other_tenant_data_leaked": leaked,
        "leaked_tenants": list(other_tenants),
        "total_latency_ms": total_ms,
        "response_text": result.response_text[:500],  # Truncate for storage
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_injection_server_side(
    server_url: str,
    probe: dict,
    store_map: dict,
    config: str,
) -> dict:
    """Run an injection probe via server-side Responses API."""
    tenant = probe["tenant"]
    query_text = probe["query_text"]
    user_idx = probe.get("user_idx", 0)

    client = get_client(server_url, tenant=tenant if config in GATED_CONFIGS else None, user_idx=user_idx)

    if config in UNGATED_CONFIGS:
        vs_ids = [store_map["shared"]]
    else:
        vs_ids = [store_map[tenant]]

    start = time.perf_counter()
    error = None
    retrieved_tenants = set()
    retrieved_chunk_count = 0
    response_text = ""

    try:
        response = client.responses.create(
            model=MODEL_ID,
            input=query_text,
            tools=[{
                "type": "file_search",
                "vector_store_ids": vs_ids,
                "max_num_results": 5,
            }],
        )
        total_ms = (time.perf_counter() - start) * 1000

        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type"):
                    if item.type == "file_search_call" and hasattr(item, "results"):
                        results = item.results or []
                        retrieved_chunk_count = len(results)
                        for r in results:
                            attrs = getattr(r, "attributes", {}) or {}
                            if isinstance(attrs, dict) and "tenant_id" in attrs:
                                retrieved_tenants.add(attrs["tenant_id"])
                    elif item.type == "message" and hasattr(item, "content"):
                        for c in item.content:
                            if hasattr(c, "text"):
                                response_text += c.text

    except Exception as e:
        total_ms = (time.perf_counter() - start) * 1000
        error = str(e)

    other_tenants = retrieved_tenants - {tenant}
    leaked = len(other_tenants) > 0

    return {
        "query_id": probe["query_id"],
        "tenant": tenant,
        "query_type": "prompt_injection",
        "query_text": query_text,
        "config": config,
        "orchestration": "server",
        "retrieved_chunk_count": retrieved_chunk_count,
        "retrieved_tenants": list(retrieved_tenants),
        "other_tenant_data_leaked": leaked,
        "leaked_tenants": list(other_tenants),
        "total_latency_ms": total_ms,
        "response_text": response_text[:500],
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run prompt injection probes")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--server-url", type=str, default="http://localhost:8321")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N probes")
    args = parser.parse_args()

    config = args.config
    store_map = load_store_map(args.data_dir, config)
    probes = load_injection_probes(args.data_dir)

    print(f"=== Running Injection Probes for Config {config} ===")
    print(f"  Orchestration: {'client-side' if config in CLIENT_SIDE_CONFIGS else 'server-side'}")
    print(f"  Retrieval: {'gated' if config in GATED_CONFIGS else 'ungated'}")
    print(f"  Total probes: {len(probes)}")
    print(f"  Progress interval: {args.progress_every} probes")
    print()

    results = []
    start = time.perf_counter()
    errors = 0
    progress_every = max(1, args.progress_every)
    for i, probe in enumerate(probes):
        if config in CLIENT_SIDE_CONFIGS:
            result = run_injection_client_side(args.server_url, probe, store_map, config)
        else:
            result = run_injection_server_side(args.server_url, probe, store_map, config)

        results.append(result)
        if result.get("error") is not None:
            errors += 1

        completed = i + 1
        if completed == 1 or completed % progress_every == 0 or completed == len(probes):
            print_progress("Injection", completed, len(probes), start, errors)

    # Save results
    results_dir = os.path.join(args.data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"config_{config}_injection_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nInjection results saved to {results_path}")

    # Summary
    total = len(results)
    leaked = sum(1 for r in results if r["other_tenant_data_leaked"])
    errors = sum(1 for r in results if r["error"] is not None)
    denied = sum(1 for r in results if r["error"] and "not found" in r["error"].lower())

    print(f"\n=== Injection Probe Summary (Config {config}) ===")
    print(f"  Total probes: {total}")
    print(f"  Cross-tenant data leaked: {leaked} ({100*leaked/total:.1f}%)")
    print(f"  Access denied (expected for gated): {denied}")
    print(f"  Other errors: {errors - denied}")

    if config in UNGATED_CONFIGS:
        print(f"\n  NOTE: Under ungated retrieval, injection probes may retrieve")
        print(f"  cross-tenant data through normal relevance-based search,")
        print(f"  not necessarily through successful prompt injection.")
    elif leaked > 0:
        print(f"\n  WARNING: Cross-tenant leakage detected under gated config!")
        print(f"  This would indicate a security vulnerability.")
    else:
        print(f"\n  PASS: No cross-tenant leakage under gated retrieval.")


if __name__ == "__main__":
    main()
