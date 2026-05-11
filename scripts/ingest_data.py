"""
Ingest synthetic documents into OGX vector stores.

For ungated configs (A, C): Creates a single shared vector store and uploads all 300 documents.
For gated configs (B, D): Creates 3 per-tenant vector stores, each via a tenant-authenticated
client, so ABAC ownership is set correctly.

Usage:
    python scripts/ingest_data.py --config A|B|C|D [--server-url http://localhost:8321]
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI


TENANTS = ["finance", "engineering", "legal"]

# Configs A and C are ungated; B and D are gated
GATED_CONFIGS = {"B", "D"}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def print_progress(label: str, completed: int, total: int, start: float) -> None:
    elapsed = time.time() - start
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = (total - completed) / rate if rate > 0 else 0
    pct = completed / total * 100 if total else 100
    print(
        f"  [{label}] {completed}/{total} ({pct:5.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)} "
        f"rate={rate * 60:.1f}/min",
        flush=True,
    )


def get_client(server_url: str, tenant: str | None = None, user_idx: int = 0) -> OpenAI:
    """Create an OpenAI client pointing at OGX."""
    if tenant:
        api_key = f"token-{tenant}-{user_idx}"
    else:
        api_key = "no-auth"

    return OpenAI(base_url=f"{server_url}/v1", api_key=api_key)


def load_manifest(data_dir: str) -> list[dict]:
    manifest_path = os.path.join(data_dir, "documents", "manifest.json")
    with open(manifest_path) as f:
        return json.load(f)


def create_vector_store(
    client: OpenAI,
    name: str,
    embedding_model: str = "openai/text-embedding-3-small",
    embedding_dimension: int = 1536,
) -> str:
    """Create a vector store and return its ID."""
    vs = client.vector_stores.create(
        name=name,
        extra_body={
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        },
    )
    print(f"  Created vector store '{name}' -> {vs.id}")
    return vs.id


def upload_file(client: OpenAI, filepath: str) -> str:
    """Upload a file and return its ID."""
    with open(filepath, "rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")
    return file_obj.id


def attach_file_to_vector_store(
    client: OpenAI,
    vector_store_id: str,
    file_id: str,
    attributes: dict | None = None,
) -> None:
    """Attach a file to a vector store with optional metadata attributes."""
    extra_body = {}
    if attributes:
        extra_body["attributes"] = attributes

    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id,
        **({"extra_body": extra_body} if extra_body else {}),
    )


def wait_for_file_processing(client: OpenAI, vector_store_id: str, file_id: str, timeout: int = 120) -> bool:
    """Poll until the file is processed in the vector store."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            vs_file = client.vector_stores.files.retrieve(
                vector_store_id=vector_store_id,
                file_id=file_id,
            )
            if vs_file.status == "completed":
                return True
            if vs_file.status == "failed":
                print(f"    File {file_id} processing failed: {vs_file.last_error}")
                return False
        except Exception:
            pass
        time.sleep(1)
    print(f"    File {file_id} processing timed out after {timeout}s")
    return False


def ingest_ungated(server_url: str, data_dir: str, embedding_model: str, embedding_dimension: int) -> dict:
    """Ingest all documents into a single shared vector store (no auth)."""
    client = get_client(server_url)
    manifest = load_manifest(data_dir)

    print("Ingesting into shared vector store (ungated)...")
    vs_id = create_vector_store(client, "shared-all-tenants", embedding_model, embedding_dimension)

    store_map = {"shared": vs_id}
    start = time.time()

    for i, doc in enumerate(manifest):
        filepath = os.path.join(data_dir, "documents", doc["filename"])
        file_id = upload_file(client, filepath)

        attach_file_to_vector_store(
            client, vs_id, file_id,
            attributes={
                "tenant_id": doc["tenant_id"],
                "department": doc["department"],
                "sensitivity": doc["sensitivity"],
            },
        )

        if (i + 1) == 1 or (i + 1) % 10 == 0 or (i + 1) == len(manifest):
            print_progress("Ingest shared", i + 1, len(manifest), start)

    print(f"  Waiting for file processing to complete...")
    # Wait a bit for batch processing
    time.sleep(5)

    print(f"  Ingested {len(manifest)} documents into shared vector store {vs_id}")
    return store_map


def ingest_gated(server_url: str, data_dir: str, embedding_model: str, embedding_dimension: int) -> dict:
    """Ingest documents into per-tenant vector stores (with auth)."""
    manifest = load_manifest(data_dir)

    print("Ingesting into per-tenant vector stores (gated)...")
    store_map = {}

    for tenant in TENANTS:
        # Create vector store as tenant user (sets owner)
        client = get_client(server_url, tenant=tenant, user_idx=0)
        vs_id = create_vector_store(client, f"vs-{tenant}", embedding_model, embedding_dimension)
        store_map[tenant] = vs_id

        # Upload only this tenant's documents
        tenant_docs = [d for d in manifest if d["tenant_id"] == tenant]
        start = time.time()

        for i, doc in enumerate(tenant_docs):
            filepath = os.path.join(data_dir, "documents", doc["filename"])
            file_id = upload_file(client, filepath)

            attach_file_to_vector_store(
                client, vs_id, file_id,
                attributes={
                    "tenant_id": doc["tenant_id"],
                    "department": doc["department"],
                    "sensitivity": doc["sensitivity"],
                },
            )

            if (i + 1) == 1 or (i + 1) % 10 == 0 or (i + 1) == len(tenant_docs):
                print_progress(f"Ingest {tenant}", i + 1, len(tenant_docs), start)

        print(f"  [{tenant}] Ingested {len(tenant_docs)} documents into {vs_id}")

    # Wait for processing
    print(f"  Waiting for file processing to complete...")
    time.sleep(5)

    return store_map


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into OGX")
    parser.add_argument("--config", type=str, required=True, choices=["A", "B", "C", "D"],
                        help="Experiment configuration")
    parser.add_argument("--server-url", type=str, default="http://localhost:8321",
                        help="OGX server URL")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory with generated documents")
    parser.add_argument("--embedding-model", type=str, default="openai/text-embedding-3-small",
                        help="Embedding model identifier (default: openai/text-embedding-3-small)")
    parser.add_argument("--embedding-dimension", type=int, default=1536,
                        help="Embedding vector dimension (default: 1536 for text-embedding-3-small)")
    args = parser.parse_args()

    is_gated = args.config in GATED_CONFIGS

    if is_gated:
        store_map = ingest_gated(args.server_url, args.data_dir, args.embedding_model, args.embedding_dimension)
    else:
        store_map = ingest_ungated(args.server_url, args.data_dir, args.embedding_model, args.embedding_dimension)

    # Save store mapping for the experiment runner
    output_path = os.path.join(args.data_dir, "results", f"store_map_{args.config}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(store_map, f, indent=2)

    print(f"\nStore mapping saved to {output_path}")
    print(f"Config {args.config} ({'gated' if is_gated else 'ungated'}) ingestion complete.")


if __name__ == "__main__":
    main()
