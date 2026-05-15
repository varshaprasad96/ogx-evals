import os
"""
Client-side RAG orchestration for Configs A and B.

Implements the manual client-side loop:
  1. Search vector store for relevant chunks
  2. Assemble context from retrieved chunks
  3. Call chat completions with context

This is the client-side alternative to server-side Responses API.

Usage:
    Imported by run_experiment.py, not run standalone.
"""

import time
from dataclasses import dataclass

from openai import OpenAI


MODEL_ID = os.environ.get("OGX_MODEL_ID", "openai/gpt-4o-mini")


@dataclass
class ClientRAGResult:
    """Result from a client-side RAG query."""
    query_text: str
    retrieved_chunks: list[dict]  # [{content, file_id, score, attributes}]
    response_text: str
    search_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    error: str | None = None


def search_vector_store(
    client: OpenAI,
    vector_store_id: str,
    query: str,
    max_results: int = 5,
) -> tuple[list[dict], float]:
    """
    Search a vector store and return chunks with timing.
    Returns (chunks, latency_ms).
    """
    start = time.perf_counter()
    try:
        results = client.vector_stores.search(
            vector_store_id=vector_store_id,
            query=query,
            max_num_results=max_results,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        chunks = []
        for item in results.data:
            chunk_content = ""
            file_id = getattr(item, "file_id", None)
            score = getattr(item, "score", 0.0)
            attributes = getattr(item, "attributes", {}) or {}

            # Extract text content
            if hasattr(item, "content") and item.content:
                for content_block in item.content:
                    if hasattr(content_block, "text"):
                        chunk_content += content_block.text

            chunks.append({
                "content": chunk_content,
                "file_id": file_id,
                "score": score,
                "attributes": attributes,
            })

        return chunks, elapsed_ms

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return [], elapsed_ms


def call_chat_completions(
    client: OpenAI,
    query: str,
    context: str,
    model: str = MODEL_ID,
) -> tuple[str, float]:
    """
    Call chat completions with retrieved context.
    Returns (response_text, latency_ms).
    """
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question "
                        "based on the provided context. If the context doesn't contain "
                        "relevant information, say so.\n\n"
                        f"Context:\n{context}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=512,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content or ""
        return text, elapsed_ms

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return f"Error: {e}", elapsed_ms


def client_side_rag(
    client: OpenAI,
    query: str,
    vector_store_id: str,
    model: str = MODEL_ID,
    max_results: int = 5,
) -> ClientRAGResult:
    """
    Execute a full client-side RAG query:
    search -> assemble context -> inference.
    """
    total_start = time.perf_counter()

    # Step 1: Search
    chunks, search_latency = search_vector_store(
        client, vector_store_id, query, max_results
    )

    if not chunks:
        total_elapsed = (time.perf_counter() - total_start) * 1000
        return ClientRAGResult(
            query_text=query,
            retrieved_chunks=chunks,
            response_text="No relevant documents found.",
            search_latency_ms=search_latency,
            inference_latency_ms=0.0,
            total_latency_ms=total_elapsed,
            error="no_results" if search_latency > 0 else "search_failed",
        )

    # Step 2: Assemble context
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[{i+1}] {chunk['content']}")
    context = "\n\n".join(context_parts)

    # Step 3: Call inference
    response_text, inference_latency = call_chat_completions(
        client, query, context, model
    )

    total_elapsed = (time.perf_counter() - total_start) * 1000

    return ClientRAGResult(
        query_text=query,
        retrieved_chunks=chunks,
        response_text=response_text,
        search_latency_ms=search_latency,
        inference_latency_ms=inference_latency,
        total_latency_ms=total_elapsed,
    )
