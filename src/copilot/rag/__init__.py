"""Document search over the maintenance knowledge base."""

from __future__ import annotations

from pathlib import Path

from copilot.rag.chunking import load_chunks
from copilot.rag.retriever import (
    BM25Retriever,
    DenseRetriever,
    Embedder,
    FastEmbedder,
    HybridRetriever,
    load_index,
    passage_text,
    save_index,
)


def build_index(knowledge_dir: Path, index_dir: Path, embedder: Embedder, model: str) -> int:
    """Chunk and embed all documents; returns the number of chunks."""
    chunks = load_chunks(knowledge_dir)
    vectors = embedder.embed([passage_text(c) for c in chunks])
    save_index(index_dir, chunks, vectors, model)
    return len(chunks)


def open_hybrid(
    index_dir: Path, embedder: Embedder | None = None, model: str | None = None, rrf_k: int = 60
) -> HybridRetriever:
    """Load a saved index and return the hybrid retriever."""
    chunks, vectors, saved_model = load_index(index_dir)
    if model and model != saved_model:
        raise ValueError(f"index built with {saved_model}, settings say {model}: rebuild it")
    embedder = embedder or FastEmbedder(saved_model)
    return HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, embedder, vectors), rrf_k)


__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "build_index", "open_hybrid"]
