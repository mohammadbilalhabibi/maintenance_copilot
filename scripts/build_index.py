"""Chunk and embed the knowledge documents into index/ (BM25 is rebuilt at load time)."""

from __future__ import annotations

import time

from copilot.config import get_settings
from copilot.rag import build_index
from copilot.rag.retriever import FastEmbedder


def main() -> None:
    """CLI entry point."""
    s = get_settings()
    t0 = time.perf_counter()
    embedder = FastEmbedder(s.embedding_model, s.embedding_cache_dir)
    n = build_index(s.knowledge_dir, s.index_dir, embedder, s.embedding_model)
    print(f"indexed {n} chunks from {s.knowledge_dir} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
