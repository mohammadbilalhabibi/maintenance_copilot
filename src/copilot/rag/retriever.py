"""Document retrievers sharing one interface: BM25, dense (bge-small), hybrid (RRF)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from copilot.rag.chunking import Chunk

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Hit:
    """A retrieved chunk with its score."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    """Common interface so evaluation can swap retrievers."""

    def search(self, query: str, k: int) -> list[Hit]:
        """Return the k best chunks for a query."""
        ...


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens."""
    return TOKEN.findall(text.lower())


class BM25Retriever:
    """Keyword search over section heading + text."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(f"{c.title} {c.section} {c.text}") for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in idx if scores[i] > 0]


class Embedder(Protocol):
    """Anything that turns texts into normalised vectors."""

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """fastembed (ONNX, no torch) wrapper; loaded once."""

    def __init__(
        self, model_name: str, cache_dir: Path | None = None, batch_size: int = 16
    ) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        # fastembed's default batch of 256 needed 3.9 GB RAM for our chunks (16: 0.85 GB).
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts, batch_size=self.batch_size)),
                        dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class DenseRetriever:
    """Cosine similarity over precomputed passage embeddings."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray) -> None:
        self.chunks, self.embedder, self.vectors = chunks, embedder, vectors

    def search(self, query: str, k: int) -> list[Hit]:
        q = self.embedder.embed([QUERY_PREFIX + query])[0]
        scores = self.vectors @ q
        idx = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in idx]


class HybridRetriever:
    """Reciprocal Rank Fusion of BM25 and dense rankings."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60) -> None:
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    @property
    def chunks(self) -> list[Chunk]:
        """All indexed chunks."""
        return self.bm25.chunks

    def search(self, query: str, k: int) -> list[Hit]:
        depth = max(20, k * 4)
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranking in (self.bm25.search(query, depth), self.dense.search(query, depth)):
            for rank, hit in enumerate(ranking):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[cid] = hit.chunk
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(by_id[c], fused[c]) for c in best]


def passage_text(c: Chunk) -> str:
    """Text embedded for a passage (title and section give context)."""
    return f"{c.title} - {c.section}: {c.text}"


def save_index(index_dir: Path, chunks: list[Chunk], vectors: np.ndarray, model: str) -> None:
    """Persist chunks and embeddings."""
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", vectors)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict()) + "\n")
    (index_dir / "meta.json").write_text(json.dumps({"model": model, "n": len(chunks)}))


def load_index(index_dir: Path) -> tuple[list[Chunk], np.ndarray, str]:
    """Load chunks, embeddings and the embedding model name."""
    chunks = [
        Chunk(**json.loads(line))
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    meta = json.loads((index_dir / "meta.json").read_text())
    return chunks, np.load(index_dir / "vectors.npy"), meta["model"]
