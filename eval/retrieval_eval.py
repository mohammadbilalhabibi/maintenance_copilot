"""Retrieval-only evaluation: BM25 vs dense vs hybrid on the document questions (no LLM).

A retrieved chunk is a hit when it comes from the expected document and contains the
question's key facts (the same keyword groups the answer grader uses).
Writes eval/results/retrieval.json.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import numpy as np

from copilot.config import PROJECT_ROOT, get_settings
from copilot.evaluation import check_keywords
from copilot.rag import BM25Retriever, DenseRetriever, HybridRetriever
from copilot.rag.retriever import FastEmbedder, load_index

K = 4


def main() -> None:
    """CLI entry point."""
    s = get_settings()
    ref = json.loads((PROJECT_ROOT / "eval" / "reference_set.json").read_text())
    items = [i for i in ref["items"] if i["category"] == "document"]
    chunks, vectors, model = load_index(s.index_dir)
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, FastEmbedder(model, s.embedding_cache_dir), vectors)
    retrievers = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense, s.rrf_k)}
    out: dict = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "k": K,
                 "n_questions": len(items), "n_chunks": len(chunks), "embedding_model": model,
                 "retrievers": {}}
    for name, r in retrievers.items():
        ranks, times = [], []
        for item in items:
            groups = item["checks"][0]["groups"]
            t0 = time.perf_counter()
            hits = r.search(item["question"], 20)
            times.append(time.perf_counter() - t0)
            rank = next(
                (i + 1 for i, h in enumerate(hits)
                 if h.chunk.doc_id == item["expected_doc"]
                 and check_keywords(f"{h.chunk.section} {h.chunk.text}", groups)),
                None,
            )
            ranks.append(rank)
        out["retrievers"][name] = {
            "hit_at_1": float(np.mean([r_ == 1 for r_ in ranks])),
            f"hit_at_{K}": float(np.mean([r_ is not None and r_ <= K for r_ in ranks])),
            "mrr_at_20": float(np.mean([1 / r_ if r_ else 0 for r_ in ranks])),
            "latency_ms_p50": float(np.percentile(times, 50) * 1000),
            "ranks": dict(zip([i["id"] for i in items], ranks, strict=True)),
        }
        m = out["retrievers"][name]
        print(f"{name:6s} hit@1 {m['hit_at_1']:.2f}  hit@{K} {m[f'hit_at_{K}']:.2f}  "
              f"MRR {m['mrr_at_20']:.2f}  {m['latency_ms_p50']:.1f} ms")
    path = PROJECT_ROOT / "eval" / "results" / "retrieval.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
