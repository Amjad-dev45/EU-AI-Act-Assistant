"""Locally score question/passage pairs, then keep the strongest evidence."""

from functools import lru_cache
import os

import numpy as np

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model():
    """Download once into the Hugging Face cache; reuse one CPU model per process."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL, device="cpu", max_length=512)


class Reranker:
    def __init__(self, enabled=None, candidates=None):
        # Read settings after the application has loaded .env.
        setting = os.getenv("RAG_RERANK_ENABLED", "true").strip().lower()
        if enabled is None:
            if setting not in {"true", "false", "1", "0"}:
                raise ValueError("RAG_RERANK_ENABLED must be true or false")
            enabled = setting in {"true", "1"}
        self.enabled = enabled
        self.candidates = int(os.getenv("RAG_RERANK_CANDIDATES", "20")) if candidates is None else candidates
        if type(self.candidates) is not int or self.candidates < 1:
            raise ValueError("RAG_RERANK_CANDIDATES must be a positive integer")

    def rank(self, question, passages, k):
        """Keep FAISS scores intact; attach a separate, non-probability relevance score."""
        if not self.enabled or not passages:
            return passages[:k]
        scores = np.asarray(load_model().predict(
            [(question, p["text"]) for p in passages],
            batch_size=16, show_progress_bar=False,
        ), dtype=float).reshape(-1)
        if len(scores) != len(passages) or not np.isfinite(scores).all():
            raise ValueError("Reranker returned invalid scores")
        ranked = [{**p, "rerank_score": float(score), "retrieval_rank": rank}
                  for rank, (p, score) in enumerate(zip(passages, scores), 1)]
        # Stable sorting preserves FAISS order for tied relevance scores.
        return sorted(ranked, key=lambda p: p["rerank_score"], reverse=True)[:k]
