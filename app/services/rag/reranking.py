"""Score question/passage pairs locally, keeping FAISS scores separate."""

from functools import lru_cache

import numpy as np

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


# Load once per process. CPU execution also works on teammates' laptops.
@lru_cache(maxsize=1)
def load_model():
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(min(4, torch.get_num_threads()))
    return CrossEncoder(
        MODEL, revision=REVISION, device="cpu", max_length=512,
        activation_fn=torch.nn.Identity(), trust_remote_code=False,
    )


def rerank(question, hits, k=5, model=None):
    """Return new hit dictionaries ordered by raw cross-encoder relevance score."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be blank.")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")
    if not hits:
        return []
    model = load_model() if model is None else model
    pairs = [(question, hit["text"]) for hit in hits]
    scores = np.asarray(model.predict(pairs, batch_size=16, show_progress_bar=False), dtype=float).reshape(-1)
    if scores.shape != (len(hits),) or not np.isfinite(scores).all():
        raise ValueError("Reranker must return one finite score per candidate.")

    # Stable sorting preserves vector order on ties; input hits stay unchanged.
    ranked = [{**hit, "rerank_score": float(score)} for hit, score in zip(hits, scores)]
    return sorted(ranked, key=lambda hit: hit["rerank_score"], reverse=True)[:k]
