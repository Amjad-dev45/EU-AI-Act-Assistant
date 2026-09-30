"""Serialized access to the existing hybrid retrieval and answer pipelines."""

from threading import RLock


# Both HTTP entry points share one lock for the reranker and embedding cache.
PIPELINE_LOCK = RLock()


def search_chunks(data_dir, query: str, k: int = 3, *, injection_test=False) -> dict:
    from .retrieval import retrieve

    with PIPELINE_LOCK:
        return retrieve(query, data_dir, k=k, **({"injection_test": True} if injection_test else {}))


def answer_query(data_dir, query: str, client=None, min_rerank_score=None, *, injection_test=False) -> dict:
    from .answering import answer_question

    with PIPELINE_LOCK:
        return answer_question(query, data_dir, client=client, min_rerank_score=min_rerank_score,
                               **({"injection_test": True} if injection_test else {}))
