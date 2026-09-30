"""Shared retrieval pipeline: vectors, optional BM25 fusion, then reranking."""

import time

from .embeddings import embed_texts
from .diagnostics import build_trace
from .injection_demo import INJECTION_TEST
from .vector_index import load_index, search_index


def retrieve(question, data_dir, k=5, candidate_k=20, use_rerank=True, use_hybrid=True, *, injection_test=INJECTION_TEST):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be blank.")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")
    if type(candidate_k) is not int or candidate_k < 1:
        raise ValueError("Candidate count must be a positive integer.")
    if use_rerank and candidate_k < k:
        raise ValueError("Candidate count must be at least k when reranking.")

    # Verify saved artifacts before embedding; unchanged queries reuse disk cache.
    started = time.perf_counter()
    index, chunks, metadata = load_index(data_dir)
    fixture_tokens = 0
    if injection_test:
        from .injection_demo import add_test_chunk
        index, chunks, fixture_tokens = add_test_chunk(index, chunks, data_dir)
    vector, tokens = embed_texts([question], data_dir)
    embedded = time.perf_counter()
    candidates = search_index(index, chunks, vector[0], candidate_k if use_rerank else k)
    vectors = candidates
    vector_hits = candidates[:k]
    vector_done = time.perf_counter()

    # Disabling reranking preserves the original vector-only baseline.
    hybrid_enabled = use_rerank and use_hybrid
    lexical_trace = {}
    if hybrid_enabled:
        from .hybrid import BM25Index, hybrid_candidates
        candidates = hybrid_candidates(index, vector[0], question, candidates, BM25Index(chunks), candidate_k,
                                       trace=lexical_trace)
    fusion_done = time.perf_counter()

    # Baseline mode never imports or loads the cross-encoder.
    if use_rerank:
        from .reranking import rerank
        # The model already scores every candidate. Keep that order for the
        # trace before taking the same top-k results used by the answer pipeline.
        ranked = rerank(question, candidates, max(1, len(candidates)))
        hits = ranked[:k]
    else:
        ranked = []
        hits = vector_hits
    finished = time.perf_counter()
    trace = build_trace(question, vectors, lexical_trace.get("keyword_hits", []), candidates,
                        ranked, hits, lexical_trace.get("query_terms", []),
                        hybrid=hybrid_enabled, rerank=use_rerank, candidate_k=candidate_k, k=k,
                        timings={"index_and_embedding": round((embedded - started) * 1000, 2),
                                 "faiss": round((vector_done - embedded) * 1000, 2),
                                 "bm25_and_fusion": round((fusion_done - vector_done) * 1000, 2),
                                 "reranking": round((finished - fusion_done) * 1000, 2)})
    trace["injection_test"] = bool(injection_test)
    trace["index"] = {"chunk_count": len(chunks), "embedding_model": metadata.get("model"),
                      "dimensions": metadata.get("dimensions"),
                      "largest_chunk_characters": max((len(c["text"]) for c in chunks), default=0)}
    return {
        "hits": hits, "vector_hits": vector_hits, "reranked": use_rerank, "hybrid": hybrid_enabled,
        "candidate_count": len(candidates), "embedding_tokens": tokens + fixture_tokens, "injection_test": bool(injection_test),
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
    }
