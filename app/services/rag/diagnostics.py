"""Record observed retrieval stages without inventing model reasoning."""


def build_trace(question, vectors, keywords, fused, reranked, final, terms, *,
                hybrid, rerank, candidate_k, k, timings):
    """Snapshots preserve full source text and separate score scales for inspection."""
    ranks = {name: {hit["chunk_id"]: rank for rank, hit in enumerate(hits, 1)}
             for name, hits in (("faiss_rank", vectors), ("bm25_rank", keywords),
                               ("fusion_rank", fused if hybrid else []),
                               ("rerank_rank", reranked), ("selected_rank", final))}

    def rows(hits):
        return [{**{key: hit[key] for key in ("chunk_id", "source", "document_id", "pages", "text",
                                             "score", "bm25_score", "rrf_score", "rerank_score") if key in hit},
                 "rank": rank, **{name: positions.get(hit["chunk_id"]) for name, positions in ranks.items()}}
                for rank, hit in enumerate(hits, 1)]

    stages = []
    for name, label, enabled, hits in (
        ("faiss", "FAISS semantic search", True, vectors),
        ("bm25", "BM25 keyword search", hybrid, keywords),
        ("rrf", "RRF rank fusion", hybrid, fused if hybrid else []),
        ("reranker", "Cross-encoder reranking", rerank, reranked),
        ("selected", "Selected retrieval results", True, final),
    ):
        stages.append({"name": name, "label": label, "enabled": enabled,
                       "count": len(hits), "results": rows(hits)})
    return {"query": question, "policy": "hybrid" if hybrid else "vector_then_rerank" if rerank else "vector_only",
            "policy_explanation": (
                "Both FAISS and BM25 run for every search. RRF merges their ranks, then the cross-encoder reranks the merged candidates. There is no keyword-versus-semantic router."
                if hybrid else "BM25 is disabled for this run."),
            "query_terms": terms,
            "numbered_references": [term for term in terms if term.startswith(("article_", "annex_", "section_"))],
            "candidate_limit": candidate_k, "result_limit": k, "stages": stages,
            "timings_ms": timings,
            "score_note": "Cosine, BM25, RRF and reranker scores use different scales; reranker scores are not probabilities."}
