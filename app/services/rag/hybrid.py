"""BM25 keyword search and rank fusion over the same chunks as FAISS."""

import re

import numpy as np
from rank_bm25 import BM25Okapi


RRF_K = 60
# Remove common question words, but retain numbers, acronyms, and negation.
STOPWORDS = frozenset("a an the and or of to in on for with by from as is are was were be been being it its this that these those what which who when where how do does did have has had can could would should".split())


def tokenize(text):
    """Use identical preprocessing for stored chunks and incoming questions."""
    text = text.casefold()
    words = [word for word in re.findall(r"\w+", text) if word not in STOPWORDS]
    # Preserve numbered references as extra terms: Article 4 differs from any
    # paragraph numbered 4. The rule also covers sections and Roman-numeral annexes.
    references = re.findall(r"\b(article|annex|section)\s+([0-9]+|[ivxlcdm]+)\b", text)
    return words + [f"{kind}_{number}" for kind, number in references]


class BM25Index:
    """Build a small in-memory keyword index; no API calls or extra saved files."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.row_by_id = {chunk["chunk_id"]: row for row, chunk in enumerate(chunks)}
        tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.token_sets = [set(words) for words in tokens]
        self.model = BM25Okapi(tokens) if any(tokens) else None

    def search(self, question, k=20):
        if type(k) is not int or k < 1:
            raise ValueError("Keyword result count must be positive.")
        tokens = tokenize(question)
        if not tokens or self.model is None:
            return []
        scores = self.model.get_scores(tokens)
        query_terms = set(tokens)
        # Overlap is required: an unknown query must not return arbitrary chunks.
        # Matching BM25 scores can be zero or negative in very small corpora.
        matching = [row for row, terms in enumerate(self.token_sets)
                    if terms & query_terms and np.isfinite(scores[row])]
        matching.sort(key=lambda row: -scores[row])
        return [{**self.chunks[row], "bm25_score": float(scores[row])} for row in matching[:k]]


def reciprocal_rank_fusion(vector_hits, keyword_hits, k=20, rank_constant=RRF_K):
    """Combine ranks, not incompatible cosine/BM25 score magnitudes."""
    if type(k) is not int or k < 1 or type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("Result count and rank constant must be positive integers.")
    merged = {}
    for branch in (vector_hits, keyword_hits):
        seen = set()
        for rank, hit in enumerate(branch, 1):
            chunk_id = hit["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in merged:
                merged[chunk_id] = {**hit, "rrf_score": 0.0}
            elif "bm25_score" in hit:
                merged[chunk_id]["bm25_score"] = hit["bm25_score"]
            merged[chunk_id]["rrf_score"] += 1 / (rank_constant + rank)
    # Stable sorting breaks equal-score ties by vector order, then keyword order.
    return sorted(merged.values(), key=lambda hit: -hit["rrf_score"])[:k]


def hybrid_candidates(index, query_vector, question, vector_hits, lexical_index, k=20, *, trace=None):
    """Fuse candidate lists and preserve a true cosine score for every result."""
    keyword_hits = lexical_index.search(question, k)
    if trace is not None:
        trace["keyword_hits"] = keyword_hits
        trace["query_terms"] = tokenize(question)
    hits = reciprocal_rank_fusion(vector_hits, keyword_hits, k)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(query)
    if not np.isfinite(query).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Query vector must be finite and nonzero.")
    query = query / norm
    for hit in hits:
        if "score" not in hit:
            row = lexical_index.row_by_id[hit["chunk_id"]]
            # IndexFlatIP stores normalized vectors, so this dot product is cosine.
            hit["score"] = float(np.dot(index.reconstruct(row), query))
    return hits
