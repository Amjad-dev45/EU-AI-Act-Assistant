"""PERSON 2: embedding/index/search foundations.

Replace `search_chunks` with real retrieval (embedding + FAISS) later. Keep the
returned hit shape: {"chunk_id","document_id","pages","text","score"}, best first.
"""
import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_chunks(chunks: list[dict], query: str, k: int) -> list[dict]:
    """Deterministic mock: score = share of query words found in the chunk.
    Ties keep fixture order, so the same query always returns the same hits."""
    q = _tokens(query)
    scored = []
    for order, chunk in enumerate(chunks):
        overlap = len(q & _tokens(chunk["text"]))
        score = round(overlap / len(q), 4) if q else 0.0
        scored.append((-score, order, chunk, score))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [
        {"chunk_id": c["chunk_id"], "document_id": c["document_id"],
         "pages": c["pages"], "text": c["text"], "score": s}
        for _, _, c, s in scored[:k]
    ]
