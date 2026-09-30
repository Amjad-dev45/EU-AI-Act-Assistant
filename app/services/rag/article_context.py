"""Optionally recover the article governing an authoritative retrieved passage."""

from time import perf_counter

from .section_expansion import (
    MAX_SECTION_CHARS, MAX_SECTION_CHUNKS, _headings, _valid_chunk, expand_section,
)
from .vector_index import load_chunks


def _valid_passage(passage):
    return (isinstance(passage, dict)
            and all(isinstance(passage.get(key), str) and passage[key].strip()
                    for key in ("chunk_id", "source", "document_id", "text"))
            and _valid_chunk(passage))


def _valid_hits(hits):
    return isinstance(hits, list) and bool(hits) and all(_valid_passage(hit) for hit in hits)


def assemble_article_context(hits, chunks, max_chars=MAX_SECTION_CHARS,
                             max_chunks=MAX_SECTION_CHUNKS):
    """Keep the complete governing article and remaining hits, or return None.

    The highest-ranked passage must fit wholly inside one verified Article.
    Failure is optional-context fallback, not evidence that a question cannot
    be answered from the original retrieved passages.
    """
    if (not _valid_hits(hits) or not isinstance(chunks, list) or not chunks
            or any(not _valid_passage(chunk) for chunk in chunks)
            or type(max_chars) is not int or max_chars <= 0
            or type(max_chunks) is not int or max_chunks <= 0):
        return None
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        return None
    for hit in hits:
        original = by_id.get(hit["chunk_id"])
        if original is None or any(hit[key] != original[key] for key in
                                   ("source", "document_id", "text", "start_char", "end_char", "pages")):
            return None

    top = hits[0]
    document = [chunk for chunk in chunks
                if (chunk["source"], chunk["document_id"]) == (top["source"], top["document_id"])]
    headings = {}
    for chunk in document:
        for offset, label in _headings(chunk):
            headings.setdefault(offset, set()).add(label)
    if any(len(labels) != 1 for labels in headings.values()):
        return None
    ordered = sorted((offset, next(iter(labels))) for offset, labels in headings.items())
    preceding = [(offset, label) for offset, label in ordered if offset <= top["start_char"]]
    if not preceding or not preceding[-1][1].startswith("Article "):
        return None
    start, heading = preceding[-1]
    if sum(label == heading for _, label in ordered) != 1:
        return None  # Repeated headings cannot establish one unambiguous section.
    following = [offset for offset, _ in ordered if offset > start]
    if not following or top["end_char"] > following[0]:
        return None
    anchor = next(chunk for chunk in document if (start, heading) in _headings(chunk))
    expanded = expand_section(None, [anchor, *hits], chunks, max_chars, max_chunks, heading=heading)
    if not expanded["complete"]:
        return None

    # A closing-heading chunk can contribute only its preceding blank separator.
    combined = [hit for hit in expanded["hits"] if hit["text"].strip()]
    article_chunk_count = len(combined)
    article_start = min(hit["excerpt_start_char"] for hit in combined)
    article_end = max(hit["excerpt_end_char"] for hit in combined)
    retained = set()
    for hit in hits:
        fully_covered = ((hit["source"], hit["document_id"]) == (top["source"], top["document_id"])
                         and article_start <= hit["start_char"] and hit["end_char"] <= article_end)
        if not fully_covered and hit["chunk_id"] not in retained:
            combined.append(dict(hit))
            retained.add(hit["chunk_id"])
    characters = sum(len(hit["text"]) for hit in combined)
    if characters > max_chars or len(combined) > max_chunks:
        return None
    return {**expanded, "mode": "article", "hits": combined,
            "article_chunk_count": article_chunk_count,
            "characters": characters, "chunk_count": len(combined)}


def expand_article_context(hits, data_dir):
    """Load source chunks once for valid hits; unavailable context stays optional."""
    if not _valid_hits(hits):
        return None
    started = perf_counter()
    try:
        chunks, _ = load_chunks(data_dir)
    except (OSError, ValueError):
        return None
    result = assemble_article_context(hits, chunks)
    if result is not None:
        result["elapsed_seconds"] = perf_counter() - started
    return result
