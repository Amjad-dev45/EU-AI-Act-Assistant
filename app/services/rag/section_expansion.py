"""Assemble a complete retrieved article or annex for explicit list questions."""

import re
from time import perf_counter

from .vector_index import load_chunks

MAX_SECTION_CHUNKS = 24
MAX_SECTION_CHARS = 20000

LIST_INTENT = re.compile(r"^\s*(?:list\b|enumerate\b|which\b|what\b[^?]*\bare\b)", re.I)
REFERENCE = re.compile(r"\b(?:Article\s+(\d+[a-z]?)|Annex\s+([IVXLCDM]+))\b", re.I)
# Headings stand on their own line; a page join can put one after '. '.
HEADING = re.compile(
    r"(?:\A|(?<=\n)|(?<=\. ))[ \t]*"
    r"(?P<label>Article[ \t]+\d+[a-z]?|ANNEX[ \t]+[IVXLCDM]+|"
    r"CHAPTER[ \t]+(?:[IVXLCDM]+|\d+)|SECTION[ \t]+(?:[IVXLCDM]+|\d+))"
    r"[ \t]*\r?(?=\n|\Z)", re.I,
)


def _canonical(label):
    kind, number = label.split()
    return f"{kind.title()} {number.upper() if kind.lower() == 'annex' else number.lower()}"


def requested_section(question):
    """Only expand list requests naming one unique article or annex."""
    if not LIST_INTENT.search(question):
        return None
    references = {
        f"Article {article.lower()}" if article else f"Annex {annex.upper()}"
        for article, annex in REFERENCE.findall(question)
    }
    return next(iter(references)) if len(references) == 1 else None


def _result(hits, heading, max_chars, max_chunks):
    return {"hits": [dict(hit) for hit in hits], "requested": heading is not None,
            "applied": False, "complete": None if heading is None else False,
            "heading": heading, "reason": None,
            "characters": sum(len(hit["text"]) for hit in hits), "chunk_count": len(hits),
            "max_characters": max_chars, "max_chunks": max_chunks}


def _valid_chunk(chunk):
    start, end, text = chunk.get("start_char"), chunk.get("end_char"), chunk.get("text")
    pages = chunk.get("pages")
    return (type(start) is int and type(end) is int and 0 <= start < end
            and isinstance(text, str) and end - start == len(text)
            and isinstance(pages, list) and bool(pages)
            and all(type(page) is int and page > 0 for page in pages))


def _headings(chunk):
    return [(chunk["start_char"] + match.start("label"), _canonical(match["label"]))
            for match in HEADING.finditer(chunk["text"])]


def expand_section(question, hits, chunks, max_chars=MAX_SECTION_CHARS,
                   max_chunks=MAX_SECTION_CHUNKS, *, heading=None):
    """Use saved offsets to include each source character at most once."""
    if type(max_chars) is not int or max_chars <= 0 or type(max_chunks) is not int or max_chunks <= 0:
        raise ValueError("Section context limits must be positive integers.")
    heading = requested_section(question) if heading is None else heading
    result = _result(hits if heading is None else [], heading, max_chars, max_chunks)
    if heading is None:
        return result

    def fail(reason):
        result["reason"] = reason
        return result

    # A section must already have a heading in a retrieved, authoritative chunk.
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        return fail("missing_context")
    anchors = set()
    scores = {}
    for hit in hits:
        chunk = by_id.get(hit.get("chunk_id"))
        if chunk is None or not _valid_chunk(chunk) or any(
            hit.get(key) != chunk.get(key)
            for key in ("source", "document_id", "text", "start_char", "end_char")
        ):
            return fail("missing_context")
        scores[chunk["chunk_id"]] = {key: hit[key] for key in
                                   ("score", "bm25_score", "rrf_score", "rerank_score") if key in hit}
        anchors.update((chunk["source"], chunk["document_id"], offset)
                       for offset, label in _headings(chunk) if label == heading)
    if not anchors:
        return fail("section_not_retrieved")
    if len(anchors) != 1:
        return fail("ambiguous_section")
    source, document_id, start = anchors.pop()
    document = sorted((chunk for chunk in chunks
                       if (chunk["source"], chunk["document_id"]) == (source, document_id)),
                      key=lambda chunk: (chunk.get("start_char", -1), chunk.get("end_char", -1)))
    if any(not _valid_chunk(chunk) for chunk in document):
        return fail("missing_context")

    # Articles stop at the next structural heading; annexes include their sections.
    boundaries = {offset for chunk in document for offset, label in _headings(chunk)
                  if offset > start and (not heading.startswith("Annex ") or label.startswith("Annex "))}
    if not boundaries:
        return fail("missing_boundary")
    end = min(boundaries)
    selected = [chunk for chunk in document if chunk["start_char"] < end and chunk["end_char"] > start]
    # Include the closing heading's page so an extraction gap cannot look complete.
    closing = [chunk for chunk in document if any(offset == end for offset, _ in _headings(chunk))]
    pages = {page for chunk in selected + closing for page in chunk["pages"]}
    if pages and len(pages) != max(pages) - min(pages) + 1:
        return fail("missing_context")

    # Check continuity and overlapping text before trimming repeated characters.
    cursor, known_text, expanded = start, [], []
    for chunk in selected:
        left, right = max(start, chunk["start_char"]), min(end, chunk["end_char"])
        if left > cursor + 2:
            return fail("missing_context")
        overlap_end = min(right, cursor)
        if left < overlap_end:
            existing = known_text[left - start:overlap_end - start]
            overlapping = chunk["text"][left - chunk["start_char"]:overlap_end - chunk["start_char"]]
            if any(old is not None and old != new for old, new in zip(existing, overlapping)):
                return fail("missing_context")
        if right <= cursor:
            continue
        # Chunking omits up to two separator characters between sentence chunks.
        known_text.extend([None] * max(0, left - cursor))
        left = max(left, cursor)
        text = chunk["text"][left - chunk["start_char"]:right - chunk["start_char"]]
        known_text.extend(text)
        expanded.append({**chunk, **scores.get(chunk["chunk_id"], {}), "text": text,
                         "excerpt_start_char": left, "excerpt_end_char": right,
                         "section_heading": heading})
        cursor = right
    if end - cursor > 2:
        return fail("missing_context")
    characters = sum(len(chunk["text"]) for chunk in expanded)
    result.update(characters=characters, chunk_count=len(expanded))
    if characters > max_chars or len(expanded) > max_chunks:
        return fail("context_limit")
    result.update(hits=expanded, applied=True, complete=True)
    return result


def expand_context(question, hits, data_dir, enabled=True):
    """Load saved chunks only when a section expansion is actually requested."""
    started = perf_counter()
    if enabled and requested_section(question) is not None:
        chunks, _ = load_chunks(data_dir)
        result = expand_section(question, hits, chunks)
    else:
        result = _result(hits, None, MAX_SECTION_CHARS, MAX_SECTION_CHUNKS)
    result["elapsed_seconds"] = perf_counter() - started
    return result
