"""PERSON 1: quote_exact.

Returns text copied unchanged from the source: never similarity, never paraphrase.
"""
from contracts.errors import ApiError, QUOTE_NOT_FOUND


def _not_found(message: str) -> ApiError:
    return ApiError(404, QUOTE_NOT_FOUND, message, extra={"found": False})


def find_quote(source: dict, document_id: str, page: int, phrase: str | None) -> dict:
    doc = next((d for d in source["documents"] if d["document_id"] == document_id), None)
    if doc is None:
        raise _not_found(f"Unknown document_id `{document_id}`.")
    page_obj = next((p for p in doc["pages"] if p["page"] == page), None)
    if page_obj is None:
        raise _not_found(f"Document `{document_id}` has no page {page} in the demo fixture.")
    text = page_obj["text"]
    match = None
    if phrase:
        start = text.find(phrase)  # exact, case-sensitive substring
        if start < 0:
            raise _not_found("Phrase not found verbatim on that page.")
        match = {"start": start, "end": start + len(phrase)}
        text = text[start:start + len(phrase)]
    return {
        "found": True,
        "text": text,
        "match": match,
        "citation": {"document_id": document_id, "title": doc["title"], "pages": [page]},
    }
