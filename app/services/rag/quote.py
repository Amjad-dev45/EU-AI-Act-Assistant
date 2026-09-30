"""Copy exact text from a page of the extracted production corpus."""

from contracts.errors import ApiError, QUOTE_NOT_FOUND


def _not_found(message: str) -> ApiError:
    return ApiError(404, QUOTE_NOT_FOUND, message, extra={"found": False})


def find_quote(source: list[dict], document_id: str, page: int, phrase: str | None) -> dict:
    # Extraction and chunking use the original source filename as document_id.
    document = [entry for entry in source if entry["source"] == document_id]
    if not document:
        raise _not_found("The requested document was not found in the corpus.")
    page_obj = next((entry for entry in document if entry["page"] == page), None)
    if page_obj is None:
        raise _not_found("The requested page was not found in that document.")
    text = page_obj["text"]
    match = None
    if phrase:
        start = text.find(phrase)
        if start < 0:
            raise _not_found("Phrase not found verbatim on that page.")
        match = {"start": start, "end": start + len(phrase)}
        text = text[start:start + len(phrase)]
    return {
        "found": True,
        "text": text,
        "match": match,
        "citation": {"document_id": document_id, "source": page_obj["source"],
                     "title": page_obj["source"], "pages": [page]},
    }
