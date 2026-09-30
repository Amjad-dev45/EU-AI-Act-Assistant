"""PERSON 1: extraction/chunking foundations.

Replace `chunk_documents` with the real chunker later. Keep the output shape:
{"chunk_id", "document_id", "pages": [int], "text"}.
"""


def chunk_documents(source: dict) -> list[dict]:
    """Dummy chunker: one chunk per paragraph of each fixture page."""
    chunks = []
    for doc in source["documents"]:
        for page in doc["pages"]:
            paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
            for i, text in enumerate(paragraphs):
                chunks.append({
                    "chunk_id": f"{doc['document_id']}:p{page['page']}:c{i}",
                    "document_id": doc["document_id"],
                    "pages": [page["page"]],
                    "text": text,
                })
    return chunks
