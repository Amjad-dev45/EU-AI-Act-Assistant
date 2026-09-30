"""Create real chunks: uv run --python 3.12 scripts/chunk.py."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag.chunking import chunk_documents


def main():
    data = ROOT / "data"
    output = data / "chunks.json"
    if output.exists():
        raise SystemExit("chunks.json already exists; preserve it and its matching index first.")

    # Adapt the corpus's page records to the reusable chunker's document format.
    corpus = json.loads((data / "corpus.json").read_text(encoding="utf-8"))
    documents = {}
    for entry in corpus:
        source = entry["source"]
        document = documents.setdefault(source, {
            "document_id": source, "source": source, "pages": [],
        })
        document["pages"].append({"page": entry["page"], "text": entry["text"]})

    chunks = chunk_documents({"documents": list(documents.values())})
    if not chunks:
        raise SystemExit("Corpus produced no text chunks; nothing was written.")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Chunks: {len(chunks)}")
    print("Limit: 1,000 characters; overlap up to 200 characters")
    print(f"Maximum length: {max(len(c['text']) for c in chunks)} characters")
    print(f"Cross-page chunks: {sum(len(c['pages']) > 1 for c in chunks)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
