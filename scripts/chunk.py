"""Chunk consecutive document pages with overlap and accurate page citations."""

import argparse
import json
import re
from pathlib import Path


# Find overlapping character spans, preferring sentence or word boundaries.
def chunk_text(text: str, chunk_size: int, overlap: int):
    """Yield exact spans; prefer sentence endings, then whitespace, then a cut."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Only search the last quarter, and leave room to advance.
            lower = start + max(overlap + 1, chunk_size * 3 // 4)
            # A heuristic, not a linguistic parser: abbreviations may match.
            sentences = [match for match in
                         re.finditer(r'[.!?][\"”’\')\]]*(?=\s)', text[lower:end + 1])
                         if lower + match.end() <= end]
            if sentences:
                end = lower + sentences[-1].end()
            else:
                for boundary in range(end, lower - 1, -1):
                    if text[boundary - 1].isspace():
                        end = boundary
                        break
        if text[start:end].strip():
            yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap


# Join consecutive source pages and map every chunk back to its page ranges.
def chunk_corpus(corpus: list, chunk_size: int, overlap: int) -> list:
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    chunks = []
    seen = set()
    documents = {}
    for entry in corpus:
        source, page, text = entry["source"], entry["page"], entry["text"]
        if not isinstance(source, str) or not source:
            raise ValueError("Each entry needs a nonempty source")
        if type(page) is not int or page < 1 or not isinstance(text, str):
            raise ValueError("Each entry needs a positive page and text string")
        if (source, page) in seen:
            raise ValueError(f"Duplicate source/page: {source}, {page}")
        seen.add((source, page))
        documents.setdefault(source, []).append(entry)

    for source, entries in sorted(documents.items()):
        entries.sort(key=lambda entry: entry["page"])
        for previous, current in zip(entries, entries[1:]):
            if current["page"] != previous["page"] + 1:
                raise ValueError(f"Missing page between {previous['page']} and {current['page']} in {source}")
        # Offsets refer to this document's joined text. The inserted newline
        # separates pages without changing the original corpus or page text.
        pieces, page_spans = [], []
        position = 0
        for entry in entries:
            if pieces:
                pieces.append("\n")
                position += 1
            page_text = entry["text"]
            page_spans.append((entry["page"], position, position + len(page_text)))
            pieces.append(page_text)
            position += len(page_text)
        text = "".join(pieces)
        for start, end, part in chunk_text(text, chunk_size, overlap):
            pages = [page for page, page_start, page_end in page_spans
                     if page_start < page_end and page_start < end and page_end > start]
            chunks.append({
                "id": len(chunks),
                "source": source,
                "page": pages[0],  # First page, for the course starter format.
                "pages": pages,  # Use ALL these pages in citations and evaluation.
                "start_char": start,
                "end_char": end,
                "text": part,
            })
    return chunks



# Command-line inputs, validation, execution, and output.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing corpus.json; chunks.json is saved here")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Maximum characters per chunk (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200,
                        help="Shared characters between chunks (default: 200)")
    # Parse options before opening files or making API requests.
    args = parser.parse_args()
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("Require 0 <= overlap < chunk-size")
    corpus = json.loads((args.data_dir / "corpus.json").read_text(encoding="utf-8"))
    chunks = chunk_corpus(corpus, args.chunk_size, args.overlap)
    if not chunks:
        parser.error("Corpus produced no nonempty chunks")
    output = args.data_dir / "chunks.json"
    if output.exists():
        parser.error("chunks.json already exists; preserve the matching FAISS index before replacing it")
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"Pages: {len(corpus)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Maximum characters/chunk: {max(len(c['text']) for c in chunks)}")
    print(f"Saved: {output}")


# Running this file explicitly invokes the command.
if __name__ == "__main__":
    main()
