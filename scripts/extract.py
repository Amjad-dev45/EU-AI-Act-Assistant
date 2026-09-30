# /// script
# requires-python = "==3.12.*"
# dependencies = ["pymupdf==1.28.2"]
# ///
"""Extract the course corpus. Run from the repo: uv run scripts/extract.py."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = DATA / "sources"
REPORT = ROOT / "scripts" / "EXTRACTION_REPORT.md"
MIN_CHARS = 50


# Text cleanup: retain words, paragraph boundaries, and mathematical exponents.
def line_text(line):
    text = ""
    for span in line["spans"]:
        part = span["text"]
        if span["flags"] & 1 and part.strip().isdigit() and re.search(r"\d$", text):
            part = "^" + part.strip()  # For example, 10 with a raised 25 -> 10^25.
        text += part
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()


def margin_key(text):
    # Normalize page counters only; article numbers and other text stay intact.
    return "<page counter>" if re.fullmatch(r"\d+\s*/\s*\d+", text) else text


def clean_block(lines):
    # Join wrapped lines within a paragraph, without guessing whether '-' is a word break.
    return " ".join(line for line in lines if line)


# Extract every PDF page; only repeated text at the extreme edges is removed.
def extract_pdf(path):
    import pymupdf

    pages, counts = [], Counter()
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDF: {path.name}")
        for page in document:
            blocks, edge_keys = [], set()
            for block in page.get_text("dict", sort=True)["blocks"]:
                lines = []
                for line in block.get("lines", []):
                    text = line_text(line)
                    # Body text and footnotes can reach low on the page: use a narrow footer.
                    at_edge = line["bbox"][3] <= page.rect.height * 0.08 or line["bbox"][1] >= page.rect.height * 0.96
                    key = margin_key(text)
                    lines.append((text, at_edge, key))
                    if text and at_edge:
                        edge_keys.add(key)
                if lines:
                    blocks.append(lines)
            pages.append(blocks)
            counts.update(edge_keys)

    repeated = {key for key, count in counts.items() if count >= max(3, (len(pages) + 1) // 2)}
    cleaned_pages, removed = [], Counter()
    for blocks in pages:
        paragraphs = []
        for block in blocks:
            kept = []
            for text, at_edge, key in block:
                if at_edge and key in repeated:
                    removed[key] += 1
                else:
                    kept.append(text)
            paragraph = clean_block(kept)
            if paragraph:
                paragraphs.append(paragraph)
        cleaned_pages.append("\n\n".join(paragraphs))
    return cleaned_pages, removed


# Build the required JSON records and an honest extraction report.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing corpus, sample, and extraction report")
    args = parser.parse_args()
    outputs = [DATA / "corpus.json", DATA / "sample.json", REPORT]
    existing = [p.name for p in outputs if p.exists() and p.stat().st_size]
    if existing and not args.overwrite:
        parser.error("Outputs already exist: " + ", ".join(existing) + ". Use --overwrite intentionally to replace them.")
    if not SOURCES.is_dir():
        parser.error(f"Create {SOURCES} and place your source PDFs there.")
    files = sorted(p for p in SOURCES.iterdir() if p.is_file())
    supported = [p for p in files if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not supported:
        parser.error("No PDF, TXT, or Markdown source files found; nothing was written.")

    entries, cleanup = [], []
    for path in supported:
        if path.suffix.lower() == ".pdf":
            try:
                pages, removed = extract_pdf(path)
            except Exception as error:
                parser.error(f"Cannot extract {path.name}: {error}")
            cleanup.extend(f"- `{path.name}`: removed edge text `{text}` ({count} occurrences)." for text, count in sorted(removed.items()))
        else:
            raw = path.read_text(encoding="utf-8")
            pages = ["\n\n".join(" ".join(block.split()) for block in re.split(r"\n\s*\n", raw.strip()))]
        for number, text in enumerate(pages, start=1):
            # Empty and near-empty pages must remain in the corpus.
            entries.append({"source": path.name, "page": number, "char_count": len(text), "text": text})

    if not entries:
        parser.error("Sources contained no pages or entries; nothing was written.")
    near_empty = [entry for entry in entries if entry["char_count"] < MIN_CHARS]
    total_chars = sum(entry["char_count"] for entry in entries)
    stats = [f"Source files: {len(supported)}", f"Total entries: {len(entries)}", f"Total characters: {total_chars}",
             f"Average characters per entry: {total_chars / len(entries):.1f}", f"Empty or near-empty entries (< {MIN_CHARS} characters): {len(near_empty)}"]
    report = ["# Extraction report", "", "## Statistics", "", *[f"- {stat}" for stat in stats], "",
              "## Cleanup", "", "Wrapped lines are joined within text blocks; paragraph and page boundaries are kept. Whitespace is normalized. Numeric superscripts after digits use caret notation (for example, 10^25). Original PDFs are unchanged.", "",
              *cleanup, "", "## Empty or near-empty entries", ""]
    report += [f"- `{e['source']}`, page {e['page']}: {e['char_count']} characters." for e in near_empty] or ["None found."]
    skipped = [p.name for p in files if p not in supported]
    if skipped:
        report += ["", "## Unsupported files", "", *[f"- Skipped `{name}`." for name in skipped]]
    report += ["", "## Review still needed", "", "Manually inspect the sample and pages containing tables, footnotes, or superscripts. Text extraction can lose layout; this report does not certify visual accuracy. No OCR is performed. Empty pages may be blank or scanned. Header detection is conservative, so some one-off headers may remain. Exact quotations should be checked against the original PDF.", ""]

    # Write only after all source files have been read successfully.
    for path, value in [(outputs[0], entries), (outputs[1], entries[:10])]:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(stats))
    for path in outputs:
        print(f"Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
