"""
Corpus extraction script (Assignment 0).
Reads the documents in data/sources/, writes data/corpus.json and
data/sample.json, and prints corpus statistics.
Run from the repo root:  python scripts/extract.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

SOURCES = Path("data/sources")
OUT_DIR = Path("data")
MIN_CHARS = 50

# PDF-specific header/footer patterns; preserve useful source text.
KNOWN_NOISE_PATTERNS = [
    r"^(EN\s+)?OJ L, \d{1,2}\.\d{1,2}\.\d{4}(\s+EN)?$",
    r"^(\d+/\d+\s+)?ELI: https?://data\.europa\.eu/eli/\S+(\s+\d+/\d+)?$",
    r"^\d+/\d+$",
    r"^EN$",
    r"^Official Journal$",
    r"^of the European Union$",
    r"^L series$",
    r"^\d{4}/\d+ \d{1,2}\.\d{1,2}\.\d{4}$",
]
KNOWN_NOISE = [re.compile(p) for p in KNOWN_NOISE_PATTERNS]


def normalise_for_repeat(line: str) -> str:
    """Replace digits so '3/144' and '4/144' count as the same line."""
    return re.sub(r"\d+", "#", line.strip())


def find_repeated_lines(pages: list[str], threshold: float = 0.5) -> set[str]:
    """Short lines that appear on more than half of the pages are headers/footers."""
    if len(pages) < 4:
        return set()
    counts = Counter()
    for text in pages:
        seen = {normalise_for_repeat(ln) for ln in text.splitlines() if ln.strip()}
        counts.update(s for s in seen if len(s) <= 80)
    return {line for line, c in counts.items() if c / len(pages) > threshold}


def is_noise(line: str, repeated: set[str]) -> bool:
    s = line.strip()
    if any(p.match(s) for p in KNOWN_NOISE):
        return True
    return normalise_for_repeat(s) in repeated

NEW_BLOCK = re.compile(
    r"^(\(\d+\)|\([a-z]{1,4}\)|\d+\.\s|—|•|"
    r"Article \d+|CHAPTER [IVXLC]+|SECTION \d+|ANNEX [IVXLC]+|Annex [IVXLC]+)"
)
HEADING = re.compile(r"^(Article \d+\w?|CHAPTER [IVXLC]+|SECTION \d+|ANNEX [IVXLC]+)$")


def fix_footnote_markers(text: str) -> str:
    """Superscripts come out as '(\\n1\\n)' or '(\\n10)'. Rebuild '(1)'."""
    text = re.sub(r"\(\s*\n\s*(\d+)\s*\n\s*\)", r"(\1)", text)
    text = re.sub(r"\(\s*\n\s*(\d+)\)", r"(\1)", text)
    return text


def join_wrapped_lines(lines: list[str]) -> list[str]:
    """Re-join lines that the PDF broke in the middle of a paragraph."""
    out: list[str] = []
    for line in lines:
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        starts_new_block = bool(NEW_BLOCK.match(line))
        prev_is_heading = bool(HEADING.match(prev))
        prev_is_title = len(out) >= 2 and bool(HEADING.match(out[-2]))
        prev_ends_sentence = prev.endswith((".", ":", ";"))
        if starts_new_block or prev_is_heading or prev_is_title or prev_ends_sentence:
            out.append(line)
        elif prev.endswith("-"):
            out[-1] = prev + line
        else:
            out[-1] = prev + " " + line
    return out


# Compose extraction cleanup before counting or storing characters.
def clean(text: str, repeated: set[str] = frozenset()) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = fix_footnote_markers(text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not is_noise(ln, repeated)]
    lines = join_wrapped_lines(lines)
    return "\n".join(lines).strip()

# Read each supported file format while preserving source page numbers.
def extract_pdf(path: Path):
    doc = fitz.open(path)
    raw_pages = [page.get_text("text") for page in doc]
    doc.close()
    repeated = find_repeated_lines(raw_pages)
    for i, raw in enumerate(raw_pages, start=1):
        yield i, clean(raw, repeated)


def extract_text_file(path: Path):
    yield 1, clean(path.read_text(encoding="utf-8"))


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".txt": extract_text_file,
    ".md": extract_text_file,
}


# Command-line inputs, validation, execution, and output.
def main():
    if not SOURCES.exists():
        raise SystemExit(f"{SOURCES} not found. Run this from the repo root.")

    entries, near_empty = [], []
    for path in sorted(SOURCES.iterdir()):
        extractor = EXTRACTORS.get(path.suffix.lower())
        if extractor is None:
            continue
        for page, text in extractor(path):
            if len(text) < MIN_CHARS:
                near_empty.append(f"{path.name} p.{page} ({len(text)} chars)")
            entries.append({
                "source": path.name,
                "page": page,
                "char_count": len(text),
                "text": text,
            })

    (OUT_DIR / "corpus.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sample.json").write_text(
        json.dumps(entries[:10], ensure_ascii=False, indent=2), encoding="utf-8")

    chars = sum(e["char_count"] for e in entries)
    files = len({e["source"] for e in entries})
    print(f"source files: {files}")
    print(f"total entries: {len(entries)}")
    print(f"total characters: {chars}")
    print(f"avg characters/entry: {chars // max(len(entries), 1)}")
    print(f"near-empty entries (<{MIN_CHARS} chars): {len(near_empty)}")
    for item in near_empty:
        print(f"  - {item}")


# Running this file explicitly invokes the command.
if __name__ == "__main__":
    main()