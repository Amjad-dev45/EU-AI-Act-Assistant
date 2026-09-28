"""Render cited PDF passages as highlighted page images and collapsible panels."""

import base64
import difflib
import html
import io
import re
from pathlib import Path

import pymupdf
from PIL import Image

from app.settings import REPO_DIR

PDF_PATH = REPO_DIR / "data/sources/OJ_L_202401689_EN_TXT.pdf"
MIN_BLOCK = 4  # Ignore short word matches likely to occur by coincidence.
MAX_GAP = 3    # Bridge small unmatched gaps, such as paragraph numbers.

# Match normalized words while retaining their positions on the PDF page.
def norm(word):
    return re.sub(r"\W+", "", word.lower())


def render_page(page_number, text, pdf_path=PDF_PATH):
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]

        # every word on the page, with its position
        words = [w for w in page.get_text("words") if norm(w[4])]
        page_tokens = [norm(w[4]) for w in words]
        chunk_tokens = [t for t in (norm(w) for w in text.split()) if t]

        # find the stretches of the page that match the chunk text
        matcher = difflib.SequenceMatcher(None, page_tokens, chunk_tokens, autojunk=False)
        matched = set()
        for a, b, size in matcher.get_matching_blocks():
            if size >= MIN_BLOCK:
                matched.update(range(a, a + size))

        # fill small holes between matches, like the "3." and "4." paragraph numbers
        ordered = sorted(matched)
        for i, j in zip(ordered, ordered[1:]):
            if 1 < j - i <= MAX_GAP + 1:
                matched.update(range(i + 1, j))

        # draw one highlight per line of consecutive matched words
        group = None
        for i in sorted(matched):
            x0, y0, x1, y1, _, block, line, _ = words[i]
            key = (block, line)
            if group and group["key"] == key and group["last"] == i - 1:
                group["rect"] |= pymupdf.Rect(x0, y0, x1, y1)
                group["last"] = i
            else:
                if group:
                    page.add_highlight_annot(group["rect"])
                group = {"rect": pymupdf.Rect(x0, y0, x1, y1), "key": key, "last": i}
        if group:
            page.add_highlight_annot(group["rect"])

        image = Image.open(io.BytesIO(page.get_pixmap(dpi=110).tobytes("png")))
        image.load()
    return image

# Encode rendered pages and attach them to the original citation IDs.
def to_data_uri(image):
    """Turn a PIL image into text that can be embedded in the page (no file needed)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_sources_html(citations, data_dir):
    data_dir = Path(data_dir).resolve()
    parts = ["<h3>Sources</h3>"]
    for c in citations:
        pages = ", ".join(str(p) for p in c["pages"])
        title = f"[{c['citation_id']}] {c['source']}, page {pages}"
        body = ""
        for page in c["pages"]:
            try:
                source_path = (data_dir / "sources" / c["source"]).resolve()
                if not source_path.is_relative_to((data_dir / "sources").resolve()):
                    raise ValueError("Source is outside the sources folder")
                image = render_page(page, c["text"], pdf_path=str(source_path))
                body += (f"<p><b>Page {page}</b></p>"
                         f"<img src='{to_data_uri(image)}' style='max-width:100%; border-radius:6px;'>")
            except Exception as e:
                print("highlight failed:", e)
        parts.append(
            "<details style='margin-bottom:10px; padding:8px; border:1px solid #555; border-radius:6px;'>"
            f"<summary style='cursor:pointer; font-weight:600;'>{html.escape(title)}</summary>"
            f"{body}</details>"
        )
    return "".join(parts)
