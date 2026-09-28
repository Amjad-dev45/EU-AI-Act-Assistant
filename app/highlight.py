import io
import re
import difflib
import pymupdf
from PIL import Image

PDF_PATH = "data/sources/OJ_L_202401689_EN_TXT.pdf"
MIN_BLOCK = 4   # ignore matches shorter than 4 words (too likely to be a coincidence)
MAX_GAP = 3     # fill up to 3 unmatched words between matches (e.g. "3." paragraph numbers)


def norm(word):
    return re.sub(r"\W+", "", word.lower())


def render_page(page_number, text, pdf_path=PDF_PATH):
    doc = pymupdf.open(pdf_path)
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
    doc.close()
    return image