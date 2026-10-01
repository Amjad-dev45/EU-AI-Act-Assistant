"""Render highlighted PDF excerpts in memory; original documents remain untouched."""
import base64
from difflib import SequenceMatcher
from pathlib import Path
import re
import threading
import unicodedata

from contracts.errors import ApiError

PDF_LOCK = threading.Lock()  # PyMuPDF document work must not run concurrently.


def _tokens(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def _highlight_rectangles(page, excerpt):
    """Match continuous word sequences, keeping each PDF word's real rectangle.

    Line wrapping and punctuation differ from cleaned corpus text. Matching runs
    avoids highlighting isolated common words throughout the page.
    """
    import pymupdf
    words = page.get_text('words', sort=True)
    tokens, positions = [], []
    for index, word in enumerate(words):
        parts = _tokens(word[4])
        tokens.extend(parts)
        positions.extend([index] * len(parts))
    wanted = _tokens(excerpt)
    if not wanted or not tokens:
        return []
    minimum = 4 if len(wanted) >= 4 else len(wanted)
    if len(''.join(wanted)) < 10:
        return []
    selected = set()
    for a, b, size in SequenceMatcher(None, tokens, wanted, autojunk=False).get_matching_blocks():
        if size >= minimum:
            selected.update(positions[a:a+size])
    # Merge adjacent selected words on the same PDF line for legible highlights.
    rectangles, previous = [], None
    for index in sorted(selected):
        word = words[index]
        rect = pymupdf.Rect(word[:4])
        line = tuple(word[5:7])
        if previous and previous == (index-1, line):
            rectangles[-1] |= rect
        else:
            rectangles.append(rect)
        previous = (index, line)
    return rectangles


def render_preview(data_dir, corpus, document_id, page_number, excerpt):
    """Only PDFs/pages in corpus.json may be opened; never accept a filesystem path."""
    if not any(p['source'] == document_id and p['page'] == page_number for p in corpus):
        raise ApiError(404, 'source_not_found', 'This source page is not in the prepared corpus.')
    base = (Path(data_dir) / 'sources').resolve()
    path = (base / document_id).resolve()
    if not path.is_relative_to(base) or path.suffix.lower() != '.pdf' or not path.is_file():
        raise ApiError(404, 'pdf_unavailable', 'The original PDF is unavailable in data/sources.')
    import pymupdf
    with PDF_LOCK, pymupdf.open(path) as pdf:
        if pdf.needs_pass or page_number > len(pdf):
            raise ApiError(404, 'page_unavailable', 'The original PDF page is unavailable.')
        page = pdf[page_number - 1]
        rects = _highlight_rectangles(page, excerpt)
        clip = page.rect
        if rects:
            annot = page.add_highlight_annot(rects)
            annot.set_colors(stroke=(1, 0.82, 0.1))
            annot.set_opacity(0.45)
            annot.update()
            bounds = pymupdf.Rect(rects[0])
            for rect in rects[1:]:
                bounds |= rect
            # Preserve full page width and a little surrounding context.
            clip = pymupdf.Rect(page.rect.x0, bounds.y0-28, page.rect.x1, bounds.y1+28) & page.rect
            if page.rotation:
                clip = page.rect  # annotation coordinates are unrotated
        scale = min(1.7, 1400 / max(clip.width, 1), 1800 / max(clip.height, 1))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False, annots=True)
        png = base64.b64encode(pix.tobytes('png')).decode('ascii')
        cropped = bool(rects) and not page.rotation
    return {'source': document_id, 'page': page_number, 'image_base64': png,
            'highlight_count': len(rects), 'cropped': cropped,
            'message': ('Matching passage text is highlighted. Highlighting may be partial across line breaks or pages.'
                        if rects else 'No reliable text match; showing the original page without highlights.')}
