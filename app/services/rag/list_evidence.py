"""Expose a verified article's leading enumeration without rewriting its text."""

import re


PARAGRAPH = re.compile(r"(?:\A|(?<=\n)|(?<=[.!?] ))[ \t]*(\d+)\.[ \t]+")
MARKER = re.compile(r"(?:\A|(?<=\n)|(?<=;))\s*(?P<marker>\((?P<label>[a-z]+)\))[ \t]+")
ROMAN = re.compile(r"[ivxlcdm]+\Z")


def reconstruct_article(hits):
    """Rejoin verified expanded excerpts, retaining absolute citation intervals."""
    if not isinstance(hits, list) or not hits or any(not isinstance(hit, dict) for hit in hits):
        return None
    heading = hits[0].get("section_heading")
    if not isinstance(heading, str) or not re.fullmatch(r"Article \d+[a-z]?", heading):
        return None
    identity = (hits[0].get("source"), hits[0].get("document_id"))
    if any(not isinstance(value, str) or not value for value in identity):
        return None
    parts, intervals, cursor, start = [], [], None, None
    for number, hit in enumerate(hits, 1):
        left, right = hit.get("excerpt_start_char"), hit.get("excerpt_end_char")
        original_left, original_right = hit.get("start_char"), hit.get("end_char")
        text = hit.get("text")
        if (hit.get("section_heading") != heading
                or (hit.get("source"), hit.get("document_id")) != identity
                or any(type(value) is not int for value in (left, right, original_left, original_right))
                or not 0 <= original_left <= left < right <= original_right
                or not isinstance(text, str) or len(text) != right - left):
            return None
        if cursor is None:
            start = cursor = left
        if not 0 <= left - cursor <= 2:
            return None
        parts.extend((" " * (left - cursor), text))
        intervals.append((left, right, number))
        cursor = right
    text = "".join(parts)
    if not text.startswith(heading + "\n"):
        return None
    return {"heading": heading, "text": text, "intervals": intervals, "start": start}


def group_article_items(hits):
    """Return the leading letter list only after complete section expansion."""
    article = reconstruct_article(hits)
    if article is None:
        return None
    heading, text = article["heading"], article["text"]
    intervals, start = article["intervals"], article["start"]
    paragraphs = list(PARAGRAPH.finditer(text))
    if not paragraphs or paragraphs[0][1] != "1":
        return None
    if len(paragraphs) > 1 and paragraphs[1][1] != "2":
        return None
    stop = paragraphs[1].start() if len(paragraphs) > 1 else len(text)
    markers = list(MARKER.finditer(text, paragraphs[0].end(), stop))
    if not markers or markers[0]["label"] != "a":
        return None
    if not text[paragraphs[0].end():markers[0].start("marker")].rstrip().endswith(":"):
        return None

    items, nested = [], False
    for marker in markers:
        label, position = marker["label"], marker.start("marker")
        # A colon introduces Roman children, including (i) after an outer (h).
        begins_children = label == "i" and text[:position].rstrip().endswith(":")
        if begins_children:
            nested = True
        elif len(label) == 1 and ord(label) == ord("a") + len(items):
            items.append((label, position))
            nested = False
        elif not (nested and ROMAN.fullmatch(label)):
            return None
    if len(items) < 2:
        return None
    grouped = []
    for index, (label, left) in enumerate(items):
        right = items[index + 1][1] if index + 1 < len(items) else stop
        item_text = text[left:right].rstrip()
        right = left + len(item_text)
        ids = [number for a, b, number in intervals if a < start + right and b > start + left]
        grouped.append({"label": label, "text": item_text, "citation_ids": ids})
    return {"heading": heading, "intro": text[:items[0][1]],
            "items": grouped, "notes": text[stop:]}
