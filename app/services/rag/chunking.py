"""Prefer complete sentences while enforcing a strict character limit."""

import re

SENTENCE_END = re.compile(r'[.!?]+["\u201d\u2019\')\]]*(?=\s|$)')
ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "no.", "nos.", "art.", "arts.",
                 "para.", "paras.", "fig.", "figs.", "p.", "pp.",
                 "mr.", "mrs.", "ms.", "dr.", "prof.", "vs.", "cf."}


# Keep each document's consecutive pages together, without bridging missing pages.
def page_groups(pages):
    group = []
    for page in sorted(pages, key=lambda item: item["page"]):
        if group and page["page"] != group[-1]["page"] + 1:
            yield group
            group = []
        group.append(page)
    if group:
        yield group


# Sentence detection is a heuristic; protect common abbreviations and list numbers.
def sentence_spans(text):
    start = 0
    for match in SENTENCE_END.finditer(text):
        if match.group().startswith("."):
            word_start = match.start()
            while word_start > 0 and not text[word_start - 1].isspace():
                word_start -= 1
            word = text[word_start:match.start() + 1]
            abbreviation = word.lstrip('([{\"\u201c\u2018')
            if (abbreviation.lower() in ABBREVIATIONS
                    or re.fullmatch(r"(?:[A-Za-z]\.)+", abbreviation)):
                continue
            line_start = text.rfind("\n", 0, word_start) + 1
            if (re.fullmatch(r"\(?\d+\)?\.", word)
                    and not text[max(start, line_start):word_start].strip()):
                continue  # For example, "1. Providers shall ..."
        left = start
        while left < match.end() and text[left].isspace():
            left += 1
        if left < match.end():
            yield left, match.end()
        start = match.end()

    # Preserve remaining text even if a document ends without punctuation.
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        yield left, right


# Split oversized sentences at whitespace; only cut a word if it exceeds the cap.
def bounded_sentence_spans(text, chunk_size):
    for left, right in sentence_spans(text):
        while right - left > chunk_size:
            limit = left + chunk_size
            cut = limit
            while cut > left and not text[cut].isspace():
                cut -= 1
            if cut == left:
                cut = limit
            end = cut
            while end > left and text[end - 1].isspace():
                end -= 1
            yield left, end
            left = cut
            while left < right and text[left].isspace():
                left += 1
        if left < right:
            yield left, right


# Pack sentences or fragments; repeat whole trailing units within the overlap budget.
def chunk_spans(text, chunk_size, overlap):
    sentences = list(bounded_sentence_spans(text, chunk_size))
    start = 0
    while start < len(sentences):
        end = start + 1
        while end < len(sentences) and sentences[end][1] - sentences[start][0] <= chunk_size:
            end += 1
        yield sentences[start][0], sentences[end - 1][1]
        if end == len(sentences):
            break

        next_start = end
        while (next_start > start + 1
               and sentences[end - 1][1] - sentences[next_start - 1][0] <= overlap):
            next_start -= 1
        # Drop overlap if needed to fit at least one new sentence.
        while next_start < end and sentences[end][1] - sentences[next_start][0] > chunk_size:
            next_start += 1
        start = next_start


def chunk_documents(source, chunk_size=1000, overlap=200):
    """Keep the shared document/chunk interface used by the processing service."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size.")
    chunks = []
    for document in source["documents"]:
        document_id = document["document_id"]
        numbers = [page["page"] for page in document["pages"]]
        if len(numbers) != len(set(numbers)) or any(number < 1 for number in numbers):
            raise ValueError(f"Invalid or duplicate page numbers in {document_id}")
        offset = 0
        for group in page_groups(document["pages"]):
            # One continuous string per consecutive page group, with a page map.
            text, pages = "", []
            for page in group:
                if pages:
                    text += " "
                begin = len(text)
                text += page["text"]
                pages.append((begin, len(text), page["page"]))

            for left, right in chunk_spans(text, chunk_size, overlap):
                chunks.append({
                    "chunk_id": f"{document_id}:{offset + left}:{offset + right}",
                    "document_id": document_id,
                    "source": document.get("source", document_id),
                    "pages": [page for a, b, page in pages if a < b and a < right and b > left],
                    "start_char": offset + left,
                    "end_char": offset + right,
                    "oversized": right - left > chunk_size,
                    "text": text[left:right],
                })
            # Offsets refer to all available document pages joined by single spaces.
            offset += len(text) + 1
    return chunks
