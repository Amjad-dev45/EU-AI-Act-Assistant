"""Locate authoritative Articles and preserve their original page boundaries."""
from collections import defaultdict
import hashlib
import re

ARTICLE = re.compile(r"(?m)^Article (\d+)\s*$")


def articles(pages, number=None):
    """Return all exact heading matches; callers refuse ambiguous documents."""
    documents = defaultdict(list)
    for page in pages:
        documents[page['source']].append(page)
    found = []
    for source, rows in documents.items():
        text, intervals = '', []
        for row in sorted(rows, key=lambda p: p['page']):
            if text:
                text += '\n\n'
            start = len(text)
            text += row['text']
            intervals.append((start, len(text), row['page']))
        headings = list(ARTICLE.finditer(text))
        for index, heading in enumerate(headings):
            if number is not None and int(heading[1]) != number:
                continue
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            # The last article ends before signatures / annexes, not at EOF.
            tail = re.search(r'(?m)^(?:This Regulation shall be binding|Done at |ANNEX\b)', text[heading.end():end])
            if tail:
                end = heading.end() + tail.start()
            start = heading.start()
            found.append({'source': source, 'article': int(heading[1]), 'text': text[start:end].rstrip(),
                          'intervals': [(a-start, b-start, page) for a,b,page in intervals if a < end and b > start]})
    return found


def evidence(section, start, end, kind):
    """A stable passage identity for direct lookups, independent of FAISS."""
    text = section['text'][start:end].strip()
    left = start + len(section['text'][start:end]) - len(section['text'][start:end].lstrip())
    right = left + len(text)
    pages = [page for a,b,page in section['intervals'] if a < right and b > left]
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return {'chunk_id': f"{kind}:{section['source']}:{digest}", 'document_id': section['source'],
            'source': section['source'], 'pages': pages, 'text': text}
