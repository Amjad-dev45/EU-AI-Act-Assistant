"""Select exact source paragraphs for qualified higher/lower calculations."""

import re

from .list_evidence import PARAGRAPH, reconstruct_article


COMPARISON = re.compile(r"\b(?:compar(?:e|ed|ing|ison|isons)|versus|vs|differences?)\b", re.I)
OPERATOR = re.compile(
    r"\bwhichever(?:\s+(?:of|these|the|two|amounts?|percentages?|values?|thereof|is|are)){0,10}"
    r"\s+(higher|lower)\b", re.I,
)
SCOPE = re.compile(
    r"^\d+\.\s*(?:in\s+the\s+case\s+of|for)\s+"
    r"([^,;:.]+(?:,\s*including\s+[^,;:.]+)?)(?=[,;:])", re.I,
)
FUNCTION_WORDS = {"a", "an", "the", "any", "all", "each", "including", "of", "for", "in", "case"}


def _words(text):
    text = re.sub(r"(?<=\w)[\-\u2010-\u2015](?=\w)", "", text.casefold())
    words = re.findall(r"[a-z]+", text)
    return [word[:-3] + "y" if len(word) > 4 and word.endswith("ies") else
            word[:-1] if len(word) > 3 and word.endswith("s")
            and not word.endswith(("ss", "us", "is")) else word for word in words]


def _scope_matches(text, question_words):
    match = SCOPE.search(text)
    if match is None:
        return False
    for phrase in re.split(r",|\band\b|\bor\b", match[1], flags=re.I):
        words = [word for word in _words(phrase) if word not in FUNCTION_WORDS]
        if words and any(question_words[index:index + len(words)] == words
                         for index in range(len(question_words) - len(words) + 1)):
            return True
    return False


def select_calculation_rules(question, hits):
    """Return quoted calculation rules only for a matching class or comparison.

    Amounts, operators, scope and cross-references remain verbatim source text.
    Unsupported source structure and ordinary maximum questions use the caller's
    existing answer path. Hits must be the complete expanded article prefix.
    """
    if not isinstance(question, str) or not question.strip():
        return None
    article = reconstruct_article(hits)
    if article is None:
        return None
    text = article["text"]
    paragraphs = list(PARAGRAPH.finditer(text))
    if len({match[1] for match in paragraphs}) != len(paragraphs):
        return None
    higher, lower = [], []
    for index, match in enumerate(paragraphs):
        left = match.start(1)
        right = paragraphs[index + 1].start(1) if index + 1 < len(paragraphs) else len(text)
        paragraph = text[left:right].rstrip()
        operators = {operator.casefold() for operator in OPERATOR.findall(paragraph)}
        if len(operators) > 1:
            return None
        if not operators:
            continue
        right = left + len(paragraph)
        ids = [number for a, b, number in article["intervals"]
               if a < article["start"] + right and b > article["start"] + left]
        item = {"label": match[1], "text": paragraph, "citation_ids": ids}
        (higher if operators == {"higher"} else lower).append(item)
    if not higher or not lower:
        return None
    words = _words(question)
    matching = [item for item in lower if _scope_matches(item["text"], words)]
    if not matching and not COMPARISON.search(question):
        return None
    return {"heading": article["heading"], "items": (matching or lower) + higher}
