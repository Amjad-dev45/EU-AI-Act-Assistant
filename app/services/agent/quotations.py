"""Complete quotation requests using verified tool text and its exact provenance."""
import re

QUOTE_REQUEST = re.compile(r'\b(?:quote|quotes|quotation|quotations|verbatim|word[ -]for[ -]word)\b', re.I)
QUOTE_REFUSAL = "I couldn't complete the requested quotation with verified source text. Please specify the document and page, or a shorter exact phrase."

# This strips only a trailing presentation request, preserving the substantive
# question, legal qualifications and article references used for retrieval.
QUOTE_SUFFIX = re.compile(
    r'(?:[,;.]?\s+(?:and|then)\s+|[,;]\s*)'
    r'(?:please\s+)?(?:include|add|provide|give)\s+(?:(?:a|an|one)\s+)?'
    r'(?:(?:verified|exact|verbatim|literal|short|direct|source)\s+)*'
    r'(?:quotation|quote)(?:\s+from\s+(?:the\s+)?(?:source|corpus))?[.!?]?\s*$', re.I)


def quotation_search_query(question):
    """Quote delivery instructions stay in the user request, not the search query."""
    if not isinstance(question, str):
        return question  # The normal tool validator handles invalid arguments.
    match = QUOTE_SUFFIX.search(question)
    return question[:match.start()].rstrip(' ,;.') if match and question[:match.start()].strip() else question


class QuoteCompletionError(ValueError):
    """Give the model bounded feedback when it skips a requested quote step."""


def finish_quotations(result, session, question):
    """Never treat a RAG-generated quotation as proof of an exact-quote lookup."""
    calls = [value for (name, _), value in session.cache.items() if name == 'quote_exact']
    quotes = [value for value in calls if value.get('found') is True and value.get('quote')]
    requested = bool(QUOTE_REQUEST.search(question)) or bool(calls)
    if not requested:
        return result
    if result['status'] != 'answered':
        # A genuine out-of-scope refusal needs no quote attempt. An attempted but
        # failed quote must be visible instead of silently returning only a summary.
        if calls and not quotes:
            return {'status': 'refused', 'answer': QUOTE_REFUSAL, 'citations': [],
                    'answer_mode': 'quote_unavailable'}
        return result
    if not quotes:
        raise QuoteCompletionError(
            'Requested quotation is not verified. Call quote_exact using a phrase copied '
            'from evidence. If its chunk spans pages, try the other cited page after a '
            'not-found result. If verification is impossible, return status=refused.')
    if result.get('answer_mode') == 'grounded_answer':
        raise QuoteCompletionError(
            'Do not select grounded_answer_id for this multi-part quote request. Set it '
            'to null and give the requested explanation with source citation_ids. Do not '
            'repeat the quotation or type source/page labels: the application appends '
            'verified quotations and their source metadata automatically.')

    parts = [result['answer']]
    citations = {row['id']: row for row in result['citations']}
    seen = set()
    for quote in quotes:
        for row in quote['evidence']:
            if row['id'] in seen:
                continue
            seen.add(row['id'])
            # Code fences preserve every source character, including Markdown-like
            # text; a longer fence prevents embedded backticks from closing it.
            text = quote['quote']
            fence = '`' * max(3, 1 + max((len(m[0]) for m in re.finditer(r'`+', text)), default=0))
            source = row['source'].replace('`', '')
            pages = ', '.join(map(str, row['pages']))
            parts.append(f"**Verified quotation** [{row['id']}] — `{source}`, page {pages}\n\n{fence}text\n{text}\n{fence}")
            citations[row['id']] = {key: row[key] for key in ('id', 'chunk_id', 'document_id', 'source', 'pages')}
    return {**result, 'answer': '\n\n'.join(parts), 'citations': list(citations.values()),
            'answer_mode': 'generated_with_quotes'}
