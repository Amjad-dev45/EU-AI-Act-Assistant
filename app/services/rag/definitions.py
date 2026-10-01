"""Fast path for structured definitions anywhere; other formats use corpus RAG."""
import re
import unicodedata
from .corpus_sections import articles, evidence

# Some entries insert a qualifier before 'means' (notably Article 3(58)).
# The numbered term marks the boundary, including when that qualifier is present.
DEFINITION = re.compile(r"(?m)^\((\d+)\)\s*[‘']([^’']+)[’'](?=\s|,)")


def normalize(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.findall(r'\w+', value))


def lookup_definition(pages, term):
    entries = []
    # No article number restriction. Keep complete numbered definitions, including
    # their internal lists, and stop at the next entry or verified article heading.
    for section in articles(pages):
        markers = list(DEFINITION.finditer(section['text']))
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(section['text'])
            if not re.search(r'\bmeans\b', section['text'][marker.end():min(end, marker.end()+300)]):
                continue
            entries.append((section, marker, end))
    terms = {normalize(marker[2]) for _, marker, _ in entries}
    key = normalize(term)
    # Only spelling/number variants, never fuzzy semantic matches such as provider/deployer.
    aliases = {'artificial intelligence system': 'ai system', 'general purpose ai': 'general purpose ai model',
               'authorized representative': 'authorised representative', 'deepfake': 'deep fake'}
    key = aliases.get(key, key)
    if key not in terms and key.endswith('s'):
        key = key[:-1]
    matches = [(section, marker, end) for section, marker, end in entries if normalize(marker[2]) == key]
    if not matches:
        return {'found': False, 'term': term, 'reason': 'not_found',
                'message': 'No structured definition matched; search the full corpus.', 'hits': []}
    if len(matches) != 1:
        return {'found': False, 'term': term, 'reason': 'ambiguous',
                'message': 'Multiple definitions match. Ask which document or context the user means, then use search_corpus.',
                'matches': [{'source': section['source'], 'article': f"Article {section['article']}({marker[1]})"}
                            for section, marker, _ in matches], 'hits': []}
    section, marker, end = matches[0]
    hit = evidence(section, marker.start(), end, 'definition')
    return {'found': True, 'term': marker[2], 'article': f"Article {section['article']}({marker[1]})",
            'mode': 'exact_definition', 'definition': hit['text'], 'hits': [hit]}
