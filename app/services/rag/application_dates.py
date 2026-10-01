"""Look up application clauses in the supplied Act, including transitional rules.

Topics select source clauses; dates and qualifications are copied from the corpus.
This is a lookup of the uploaded version, not a live update of EU law.
"""
import re
from .corpus_sections import articles, evidence
from .definitions import normalize

TOPICS = {
    'general': ('General application', 113, 'general'),
    'ai_literacy': ('AI literacy / Chapter I', 113, 'a'),
    'prohibited_practices': ('Prohibited practices / Chapter II', 113, 'a'),
    'gpai': ('General-purpose AI / Chapter V', 113, 'b'),
    'governance': ('Governance / Chapter VII', 113, 'b'),
    'penalties': ('Chapter XII penalties; Article 101 is excepted', 113, 'b'),
    'gpai_fines': ('Article 101 follows the general date; excluded from the early date', 113, 'general+b'),
    'conformity_bodies': ('Notifying authorities and notified bodies / Chapter III Section 4', 113, 'b'),
    'confidentiality': ('Confidentiality / Article 78', 113, 'b'),
    'high_risk_products': ('Article 6(1) and corresponding obligations', 113, 'c'),
    'high_risk': ('High-risk systems: general date and Article 6(1) exception', 113, 'general+c'),
    'transparency': ('Transparency / Article 50: general application', 113, 'general'),
    'legacy_gpai': ('GPAI models placed on the market before the cutoff', 111, '3'),
    'legacy_high_risk': ('Existing high-risk systems, design changes and public authorities', 111, '2'),
    'large_scale_it': ('Existing large-scale IT systems under Annex X', 111, '1'),
}
ALIASES = {
    'general application': 'general', 'ai act': 'general', 'article 113': 'general',
    'article 4': 'ai_literacy', 'ai literacy': 'ai_literacy', 'chapter i': 'ai_literacy',
    'article 5': 'prohibited_practices', 'prohibited practices': 'prohibited_practices', 'chapter ii': 'prohibited_practices',
    'general purpose ai': 'gpai', 'general purpose ai models': 'gpai', 'chapter v': 'gpai',
    'chapter vii': 'governance', 'chapter xii': 'penalties', 'article 99': 'penalties',
    'article 101': 'gpai_fines', 'gpai fines': 'gpai_fines',
    'chapter iii section 4': 'conformity_bodies', 'notified bodies': 'conformity_bodies',
    'article 78': 'confidentiality', 'article 6 1': 'high_risk_products',
    'high risk ai systems': 'high_risk', 'high risk': 'high_risk', 'annex iii': 'general',
    'article 50': 'transparency', 'chatbots': 'transparency',
    'existing gpai models': 'legacy_gpai', 'article 111 3': 'legacy_gpai',
    'existing high risk systems': 'legacy_high_risk', 'article 111 2': 'legacy_high_risk',
    'annex x': 'large_scale_it', 'article 111 1': 'large_scale_it',
}


def _clauses(section, article):
    text = section['text']
    result = {}
    if article == 113:
        general = re.search(r'It shall apply from[^.]+\.', text)
        if general:
            result['general'] = (general.start(), general.end())
        markers = list(re.finditer(r'(?m)^\(([a-c])\)\s', text))
    else:
        markers = list(re.finditer(r'(?m)^(\d+)\.\s', text))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        # Footnotes are not part of the transitional obligation.
        footnote = re.search(r'(?m)^\(\d+\)\s', text[marker.end():end])
        if footnote:
            end = marker.end() + footnote.start()
        result[marker[1]] = (marker.start(), end)
    return result


def get_application_dates(pages, topic):
    key = normalize(topic)
    canonical = {normalize(name): name for name in TOPICS}
    key = canonical.get(key, ALIASES.get(key))
    if key is None:
        return {'found': False, 'topic': topic, 'message': 'Choose a supported topic; use search_corpus for other deadlines.',
                'available_topics': {k:v[0] for k,v in TOPICS.items()}, 'hits': []}
    label, number, selection = TOPICS[key]
    sections = articles(pages, number)
    if len(sections) != 1:
        return {'found': False, 'message': f'Article {number} is missing or ambiguous in the corpus.', 'hits': []}
    section = sections[0]
    clauses = _clauses(section, number)
    wanted = selection.split('+')
    if any(item not in clauses for item in wanted):
        return {'found': False, 'message': 'The expected application clause could not be verified.', 'hits': []}
    hits = [evidence(section, *clauses[item], 'application-date') for item in wanted]
    rules = [{'article': f'Article {number}' + (f'({item})' if item != 'general' else ''),
              'text': hit['text'],
              'dates_in_clause': re.findall(r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b', hit['text'])}
             for item, hit in zip(wanted, hits)]
    return {'found': True, 'topic': key, 'label': label, 'rules': rules, 'hits': hits,
            'scope': 'Dates in the uploaded Act. Preserve the conditions and exceptions; these are not live legal updates.'}
