"""Lookups must preserve exact legal scope, page provenance and missing matches."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.rag.definitions import lookup_definition
from services.rag.application_dates import get_application_dates
from services.agent.tools.runtime import ToolSession, TOOLS
from services.agent.loop import final_answer
from services.rag import handlers

PAGES = [
    {'source': 'act.pdf', 'page': 1, 'text': "Article 3\nDefinitions\n(1) ‘provider’ means the person developing\n"},
    {'source': 'act.pdf', 'page': 2, 'text': "an AI system;\n(2) ‘deployer’ means the person using it;\nArticle 4\nOther rule."},
    {'source': 'act.pdf', 'page': 3, 'text': 'Article 111\nTransition\n1. Existing large-scale systems before 2 August 2027 comply by 31 December 2030.\n2. Existing high-risk systems before 2 August 2026 need significant changes; public authorities comply by 2 August 2030.\n3. Models placed on the market before 2 August 2025 comply by 2 August 2027.\nArticle 112\nReviews.'},
    {'source': 'act.pdf', 'page': 4, 'text': 'Article 113\nApplication\nIt shall apply from 2 August 2026.\nHowever:\n(a) Chapters I and II shall apply from 2 February 2025;\n(b) Chapter V and Chapter XII shall apply from 2 August 2025, with the exception of Article 101;\n(c) Article 6(1) and the corresponding obligations shall apply from 2 August 2027.\nThis Regulation shall be binding in its entirety.\nANNEX I\nUnrelated date 1 January 2000.'},
]


class CorpusLookupTests(unittest.TestCase):
    def test_definitions_distinguish_terms_preserve_cross_page_text_and_stop_at_heading(self):
        provider = lookup_definition(PAGES, 'PROVIDERS')
        deployer = lookup_definition(PAGES, 'deployer')
        self.assertEqual(provider['article'], 'Article 3(1)')
        self.assertEqual(provider['hits'][0]['pages'], [1, 2])
        self.assertIn('developing\n\n\nan AI system', provider['definition'])
        self.assertEqual(deployer['hits'][0]['pages'], [2])
        self.assertNotIn('Article 4', deployer['definition'])
        self.assertNotEqual(provider['hits'][0]['chunk_id'], deployer['hits'][0]['chunk_id'])

    def test_unknown_term_and_duplicate_documents_do_not_guess(self):
        self.assertFalse(lookup_definition(PAGES, 'service provider')['found'])
        duplicate = PAGES + [{**p, 'source': 'other.pdf'} for p in PAGES]
        self.assertFalse(lookup_definition(duplicate, 'provider')['found'])

    def test_qualified_definition_has_its_own_boundary(self):
        pages = [{'source': 'act.pdf', 'page': 1, 'text':
            "Article 3\n(57) ‘sandbox’ means a testing environment;\n"
            "(58) ‘subject’, for the purpose of real-world testing, means a participant;\n"
            "(59) ‘informed consent’ means the subject's agreement;\nArticle 4\nOther rule."}]
        self.assertNotIn('subject', lookup_definition(pages, 'sandbox')['definition'])
        subject = lookup_definition(pages, 'subject')
        self.assertTrue(subject['found'])
        self.assertEqual(subject['article'], 'Article 3(58)')
        self.assertIn('for the purpose of real-world testing', subject['definition'])
        self.assertNotIn('(59)', subject['definition'])

    def test_structured_definition_is_not_restricted_to_article_three(self):
        pages = [{'source': 'other.pdf', 'page': 8, 'text':
                  "Article 19\nDefinitions\n(1) ‘widget’ means a device for this chapter;\nArticle 20\nOther rule."}]
        result = lookup_definition(pages, 'widget')
        self.assertTrue(result['found'])
        self.assertEqual(result['article'], 'Article 19(1)')
        self.assertEqual(result['hits'][0]['pages'], [8])
        self.assertNotIn('Other rule', result['definition'])

    def test_unstructured_definition_uses_full_corpus_rag_and_records_usage(self):
        passage = {'chunk_id': 'glossary:1', 'document_id': 'glossary.pdf', 'source': 'glossary.pdf',
                   'pages': [12], 'text': 'Widgets are devices for the task.'}
        result = {'status': 'answered', 'answer': 'Devices for the task. [1]',
                  'citations': [{'id': 1, **{k: passage[k] for k in ('chunk_id', 'source', 'pages')}}],
                  'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                  'answer_context': {'hits': [passage]},
                  'retrieval': {'embedding_tokens': 4, 'hybrid': True, 'reranked': True}}
        def transport(_method, _url, payload, *_args):
            return handlers.definition(payload)
        with patch.object(handlers.store, 'load_source', return_value=[]), \
             patch.object(handlers, 'answer_query', return_value=result) as search:
            session = ToolSession('test', transport, 'http://rag', time.monotonic()+10)
            output, _ = session.execute('lookup_definition', {'term': 'widgets'})
        query = search.call_args.args[1]
        self.assertIn('widgets', query)
        self.assertIn('mere mention', query)
        self.assertNotIn('Article 3', query)
        self.assertTrue(output['found'])
        self.assertEqual(output['evidence'][0]['pages'], [12])
        self.assertEqual(session.embedding_tokens, 4)
        self.assertEqual(session.usage['total_tokens'], 15)
        self.assertTrue(session.last_diagnostics['retrieval']['hybrid'])
        answer = final_answer({'answer': 'See cited answer.', 'grounded_answer_id': output['grounded_answer']['id'],
                              'citation_ids': [], 'status': 'answered'}, session)
        self.assertEqual(answer['answer'], 'Devices for the task. [1]')

    def test_failed_definition_search_does_not_offer_unrelated_passages_as_evidence(self):
        result = {'status': 'refused', 'answer': 'No definition found.', 'citations': [],
                  'usage': {}, 'answer_context': {'hits': [{'text': 'Unrelated mention.'}]}, 'retrieval': {}}
        with patch.object(handlers.store, 'load_source', return_value=PAGES), \
             patch.object(handlers, 'answer_query', return_value=result):
            status, output = handlers.definition({'request_id': 'missing', 'term': 'unlisted term'})
        self.assertEqual(status, 200)
        self.assertFalse(output['found'])
        self.assertEqual(output['hits'], [])

    def test_ambiguous_definition_does_not_silently_select_a_document(self):
        duplicate = PAGES + [{**p, 'source': 'other.pdf'} for p in PAGES]
        with patch.object(handlers.store, 'load_source', return_value=duplicate), \
             patch.object(handlers, 'answer_query', side_effect=AssertionError('Ambiguity must be explicit')):
            _, result = handlers.definition({'request_id': 'ambiguous', 'term': 'provider'})
        self.assertFalse(result['found'])
        self.assertEqual(result['reason'], 'ambiguous')
        self.assertEqual(len(result['matches']), 2)

    def test_date_exception_is_returned_with_general_date_for_article_101(self):
        result = get_application_dates(PAGES, 'Article 101')
        self.assertEqual(result['rules'][0]['dates_in_clause'], ['2 August 2026'])
        self.assertIn('exception of Article 101', result['rules'][1]['text'])
        self.assertEqual([h['pages'] for h in result['hits']], [[4], [4]])
        self.assertNotIn('2000', json.dumps(result))

    def test_legacy_cutoff_is_not_silently_reported_as_application_date(self):
        result = get_application_dates(PAGES, 'legacy_gpai')
        self.assertEqual(result['rules'][0]['dates_in_clause'], ['2 August 2025', '2 August 2027'])
        self.assertIn('before', result['rules'][0]['text'])
        self.assertIn('comply by', result['rules'][0]['text'])
        self.assertEqual(result['hits'][0]['pages'], [3])

    def test_high_risk_includes_exception_and_missing_topics_return_no_evidence(self):
        self.assertEqual(len(get_application_dates(PAGES, 'high_risk')['hits']), 2)
        self.assertFalse(get_application_dates(PAGES, 'unrelated deadline')['found'])
        missing = [p for p in PAGES if p['page'] != 4]
        self.assertFalse(get_application_dates(missing, 'general')['found'])

    def test_tools_register_citable_evidence_and_cache_without_model_or_embedding_calls(self):
        calls = []
        def transport(method, url, payload, timeout, dependency):
            calls.append(url)
            return 200, (lookup_definition(PAGES, payload['term']) if url.endswith('/definition')
                         else get_application_dates(PAGES, payload['topic']))
        session = ToolSession('test', transport, 'http://rag', time.monotonic()+10)
        names = {t['function']['name'] for t in TOOLS}
        self.assertEqual(len(names), 5)
        definition, _ = session.execute('lookup_definition', {'term': 'provider'})
        dates, _ = session.execute('get_application_dates', {'topic': 'general'})
        _, cached = session.execute('lookup_definition', {'term': 'provider'})
        self.assertTrue(cached)
        self.assertEqual(len(calls), 2)
        self.assertEqual(session.embedding_tokens, 0)
        ids = [definition['evidence'][0]['id'], dates['evidence'][0]['id']]
        answer = final_answer({'answer': 'Supported.', 'citation_ids': ids, 'grounded_answer_id': None, 'status': 'answered'}, session)
        self.assertEqual(len(answer['citations']), 2)
        with self.assertRaises(ValueError):
            session.execute('lookup_definition', {'term': 'provider', 'path': '/etc/passwd'})


if __name__ == '__main__':
    unittest.main()
