"""Regression coverage for skipped, lost, wrong-page, and failed quotations."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.agent.loop import run_agent, MAX_STEPS
from services.agent.quotations import finish_quotations, QUOTE_REFUSAL
from services.agent.tools.runtime import ToolSession
from test_agent_loop import client_for, response, final, rag_response

QUESTION = 'Explain the rule, then use the exact-quote tool to quote its obligation.'
PHRASE = 'A precise obligation.'
ARGS = {'document_id': 'law', 'page': 51, 'phrase': PHRASE}


def source_result():
    result = rag_response()
    result['hits'][0].update(text='Previous provision. ' + PHRASE, pages=[50,51])
    result['grounded_answer']['citations'][0]['pages'] = [50,51]
    return result


def quote_result(text=PHRASE):
    return {'found': True, 'text': text, 'citation': {'document_id': 'law', 'pages': [51]}}


class QuotationTests(unittest.TestCase):
    def run_case(self, replies, transport, question=QUESTION):
        client, create = client_for(*replies)
        with TemporaryDirectory() as directory:
            result = run_agent({'request_id':'quotation-test','message':question},
                               client=client, transport=transport, log_dir=directory)
        return result, create

    def test_skipping_quote_cannot_return_a_rag_summary_as_complete(self):
        http = Mock(side_effect=[(200,source_result()), (200,quote_result())])
        result, _ = self.run_case([
            response(name='search_corpus',arguments={'query':'rule'}),
            response(final(grounded='rag-1')),
            response(name='quote_exact',arguments=ARGS),
            response(final('The rule requires this obligation.',[1]))],http)
        self.assertEqual(result['status'],'answered')
        self.assertIn('```text\n'+PHRASE+'\n```',result['answer'])
        self.assertEqual(result['citations'][-1]['pages'],[51])
        self.assertEqual([t['name'] for t in result['trace'] if t['type']=='tool'],['search_corpus','quote_exact'])
        self.assertTrue(any(t.get('error')=='incomplete_quote_request' for t in result['trace']))

    def test_a_stored_answer_cannot_discard_a_successful_quote(self):
        result, create = self.run_case([
            response(name='search_corpus',arguments={'query':'rule'}),
            response(name='quote_exact',arguments=ARGS),
            response(final(grounded='rag-1')),
            response(final('Explanation of the supported rule.',[1]))],
            Mock(side_effect=[(200,source_result()),(200,quote_result())]))
        self.assertEqual(create.call_count,4)
        self.assertIn('Explanation of the supported rule.',result['answer'])
        self.assertIn('```text\n'+PHRASE+'\n```',result['answer'])
        self.assertIn('page 51',result['answer'])

    def test_failed_page_suggests_the_other_page_and_retry_has_exact_provenance(self):
        http = Mock(side_effect=[(200,source_result()),(404,{'found':False}),(200,quote_result())])
        result, _ = self.run_case([
            response(name='search_corpus',arguments={'query':'rule'}),
            response(name='quote_exact',arguments={**ARGS,'page':50}),
            response(name='quote_exact',arguments=ARGS),
            response(final('The obligation is explained above.',[1]))],http)
        tools = [t for t in result['trace'] if t['type']=='tool']
        self.assertIn('pages [51]',tools[1]['output']['message'])
        self.assertEqual(result['citations'][-1]['pages'],[51])
        self.assertEqual([call.args[2]['page'] for call in http.call_args_list[1:]],[50,51])

    def test_repeated_unverified_quotes_end_in_a_clear_refusal_within_budget(self):
        result, create = self.run_case([
            response(name='search_corpus',arguments={'query':'rule'}),
            *[response(final(grounded='rag-1')) for _ in range(MAX_STEPS-1)]],
            Mock(return_value=(200,source_result())))
        self.assertEqual(create.call_count,MAX_STEPS)
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['answer'],QUOTE_REFUSAL)
        self.assertEqual(result['citations'],[])

    def test_provider_failure_during_quote_recovery_cannot_expose_rejected_summary(self):
        result, _ = self.run_case([
            response(name='search_corpus',arguments={'query':'rule'}),
            response(final(grounded='rag-1')),
            TimeoutError('provider unavailable')],Mock(return_value=(200,source_result())))
        self.assertEqual(result['status'],'error')
        self.assertNotIn('exact grounded answer',result['answer'])

    def test_failed_quote_refusal_is_visible_instead_of_summary_only(self):
        result, _ = self.run_case([
            response(name='quote_exact',arguments=ARGS),
            response(final('Not found.',status='refused'))],Mock(return_value=(404,{})))
        self.assertEqual(result['answer'],QUOTE_REFUSAL)

    def test_refusing_an_unrelated_quote_request_does_not_force_a_tool(self):
        result, _ = self.run_case([response(final('Outside the corpus.',status='refused'))],
                                 Mock(side_effect=AssertionError('No HTTP')),question='Quote an unrelated book.')
        self.assertEqual(result['status'],'refused')
        self.assertFalse(any(t['type']=='tool' for t in result['trace']))

    def test_quote_characters_are_preserved_and_duplicates_render_once(self):
        import time
        text = 'A *literal* rule.\n```\nNot a formatting instruction.'
        http = Mock(return_value=(200,quote_result(text)))
        session = ToolSession('test',http,'http://rag',time.monotonic()+10)
        session.execute('quote_exact',{**ARGS,'phrase':text})
        session.execute('quote_exact',{**ARGS,'phrase':text})
        final_result = finish_quotations({'status':'answered','answer':'Verified below.',
                                         'citations':[],'answer_mode':'generated'},session,QUESTION)
        self.assertIn('````text\n'+text+'\n````',final_result['answer'])
        self.assertEqual(final_result['answer'].count('**Verified quotation**'),1)
        self.assertEqual(http.call_count,1)
