"""Trace fidelity and separation from the agent's model context."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.rag.diagnostics import build_trace
from services.agent.tools.runtime import ToolSession
from services.ui.retrieval_trace import format_retrieval
from services.ui.tool_calls import format_tool_calls
from services.ui.security_notice import tool_security_events
from services.ui.execution_steps import format_execution_steps
from test_agent_loop import client_for, response, final, rag_response


def diagnostic():
    a = {'chunk_id': 'a', 'source': 'act.pdf', 'pages': [1], 'text': 'First passage.', 'score': 0.9}
    b = {'chunk_id': 'b', 'source': 'act.pdf', 'pages': [51], 'text': '<script>bad</script> [link](https://bad)',
         'score': 0.3, 'bm25_score': 2}
    return build_trace('Article 4?', [a, b], [b], [a, b], [{**b, 'rerank_score': 9}, a], [b],
                       ['article', '4', 'article_4'], hybrid=True, rerank=True, candidate_k=20, k=1, timings={})


class TraceTests(unittest.TestCase):
    def test_injection_guard_is_a_collapsed_substep_in_the_correct_tool_call(self):
        data = {'retrieval': {'trace': diagnostic()}, 'answer': {'status': 'answered'},
                'context': {'security': {'blocked': [{'chunk_id': 'b', 'source': 'act.pdf',
                    'pages': [51], 'reasons': ['instruction_override'], 'stage': 'retrieval_screening'}]}}}
        call = {'type': 'tool', 'name': 'search_corpus', 'diagnostics': data, 'output': {}}
        steps = format_execution_steps(data, 2, security_events=tool_security_events(call))
        self.assertIn('2.7', steps)
        self.assertIn('es-security', steps)
        self.assertLess(steps.index('Select top passages'), steps.index('Source screening after retrieval'))
        self.assertLess(steps.index('Source screening after retrieval'), steps.index('RAG answer'))
        self.assertIn('Flagged passages appeared in: FAISS, BM25', steps)
        cards = format_tool_calls([{'type': 'tool', 'name': 'calculator', 'output': {'result': 4}}, call])
        self.assertIn('2. RAG search', cards)
        self.assertIn('Injection blocked', cards)
        self.assertNotIn(' open', cards)

    def test_agent_evidence_check_is_visible_even_without_a_retrieval_trace(self):
        call = {'type': 'tool', 'name': 'quote_exact', 'output': {'error': 'SourceInjectionError',
                'security': {'blocked': [{'source': 'law.pdf', 'pages': [1], 'chunk_id': 'bad',
                    'reasons': ['instruction_override'], 'stage': 'agent_evidence'}]}}}
        cards = format_tool_calls([call])
        self.assertIn('Agent evidence validation', cards)
        self.assertIn('es-security', cards)
        self.assertNotIn('FAISS · semantic search', cards)

    def test_rank_changes_and_full_candidates_are_observed(self):
        trace = diagnostic()
        stages = {s['name']: s for s in trace['stages']}
        selected = stages['selected']['results'][0]
        self.assertEqual([selected[k] for k in ('faiss_rank', 'bm25_rank', 'fusion_rank', 'rerank_rank')], [2, 1, 2, 1])
        self.assertEqual(stages['reranker']['count'], 2)
        self.assertEqual(trace['numbered_references'], ['article_4'])
        ui = format_retrieval([{'diagnostics': {'retrieval': {'trace': trace}}, 'cached': True}])
        self.assertIn('Both FAISS and BM25', ui)
        self.assertIn('Cached result', ui)
        self.assertIn('Selected chunks', ui)
        self.assertNotIn('<script>', ui)
        # This is now HTML rather than Markdown; source strings must remain
        # inert text in the inspector, never active HTML links or scripts.
        self.assertNotIn('<a href=', ui)

    def test_tool_cache_restores_diagnostics_and_calculator_clears_them(self):
        data = rag_response()
        data['retrieval']['trace'] = diagnostic()
        http = Mock(return_value=(200, data))
        session = ToolSession('test', http, 'http://rag:8002', time.monotonic() + 10)
        first, cached = session.execute('search_corpus', {'query': 'rule'})
        observed = session.last_diagnostics
        self.assertNotIn('diagnostics', first)
        session.execute('calculator', {'expression': '2+2'})
        self.assertIsNone(session.last_diagnostics)
        result, cached = session.execute('search_corpus', {'query': 'rule'})
        self.assertTrue(cached)
        self.assertEqual(observed, session.last_diagnostics)
        self.assertEqual(result['result'], first)
        self.assertEqual(http.call_count, 1)


class TraceAgentTests(unittest.TestCase):
    def test_diagnostics_displayed_but_never_sent_to_model(self):
        data = rag_response()
        data['retrieval']['trace'] = diagnostic()
        client, create = client_for(response(name='search_corpus', arguments={'query': 'rule'}),
                                    response(final(grounded='rag-1')))
        from services.agent.loop import run_agent
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            result = run_agent({'request_id': 'trace-test', 'message': 'rule'}, client=client,
                               transport=Mock(return_value=(200, data)), log_dir=directory)
        tool = next(r for r in result['trace'] if r['type'] == 'tool')
        self.assertEqual(tool['diagnostics']['retrieval']['trace'], diagnostic())
        messages = create.call_args.kwargs['messages']
        self.assertNotIn('policy_explanation', json.dumps(messages))
        self.assertNotIn('diagnostics', json.dumps(messages))
        self.assertEqual(json.loads(next(m['content'] for m in messages if m['role'] == 'tool')), tool['output'])
