"""Context bars must reflect one measured request, not cumulative API billing."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from contracts.model_usage import context_snapshot
from services.agent.loop import run_agent
from services.ui.context_usage import format_context
from services.ui.rag_demo import format_rag_demo
from test_agent_loop import client_for, response, final, rag_response


class ContextTests(unittest.TestCase):
    def test_unknown_counts_are_unavailable_and_high_usage_is_bounded_visually(self):
        for value in (None, {}, {'prompt_tokens': None, 'completion_tokens': 10}):
            self.assertIsNone(context_snapshot(value)['used_tokens'])
        unknown = format_context()
        self.assertNotIn('aria-valuenow', unknown)
        self.assertIn('128,000', unknown)
        body = {'context_window': {'agent': context_snapshot({'prompt_tokens': 125000, 'completion_tokens': 1000})}}
        html = format_context(body)
        self.assertIn('126,000 / 128,000', html)
        self.assertIn('cu-high', html)
        self.assertIn('aria-valuenow="126000"', html)
        self.assertIn('width: 98.4375%', html)
        self.assertIsNone(context_snapshot({'prompt_tokens': 5, 'completion_tokens': 3}, 'unknown-model')['limit_tokens'])

    def test_latest_agent_request_and_rag_request_are_separate_from_billing(self):
        client, create = client_for(response(name='search_corpus', arguments={'query': 'rule'}), response(final(grounded='rag-1')))
        body = rag_response()
        body['grounded_answer']['model'] = 'gpt-4o-mini'
        with TemporaryDirectory() as directory:
            result = run_agent({'request_id': 'context-test', 'message': 'rule'}, client=client,
                               transport=Mock(return_value=(200, body)), log_dir=directory)
        self.assertEqual(result['usage']['agent']['total_tokens'], 30)
        self.assertEqual(result['context_window']['agent']['used_tokens'], 15)
        self.assertEqual(result['context_window']['rag']['used_tokens'], 120)
        self.assertEqual(result['context_window']['history_messages_sent'], 0)
        messages = create.call_args.kwargs['messages']
        self.assertNotIn('context_usage', json.dumps(messages))
        self.assertNotIn('context_window', json.dumps(messages))

    def test_demo_does_not_fabricate_retrieval_for_messages_without_search(self):
        self.assertNotIn('class="rd-run"', format_rag_demo([]))
        self.assertNotIn('class="rd-step"', format_rag_demo([{'type': 'tool', 'name': 'calculator'}]))


if __name__ == '__main__': unittest.main()
