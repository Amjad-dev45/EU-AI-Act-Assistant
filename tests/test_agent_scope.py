"""A plausible unsupported model reply must never reach the conversation."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.agent.loop import run_agent
from services.agent.scope import enforce_scope, REFUSAL
from services.ui.tool_calls import format_tool_calls
from test_agent_loop import client_for, response, final, rag_response


class ScopeTests(unittest.TestCase):
    def run_case(self, replies, message, *, history=None, transport=None):
        client, calls = client_for(*replies)
        with TemporaryDirectory() as directory:
            result = run_agent({'request_id': 'scope-test', 'message': message,
                                'history': history or []}, client=client,
                               transport=transport or Mock(side_effect=AssertionError('No HTTP expected')),
                               log_dir=directory)
        return result, calls

    def test_unrelated_answers_cannot_bypass_rag_with_or_without_history(self):
        for question in ('How do I train a neural network in PyTorch?',
                         'Give me a cake recipe.', 'Hello, explain quantum mechanics.',
                         'What is the capital of France?'):
            for history in ([], [{'role': 'assistant', 'content': 'Here is a general tutorial.'}]):
                with self.subTest(question=question, history=history):
                    result, calls = self.run_case([response(final('Here is an unsupported tutorial.'))],
                                                 question, history=history)
                    self.assertEqual(result['status'], 'refused')
                    self.assertEqual(result['answer'], REFUSAL)
                    self.assertEqual(calls.call_count, 1)
                    self.assertEqual(result['trace'][-1]['type'], 'scope')
                    self.assertIn('No supported answer', format_tool_calls(result['trace']))

    def test_corpus_facts_also_need_current_evidence(self):
        result, _ = self.run_case([response(final('Article 4 is about AI literacy.'))], 'What does Article 4 say?')
        self.assertEqual(result['status'], 'refused')

    def test_greetings_thanks_and_capabilities_still_need_no_tools(self):
        for question in ('Hello!', 'Thanks!', 'What can you do?', 'Bonjour!'):
            with self.subTest(question=question):
                result, _ = self.run_case([response(final('Hello!'))], question)
                self.assertEqual(result['status'], 'answered')
                self.assertFalse(any(t['type'] == 'tool' for t in result['trace']))

    def test_arithmetic_followup_uses_only_actual_calculator_result(self):
        result, _ = self.run_case([
            response(name='calculator', arguments={'expression': '2450000 / 2'}),
            response(final('123 is the answer. Here is a PyTorch tutorial.'))],
            'Now divide that by two.', history=[{'role': 'assistant', 'content': 'The result is 2450000.'}])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['answer'], '2450000 / 2 = 1,225,000.0')
        self.assertNotIn('PyTorch', result['answer'])

    def test_failed_calculator_is_not_evidence(self):
        result, _ = self.run_case([response(name='calculator', arguments={'expression': '1/0'}),
                                  response(final('An invented answer.'))], 'Calculate 1/0.')
        self.assertEqual(result['status'], 'refused')

    def test_failed_search_cannot_fall_back_to_model_knowledge(self):
        result, _ = self.run_case([response(name='search_corpus', arguments={'query': 'question'}),
                                  response(final('Use torch.nn.Linear.'))], 'How do I train in PyTorch?',
                                 transport=Mock(return_value=(503, {})))
        self.assertEqual(result['status'], 'refused')

    def test_grounded_answers_and_validated_citations_are_preserved(self):
        result, _ = self.run_case([response(name='search_corpus', arguments={'query': 'rule'}),
                                  response(final(grounded='rag-1'))], 'What is the rule?',
                                 transport=Mock(return_value=(200, rag_response())))
        self.assertEqual(result['answer'], 'The exact grounded answer. [1]')
        session = SimpleNamespace(cache={})
        result = {'answer': 'Supported claim.', 'citations': [{'id': 1}], 'status': 'answered'}
        self.assertEqual(enforce_scope(result, session, 'What is the rule?'), (result, None))

    def test_refusal_label_cannot_smuggle_an_unsupported_tutorial(self):
        result, _ = self.run_case([response(final('I cannot help, but here is the training loop.', status='refused'))],
                                 'How do I train a neural network?')
        self.assertEqual(result['answer'], REFUSAL)
