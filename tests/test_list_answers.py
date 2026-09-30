"""List answers must retain source labels, provenance, and refusal semantics."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services.rag import answering
from services.rag.list_answers import ITEM_PROMPT, list_payload, list_response_format
from test_answering import completion, fake_client, make_hits


ITEMS = [
    {'label': 'a', 'text': '(a) Source rule and condition.', 'citation_ids': [1, 2]},
    {'label': 'b', 'text': '(b) Other rule.', 'citation_ids': [2]},
]


def payload():
    # Deliberately reverse the response keys; rendering must follow source order.
    return {'answerable': True, 'items': {
        'b': {'text': 'Second rule.', 'citation_ids': [2]},
        'a': {'text': 'First rule and condition.', 'citation_ids': [2]},
    }}


class ListAnswerTests(unittest.TestCase):
    def test_schema_requires_source_labels_instead_of_a_fixed_eight_items(self):
        result = list_response_format(ITEMS, [{'id': 1}, {'id': 2}])
        schema = result['json_schema']['schema']['properties']['items']
        self.assertEqual(schema['required'], ['a', 'b'])
        self.assertEqual(set(schema['properties']), {'a', 'b'})
        self.assertFalse(schema['additionalProperties'])

    def test_source_order_and_cross_chunk_citations_survive_model_omissions(self):
        source, answer = deepcopy(ITEMS), payload()
        original = deepcopy(answer)
        result = list_payload(answer, source)
        self.assertEqual(result['statements'], [
            {'text': '(a) First rule and condition.', 'citation_ids': [1, 2]},
            {'text': '(b) Second rule.', 'citation_ids': [2]},
        ])
        self.assertEqual(source, ITEMS)
        self.assertEqual(answer, original)

    def test_missing_extra_empty_and_invalid_items_are_rejected(self):
        missing = payload(); del missing['items']['b']
        extra = payload(); extra['items']['c'] = extra['items']['b']
        empty = payload(); empty['items']['a']['text'] = ''
        uncited = payload(); uncited['items']['a']['citation_ids'] = []
        invalid = payload(); invalid['items']['a']['citation_ids'] = [True]
        for value in (missing, extra, empty, uncited, invalid):
            with self.subTest(value=value), self.assertRaises(ValueError):
                list_payload(value, ITEMS)

    def test_refusal_has_no_hidden_statements(self):
        refusal = {'answerable': False, 'items': {
            label: {'text': '', 'citation_ids': []} for label in ('a', 'b')}}
        self.assertEqual(list_payload(refusal, ITEMS), {'answerable': False, 'statements': []})
        refusal['items']['a']['text'] = 'Unsupported assertion.'
        with self.assertRaises(ValueError):
            list_payload(refusal, ITEMS)

    def test_generation_uses_item_contract_and_resolves_original_source_citations(self):
        client, create = fake_client(completion(payload()))
        with patch.object(answering, 'group_article_items', return_value={'items': ITEMS}):
            result = answering.generate_answer('List the article rules.', make_hits(), client, expanded=True)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['list_item_labels'], ['a', 'b'])
        self.assertIn('(a) First rule and condition. [1] [2]', result['answer'])
        self.assertEqual(result['citations'][0]['chunk_id'], 'chunk-0')
        request = create.call_args.kwargs
        self.assertEqual(request['messages'][0]['content'], ITEM_PROMPT)
        context = request['messages'][1]['content'].split('<context>\n')[1].split('\n</context>')[0]
        self.assertEqual(json.loads(context)['list_items'], ITEMS)
        self.assertEqual(request['max_completion_tokens'], 2400)
        create.assert_called_once()

    def test_fabricated_extra_citation_is_still_rejected_after_adding_source_citations(self):
        response = payload(); response['items']['a']['citation_ids'] = [999]
        client, _ = fake_client(completion(response))
        with patch.object(answering, 'group_article_items', return_value={'items': ITEMS}):
            result = answering.generate_answer('List the article rules.', make_hits(), client, expanded=True)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['citations'], [])


if __name__ == '__main__':
    unittest.main()
