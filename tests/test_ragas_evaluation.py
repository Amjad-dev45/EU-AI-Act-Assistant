"""No-network evaluation checks: evidence provenance, exclusions, and failures."""
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from evaluation.ragas_inputs import load_samples
from evaluation.ragas_judge import judge_sample, metric_inputs, summarize


class RagasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        (self.data / 'answer_runs').mkdir()
        self.question = {'question': 'What is the rule?', 'expected_answer': 'Student reference.', 'category': 'factual'}
        self.write('questions.json', [self.question])
        self.answer = {'question_number': 1, 'question': self.question['question'], 'status': 'answered',
                       'answer': 'Answer [1]', 'retrieval': {'hits': [{'text': 'Ranked first'}, {'text': 'Ranked second'}]},
                       'answer_context': {'hits': [{'text': 'Expanded article evidence'}]}}
        self.write('answer_runs/20260101.json', self.answer)

    def write(self, name, data):
        (self.data / name).write_text(json.dumps(data))

    def test_uses_saved_full_generation_context_and_ordered_retrieval_separately(self):
        sample = load_samples(self.data, [1])[0]
        inputs = metric_inputs(sample)
        self.assertEqual(inputs['faithfulness']['retrieved_contexts'], ['Expanded article evidence'])
        self.assertEqual(inputs['context_precision']['retrieved_contexts'], ['Ranked first', 'Ranked second'])
        self.assertEqual(inputs['context_recall']['reference'], 'Student reference.')
        self.assertEqual(inputs['factual_correctness']['response'], 'Answer [1]')
        self.assertEqual(len(sample['answer_sha256']), 64)

    def test_selects_latest_question_match_and_refuses_number_misalignment(self):
        self.write('answer_runs/20260102.json', {**self.answer, 'answer': 'New answer'})
        self.assertEqual(load_samples(self.data, [1])[0]['response'], 'New answer')
        self.write('answer_runs/20260103.json', {**self.answer, 'question_number': 2})
        with self.assertRaises(ValueError):
            load_samples(self.data, [1])

    def test_refusal_and_missing_runs_are_explicitly_unscored(self):
        self.write('questions.json', [{**self.question, 'category': 'out-of-scope'}, {'question': 'Missing'}])
        rows = load_samples(self.data, [1, 2])
        self.assertTrue(all(r['skip_reason'] for r in rows))
        self.assertEqual(summarize(rows)['metrics']['faithfulness'], {'mean': None, 'scored': 0, 'unscored': 2})

    def test_missing_generation_context_is_not_replaced_by_retrieved_or_cited_text(self):
        self.answer.pop('answer_context')
        self.write('answer_runs/20260101.json', self.answer)
        with self.assertRaises(ValueError):
            load_samples(self.data, [1])

    def test_errors_and_nan_excluded_negative_cosine_preserved(self):
        class Scorer:
            def __init__(self, value): self.value = value
            async def ascore(self, **kwargs):
                if isinstance(self.value, Exception): raise self.value
                return SimpleNamespace(value=self.value)
        sample = load_samples(self.data, [1])[0]
        row = asyncio.run(judge_sample(sample, {'faithfulness': Scorer(float('nan')),
                        'context_recall': Scorer(TimeoutError('secret must not leak')),
                        'answer_relevancy': Scorer(-0.1), 'factual_correctness': Scorer(0.75)}))
        self.assertIsNone(row['scores']['faithfulness'])
        self.assertEqual(row['errors']['context_recall'], 'TimeoutError')
        self.assertNotIn('secret', json.dumps(row))
        stats = summarize([row, {'skip_reason': 'refused'}])
        self.assertEqual(stats['metrics']['answer_relevancy']['mean'], -0.1)
        self.assertEqual(stats['metrics']['factual_correctness']['scored'], 1)
        self.assertIsNone(stats['metrics']['faithfulness']['mean'])


if __name__ == '__main__': unittest.main()
