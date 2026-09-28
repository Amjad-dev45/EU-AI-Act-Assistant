"""Check selection, citation integrity, logging, and failure behavior without API calls."""

import unittest
from unittest.mock import Mock, patch
import numpy as np

import test_observability
from app.rag.reranking import Reranker


class RerankingTests(unittest.TestCase):
    setUp = test_observability.ObservabilityTests.setUp

    def test_reranking_selects_lower_faiss_candidate_and_preserves_citations(self):
        self.retriever.reranker = Reranker(enabled=True, candidates=20)
        model = Mock()
        model.predict.return_value = np.array([-2., 1., 5.])
        with patch('app.rag.reranking.load_model', return_value=model):
            result = self.answerer.answer('Amount?', k=1)
        self.assertEqual(result['citations'][0]['chunk_id'], 2)
        self.assertEqual(result['citations'][0]['citation_id'], 1)
        self.assertEqual(result['citations'][0]['score'], 0.)
        self.assertEqual(result['citations'][0]['rerank_score'], 5.)
        t = result['telemetry']
        self.assertEqual([p['chunk_id'] for p in t['retrieval_candidates']], [0, 1, 2])
        self.assertEqual(t['context_chunk_ids'], [2])
        self.assertEqual(t['retrieval'][0]['retrieval_rank'], 3)
        self.assertGreaterEqual(t['timings_ms']['reranking_time_ms'], 0)
        self.assertEqual(len(t['api_calls']), 2)
        self.assertAlmostEqual(t['cost']['total_usd'], .00002414)
        pairs = model.predict.call_args.args[0]
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(q == 'Amount?' for q, _ in pairs))
        prompt = self.client.chat.completions.create.call_args.kwargs['messages'][1]['content']
        self.assertIn('PRIVATE PASSAGE C', prompt)
        self.assertNotIn('PRIVATE PASSAGE A', prompt)

    def test_disabled_reranking_does_not_load_model(self):
        with patch('app.rag.reranking.load_model') as load:
            result = self.answerer.answer('Amount?', k=1)
        load.assert_not_called()
        self.assertEqual(result['retrieved_passages'][0]['chunk_id'], 0)
        self.assertIsNone(result['telemetry']['timings_ms']['reranking_time_ms'])
        self.assertEqual(len(result['telemetry']['retrieval_candidates']), 1)

    def test_invalid_scores_fail_before_generation_and_preserve_embedding_usage(self):
        self.retriever.reranker = Reranker(enabled=True)
        for scores in ([float('nan'), 1, 2], [1]):
            with self.subTest(scores=scores), patch('app.rag.reranking.load_model') as load:
                load.return_value.predict.return_value = scores
                with self.assertRaisesRegex(ValueError, 'invalid scores'):
                    self.answerer.answer('Amount?')
        self.client.chat.completions.create.assert_not_called()
        self.assertTrue(all(r['status'] == 'error' for r in self.monitor.records()))

    def test_ties_preserve_original_order_and_k_can_exceed_candidate_setting(self):
        self.retriever.reranker = Reranker(enabled=True, candidates=1)
        with patch('app.rag.reranking.load_model') as load:
            load.return_value.predict.return_value = [1, 1, 1]
            result = self.answerer.answer('Amount?', k=5)
        self.assertEqual([p['chunk_id'] for p in result['retrieved_passages']], [0, 1, 2])

    def test_environment_configuration(self):
        with patch.dict('os.environ', {'RAG_RERANK_ENABLED': 'false', 'RAG_RERANK_CANDIDATES': '30'}):
            r = Reranker()
            self.assertFalse(r.enabled)
            self.assertEqual(r.candidates, 30)
        with patch.dict('os.environ', {'RAG_RERANK_ENABLED': 'invalid'}):
            with self.assertRaises(ValueError):
                Reranker()
        with self.assertRaises(ValueError):
            Reranker(candidates=0)
