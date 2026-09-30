"""Offline reranking checks: fake model and cached-query pipeline boundaries."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))

from services.rag import embeddings, reranking, retrieval
from retrieval_checker import first_expected_rank


class FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return self.scores


def make_hits(count):
    return [{"chunk_id": f"chunk-{i}", "document_id": f"document-{i}",
             "source": f"source-{i}.pdf", "pages": [i + 1, i + 2],
             "text": f"passage {i}", "score": 0.9 - i / 100}
            for i in range(count)]


class RerankingTests(unittest.TestCase):
    def setUp(self):
        # Any accidental network or model-loading path fails immediately.
        for module, name in ((embeddings, "create_client"), (reranking, "load_model")):
            guard = patch.object(module, name, side_effect=AssertionError("unexpected external call"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_descending_stable_scores_preserve_sources_and_input(self):
        hits = make_hits(3)
        original = deepcopy(hits)
        model = FakeModel([-7.0, 4.0, 4.0])
        ranked = reranking.rerank("question", hits, k=3, model=model)
        self.assertEqual(model.pairs, [("question", hit["text"]) for hit in hits])
        self.assertEqual([hit["chunk_id"] for hit in ranked], ["chunk-1", "chunk-2", "chunk-0"])
        self.assertEqual([hit["rerank_score"] for hit in ranked], [4.0, 4.0, -7.0])
        for result, position in zip(ranked, [1, 2, 0]):
            self.assertEqual({key: result[key] for key in original[position]}, original[position])
            self.assertIsNot(result, hits[position])
        self.assertEqual(hits, original)

    def test_invalid_model_scores_fail(self):
        for scores in ([1, 2], [1, 2, 3, 4], [1, np.nan, 3],
                       [1, np.inf, 3], [1, -np.inf, 3]):
            with self.subTest(scores=scores):
                with self.assertRaisesRegex(ValueError, "one finite score"):
                    reranking.rerank("question", make_hits(3), model=FakeModel(scores))

    def test_empty_candidates_do_not_load_model(self):
        self.assertEqual(reranking.rerank("question", []), [])
        reranking.load_model.assert_not_called()

    def test_pipeline_fetches_twenty_and_keeps_best_five(self):
        hits = make_hits(20)
        model = FakeModel(list(range(20)))
        query_vector = np.array([[1.0, 0.0]])
        with patch.object(retrieval, "load_index", return_value=("index", hits, {})), \
             patch.object(retrieval, "embed_texts", return_value=(query_vector, 0)), \
             patch.object(retrieval, "search_index", return_value=hits) as search, \
             patch.object(reranking, "load_model", return_value=model):
            result = retrieval.retrieve("question", Path("unused"), use_hybrid=False)
        self.assertEqual(search.call_args.args[3], 20)
        self.assertEqual([hit["chunk_id"] for hit in result["hits"]],
                         [f"chunk-{i}" for i in range(19, 14, -1)])
        self.assertEqual(result["vector_hits"], hits[:5])
        self.assertEqual(result["candidate_count"], 20)
        self.assertEqual(result["embedding_tokens"], 0)
        self.assertTrue(result["reranked"])

    def test_baseline_returns_vector_five_without_reranker(self):
        hits = make_hits(5)
        with patch.object(retrieval, "load_index", return_value=("index", hits, {})), \
             patch.object(retrieval, "embed_texts", return_value=(np.array([[1.0, 0.0]]), 0)), \
             patch.object(retrieval, "search_index", return_value=hits) as search, \
             patch.object(reranking, "rerank", side_effect=AssertionError("baseline reranked")):
            result = retrieval.retrieve("question", Path("unused"), use_rerank=False)
        self.assertEqual(search.call_args.args[3], 5)
        self.assertEqual(result["hits"], hits)
        self.assertFalse(result["reranked"])
        self.assertTrue(all("rerank_score" not in hit for hit in result["hits"]))
        reranking.load_model.assert_not_called()

    def test_invalid_candidate_limit_fails_before_loading_index(self):
        with patch.object(retrieval, "load_index") as load:
            with self.assertRaisesRegex(ValueError, "at least k"):
                retrieval.retrieve("question", Path("unused"), k=5, candidate_k=4)
        load.assert_not_called()

    def test_expected_rank_matches_source_and_any_recorded_page(self):
        hits = make_hits(3)
        self.assertEqual(first_expected_rank(hits, [{"source": "source-1.pdf", "page": 3}]), 2)
        self.assertIsNone(first_expected_rank(hits, [{"source": "other.pdf", "page": 3}]))
        self.assertIsNone(first_expected_rank(hits, []))
        self.assertEqual(first_expected_rank(hits, [{"source": "source-2.pdf", "page": 4},
                                                  {"source": "source-0.pdf", "page": 1}]), 1)


if __name__ == "__main__":
    unittest.main()
