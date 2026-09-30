"""Offline checks for keyword retrieval, rank fusion, and real FAISS scores."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag import embeddings, hybrid, reranking, retrieval


def chunk(identifier, text, page=1):
    return {"chunk_id": identifier, "document_id": "act", "source": "act.pdf",
            "pages": [page, page + 1], "text": text}


class KeywordSearchTests(unittest.TestCase):
    def test_tokenization_ignores_case_but_keeps_numbers_and_negation(self):
        tokens = hybrid.tokenize("ARTICLE 50(1): do NOT disclose.")
        self.assertEqual(tokens, hybrid.tokenize("article 50(1): do not disclose."))
        self.assertTrue({"article", "50", "1", "not", "disclose"}.issubset(tokens))

    def test_numbered_reference_tokens_keep_article_identity(self):
        tokens = hybrid.tokenize("ARTICLE\n4; Annex IV; Section 2; Article 41.")
        self.assertTrue({"article_4", "annex_iv", "section_2", "article_41"}.issubset(tokens))
        self.assertTrue({"article", "4", "annex", "iv", "section", "2", "41"}.issubset(tokens))
        self.assertNotIn("article_4", hybrid.tokenize("Article 41"))
        self.assertNotIn("article_4", hybrid.tokenize("Paragraph 4"))
        self.assertFalse(any("_" in token for token in hybrid.tokenize("Article, Annex, Section")))

        # The same reference token must reach BM25 from both the document
        # heading (including a line break) and the differently cased query.
        chunks = [chunk("target", "ARTICLE\n4 literacy"),
                  chunk("other_article", "Article 41 documentation"),
                  chunk("paragraph", "Paragraph 4 requirements"),
                  chunk("weather", "weather sunny skies"),
                  chunk("medicine", "medical clinical diagnosis")]
        self.assertEqual(hybrid.BM25Index(chunks).search("article 4", k=1)[0]["chunk_id"], "target")

    def test_empty_and_out_of_vocabulary_queries_have_no_matches(self):
        lexical = hybrid.BM25Index([chunk("a", "article 50 transparency"),
                                    chunk("b", "training documentation")])
        for question in ("", "!!!", "xylophonist"):
            with self.subTest(question=question):
                self.assertEqual(lexical.search(question), [])
        self.assertEqual(hybrid.BM25Index([]).search("article"), [])

    def test_actual_overlap_keeps_zero_or_negative_scores_and_source_metadata(self):
        chunks = [chunk("a", "transparency notice", 4),
                  chunk("b", "transparency information", 8),
                  chunk("c", "unrelated weather", 12)]
        original = deepcopy(chunks)
        # BM25 IDF can be zero or negative in small corpora. Token overlap,
        # rather than a positive-score threshold, determines eligible matches.
        with patch.object(hybrid, "BM25Okapi") as backend:
            backend.return_value.get_scores.return_value = np.array([0.0, -1.0, 10.0])
            lexical = hybrid.BM25Index(chunks)
            hits = lexical.search("TRANSPARENCY", k=3)
            self.assertEqual([hit["chunk_id"] for hit in hits], ["a", "b"])
            self.assertEqual([hit["bm25_score"] for hit in hits], [0.0, -1.0])
            self.assertEqual(len(lexical.search("transparency", k=1)), 1)
            backend.assert_called_once()
        for hit, source in zip(hits, original):
            self.assertEqual({key: hit[key] for key in source}, source)
            self.assertIsNot(hit, chunks[original.index(source)])
        self.assertEqual(chunks, original)


class RankFusionTests(unittest.TestCase):
    def test_one_based_ranks_deduplicate_each_branch_and_merge_metadata(self):
        vector_a = {**chunk("a", "alpha"), "score": 0.7}
        vector_b = {**chunk("b", "beta"), "score": 0.6}
        keyword_a = {**chunk("a", "alpha"), "bm25_score": 12.0}
        keyword_b = {**chunk("b", "beta"), "bm25_score": 20.0}
        vectors = [vector_a, deepcopy(vector_a), vector_b]
        keywords = [keyword_b, keyword_a, deepcopy(keyword_a)]
        originals = deepcopy((vectors, keywords))
        hits = hybrid.reciprocal_rank_fusion(vectors, keywords, k=5)
        self.assertEqual([hit["chunk_id"] for hit in hits], ["a", "b"])
        self.assertAlmostEqual(hits[0]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(hits[1]["rrf_score"], 1 / 63 + 1 / 61)
        self.assertEqual(hits[0]["score"], 0.7)
        self.assertEqual(hits[0]["bm25_score"], 12.0)
        self.assertEqual(hits[0]["pages"], [1, 2])
        self.assertIsNot(hits[0], vector_a)
        self.assertEqual((vectors, keywords), originals)

    def test_fusion_uses_ranks_not_score_scales(self):
        vectors = [{**chunk("a", "alpha"), "score": 0.9},
                   {**chunk("b", "beta"), "score": 0.2}]
        keywords = [{**chunk("b", "beta"), "bm25_score": 1.0},
                    {**chunk("c", "gamma"), "bm25_score": 0.1}]
        before = hybrid.reciprocal_rank_fusion(vectors, keywords, k=3)
        keywords[0]["bm25_score"] = 1_000_000.0
        keywords[1]["bm25_score"] = -50.0
        after = hybrid.reciprocal_rank_fusion(vectors, keywords, k=3)
        self.assertEqual([(hit["chunk_id"], hit["rrf_score"]) for hit in before],
                         [(hit["chunk_id"], hit["rrf_score"]) for hit in after])
        self.assertEqual(before[0]["chunk_id"], "b")

    def test_ties_preserve_vector_then_keyword_order_and_respect_limit(self):
        vectors = [{**chunk("a", "alpha"), "score": 0.4}]
        keywords = [{**chunk("b", "beta"), "bm25_score": 10.0}]
        hits = hybrid.reciprocal_rank_fusion(vectors, keywords, k=2)
        self.assertEqual([hit["chunk_id"] for hit in hits], ["a", "b"])
        self.assertEqual(hits[0]["rrf_score"], hits[1]["rrf_score"])
        self.assertEqual(len(hybrid.reciprocal_rank_fusion(vectors, keywords, k=1)), 1)
        self.assertEqual(hybrid.reciprocal_rank_fusion([], [], k=1), [])

    def test_invalid_fusion_limits_fail(self):
        for options in ({"k": 0}, {"k": -1}, {"rank_constant": 0}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    hybrid.reciprocal_rank_fusion([], [], **options)


class HybridCandidateTests(unittest.TestCase):
    def test_keyword_only_hit_gets_true_cosine_from_faiss_without_numpy_file(self):
        chunks = [chunk("a", "weather sunshine", 2),
                  chunk("b", "article 99 fines", 7)]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32))
        query_vector = np.array([3.0, 4.0], dtype=np.float32)
        vectors = [{**chunks[0], "score": 0.6}]
        originals = deepcopy((chunks, vectors))
        lexical = hybrid.BM25Index(chunks)
        # Search must work from the saved FAISS vectors without embeddings.npy.
        with patch.object(np, "load", side_effect=AssertionError("unexpected file read")):
            hits = hybrid.hybrid_candidates(index, query_vector, "article 99",
                                            vectors, lexical, k=2)
        by_id = {hit["chunk_id"]: hit for hit in hits}
        self.assertEqual(set(by_id), {"a", "b"})
        self.assertEqual(by_id["a"]["score"], 0.6)
        self.assertAlmostEqual(by_id["b"]["score"], 1.0, places=6)
        self.assertIn("bm25_score", by_id["b"])
        self.assertIn("rrf_score", by_id["b"])
        self.assertEqual(by_id["b"]["source"], "act.pdf")
        self.assertEqual(by_id["b"]["pages"], [7, 8])
        np.testing.assert_array_equal(query_vector, [3.0, 4.0])
        self.assertEqual((chunks, vectors), originals)
        self.assertEqual(len(hybrid.hybrid_candidates(index, query_vector, "article 99",
                                                     vectors, lexical, k=1)), 1)

    def test_default_pipeline_passes_keyword_only_candidate_to_reranker(self):
        chunks = [chunk("a", "weather clouds"), chunk("b", "weather sunshine"),
                  chunk("c", "article 99 fines", 7)]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32))

        def choose_keyword(question, candidates, k):
            return [{**hit, "rerank_score": 9.0} for hit in candidates
                    if hit["chunk_id"] == "c"][:k]

        with patch.object(retrieval, "load_index", return_value=(index, chunks, {})), \
             patch.object(retrieval, "embed_texts", return_value=(np.array([[1.0, 0.0]]), 0)), \
             patch.object(embeddings, "create_client", side_effect=AssertionError("API called")), \
             patch.object(reranking, "load_model", side_effect=AssertionError("model loaded")), \
             patch.object(reranking, "rerank", side_effect=choose_keyword) as rerank:
            result = retrieval.retrieve("article 99", Path("unused"), k=1, candidate_k=2)
        # FAISS alone returns a and b. Fusion admits c before reranking.
        self.assertEqual([hit["chunk_id"] for hit in rerank.call_args.args[1]], ["a", "c"])
        self.assertEqual([hit["chunk_id"] for hit in result["vector_hits"]], ["a"])
        self.assertEqual([hit["chunk_id"] for hit in result["hits"]], ["c"])
        self.assertAlmostEqual(result["hits"][0]["score"], 0.6, places=6)
        self.assertEqual(result["hits"][0]["pages"], [7, 8])
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["embedding_tokens"], 0)
        self.assertTrue(result["hybrid"])
        self.assertTrue(result["reranked"])

    def test_vector_only_pipeline_does_not_build_keyword_index(self):
        chunks = [chunk("a", "article 99")]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1.0, 0.0]], dtype=np.float32))
        with patch.object(retrieval, "load_index", return_value=(index, chunks, {})), \
             patch.object(retrieval, "embed_texts", return_value=(np.array([[1.0, 0.0]]), 0)), \
             patch.object(hybrid, "BM25Index", side_effect=AssertionError("keyword index built")), \
             patch.object(reranking, "rerank", side_effect=AssertionError("baseline reranked")):
            result = retrieval.retrieve("article 99", Path("unused"), k=1, use_rerank=False)
        self.assertEqual([hit["chunk_id"] for hit in result["hits"]], ["a"])
        self.assertFalse(result["hybrid"])
        self.assertFalse(result["reranked"])


if __name__ == "__main__":
    unittest.main()
