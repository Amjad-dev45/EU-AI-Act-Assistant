"""Offline evaluation checks with synthetic page labels and retrieval dependencies."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from evaluation import metrics, runner
from services.rag import embeddings


LABELS = [{"source": "synthetic.pdf", "page": 2}]


def hits_at(rank):
    return [{"chunk_id": f"chunk-{i}", "source": "synthetic.pdf",
             "pages": [1, 2] if i == rank else [10 + i], "score": 1 / i}
            for i in range(1, 7)]


def row(number, rank, after=None, labeled=True, category="a", difficulty="easy"):
    return {"question_number": number, "expected_chunks": LABELS if labeled else [],
            "category": category, "difficulty": difficulty,
            "vector": {"hits": hits_at(rank), "elapsed_seconds": number},
            "reranked": {"hits": hits_at(after), "elapsed_seconds": number}}


class MetricsTests(unittest.TestCase):
    def test_any_page_and_any_label_match_but_source_must_match(self):
        hits = hits_at(3)
        self.assertEqual(metrics.first_expected_rank(hits, LABELS), 3)
        self.assertIsNone(metrics.first_expected_rank(hits, [{"source": "other.pdf", "page": 2}]))
        self.assertEqual(metrics.first_expected_rank(hits, [
            {"source": "absent.pdf", "page": 99}, *LABELS]), 3)
        self.assertIsNone(metrics.first_expected_rank(hits, []))

    def test_hit_rates_mrr_and_latency_use_their_correct_denominators(self):
        rows = [row(1, 1), row(2, 3), row(3, 5), row(4, 6), row(5, None, labeled=False)]
        stats = metrics.summarize(rows, "vector")
        self.assertEqual((stats["total_questions"], stats["scored_questions"],
                          stats["unscored_questions"]), (5, 4, 1))
        for k, expected in ((1, .25), (3, .5), (5, .75)):
            self.assertEqual(stats[f"recall_at_{k}"], expected)
        self.assertAlmostEqual(stats["mrr_at_5"], (1 + 1 / 3 + 1 / 5) / 4)
        self.assertEqual(stats["mean_latency_seconds"], 3)

    def test_no_labels_leave_scores_unknown(self):
        unlabeled = row(1, 1, labeled=False)
        unlabeled["expected_chunks"] = None
        for rows in ([], [row(1, 1, labeled=False)], [unlabeled]):
            with self.subTest(rows=len(rows)):
                stats = metrics.summarize(rows, "vector")
                for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"):
                    self.assertIsNone(stats[key])
                self.assertEqual(stats["scored_questions"], 0)
        self.assertIsNone(metrics.summarize([], "vector")["mean_latency_seconds"])

    def test_category_and_difficulty_denominators_are_local(self):
        rows = [row(1, 1), row(2, None, difficulty="hard"),
                row(3, None, labeled=False), row(4, 3, category="b", difficulty="hard")]
        categories = metrics.breakdown(rows, "vector", "category")
        self.assertEqual(categories["a"]["scored_questions"], 2)
        self.assertEqual(categories["a"]["unscored_questions"], 1)
        self.assertEqual(categories["a"]["recall_at_3"], .5)
        self.assertEqual(categories["b"]["recall_at_3"], 1)
        difficulties = metrics.breakdown(rows, "vector", "difficulty")
        self.assertEqual(difficulties["easy"]["recall_at_1"], 1)
        self.assertEqual(difficulties["hard"]["recall_at_3"], .5)
        self.assertAlmostEqual(difficulties["hard"]["mrr_at_5"], 1 / 6)

    def test_refusal_cases_with_context_pages_are_not_scored(self):
        for field, value in (("sub_type", "on_topic_unanswerable"),
                             ("category", "out-of-scope"), ("category", "ambiguous")):
            with self.subTest(value=value):
                refusal = {**row(2, 1, None), field: value}
                stats = metrics.summarize([row(1, 3, 1), refusal], "vector")
                self.assertEqual(stats["scored_questions"], 1)
                self.assertEqual(stats["unscored_questions"], 1)
                self.assertEqual(stats["recall_at_1"], 0)
                self.assertAlmostEqual(stats["mrr_at_5"], 1 / 3)
                self.assertEqual(metrics.compare_results([refusal])["recall_at_5"]["regressions"], [])
                self.assertEqual(metrics.retrieval_labels(refusal), [])

    def test_rank_and_each_cutoff_regression_ignore_unlabeled_rows(self):
        rows = [row(1, 1, 2), row(2, 3, 4), row(3, 5, 6), row(4, 4, 3),
                row(5, 6, 1), row(6, 1, 6, labeled=False)]
        changes = metrics.compare_results(rows)
        self.assertEqual(changes["recall_at_1"], {"regressions": [1], "improvements": [5]})
        self.assertEqual(changes["recall_at_3"], {"regressions": [2], "improvements": [4, 5]})
        self.assertEqual(changes["recall_at_5"], {"regressions": [3], "improvements": [5]})
        self.assertEqual(changes["rank"], {"regressions": [1, 2, 3], "improvements": [4, 5]})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name)
        self.questions = [{"question": "Synthetic question one?", "expected_chunks": LABELS},
                          {"question": "Synthetic question two?", "expected_chunks": None}]
        self.write_questions()
        self.index, self.model, self.lexical = object(), object(), object()
        self.chunks = hits_at(1)
        self.fused = [{**hit, "bm25_score": 4.5, "rrf_score": .03} for hit in self.chunks]
        self.vectors = [object(), object()]
        dependencies = {
            "load_index": {"return_value": (self.index, self.chunks, {"synthetic": True})},
            "load_model": {"return_value": self.model},
            "BM25Index": {"return_value": self.lexical},
            "hybrid_candidates": {"return_value": self.fused},
            "embed_texts": {"side_effect": [([self.vectors[0]], 7), ([self.vectors[1]], 11)]},
            "search_index": {"side_effect": lambda index, chunks, vector, k: chunks[:k]},
            "rerank": {"side_effect": lambda text, hits, k, model:
                       [{**hit, "rerank_score": 2.0} for hit in hits[:k]]},
        }
        for name, options in dependencies.items():
            patcher = patch.object(runner, name, **options)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        guard = patch.object(embeddings, "create_client", side_effect=AssertionError("unexpected API"))
        self.client = guard.start()
        self.addCleanup(guard.stop)

    def write_questions(self):
        (self.data_dir / "questions.json").write_text(json.dumps(self.questions))

    def evaluate(self):
        return runner.evaluate(self.data_dir, cached_only=True, progress=lambda message: None)

    def test_load_once_embed_once_per_question_and_publish_complete_report(self):
        report = self.evaluate()
        self.load_index.assert_called_once_with(self.data_dir)
        self.load_model.assert_called_once_with()
        self.BM25Index.assert_called_once_with(self.chunks)
        self.assertEqual(self.embed_texts.call_args_list, [
            call([question["question"]], self.data_dir, allow_api=False) for question in self.questions])
        self.assertEqual(self.search_index.call_count, 4)
        self.assertEqual(self.rerank.call_count, 4)
        self.assertEqual(self.hybrid_candidates.call_count, 2)
        for i, vector in enumerate(self.vectors):
            baseline, candidates = self.search_index.call_args_list[2 * i:2 * i + 2]
            self.assertIs(baseline.args[2], vector)
            self.assertIs(candidates.args[2], vector)
            self.assertEqual((baseline.args[3], candidates.args[3]), (5, 20))
            self.assertIs(baseline.args[0], self.index)
            question = self.questions[i]["question"]
            self.assertEqual(self.rerank.call_args_list[2 * i:2 * i + 2], [
                call(question, self.chunks, 5, model=self.model),
                call(question, self.fused, 5, model=self.model)])
            self.assertEqual(self.hybrid_candidates.call_args_list[i],
                             call(self.index, vector, question, self.chunks, self.lexical, k=20))
        self.assertEqual(report["new_embedding_tokens"], 18)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["mode"], "retrieval-only")
        self.assertEqual(set(report["summary"]), {"vector", "reranked", "hybrid"})
        self.assertEqual(report["config"]["rrf_k"], runner.RRF_K)
        self.assertGreaterEqual(report["lexical_load_seconds"], 0)
        for variant in ("vector", "reranked", "hybrid"):
            self.assertEqual(report["summary"][variant]["overall"]["recall_at_1"], 1)
            self.assertIn("by_sub_type", report["summary"][variant])
        for result in report["rows"]:
            for key in ("generated_answer", "answer_correct", "rewritten_query"):
                self.assertIsNone(result[key])
        self.assertIsNone(report["rows"][1]["vector"]["reciprocal_rank"])
        self.assertEqual(report["rows"][1]["expected_chunks"], [])
        saved = json.loads((self.data_dir / "eval_results.json").read_text())
        checkpoint, = (self.data_dir / "evaluation_runs").glob("*.json")
        self.assertEqual(saved, report)
        self.assertEqual(json.loads(checkpoint.read_text()), report)

    def test_hybrid_scores_survive_compact_report_without_replacing_cosine(self):
        report = self.evaluate()
        hit = report["rows"][0]["hybrid"]["hits"][0]
        self.assertEqual(hit["score"], self.chunks[0]["score"])
        self.assertEqual(hit["bm25_score"], 4.5)
        self.assertEqual(hit["rrf_score"], .03)
        self.assertEqual(hit["rerank_score"], 2.0)

    def test_comparisons_keep_reranking_and_hybrid_changes_separate(self):
        self.rerank.side_effect = [hits_at(3)[:5], hits_at(2)[:5]]
        report = runner.evaluate(self.data_dir, limit=1, cached_only=True, progress=lambda message: None)
        self.assertEqual(report["comparison"]["rank"], {"improvements": [], "regressions": [1]})
        self.assertEqual(report["hybrid_comparison"]["rank"], {"improvements": [1], "regressions": []})
        self.assertAlmostEqual(report["summary"]["reranked"]["overall"]["mrr_at_5"], 1 / 3)
        self.assertEqual(report["summary"]["hybrid"]["overall"]["mrr_at_5"], .5)

    def test_hybrid_latency_reuses_search_time_but_excludes_other_reranker_call(self):
        elapsed = [0.0]

        def timed(seconds, result):
            elapsed[0] += seconds
            return result

        self.embed_texts.side_effect = lambda *args, **kwargs: timed(2, ([self.vectors[0]], 0))
        self.search_index.side_effect = lambda index, chunks, vector, k: timed(3 if k == 5 else 5, chunks[:k])
        self.hybrid_candidates.side_effect = lambda *args, **kwargs: timed(13, self.fused)
        self.rerank.side_effect = lambda text, hits, k, model: timed(11 if hits is self.fused else 7, hits[:k])
        with patch.object(runner.time, "perf_counter", side_effect=lambda: elapsed[0]):
            report = runner.evaluate(self.data_dir, limit=1, cached_only=True, progress=lambda message: None)
        result = report["rows"][0]
        self.assertEqual(result["vector"]["elapsed_seconds"], 2 + 3)
        self.assertEqual(result["reranked"]["elapsed_seconds"], 2 + 5 + 7)
        self.assertEqual(result["hybrid"]["elapsed_seconds"], 2 + 5 + 13 + 11)

    def test_failure_checkpoints_progress_and_preserves_prior_completed_output(self):
        output = self.data_dir / "eval_results.json"
        previous = b'{"status": "complete", "run_id": "previous"}\n'
        output.write_bytes(previous)
        self.embed_texts.side_effect = [([self.vectors[0]], 7), RuntimeError("private provider detail")]
        with self.assertRaisesRegex(RuntimeError, "private provider detail"):
            self.evaluate()
        self.assertEqual(output.read_bytes(), previous)
        checkpoint, = (self.data_dir / "evaluation_runs").glob("*.json")
        report = json.loads(checkpoint.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertEqual(len(report["rows"]), 1)
        self.assertEqual(report["new_embedding_tokens"], 7)
        self.assertNotIn("private provider detail", checkpoint.read_text())

    def test_refusal_context_references_stay_in_report_but_not_in_scores(self):
        self.questions[1].update(sub_type="on_topic_unanswerable", expected_chunks=LABELS)
        self.write_questions()
        report = self.evaluate()
        refusal = report["rows"][1]
        self.assertEqual(refusal["expected_chunks"], LABELS)
        self.assertFalse(refusal["retrieval_scored"])
        for variant in ("vector", "reranked", "hybrid"):
            self.assertIsNone(refusal[variant]["first_expected_rank"])
            self.assertIsNone(refusal[variant]["reciprocal_rank"])
            self.assertIsNone(refusal[variant]["failure_reason"])
            self.assertEqual(report["summary"][variant]["overall"]["scored_questions"], 1)
            self.assertEqual(report["summary"][variant]["by_sub_type"]["on_topic_unanswerable"]["scored_questions"], 0)
        self.assertEqual(report["hybrid_comparison"]["rank"], {"improvements": [], "regressions": []})

    def test_limited_run_does_not_replace_full_scorecard(self):
        output = self.data_dir / "eval_results.json"
        previous = b'{"status": "complete", "run_id": "previous"}\n'
        output.write_bytes(previous)
        report = runner.evaluate(self.data_dir, limit=1, cached_only=True, progress=lambda message: None)
        self.assertEqual(report["selected_questions"], 1)
        self.assertEqual(output.read_bytes(), previous)
        self.assertEqual(json.loads((self.data_dir / "eval_smoke.json").read_text()), report)

    def test_bad_labels_fail_before_embedding_or_model_loading(self):
        invalid = [{}, "invalid", [{"source": "missing.pdf", "page": 2}],
                   [{"source": "synthetic.pdf", "page": 99}],
                   [{"source": "synthetic.pdf", "page": True}], [{"source": "synthetic.pdf"}]]
        for labels in invalid:
            with self.subTest(labels=labels):
                self.questions[0]["expected_chunks"] = labels
                self.write_questions()
                with self.assertRaises(ValueError):
                    self.evaluate()
        self.embed_texts.assert_not_called()
        self.load_model.assert_not_called()
        self.assertFalse((self.data_dir / "eval_results.json").exists())

    def test_cached_only_missing_embedding_never_creates_api_client(self):
        with self.assertRaisesRegex(ValueError, "missing from cache"):
            embeddings.embed_texts(["uncached synthetic question"], self.data_dir, allow_api=False)
        self.client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
