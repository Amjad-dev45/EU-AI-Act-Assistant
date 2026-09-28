"""Offline tests: real FAISS and SDK parsing, mocked paid provider responses."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import faiss
import httpx
import numpy as np
from openai import OpenAI

from app.answering import Answerer, NO_EVIDENCE
from app.evaluation import labelled_metrics, validate_labels
from app.monitoring import Monitor, Pricing, provider_usage
from app.retrieval import Retriever


def chat_response(text="The amount is 100 euros. [1]", finish="stop", usage=True):
    return NS(id="chat-test", model="gpt-4o-mini-2024-07-18",
              usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                       prompt_tokens_details=NS(cached_tokens=40)) if usage else None,
              choices=[NS(finish_reason=finish, message=NS(content=text, refusal=None))])


def embedding_response():
    return NS(model="text-embedding-3-small", usage=NS(prompt_tokens=7, total_tokens=7),
              data=[NS(embedding=[1., 0., 0.])])


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        chunks = [
            {"id": 0, "source": "a.pdf", "pages": [1, 2], "text": "PRIVATE PASSAGE A: amount 100 euros"},
            {"id": 1, "source": "a.pdf", "pages": [3], "text": "PRIVATE PASSAGE B: exceptions"},
            {"id": 2, "source": "b.pdf", "pages": [1], "text": "PRIVATE PASSAGE C: other policy"},
        ]
        raw = json.dumps(chunks).encode()
        (self.root / "chunks.json").write_bytes(raw)
        (self.root / "index_metadata.json").write_text(json.dumps({
            "model": "text-embedding-3-small", "chunks_sha256": hashlib.sha256(raw).hexdigest(),
            "chunk_count": 3, "dimensions": 3,
        }))
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[1, 0, 0], [.8, .6, 0], [0, 0, 1]], dtype="float32"))
        faiss.write_index(index, str(self.root / "my_index.faiss"))
        self.client = Mock()
        self.client.embeddings.create.return_value = embedding_response()
        self.client.chat.completions.create.return_value = chat_response()
        with patch("app.retrieval.tiktoken.encoding_for_model", return_value=NS(encode=lambda *a, **kw: [1, 2])):
            self.retriever = Retriever(self.root, self.client)
        self.monitor = Monitor(self.root / "metrics.sqlite3")
        self.answerer = Answerer(self.retriever, self.client, self.monitor)

    def test_request_accounting_sources_and_durable_session_aggregation(self):
        first = self.answerer.answer("  Amount?  ", conversation_id="one")
        t = first["telemetry"]
        self.assertEqual(t["usage"]["total_tokens"], 127)
        self.assertEqual(t["usage"]["main_llm"]["total_tokens"], 120)
        self.assertEqual(t["usage"]["embedding"]["total_tokens"], 7)
        self.assertEqual(t["usage"]["additional_llm"]["total_tokens"], 0)
        # 60 uncached + 40 cached prompt tokens, 20 output, 7 embedding.
        self.assertAlmostEqual(t["cost"]["total_usd"], .00002414, places=10)
        self.assertEqual(t["original_query"], "  Amount?  ")
        self.assertEqual(t["search_query"], "Amount?")
        self.assertIsNone(t["rewritten_query"])
        self.assertEqual(t["metrics"]["retrieved_documents"], 2)
        self.assertEqual(t["metrics"]["retrieved_chunks"], 3)
        self.assertEqual(t["metrics"]["cited_documents"], 1)
        self.assertEqual(t["context_chunk_ids"], [0, 1, 2])
        self.assertEqual(t["cited_chunk_ids"], [0])
        self.assertEqual([p["rank"] for p in t["retrieval"]], [1, 2, 3])
        self.assertTrue(t["retrieval"][0]["source_reference"].endswith("a.pdf"))
        for key in ("query_preprocessing", "embedding_generation", "retrieval", "context_building", "llm_generation"):
            self.assertGreaterEqual(t["timings_ms"][key + "_time_ms"], 0)
        self.assertIsNone(t["timings_ms"]["reranking_time_ms"])
        self.assertIsNone(t["timings_ms"]["evaluation_time_ms"])
        self.assertGreaterEqual(t["timings_ms"]["total_latency_ms"], sum(v or 0 for k, v in t["timings_ms"].items() if k != "total_latency_ms"))
        self.answerer.answer("Again?", conversation_id="one")
        self.answerer.answer("Another session?", conversation_id="two")
        restarted = Monitor(self.monitor.path)
        summary = restarted.summary("one")
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["total_tokens"], 254)
        self.assertAlmostEqual(summary["average_cost_per_question_usd"], .00002414)
        saved = json.dumps(restarted.records())
        self.assertNotIn("PRIVATE PASSAGE", saved)
        self.assertNotIn("SYSTEM_PROMPT", saved)

    def test_failed_generation_keeps_embedding_and_unknown_cost(self):
        self.client.chat.completions.create.side_effect = RuntimeError("secret payload")
        with self.assertRaises(RuntimeError):
            self.answerer.answer("Amount?")
        record = self.monitor.records()[0]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["usage"]["known_tokens"]["total_tokens"], 7)
        self.assertIsNone(record["usage"]["total_tokens"])
        self.assertIsNone(record["cost"]["total_usd"])
        self.assertGreater(record["cost"]["known_total_usd"], 0)
        self.assertNotIn("secret payload", json.dumps(record))
        self.assertEqual(self.monitor.summary()["requests_with_unknown_cost"], 1)

    def test_failed_embedding_logs_attempt_without_generation(self):
        self.client.embeddings.create.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            self.answerer.answer("Amount?")
        self.client.chat.completions.create.assert_not_called()
        record = self.monitor.records()[0]
        self.assertIsNone(record["usage"]["total_tokens"])
        self.assertIsNone(record["timings_ms"]["llm_generation_time_ms"])

    def test_invalid_citation_is_billed_and_logged_as_error(self):
        self.client.chat.completions.create.return_value = chat_response("Amount 100. [99]")
        with self.assertRaisesRegex(ValueError, "unknown passage"):
            self.answerer.answer("Amount?")
        t = self.monitor.records()[0]
        self.assertEqual(t["invalid_citation_ids"], [99])
        self.assertEqual(t["metrics"]["citation_validity"], 0)
        self.assertEqual(t["usage"]["total_tokens"], 127)
        self.assertTrue(t["cost"]["complete"])

    def test_truncated_response_is_billed(self):
        self.client.chat.completions.create.return_value = chat_response(finish="length")
        with self.assertRaises(ValueError):
            self.answerer.answer("Amount?")
        self.assertEqual(self.monitor.records()[0]["usage"]["total_tokens"], 127)

    def test_missing_provider_usage_is_not_estimated(self):
        self.client.chat.completions.create.return_value = chat_response(usage=False)
        result = self.answerer.answer("Amount?")
        self.assertIsNone(result["usage"]["chat_input_tokens"])
        self.assertIsNone(result["telemetry"]["cost"]["total_usd"])
        self.assertEqual(result["telemetry"]["usage"]["known_tokens"]["total_tokens"], 7)

    def test_empty_query_logs_no_paid_calls(self):
        with self.assertRaises(ValueError):
            self.answerer.answer("  ")
        self.client.embeddings.create.assert_not_called()
        t = self.monitor.records()[0]
        self.assertEqual(t["cost"]["total_usd"], 0)
        self.assertEqual(t["status"], "error")

    def judge_data(self, claims=None):
        return {"passages": [{"citation_id": 1, "relevance": 1}, {"citation_id": 2, "relevance": .5},
                             {"citation_id": 3, "relevance": 0}], "answer_relevance": .9,
                "claims": [{"claim": "Amount is 100 euros", "supported": True, "citation_ids": [1]}]
                    if claims is None else claims,
                "reference_correctness": None, "reason": "Cited evidence supports the amount."}

    def test_judge_usage_is_separate_and_quality_is_estimated(self):
        self.client.chat.completions.create.side_effect = [chat_response(), chat_response(json.dumps(self.judge_data()))]
        result = self.answerer.answer("Amount?", evaluate=True)
        t = result["telemetry"]
        self.assertEqual(t["usage"]["total_tokens"], 247)
        self.assertEqual(t["usage"]["additional_llm"]["total_tokens"], 120)
        self.assertAlmostEqual(t["cost"]["by_operation_usd"]["evaluation"], .000024)
        self.assertEqual(t["semantic_evaluation"]["method"], "llm_judge_estimate")
        self.assertEqual(t["semantic_evaluation"]["faithfulness"], 1)
        self.assertEqual(t["semantic_evaluation"]["context_relevance"], .5)
        self.assertTrue(t["semantic_evaluation"]["answer_supported"])

    def test_invalid_judge_preserves_answer_and_billed_usage(self):
        self.client.chat.completions.create.side_effect = [chat_response(), chat_response("not json")]
        t = self.answerer.answer("Amount?", evaluate=True)["telemetry"]
        self.assertEqual(t["status"], "answered")
        self.assertEqual(t["semantic_evaluation"]["status"], "error")
        self.assertEqual(t["usage"]["total_tokens"], 247)

    def test_judge_api_failure_preserves_answer_but_cost_is_incomplete(self):
        self.client.chat.completions.create.side_effect = [chat_response(), RuntimeError()]
        t = self.answerer.answer("Amount?", evaluate=True)["telemetry"]
        self.assertEqual(t["status"], "answered")
        self.assertIsNone(t["cost"]["total_usd"])
        self.assertEqual(t["semantic_evaluation"]["status"], "error")

    def test_judge_cannot_invent_cited_evidence(self):
        data = self.judge_data([{"claim": "Amount", "supported": True, "citation_ids": [2]}])
        self.client.chat.completions.create.side_effect = [chat_response(), chat_response(json.dumps(data))]
        t = self.answerer.answer("Amount?", evaluate=True)["telemetry"]
        self.assertEqual(t["semantic_evaluation"]["status"], "error")

    def test_refusal_has_no_groundedness_score(self):
        self.client.chat.completions.create.side_effect = [chat_response(NO_EVIDENCE), chat_response(json.dumps(self.judge_data([])))]
        result = self.answerer.answer("Weather?", evaluate=True, label={"category": "out-of-scope"})
        self.assertTrue(result["labelled_evaluation"]["expected_status_match"])
        self.assertIsNone(result["telemetry"]["semantic_evaluation"]["faithfulness"])
        self.assertEqual(result["telemetry"]["metrics"]["cited_documents"], 0)

    def test_labelled_metrics_distinguish_page_and_chunk_evidence(self):
        result = self.answerer.answer("Amount?")
        label = {"expected_chunks": [{"source": "a.pdf", "page": 2}, {"source": "missing.pdf", "page": 1}],
                 "expected_chunk_ids": [1, 88], "expected_answer": "The amount is 100 euros."}
        m = labelled_metrics(result, label)
        self.assertEqual(m["expected_page_recall"], .5)
        self.assertEqual(m["chunk_recall_at_k"], .5)
        self.assertEqual(m["chunk_precision_at_k"], 1 / 3)
        self.assertEqual(m["chunk_mrr"], .5)
        self.assertAlmostEqual(m["chunk_ndcg_at_k"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertTrue(m["answer_exact_match"])
        self.assertEqual(m["answer_token_f1"], 1)
        self.assertTrue(all(value is None for value in labelled_metrics(result, {}).values()))

    def test_bad_labels_rejected_before_run(self):
        for labels in ([], {}, [{"question": "x", "expected_chunks": [{"source": "a", "page": -1}]}],
                       [{"question": "x", "expected_status": "wrong"}], [{"question": "x", "expected_chunk_ids": [{}]}]):
            with self.assertRaises(ValueError):
                validate_labels(labels)

    def test_unknown_model_prices_are_not_zero(self):
        response = chat_response()
        response.model = "unpriced-model"
        self.client.chat.completions.create.return_value = response
        t = self.answerer.answer("Amount?")["telemetry"]
        self.assertIsNone(t["cost"]["total_usd"])
        self.assertEqual(t["usage"]["total_tokens"], 127)

    def test_sqlite_concurrent_writes_and_session_isolation(self):
        def save(i):
            trace = self.monitor.trace("question", str(i % 2))
            trace.finish("answered")
            return self.monitor.save(trace)
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertTrue(all(pool.map(save, range(32))))
        self.assertEqual(self.monitor.summary()["requests"], 32)
        self.assertEqual(self.monitor.summary("0")["requests"], 16)
        self.assertEqual(self.monitor.summary("1")["requests"], 16)

    def test_persistence_failure_does_not_discard_answer(self):
        self.monitor.path = self.root / "chunks.json" / "bad.sqlite3"
        with self.assertLogs("app.monitoring", level="ERROR"):
            result = self.answerer.answer("Amount?")
        self.assertFalse(result["telemetry_persisted"])
        self.assertEqual(result["status"], "answered")

    def test_actual_sdk_response_usage_parsing_without_network(self):
        body = {"id": "chat-test", "object": "chat.completion", "created": 1,
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "yes"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                          "prompt_tokens_details": {"cached_tokens": 40}}}
        with OpenAI(api_key="offline-test", http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)))) as client:
            response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(provider_usage(response, "main_llm")["cached_input_tokens"], 40)
        self.assertAlmostEqual(Pricing().estimate(response.model, provider_usage(response, "main_llm"))["total_usd"], .000024)

    def test_preview_evaluation_works_without_api_key(self):
        questions = self.root / "questions.json"
        questions.write_text(json.dumps([{"question": "Amount?", "expected_answer": "100"}]))
        process = subprocess.run([sys.executable, "scripts/evaluate.py", "--data-dir", str(self.root)],
                                 capture_output=True, text=True, check=True)
        self.assertIn("Preview only", process.stdout)

    def test_evaluation_runner_checkpoints_success_and_failure(self):
        from scripts.evaluate import main
        questions = [{"question": "Amount?", "category": "factual", "difficulty": "easy",
                      "expected_chunks": [{"source": "a.pdf", "page": 2}],
                      "expected_answer": "The amount is 100 euros."},
                     {"question": "Another amount?", "category": "factual"}]
        (self.root / "questions.json").write_text(json.dumps(questions))
        output = self.root / "report.json"
        self.client.chat.completions.create.side_effect = [chat_response(), RuntimeError()]
        with patch("sys.argv", ["evaluate.py", "--data-dir", str(self.root), "--output", str(output), "--run"]), \
             patch("openai.OpenAI", return_value=self.client), \
             patch("app.retrieval.Retriever", return_value=self.retriever), \
             patch("app.answering.Answerer", return_value=self.answerer), \
             patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        report = json.loads(output.read_text())
        self.assertEqual(report["completed_questions"], 2)
        self.assertEqual(report["monitoring"]["error_rate"], .5)
        self.assertEqual(report["metrics"]["expected_page_recall"], {"mean": 1.0, "samples": 1})
        self.assertEqual(report["rows"][1]["status"], "error")
        self.assertNotIn("PRIVATE PASSAGE", json.dumps(report))

    def test_gradio_integration_and_independent_session_ids(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test", "GRADIO_ANALYTICS_ENABLED": "False"}), \
             patch("app.retrieval.Retriever", return_value=self.retriever), \
             patch("app.answering.Answerer", return_value=self.answerer):
            sys.modules.pop("app.app", None)
            ui = importlib.import_module("app.app")
        from gradio.state_holder import SessionState
        first, second = SessionState(ui.demo), SessionState(ui.demo)
        self.assertNotEqual(first[ui.conversation._id], second[ui.conversation._id])
        self.assertEqual(first[ui.conversation._id], first[ui.conversation._id])
        with patch.object(ui, "build_sources_html", return_value="sources"):
            answer, sources, stats, totals = ui.ask("Amount?", "browser-session", False)
        self.assertIn("100 euros", answer)
        self.assertEqual(sources, "sources")
        self.assertEqual(stats["usage"]["total_tokens"], 127)
        self.assertEqual(totals["requests"], 1)
        self.assertEqual(self.monitor.summary("other-session")["requests"], 0)


if __name__ == "__main__":
    unittest.main()
