"""Offline production-service checks with temporary data and fake providers."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from contracts.errors import ApiError
from services.rag import answering, handlers, retrieval, search, server, store
from services.rag.quote import find_quote


def hit(number=1, score=2.5):
    return {"chunk_id": f"law:{number}", "document_id": "law.pdf", "source": "law.pdf",
            "pages": [number], "text": f"Source passage {number}.", "rerank_score": score,
            "score": 0.25}


class RagServiceTests(unittest.TestCase):
    def setUp(self):
        self.client_guard = patch.object(handlers, "create_client",
                                        side_effect=AssertionError("unexpected model client"))
        self.client_guard.start()
        self.addCleanup(self.client_guard.stop)
        self.environment = patch.dict(os.environ, {"RAG_MIN_RERANK_SCORE": "-5.0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_health_and_ping_do_not_touch_data_retrieval_or_model(self):
        with patch.object(store, "load_source", side_effect=AssertionError("data read")), \
             patch.object(handlers, "search_chunks", side_effect=AssertionError("retrieval")), \
             patch.object(handlers, "answer_query", side_effect=AssertionError("generation")):
            self.assertEqual(handlers.health(None), (200, {"status": "ok", "service": "rag", "mock": False}))
            status, response = handlers.ping({"request_id": "test", "message": "hello"})
        self.assertEqual(status, 200)
        self.assertEqual(response["echo"], "hello")
        self.assertFalse(response["mock"])

    def test_search_uses_real_pipeline_shape_and_default_three_hits(self):
        payload = {"hits": [hit()], "vector_hits": [hit()], "hybrid": True,
                   "reranked": True, "embedding_tokens": 0, "elapsed_seconds": 0.1}
        original = deepcopy(payload)
        with patch.object(handlers, "search_chunks", return_value=payload) as run:
            status, response = handlers.search({"request_id": "test", "query": "Question?"})
        run.assert_called_once_with(store.DATA_DIR, "Question?", 3)
        self.assertEqual(status, 200)
        self.assertEqual(response["hits"], payload["hits"])
        self.assertTrue(response["retrieval"]["hybrid"])
        self.assertFalse(response["mock"])
        self.assertEqual(payload, original)

    def test_search_gate_is_configurable_and_preserves_score_diagnostics(self):
        with patch.object(handlers, "search_chunks", return_value={"hits": [hit(score=-6.0)]}) as run:
            _, response = handlers.search({"request_id": "test", "query": "Question?", "k": 2})
            self.assertEqual(response["hits"], [])
            self.assertEqual(response["refusal_reason"], "low_relevance")
            self.assertEqual(response["retrieval"]["top_rerank_score"], -6.0)
            with patch.dict(os.environ, {"RAG_MIN_RERANK_SCORE": "-7"}):
                _, response = handlers.search({"request_id": "test", "query": "Question?", "k": 2})
            self.assertEqual(response["hits"][0]["rerank_score"], -6.0)
        self.assertEqual(run.call_count, 2)

    def test_answer_returns_the_complete_prepared_context_and_original_citation_ids(self):
        passages = [hit(index) for index in range(1, 8)]
        result = {"status": "answered", "answer": "A rule. [7]", "citations": [{"id": 7, "pages": [7]}],
                  "usage": {"total_tokens": 0}, "answer_mode": "source_quotes",
                  "answer_context": {"hits": passages, "mode": "article", "applied": True},
                  "retrieval": {"hits": [passages[3]], "hybrid": True, "embedding_tokens": 0}}
        original = deepcopy(result)
        with patch.object(handlers, "answer_query", return_value=result) as run:
            status, response = handlers.answer({"request_id": "test", "query": "Question?"})
        self.assertEqual(status, 200)
        self.assertEqual(response["hits"], passages)
        self.assertEqual(response["grounded_answer"]["citations"], [{"id": 7, "pages": [7]}])
        self.assertEqual(response["context"]["mode"], "article")
        self.assertEqual(run.call_args.kwargs["min_rerank_score"], -5.0)
        self.assertEqual(result, original)

    def test_chat_client_is_bounded_lazy_and_closed(self):
        bounded = SimpleNamespace(chat=object(), close=Mock())
        original = SimpleNamespace(with_options=Mock(return_value=bounded))
        with patch.object(handlers, "create_client", return_value=original) as create:
            client = handlers._AnswerClient()
            create.assert_not_called()
            self.assertIs(client.chat, bounded.chat)
            self.assertIs(client.chat, bounded.chat)
            client.close()
        create.assert_called_once()
        original.with_options.assert_called_once_with(timeout=45, max_retries=0)
        bounded.close.assert_called_once()

    def test_provider_error_messages_are_not_exposed(self):
        with patch.object(handlers, "answer_query", side_effect=RuntimeError("SECRET_PROVIDER_PAYLOAD")):
            with self.assertRaises(ApiError) as error:
                handlers.answer({"request_id": "test", "query": "Question?"})
        self.assertEqual(error.exception.status, 503)
        self.assertNotIn("SECRET_PROVIDER_PAYLOAD", error.exception.message)

    def test_invalid_requests_fail_before_data_or_provider_access(self):
        with patch.object(handlers, "search_chunks", side_effect=AssertionError("retrieval")), \
             patch.object(handlers, "answer_query", side_effect=AssertionError("generation")):
            for route in (handlers.search, handlers.answer):
                for body in (None, {"request_id": "test", "query": ""},
                             {"request_id": "test", "query": "Question?", "k": True}):
                    with self.subTest(route=route.__name__, body=body), self.assertRaises(ApiError) as error:
                        route(body)
                    self.assertEqual(error.exception.status, 400)

    def test_quotes_use_exact_extracted_page_text(self):
        pages = [{"source": "law.pdf", "page": 4, "char_count": 29,
                  "text": "Before.\nA precise rule.\nAfter."}]
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "corpus.json").write_text(json.dumps(pages))
            with patch.object(store, "DATA_DIR", data_dir):
                status, result = handlers.quote({"request_id": "test", "document_id": "law.pdf",
                                                 "page": 4, "phrase": "A precise rule."})
        self.assertEqual(status, 200)
        self.assertEqual(result["text"], "A precise rule.")
        self.assertEqual(result["match"], {"start": 8, "end": 23})
        self.assertEqual(result["citation"]["pages"], [4])
        self.assertEqual(result["citation"]["document_id"], "law.pdf")
        self.assertEqual(find_quote(pages, "law.pdf", 4, None)["text"], pages[0]["text"])

    def test_unknown_documents_pages_and_nonverbatim_phrases_return_not_found(self):
        pages = [{"source": "law.pdf", "page": 1, "text": "Exact text."}]
        for document, page, phrase in (("absent.pdf", 1, None), ("law.pdf", 2, None),
                                       ("law.pdf", 1, "exact text.")):
            with self.subTest(document=document, page=page, phrase=phrase), self.assertRaises(ApiError) as error:
                find_quote(pages, document, page, phrase)
            self.assertEqual(error.exception.status, 404)
            self.assertEqual(error.exception.extra, {"found": False})

    def test_missing_or_invalid_corpus_never_falls_back_to_demo_fixtures(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(store, "DATA_DIR", Path(directory)):
            with self.assertRaises(ApiError) as error:
                store.load_source()
            self.assertEqual(error.exception.status, 503)
            page = {"source": "law.pdf", "page": 1, "text": "A rule."}
            for contents in ({"documents": []}, [], [page, page], [{**page, "page": True}]):
                (Path(directory) / "corpus.json").write_text(json.dumps(contents))
                with self.subTest(contents=contents), self.assertRaises(ApiError):
                    store.load_source()

    def test_low_score_skips_expansion_and_model_but_retains_retrieval_diagnostics(self):
        retrieved = {"hits": [hit(score=-8)], "embedding_tokens": 0}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=AssertionError("unexpected model call")))))
        with patch.object(answering, "retrieve", return_value=retrieved) as retrieve, \
             patch.object(answering, "expand_article_context", side_effect=AssertionError("article read")):
            result = answering.answer_question("Question?", Path("unused"), client=client,
                                                min_rerank_score=-5.0)
        retrieve.assert_called_once()
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["refusal_reason"], "low_relevance")
        self.assertEqual(result["answer_context"]["hits"], [])
        self.assertEqual(result["retrieval"], retrieved)
        self.assertEqual(result["usage"]["total_tokens"], 0)
        client.chat.completions.create.assert_not_called()

    def test_default_answer_pipeline_keeps_its_existing_no_threshold_behavior(self):
        retrieved = {"hits": [hit(score=-8)], "embedding_tokens": 0}
        context = {"hits": retrieved["hits"], "complete": None, "applied": False}
        with patch.object(answering, "retrieve", return_value=retrieved), \
             patch.object(answering, "expand_context", return_value=context), \
             patch.object(answering, "expand_article_context", return_value=None), \
             patch.object(answering, "generate_answer", return_value={"status": "answered"}) as generate:
            result = answering.answer_question("Question?", Path("unused"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(generate.call_args.args[1], retrieved["hits"])

    def test_search_and_answer_share_the_pipeline_lock(self):
        first_entered, release_first, second_started, second_entered = Event(), Event(), Event(), Event()

        def first(*args, **kwargs):
            first_entered.set()
            self.assertTrue(release_first.wait(2))
            return {"hits": []}

        def second(*args, **kwargs):
            second_entered.set()
            return {"status": "refused"}

        def call_second():
            second_started.set()
            return search.answer_query(Path("unused"), "Question?")

        with patch.object(retrieval, "retrieve", side_effect=first), \
             patch.object(answering, "answer_question", side_effect=second), \
             ThreadPoolExecutor(max_workers=2) as pool:
            first_call = pool.submit(search.search_chunks, Path("unused"), "Question?")
            self.assertTrue(first_entered.wait(1))
            second_call = pool.submit(call_second)
            self.assertTrue(second_started.wait(1))
            try:
                self.assertFalse(second_entered.wait(0.05))
            finally:
                release_first.set()
            first_call.result(timeout=2)
            second_call.result(timeout=2)
        self.assertTrue(second_entered.is_set())

    def test_server_routes_include_real_answer_and_preserved_ping(self):
        self.assertIs(server.ROUTES[("POST", "/v1/answer")], handlers.answer)
        self.assertIs(server.ROUTES[("POST", "/v1/ping")], handlers.ping)


if __name__ == "__main__":
    unittest.main()
