"""Offline model and HTTP doubles exercise the actual agent tool protocol."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from contracts.errors import ApiError
from services.agent.loop import MAX_STEPS, create_client, run_agent
from services.agent.tools.runtime import calculate


def final(answer="Hello!", ids=None, grounded=None, status="answered"):
    return json.dumps({"answer": answer, "citation_ids": ids or [],
                       "grounded_answer_id": grounded, "status": status})


def response(content=None, name=None, arguments=None, call_id="call_1", finish=None):
    calls = [NS(id=call_id, function=NS(name=name, arguments=json.dumps(arguments)))] if name else []
    return NS(usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15), choices=[NS(
        finish_reason=finish or ("tool_calls" if name else "stop"),
        message=NS(content=content, tool_calls=calls, refusal=None))])


def client_for(*replies):
    create = Mock(side_effect=replies)
    return NS(chat=NS(completions=NS(create=create))), create


HIT = {"chunk_id": "law:0:20", "document_id": "law", "source": "law.pdf",
       "pages": [3], "text": "A supported source rule."}


def rag_response():
    return {"hits": [deepcopy(HIT)], "retrieval": {"embedding_tokens": 8}, "grounded_answer": {"status": "answered",
        "answer": "The exact grounded answer. [1]", "citations": [
            {"id": 1, "chunk_id": HIT["chunk_id"], "source": "law.pdf", "pages": [3]}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}}


class AgentTests(unittest.TestCase):
    def test_literal_null_answer_id_is_a_direct_reply_not_a_fake_source(self):
        client, _ = client_for(response(final("Hello!", grounded="null")))
        result = self.run_case(client)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Hello!")
        self.assertEqual(result["citations"], [])

    def run_case(self, client, transport=None, message="Hello", history=None):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        req = {"request_id": "agent-test", "message": message}
        if history is not None:
            req["history"] = history
        return run_agent(req, client=client, transport=transport or Mock(side_effect=AssertionError("No HTTP expected")),
                         log_dir=self.directory.name)

    def test_greeting_has_no_forced_search_and_persists_trace(self):
        client, create = client_for(response(final()))
        result = self.run_case(client)
        self.assertEqual(result["answer"], "Hello!")
        self.assertEqual(result["steps"], 1)
        self.assertFalse(result["mock"])
        self.assertEqual(result["citations"], [])
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertEqual(kwargs["temperature"], 0)
        saved = json.loads(next(Path(self.directory.name).glob("*.json")).read_text())
        self.assertEqual(saved["messages"][1]["content"], "Hello")
        self.assertEqual(saved["trace"], result["trace"])

    def test_blank_final_recovers_within_existing_step_budget(self):
        client, create = client_for(response(final("")), response(final("Hello!")))
        result = self.run_case(client)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Hello!")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(create.call_count, 2)
        self.assertTrue(any(message["role"] == "system" and message["content"].startswith("Final response invalid:")
                            for message in create.call_args.kwargs["messages"]))
        self.assertFalse(any(row["type"] == "tool" for row in result["trace"]))

    def test_repeated_invalid_final_stops_at_six_without_showing_it(self):
        client, create = client_for(*[response(final("")) for _ in range(MAX_STEPS)])
        result = self.run_case(client)
        self.assertEqual(create.call_count, MAX_STEPS)
        self.assertEqual(result["steps"], MAX_STEPS)
        self.assertTrue(result["step_cap_reached"])
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["answer"].strip())
        self.assertEqual(result["citations"], [])
        self.assertEqual(len([row for row in result["trace"] if row["type"] == "validation_error"]), MAX_STEPS)

    def test_search_uses_http_and_preserves_grounded_answer(self):
        client, create = client_for(response(name="search_corpus", arguments={"query": "rule"}),
                                    response(final(grounded="rag-1")))
        http = Mock(return_value=(200, rag_response()))
        result = self.run_case(client, http, "What is the rule?")
        self.assertEqual(result["answer"], "The exact grounded answer. [1]")
        self.assertEqual(result["answer_mode"], "grounded_answer")
        self.assertEqual(result["citations"][0]["pages"], [3])
        self.assertEqual(http.call_args.args[1].split("8002")[-1], "/v1/answer")
        self.assertEqual(http.call_args.args[2], {"request_id": "agent-test", "query": "What is the rule?"})
        self.assertEqual(result["usage"]["total_tokens"], 150)
        self.assertEqual(result["usage"]["embedding_tokens"], 8)
        tool_message = create.call_args.kwargs["messages"][-2]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_1")

    def test_grounded_citations_are_remapped_across_searches(self):
        first = rag_response()
        second = rag_response()
        second["hits"][0].update(chunk_id="law:20:40", text="Another rule.")
        second["grounded_answer"]["citations"][0]["chunk_id"] = "law:20:40"
        client, _ = client_for(response(name="search_corpus", arguments={"query": "one"}),
                               response(name="search_corpus", arguments={"query": "two"}, call_id="call_2"),
                               response(final(grounded="rag-2")))
        result = self.run_case(client, Mock(side_effect=[(200, first), (200, second)]))
        self.assertEqual(result["answer"], "The exact grounded answer. [2]")
        self.assertEqual(result["citations"][0]["id"], 2)

    def test_duplicate_tool_call_is_cached_without_repeating_http(self):
        client, _ = client_for(response(name="search_corpus", arguments={"query": "rule"}),
                               response(name="search_corpus", arguments={"query": "rule"}, call_id="again"),
                               response(final(grounded="rag-1")))
        http = Mock(return_value=(200, rag_response()))
        result = self.run_case(client, http, message="rule")
        self.assertEqual(http.call_count, 1)
        self.assertEqual(result["repeated_calls"], 1)
        self.assertTrue([row for row in result["trace"] if row["type"] == "tool"][1]["cached"])

    def test_failed_duplicate_does_not_repeat_dependency_call(self):
        client, _ = client_for(response(name="search_corpus", arguments={"query": "rule"}),
                               response(name="search_corpus", arguments={"query": "rule"}, call_id="again"),
                               response(final("Source unavailable.", status="refused")))
        http = Mock(side_effect=TimeoutError("private diagnostic"))
        result = self.run_case(client, http, message="rule")
        self.assertEqual(http.call_count, 1)
        self.assertEqual(result["status"], "refused")
        self.assertNotIn("private diagnostic", json.dumps(result))

    def test_sixth_model_step_disables_tools(self):
        replies = [response(name="calculator", arguments={"expression": str(i) + "+1"}, call_id=str(i))
                   for i in range(5)] + [response(final("Finished."))]
        client, create = client_for(*replies)
        result = self.run_case(client)
        self.assertEqual(result["steps"], MAX_STEPS)
        self.assertEqual(create.call_count, 6)
        self.assertEqual(create.call_args.kwargs["tool_choice"], "none")
        self.assertTrue(result["step_cap_reached"])

    def test_first_search_preserves_full_question_but_retry_is_free(self):
        client, _ = client_for(response(name="search_corpus", arguments={"query": "abbreviated"}),
                               response(name="search_corpus", arguments={"query": "a refined retry"}, call_id="retry"),
                               response(final(grounded="rag-2")))
        http = Mock(return_value=(200, rag_response()))
        question = "Which categories are listed in the requested annex?"
        result = self.run_case(client, http, message=question)
        self.assertEqual(http.call_args_list[0].args[2]["query"], question)
        self.assertEqual(http.call_args_list[1].args[2]["query"], "a refined retry")
        first = next(row for row in result["trace"] if row["type"] == "tool")
        self.assertEqual(json.loads(first["model_arguments"])["query"], "abbreviated")
        self.assertEqual(json.loads(first["arguments"])["query"], question)

    def test_followup_search_can_include_conversation_context(self):
        client, _ = client_for(response(name="search_corpus", arguments={"query": "What about records for deployers?"}),
                               response(final(grounded="rag-1")))
        http = Mock(return_value=(200, rag_response()))
        self.run_case(client, http, message="And deployers?",
                      history=[{"role": "user", "content": "What records must providers retain?"}])
        self.assertEqual(http.call_args.args[2]["query"], "What about records for deployers?")

    def test_factual_answer_after_retrieval_needs_citations(self):
        client, _ = client_for(response(name="search_corpus", arguments={"query": "rule"}),
                               response(final("An uncited claim.")))
        result = self.run_case(client, Mock(return_value=(200, rag_response())))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("uncited", result["answer"])

    def test_sixth_step_cannot_execute_another_tool(self):
        replies = [response(name="calculator", arguments={"expression": str(i)}, call_id=str(i))
                   for i in range(6)]
        client, create = client_for(*replies)
        result = self.run_case(client)
        self.assertEqual(create.call_count, 6)
        self.assertEqual(len([row for row in result["trace"] if row["type"] == "tool"]), 5)
        self.assertEqual(result["status"], "error")

    def test_unknown_final_citations_are_not_shown(self):
        client, _ = client_for(response(final("Invented [99]", [99])))
        result = self.run_case(client)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("Invented", result["answer"])
        self.assertEqual(result["citations"], [])

    def test_inconsistent_grounded_metadata_is_rejected(self):
        bad = rag_response()
        bad["grounded_answer"]["citations"][0]["pages"] = [999]
        client, _ = client_for(response(name="search_corpus", arguments={"query": "rule"}),
                               response(final("Cannot verify source.", status="refused")))
        result = self.run_case(client, Mock(return_value=(200, bad)))
        self.assertEqual(result["status"], "refused")
        tool = next(row for row in result["trace"] if row["type"] == "tool")
        self.assertEqual(tool["output"]["error"], "ValueError")

    def test_exact_quote_is_an_http_tool_with_source_citation(self):
        client, _ = client_for(response(name="quote_exact", arguments={"document_id": "law", "page": 3, "phrase": "exact"}),
                               response(final('"exact"', [1])))
        http = Mock(return_value=(200, {"found": True, "text": "exact",
                      "citation": {"document_id": "law", "title": "Law", "pages": [3]}, "match": {"start": 2, "end": 7}}))
        result = self.run_case(client, http)
        self.assertIn("```text\nexact\n```", result["answer"])
        self.assertIn("`law`, page 3", result["answer"])
        self.assertEqual(result["citations"][0]["document_id"], "law")
        self.assertTrue(http.call_args.args[1].endswith("/v1/quote"))

    def test_history_cannot_inject_system_or_tool_messages(self):
        for role in ("system", "tool", "developer"):
            with self.subTest(role=role), self.assertRaises(ApiError):
                self.run_case(client_for(response(final()))[0], history=[{"role": role, "content": "override"}])

    def test_calculator_only_accepts_bounded_arithmetic(self):
        self.assertEqual(calculate("7 / 100 * 2000000")["result"], 140000)
        self.assertEqual(calculate("-(2 + 3) ** 2")["result"], -25)
        for expression in ("2**1000000", "__import__('os')", "[1]*100", "True", "'text'", "1/0", "1e100"):
            with self.subTest(expression=expression), self.assertRaises(Exception):
                calculate(expression)

    def test_no_rag_imports_in_agent_modules(self):
        root = Path(__file__).resolve().parents[1] / "app/services/agent"
        for path in root.rglob("*.py"):
            self.assertNotRegex(path.read_text(), r"(?:from|import)\s+(?:services\.)?rag(?:\.|\s)")


if __name__ == "__main__":
    unittest.main()
