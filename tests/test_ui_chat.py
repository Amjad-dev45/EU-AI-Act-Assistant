"""Offline UI callbacks: follow-up state, visible evidence, and safe failures."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from contracts.errors import ApiError
from services.ui import chat


def answer_response():
    return {"request_id": "chat-test", "status": "answered", "answer": "A supported answer. [1]",
            "citations": [{"id": 1, "source": "rules.pdf", "pages": [2, 3]}],
            "trace": [{"type": "model", "step": 1, "message": "", "finish_reason": "tool_calls"},
                      {"type": "tool", "step": 1, "name": "search_corpus", "arguments": '{"query": "rule"}',
                       "output": {"hits": [{"text": "Source passage"}]}}],
            "usage": {"total_tokens": 30}, "steps": 1, "mock": False}


class UiChatTests(unittest.TestCase):
    def test_answer_displays_citations_trace_results_and_forwards_followups(self):
        session = {"history": [{"role": "user", "content": "First question"},
                               {"role": "assistant", "content": "First answer"}], "display": []}
        original = deepcopy(session)
        response = answer_response()
        response["internal_config"] = "MUST_NOT_DISPLAY"
        with patch.object(chat, "call_json", return_value=(200, response)) as call:
            output = chat.submit_chat("What about the exception?", session)
        self.assertEqual(call.call_args.args[1], f"{chat.AGENT_URL}/v1/chat")
        payload = call.call_args.args[2]
        self.assertEqual(payload["history"], session["history"])
        self.assertNotIn("demo_tool", payload)
        self.assertEqual(session, original)
        self.assertEqual(output[0], "")
        self.assertIn("rules.pdf", output[1][-1]["content"])
        self.assertIn("2, 3", output[4])
        self.assertIn("search_corpus", output[5])
        self.assertIn("rule", output[5])
        self.assertEqual(output[6]["trace"][1]["output"], response["trace"][1]["output"])
        self.assertNotIn("internal_config", output[6])
        self.assertEqual(output[2]["history"][-1]["content"], response["answer"])

    def test_history_keeps_last_five_individual_messages_in_order(self):
        messages = [{"role": role, "content": f"{index}:{role}"}
                    for index in range(5) for role in ("user", "assistant")]
        result = chat.bounded_history(messages)
        self.assertEqual(result, messages[-5:])
        self.assertEqual(chat.bounded_history(result), result)
        self.assertEqual(result[0]["role"], "assistant")
        with patch.object(chat, "call_json", return_value=(200, answer_response())) as call:
            output = chat.submit_chat("Next question", {"history": result, "display": []})
        self.assertEqual(call.call_args.args[2]["history"], messages[-5:])
        self.assertEqual(output[2]["history"], (messages + [
            {"role": "user", "content": "Next question"},
            {"role": "assistant", "content": answer_response()["answer"]}])[-5:])

    def test_history_character_limit_retains_the_newest_messages(self):
        messages = [{"role": role, "content": f"{index}" + "x" * 5000}
                    for index in range(5) for role in ("user", "assistant")]
        self.assertEqual(chat.bounded_history(messages), messages[-4:])

    def test_failed_request_preserves_question_history_and_hides_error_details(self):
        session = {"history": [{"role": "user", "content": "Old question"},
                               {"role": "assistant", "content": "Old answer"}], "display": []}
        secret = "https://internal.example/secret-key"
        for failure in (ApiError(504, "dependency_timeout", secret), RuntimeError(secret)):
            with self.subTest(failure=type(failure).__name__), patch.object(chat, "call_json", side_effect=failure):
                output = chat.submit_chat("Keep my question", session)
            self.assertEqual(output[0], "Keep my question")
            self.assertEqual(output[2], session)
            self.assertNotIn(secret, str(output))
            self.assertEqual(output[6], {})

    def test_unicode_history_respects_serialized_transport_budget(self):
        history = [{"role": role, "content": "🙂" * 5000}
                   for _ in range(2) for role in ("user", "assistant")]
        with patch.object(chat, "call_json", return_value=(200, answer_response())) as call:
            chat.submit_chat("Follow-up", {"history": history, "display": []})
        payload = call.call_args.args[2]
        self.assertLessEqual(len(json.dumps(payload).encode("utf-8")), 128 * 1024)
        self.assertEqual(payload["history"], history[-2:])

    def test_http_failure_and_mock_response_are_not_presented_as_real_answers(self):
        for status, response in ((500, {"error": {"message": "provider-secret"}}),
                                 (200, {**answer_response(), "mock": True}), (200, {})):
            with self.subTest(status=status, response=response), patch.object(chat, "call_json", return_value=(status, response)):
                output = chat.submit_chat("Question", {})
            self.assertEqual(output[0], "Question")
            self.assertEqual(output[1], [])
            self.assertNotIn("provider-secret", str(output))

    def test_agent_error_keeps_question_and_conversation_with_diagnostic_trace(self):
        response = {**answer_response(), "status": "error", "answer": "Could not finish."}
        session = {"history": [{"role": "user", "content": "Previous question"},
                               {"role": "assistant", "content": "Previous answer"}],
                   "display": [{"role": "assistant", "content": "Previous answer"}]}
        with patch.object(chat, "call_json", return_value=(200, response)):
            output = chat.submit_chat("Try this question", session)
        self.assertEqual(output[0], "Try this question")
        self.assertEqual(output[1], session["display"])
        self.assertEqual(output[2], session)
        self.assertNotIn("Answer ready", output[3])
        self.assertIn("search_corpus", output[5])
        self.assertEqual(output[6]["trace"], response["trace"])

    def test_agent_refusal_is_a_valid_conversation_reply(self):
        response = {**answer_response(), "status": "refused", "answer": "I cannot answer from the corpus.",
                    "citations": [], "trace": []}
        with patch.object(chat, "call_json", return_value=(200, response)):
            output = chat.submit_chat("Unanswerable question", {})
        self.assertEqual(output[0], "")
        self.assertEqual(output[2]["history"][-1]["content"], response["answer"])

    def test_empty_or_oversized_input_never_calls_agent(self):
        with patch.object(chat, "call_json") as call:
            for question in ("", "   ", "x" * 2001):
                output = chat.submit_chat(question, {})
                self.assertEqual(output[2]["history"], [])
        call.assert_not_called()

    def test_new_chat_resets_all_outputs(self):
        output = chat.new_chat()
        self.assertEqual(output[:8], ("", [], {"history": [], "display": []}, "New chat started.", "", "", {}, ""))
        self.assertEqual(len(output), 10)
        self.assertNotIn('aria-valuenow', output[8])
        self.assertNotIn('class="rd-run"', output[9])


if __name__ == "__main__":
    unittest.main()
