"""Editing removes the future from both visible chat and model context."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.ui import chat
from services.ui.editing import edit_with_sources, history_before_edit
from test_ui_chat import answer_response


def conversation(turns=6):
    state = {}
    for i in range(turns):
        response = {**answer_response(), "request_id": f"old-{i}", "answer": f"Answer {i}."}
        with patch.object(chat, "call_json", return_value=(200, response)):
            state = chat.submit_chat(f"Question {i}", state)[2]
    return state


class UiEditingTests(unittest.TestCase):
    def test_first_middle_and_latest_edits_keep_only_the_prefix(self):
        original = conversation()
        snapshot = deepcopy(original)
        for index in (0, 2, 10):
            with self.subTest(index=index):
                response = {**answer_response(), "answer": "Replacement answer."}
                with patch.object(chat, "call_json", return_value=(200, response)) as call:
                    outputs = list(edit_with_sources(original, False, index,
                                   original["messages"][index]["content"], "Edited question"))
                self.assertEqual(len(outputs), 2)
                pending, result = outputs
                self.assertEqual(len(pending), 15)
                self.assertEqual(pending[1][-1], {"role": "user", "content": "Edited question"})
                self.assertEqual(pending[1][:-1], original["display"][:index])
                self.assertEqual(pending[4]["choices"], [])
                self.assertEqual(pending[6], {})  # Old response/tool JSON gone.
                self.assertEqual(pending[10], {})  # Old Teams approval gone.
                self.assertFalse(pending[11]["interactive"])
                self.assertEqual(pending[13:], ({}, ""))  # Old PDF preview gone.
                payload = call.call_args.args[2]
                self.assertEqual(payload["history"], original["messages"][max(0, index-5):index])
                self.assertEqual(payload["message"], "Edited question")
                self.assertNotIn("**Sources**", str(payload))
                self.assertEqual(result[1][:-2], original["display"][:index])
                self.assertEqual(result[2]["messages"], original["messages"][:index] + [
                    {"role": "user", "content": "Edited question"},
                    {"role": "assistant", "content": "Replacement answer."}])
                self.assertEqual(result[2]["history"], result[2]["messages"][-5:])
        self.assertEqual(original, snapshot)

    def test_later_requests_and_another_edit_cannot_resurrect_deleted_messages(self):
        state = conversation()
        with patch.object(chat, "call_json", return_value=(200, answer_response())) as call:
            revised = list(edit_with_sources(state, True, 2, "Question 1", "New branch"))[-1][2]
            self.assertTrue(call.call_args.args[2]["injection_test"])
            continued = chat.submit_chat("Follow-up", revised)[2]
            self.assertNotIn("Question 1", str(call.call_args.args[2]))
            self.assertNotIn("Answer 5", str(call.call_args.args[2]))
            twice = list(edit_with_sources(continued, False, 2, "New branch", "Another branch"))[-1][2]
        self.assertEqual(twice["messages"][:2], state["messages"][:2])
        self.assertNotIn("Follow-up", str(twice))
        self.assertNotIn("New branch", str(twice))

    def test_archive_tracks_display_limit_and_can_edit_outside_current_model_history(self):
        state = conversation(24)
        self.assertEqual(len(state["messages"]), chat.MAX_DISPLAY_MESSAGES)
        self.assertEqual(len(state["display"]), chat.MAX_DISPLAY_MESSAGES)
        self.assertEqual(len(state["history"]), 5)
        self.assertEqual(state["messages"][0]["content"], "Question 4")
        branch = history_before_edit(state, 2, "Question 5", "Replacement")
        self.assertEqual(branch["history"], state["messages"][:2])
        self.assertNotIn("Question 23", str(branch))

    def test_failed_regeneration_keeps_truncated_history_and_edited_input(self):
        state = conversation()
        for response in (None, {**answer_response(), "status": "error", "citations": []}):
            with self.subTest(response=response):
                kwargs = {"return_value": (200, response)} if response else {"side_effect": TimeoutError("SECRET")}
                with patch.object(chat, "call_json", **kwargs):
                    result = list(edit_with_sources(state, False, 2, "Question 1", "Try again"))[-1]
                self.assertEqual(result[0], "Try again")
                self.assertEqual(result[1], state["display"][:2])
                self.assertEqual(result[2]["messages"], state["messages"][:2])
                self.assertEqual(result[10], {})
                self.assertEqual(result[13:], ({}, ""))
                self.assertNotIn("SECRET", str(result))

    def test_invalid_or_stale_edits_do_not_change_state_or_call_agent(self):
        state = conversation(2)
        cases = [(-1, "Question 0", "New"), (True, "Question 0", "New"),
                 (99, "Question 0", "New"), (1, "Answer 0.", "New"),
                 (0, "Stale text", "New"), (0, "Question 0", " "),
                 (0, "Question 0", "x" * 2001)]
        with patch.object(chat, "call_json") as call:
            for index, old, new in cases:
                with self.subTest(index=index, old=old, size=len(new)):
                    outputs = list(edit_with_sources(state, False, index, old, new))
                    self.assertEqual(len(outputs), 1)
                    self.assertEqual(outputs[0][1], state["display"])
                    self.assertEqual(outputs[0][2], {"__type__": "update"})
                    self.assertEqual(outputs[0][13], {"__type__": "update"})
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
