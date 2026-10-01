"""UI approval is tied to the displayed answer, including queued-click races."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.ui import chat, sharing

RESPONSE = {"request_id": "chat-share-test", "status": "answered", "mock": False,
            "answer": "A supported answer. [1]", "citations": [], "trace": [], "usage": {},
            "sharing": {"available": True, "token": "a" * 32, "message": "Click to share."}}


class UiSharingTests(unittest.TestCase):
    def answer(self, response=None):
        with patch.object(chat, "call_json", return_value=(200, response or RESPONSE)):
            return sharing.submit_with_sharing("Question", {})

    def test_generation_saves_token_only_in_session_until_user_click(self):
        with patch.object(sharing, "call_json") as post:
            output = self.answer()
        post.assert_not_called()
        state = output[10]
        self.assertTrue(output[11]["interactive"])
        self.assertNotIn("sharing", output[6])
        self.assertNotIn(RESPONSE["sharing"]["token"], str(output[:10]))
        with patch.object(sharing, "call_json", return_value=(200, {"status": "accepted"})) as post:
            result = sharing.send_to_teams(state, output[6])
            self.assertTrue(post.call_args.args[2]["approved"])
            self.assertEqual(post.call_args.args[2]["token"], state["token"])
            self.assertNotIn("answer", post.call_args.args[2])
            sharing.send_to_teams(result[0], output[6])
        self.assertEqual(post.call_count, 1)
        self.assertFalse(result[1]["interactive"])
        self.assertIn("Submitted to Teams", result[2])

    def test_refusal_failure_and_new_chat_clear_old_share(self):
        with patch.object(sharing, "call_json") as post:
            for response in ({**RESPONSE, "status": "refused"}, {**RESPONSE, "status": "error"}):
                output = self.answer(response)
                self.assertEqual(output[10], {})
                self.assertFalse(output[11]["interactive"])
            with patch.object(chat, "call_json", side_effect=RuntimeError("SECRET")):
                output = sharing.submit_with_sharing("Question", {})
            self.assertEqual(output[10], {})
            self.assertFalse(output[11]["interactive"])
            reset = sharing.clear_with_sharing()
        self.assertEqual(reset[10], {})
        self.assertFalse(reset[11]["interactive"])
        self.assertFalse(reset[-1])
        post.assert_not_called()

    def test_queued_click_does_not_approve_new_unseen_answer(self):
        output = self.answer()
        with patch.object(sharing, "call_json") as post:
            result = sharing.send_to_teams(output[10], {"request_id": "older-answer"})
            sharing.send_to_teams({}, output[6])
        post.assert_not_called()
        self.assertTrue(result[1]["interactive"])
        self.assertIn("answer changed", result[2])

    def test_network_failure_and_unknown_result_disable_retries_and_hide_secrets(self):
        output = self.answer()
        for response in ((200, {"status": "unknown", "message": "SECRET"}), (500, {"error": "SECRET"})):
            with patch.object(sharing, "call_json", return_value=response):
                result = sharing.send_to_teams(output[10], output[6])
            self.assertTrue(result[0]["attempted"])
            self.assertFalse(result[1]["interactive"])
            self.assertNotIn("SECRET", str(result))
        with patch.object(sharing, "call_json", side_effect=TimeoutError("SECRET")):
            result = sharing.send_to_teams(output[10], output[6])
        self.assertTrue(result[0]["attempted"])
        self.assertNotIn("SECRET", str(result))


if __name__ == "__main__":
    unittest.main()
