"""Offline CLI checks: selection, records, usage display, and safe API failures."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from openai import OpenAIError

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("answer_checker", ROOT / "scripts/answer_checker.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def result(question="What is literacy?", status="answered", usage=None):
    return {"question": question, "status": status, "answer": "Staff should be trained. [1]",
            "citations": [{"id": 1, "chunk_id": "chunk-1", "source": "synthetic.pdf", "pages": [50, 51]}],
            "refusal_reason": None, "error": None, "model": "gpt-4o-mini", "generation_seconds": 0.2,
            "elapsed_seconds": 0.5, "top_rerank_score": 1.23456, "usage": usage,
            "retrieval": {"hits": [], "embedding_tokens": 7, "elapsed_seconds": 0.3}}


class AnswerCheckerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        root_patch = patch.object(checker, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        answer_patch = patch.object(checker, "answer_question", side_effect=lambda question, _: result(question))
        self.answer = answer_patch.start()
        self.addCleanup(answer_patch.stop)
        self.stdout, self.stderr = StringIO(), StringIO()
        self.entries = [
            {"question": "First question?", "priority": True, "expected_answer": "NEVER SEND THIS", "expected_chunks": [1]},
            {"question": "Second question?", "priority": False},
            {"question": "Third question?", "priority": True},
            {"question": "Not boolean true?", "priority": "true"},
        ]
        self.write_questions(self.entries)

    def write_questions(self, entries):
        (self.root / "data/questions.json").write_text(json.dumps(entries), encoding="utf-8")

    def run_cli(self, args):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return checker.main(args)

    def records(self):
        return [json.loads(path.read_text()) for path in sorted((self.root / "data/answer_runs").glob("*.json"))]

    def test_invalid_arguments_make_no_generation_calls(self):
        for args in ([], ["   "], ["question", "--question-number", "1"],
                     ["--priority", "--question-number", "1"], ["--question-number", "0"],
                     ["--question-number", "99"], ["--question-number", "text"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                self.run_cli(args)
            self.assertEqual(raised.exception.code, 2)
        self.answer.assert_not_called()

    def test_malformed_selected_question_prevents_all_requests(self):
        self.entries[2]["question"] = None
        self.write_questions(self.entries)
        with self.assertRaises(SystemExit):
            self.run_cli(["--priority"])
        self.answer.assert_not_called()

    def test_missing_or_invalid_question_file_is_argparse_error(self):
        path = self.root / "data/questions.json"
        for value in ("{", "{}", "[null]"):
            with self.subTest(value=value):
                path.write_text(value)
                with self.assertRaises(SystemExit):
                    self.run_cli(["--question-number", "1"])
        path.unlink()
        with self.assertRaises(SystemExit):
            self.run_cli(["--question-number", "1"])
        self.answer.assert_not_called()

    def test_direct_question_does_not_need_question_file(self):
        (self.root / "data/questions.json").unlink()
        self.assertEqual(self.run_cli(["What is literacy?"]), 0)
        self.answer.assert_called_once_with("What is literacy?", self.root / "data")
        self.assertIsNone(self.records()[0]["question_number"])

    def test_only_selected_question_text_reaches_generation(self):
        self.assertEqual(self.run_cli(["--question-number", "1"]), 0)
        self.answer.assert_called_once_with("First question?", self.root / "data")
        record, = self.records()
        self.assertEqual(record, {**result("First question?"), "question_number": 1})
        self.assertNotIn("NEVER SEND THIS", json.dumps(record))

    def test_priority_uses_only_true_entries_and_preserves_each_attempt(self):
        self.assertEqual(self.run_cli(["--priority"]), 0)
        self.assertEqual(self.answer.call_args_list, [call("First question?", self.root / "data"),
                                                    call("Third question?", self.root / "data")])
        self.assertEqual([row["question_number"] for row in self.records()], [1, 3])

    def test_no_priority_entries_is_an_input_error(self):
        self.write_questions([{"question": "Not selected?", "priority": False}])
        with self.assertRaises(SystemExit):
            self.run_cli(["--priority"])
        self.answer.assert_not_called()

    def test_citations_tokens_and_relevance_are_displayed_without_accuracy_claim(self):
        self.answer.side_effect = None
        self.answer.return_value = result(usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        self.assertEqual(self.run_cli(["What is literacy?"]), 0)
        output = self.stdout.getvalue()
        for text in ("[1] synthetic.pdf | pages 50, 51", "1.2346 (relevance, not probability)",
                     "New embedding tokens: 7", "chat tokens: prompt: 100, completion: 20, total: 120",
                     "answer accuracy has not been measured"):
            self.assertIn(text, output)

    def test_unknown_usage_is_not_reported_as_zero(self):
        self.assertEqual(self.run_cli(["Question?"]), 0)
        self.assertIn("chat tokens: unknown", self.stdout.getvalue())

    def test_refusal_is_valid_and_priority_continues(self):
        self.answer.side_effect = [result("First question?", "refused"), result("Third question?")]
        self.assertEqual(self.run_cli(["--priority"]), 0)
        self.assertEqual(self.answer.call_count, 2)
        self.assertEqual(self.records()[0]["status"], "refused")

    def test_error_result_is_saved_and_stops_priority(self):
        self.answer.side_effect = [result("First question?", "error")]
        self.assertEqual(self.run_cli(["--priority"]), 1)
        self.assertEqual(self.answer.call_count, 1)
        record, = self.records()
        self.assertEqual(record["status"], "error")

    def test_api_exception_is_sanitized_and_stops_priority(self):
        self.answer.side_effect = OpenAIError("secret-key sensitive provider message")
        self.assertEqual(self.run_cli(["--priority"]), 1)
        self.assertEqual(self.answer.call_count, 1)
        self.assertEqual(self.records(), [])
        self.assertIn("OpenAIError", self.stderr.getvalue())
        self.assertNotIn("secret-key", self.stderr.getvalue() + self.stdout.getvalue())

    def test_configuration_error_is_sanitized_without_traceback(self):
        self.answer.side_effect = ValueError("sensitive configuration")
        self.assertEqual(self.run_cli(["Question?"]), 1)
        self.assertIn("ValueError", self.stderr.getvalue())
        self.assertNotIn("sensitive configuration", self.stderr.getvalue())

    def test_no_expansion_flag_requests_the_five_passage_baseline(self):
        self.answer.side_effect = None
        self.answer.return_value = result()
        self.assertEqual(self.run_cli(["What are the Annex III areas?", "--no-expansion"]), 0)
        self.answer.assert_called_once_with("What are the Annex III areas?", self.root / "data", use_expansion=False)


if __name__ == "__main__":
    unittest.main()
