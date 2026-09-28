"""Regression checks for course labels, package boundaries, and source rendering."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pymupdf

from app.evaluation.metrics import labelled_metrics, validate_labels
from app.settings import REPO_DIR, configured_path
from app.ui.sources import render_page, build_sources_html


class RefactorTests(unittest.TestCase):
    # The course intentionally has null answers/evidence for refusal questions.
    def test_nullable_course_labels_validate_and_score_without_false_zeroes(self):
        for category, status in [('out-of-scope', 'insufficient_evidence'),
                                 ('ambiguous', 'needs_clarification')]:
            label = {'question': 'Example?', 'category': category,
                     'expected_answer': None, 'expected_chunks': None,
                     'expected_chunk_ids': None}
            validate_labels([label])
            result = {'answer': 'No evidence.', 'status': status, 'retrieved_passages': []}
            scores = labelled_metrics(result, label)
            self.assertTrue(scores['expected_status_match'])
            self.assertIsNone(scores['answer_exact_match'])
            self.assertIsNone(scores['expected_page_recall'])
            self.assertIsNone(scores['chunk_recall_at_k'])

    def test_nullable_labels_still_reject_wrong_types(self):
        for key, value in [('expected_answer', 42), ('expected_chunks', {}),
                           ('expected_chunk_ids', '1'), ('category', None)]:
            with self.assertRaises(ValueError):
                validate_labels([{'question': 'Example?', key: value}])

    # A new process confirms import alone needs neither credentials nor data.
    def test_entrypoint_import_has_no_runtime_initialization(self):
        code = '''
from unittest.mock import patch
with patch('openai.OpenAI', side_effect=AssertionError('Client initialized')):
    import app.app
assert callable(app.app.create_app)
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=REPO_DIR,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_environment_paths_resolve_against_repository(self):
        with patch.dict(os.environ, {'RAG_METRICS_DB': 'logs/custom.sqlite3'}):
            self.assertEqual(configured_path('RAG_METRICS_DB', 'unused'),
                             REPO_DIR / 'logs/custom.sqlite3')

    # Exercise the real PDF/image pipeline with synthetic public text, no API.
    def test_pdf_highlighting_and_source_panel(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sources').mkdir()
            pdf = root / 'sources/example.pdf'
            text = 'This is a simple source passage with enough matching words.'
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), text)
                document.save(pdf)
            image = render_page(1, text, pdf)
            self.assertGreater(image.width, 0)
            html = build_sources_html([{
                'citation_id': 3, 'source': 'example.pdf', 'pages': [1], 'text': text,
            }], root)
            self.assertIn('[3] example.pdf', html)
            self.assertIn('<details', html)
            self.assertIn('data:image/jpeg;base64,', html)
            self.assertFalse(list(root.rglob('*.png')))

    def test_course_named_eval_command_accepts_null_labels_without_spending(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'questions.json').write_text(json.dumps([{
                'question': 'Example?', 'expected_answer': None, 'expected_chunks': None,
            }]))
            result = subprocess.run([
                sys.executable, 'scripts/eval_runner.py', '--data-dir', folder,
            ], cwd=REPO_DIR, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Preview only', result.stdout)
            self.assertFalse((path / 'eval_results.json').exists())


if __name__ == '__main__':
    unittest.main()
