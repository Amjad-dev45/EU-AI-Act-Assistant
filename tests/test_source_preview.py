"""PDF rendering must highlight real text and never change or expose other files."""
import base64
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from contracts.errors import ApiError
from contracts.schemas import validate_source_preview
from services.rag.source_preview import render_preview
from services.ui import source_viewer


class SourcePreviewTests(unittest.TestCase):
    def setUp(self):
        import pymupdf
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'sources').mkdir()
        self.pdf = self.root / 'sources' / 'law.pdf'
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((40, 80), 'Providers shall ensure sufficient AI literacy of their staff.')
            page.insert_text((40, 250), 'This is an unrelated paragraph which must not be highlighted.')
            doc.save(self.pdf)
        self.corpus = [{'source': 'law.pdf', 'page': 1, 'text': 'Providers shall ensure sufficient AI literacy of their staff.'}]

    def test_cropped_highlights_and_no_match_fallback_leave_original_unchanged(self):
        before = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        result = render_preview(self.root, self.corpus, 'law.pdf', 1, self.corpus[0]['text'])
        self.assertEqual(result['highlight_count'], 1)
        self.assertTrue(result['cropped'])
        self.assertTrue(base64.b64decode(result['image_base64']).startswith(b'\x89PNG'))
        missing = render_preview(self.root, self.corpus, 'law.pdf', 1, 'Entirely missing words appear nowhere here')
        self.assertEqual(missing['highlight_count'], 0)
        self.assertFalse(missing['cropped'])
        self.assertEqual(before, hashlib.sha256(self.pdf.read_bytes()).hexdigest())

    def test_unlisted_pages_path_traversal_symlinks_and_invalid_requests_are_rejected(self):
        for document, page in [('other.pdf', 1), ('law.pdf', 2), ('../secret.pdf', 1)]:
            with self.assertRaises(ApiError):
                render_preview(self.root, self.corpus, document, page, '')
        outside = self.root / 'outside.pdf'
        outside.write_bytes(self.pdf.read_bytes())
        (self.root/'sources'/'link.pdf').symlink_to(outside)
        with self.assertRaises(ApiError):
            render_preview(self.root, [{'source': 'link.pdf', 'page': 1}], 'link.pdf', 1, '')
        for fields in ({'page': True}, {'excerpt': 'a'*12001}, {'path': '/etc/passwd'}):
            with self.assertRaises(ApiError):
                validate_source_preview({'request_id': 'test', 'document_id': 'law.pdf', 'page': 1, **fields})

    def test_citations_resolve_each_page_and_stale_click_does_not_fetch(self):
        hit = {'id': 1, 'chunk_id': 'law:1', 'source': 'law.pdf', 'document_id': 'law.pdf',
               'pages': [1, 2], 'text': '<script>unsafe()</script> Evidence.'}
        response = {'request_id': 'current', 'citations': [hit], 'trace': [
            {'type': 'tool', 'output': {'cached': True, 'result': {'evidence': [hit]}}}]}
        choices, items = source_viewer.source_options(response)
        self.assertEqual(len(choices), 2)
        self.assertEqual(items[choices[1][1]]['page'], 2)
        with patch.object(source_viewer, 'call_json') as call:
            source_viewer.open_source(choices[0][1], items, {'request_id': 'new'})
        call.assert_not_called()
        preview = render_preview(self.root, self.corpus, 'law.pdf', 1, self.corpus[0]['text'])
        with patch.object(source_viewer, 'call_json', return_value=(200, preview)):
            html = source_viewer.open_source(choices[0][1], items, response)
        self.assertIn('data:image/png;base64,', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertIn('data-preview-open', html)

    def test_new_answer_and_reset_clear_preview_and_source_selection(self):
        fake = ['', [], {}, '', '', '', {}, '', '', '', {}, {}, '']
        with patch.object(source_viewer, 'submit_with_sharing', return_value=tuple(fake)):
            result = source_viewer.submit_with_sources('Hi', {})
        self.assertEqual(len(result), 15)
        self.assertEqual(result[4]['choices'], [])
        self.assertEqual(result[-2:], ({}, ''))
        reset = source_viewer.clear_with_sources()
        self.assertEqual(len(reset), 16)
        self.assertFalse(reset[13])
        self.assertEqual(reset[-2:], ({}, ''))


if __name__ == '__main__':
    unittest.main()
