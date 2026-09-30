"""Offline integration checks using real NumPy/FAISS and a fake embeddings API."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np

from services.rag import embeddings, vector_index


def vector(x, y):
    result = np.zeros(embeddings.DIMENSIONS, dtype="float32")
    result[:2] = (x, y)
    return result


class FakeClient:
    def __init__(self, vectors, fail_call=None):
        self.vectors = vectors
        self.fail_call = fail_call
        self.calls = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, *, input, **kwargs):
        self.calls.append(list(input))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("simulated API failure")
        data = [SimpleNamespace(index=i, embedding=self.vectors[text].tolist())
                for i, text in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)),
                               usage=SimpleNamespace(total_tokens=11))


class VectorIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name)
        self.vectors = {"alpha": vector(4, 0), "beta": vector(0, 3),
                        "gamma": vector(1, 1), "delta": vector(-2, 0)}
        self.chunks = [
            {"chunk_id": f"chunk-{i}", "text": text, "source": f"{text}.pdf",
             "document_id": f"document-{i}", "pages": [i + 1, i + 2]}
            for i, text in enumerate(("alpha", "beta", "gamma"))
        ]
        self.write_chunks()
        self.client = FakeClient(self.vectors)
        # Any unexpected attempt to read credentials or use the network fails.
        guard = patch.object(embeddings, "create_client",
                             side_effect=AssertionError("unexpected real API client"))
        guard.start()
        self.addCleanup(guard.stop)

    def write_chunks(self):
        (self.data_dir / "chunks.json").write_text(json.dumps(self.chunks))

    def test_saved_faiss_search_preserves_sources_and_cosine_scores(self):
        vector_index.build_index(self.data_dir, client=self.client)
        # The exported matrix is not required to load the persisted FAISS index.
        (self.data_dir / "embeddings.npy").unlink()
        index, chunks, _ = vector_index.load_index(self.data_dir)
        results = vector_index.search_index(index, chunks, vector(0, 7), k=9)
        self.assertEqual(len(results), 3)
        for result, expected in zip(results, [self.chunks[1], self.chunks[2], self.chunks[0]]):
            self.assertEqual({key: result[key] for key in expected}, expected)
        np.testing.assert_allclose([result["score"] for result in results],
                                   [1, 1 / np.sqrt(2), 0], atol=1e-6)

    def test_valid_index_rerun_uses_no_api_and_reports_zero_new_tokens(self):
        first = vector_index.build_index(self.data_dir, client=self.client)
        self.assertEqual(first["embedding_tokens"], 11)
        second = vector_index.build_index(self.data_dir)
        self.assertTrue(second["reused"])
        self.assertEqual(second["embedding_tokens"], 0)
        self.assertEqual(self.client.calls, [["alpha", "beta", "gamma"]])

    def test_stale_chunks_rejected_before_api_and_rebuild_reuses_cache(self):
        vector_index.build_index(self.data_dir, client=self.client)
        self.chunks[0]["text"] = "delta"
        self.write_chunks()
        with self.assertRaisesRegex(ValueError, "--rebuild"):
            vector_index.load_index(self.data_dir)
        with self.assertRaisesRegex(ValueError, "--rebuild"):
            vector_index.build_index(self.data_dir, client=self.client)
        self.assertEqual(len(self.client.calls), 1)
        rebuilt = vector_index.build_index(self.data_dir, client=self.client, rebuild=True)
        self.assertFalse(rebuilt["reused"])
        self.assertEqual(rebuilt["embedding_tokens"], 11)
        self.assertEqual(self.client.calls[-1], ["delta"])
        index, chunks, _ = vector_index.load_index(self.data_dir)
        self.assertEqual(vector_index.search_index(index, chunks, vector(-1, 0), k=1)[0]["text"],
                         "delta")

    def test_reversed_response_is_reordered_and_duplicate_text_requested_once(self):
        vectors, tokens = embeddings.embed_texts(
            ["alpha", "beta", "alpha"], self.data_dir, client=self.client)
        self.assertEqual(self.client.calls, [["alpha", "beta"]])
        self.assertEqual(tokens, 11)
        np.testing.assert_array_equal(vectors, [self.vectors[text]
                                              for text in ("alpha", "beta", "alpha")])

    def test_successful_batches_are_cached_and_resume_after_failure(self):
        failing = FakeClient(self.vectors, fail_call=2)
        with patch.object(embeddings, "BATCH_SIZE", 2):
            with self.assertRaisesRegex(RuntimeError, "simulated API failure"):
                vector_index.build_index(self.data_dir, client=failing)
            self.assertEqual(failing.calls, [["alpha", "beta"], ["gamma"]])
            self.assertEqual(len(list((self.data_dir / "embedding_cache").glob("*.npy"))), 2)
            self.assertFalse((self.data_dir / "my_index.faiss").exists())
            resumed = vector_index.build_index(self.data_dir, client=self.client)
        self.assertEqual(self.client.calls, [["gamma"]])
        self.assertEqual(resumed["embedding_tokens"], 11)
        index, chunks, _ = vector_index.load_index(self.data_dir)
        self.assertEqual(vector_index.search_index(index, chunks, vector(1, 0), k=1)[0]["text"],
                         "alpha")

    def test_missing_binary_or_metadata_has_actionable_error(self):
        vector_index.build_index(self.data_dir, client=self.client)
        for name in ("my_index.faiss", "index_meta.json"):
            with self.subTest(missing=name):
                path = self.data_dir / name
                contents = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, f"Missing.*{name}.*--rebuild"):
                    vector_index.load_index(self.data_dir)
                path.write_bytes(contents)

    def test_malformed_vectors_are_not_cached(self):
        self.vectors["alpha"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite, nonzero"):
            embeddings.embed_texts(["alpha"], self.data_dir, client=self.client)
        self.assertEqual(list((self.data_dir / "embedding_cache").glob("*.npy")), [])


if __name__ == "__main__":
    unittest.main()
