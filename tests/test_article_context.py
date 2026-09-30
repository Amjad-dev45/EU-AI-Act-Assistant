"""Offline checks for optional context around a retrieved Article passage.

Synthetic passages exercise authority, source boundaries, and citation metadata
without corpus fixtures, model calls, or question-specific rules.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag import article_context


def chunks_for(parts, *, source="fictional-act.pdf", document_id="fictional-act", overlap=0):
    text = "".join(parts)
    chunks, end = [], 0
    for index, part in enumerate(parts):
        start = max(0, end - overlap) if index else 0
        end += len(part)
        chunks.append({
            "chunk_id": f"{document_id}:{start}:{end}",
            "source": source,
            "document_id": document_id,
            "pages": [index + 1] if not overlap or not index else [index, index + 1],
            "start_char": start,
            "end_char": end,
            "text": text[start:end],
            "oversized": False,
        })
    return text, chunks


def article_fixture(*, overlap=0):
    return chunks_for([
        "Preceding material.\nArticle 17\nRecords\n",
        "1. Operators keep ordinary records of the process.\n",
        "2. A limited allowance applies to a defined class of operators.\n",
        "3. The allowance requires an approved scheme and ongoing supervision.\n"
        "Article 18\nFollowing duties\n",
        "Operators must also follow the independent reporting duty.\n",
    ], overlap=overlap)


def scored(chunk, score=0.7):
    return {**deepcopy(chunk), "score": score, "rerank_score": score + 2,
            "rrf_score": 0.03, "bm25_score": 4.0}


class ArticleContextTests(unittest.TestCase):
    def test_separator_only_tail_is_not_sent_as_an_evidence_passage(self):
        from test_section_expansion import chunks_for, scored
        _, chunks = chunks_for(['Article 47\nRule text.\n', '\nArticle 48\nNext rule.\n'])
        result = article_context.assemble_article_context([scored(chunks[0])], chunks)
        self.assertIsNotNone(result)
        self.assertTrue(all(hit['text'].strip() for hit in result['hits']))

    def test_heading_free_top_hit_gets_its_complete_governing_article(self):
        document, chunks = article_fixture(overlap=8)
        hits = [scored(chunks[2])]
        original = deepcopy((hits, chunks))
        result = article_context.assemble_article_context(hits, list(reversed(chunks)))

        self.assertIsNotNone(result)
        self.assertTrue(result["applied"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["mode"], "article")
        self.assertEqual(result["heading"], "Article 17")
        self.assertEqual(result["article_chunk_count"], 4)
        self.assertEqual(result["chunk_count"], 4)
        selected = result["hits"]
        self.assertEqual([hit["chunk_id"] for hit in selected],
                         [chunk["chunk_id"] for chunk in chunks[:4]])
        covered = []
        for hit, chunk in zip(selected, chunks):
            start, end = hit["excerpt_start_char"], hit["excerpt_end_char"]
            self.assertEqual(hit["text"], document[start:end])
            self.assertEqual(hit["section_heading"], "Article 17")
            covered.extend(range(start, end))
            for key in ("chunk_id", "source", "document_id", "pages", "start_char", "end_char"):
                self.assertEqual(hit[key], chunk[key])
        expected = set(range(document.index("Article 17"), document.index("Article 18")))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertTrue(set(covered).issubset(expected))
        self.assertTrue(all(document[position].isspace() for position in expected - set(covered)))
        self.assertIn("ongoing supervision", "".join(hit["text"] for hit in selected))
        self.assertEqual(result["characters"], sum(len(hit["text"]) for hit in selected))
        for key in ("score", "rerank_score", "rrf_score", "bm25_score"):
            self.assertEqual(selected[2][key], hits[0][key])
        self.assertEqual((hits, chunks), original)

    def test_other_base_evidence_is_retained_after_the_article_in_retrieval_order(self):
        _, chunks = article_fixture()
        _, other = chunks_for(["Independent technical guidance.\n"],
                              source="guidance.pdf", document_id="guidance")
        hits = [scored(chunks[2]), scored(other[0], 0.6),
                scored(chunks[1], 0.5), scored(chunks[4], 0.4),
                scored(other[0], 0.6)]
        original = deepcopy((hits, chunks, other))
        result = article_context.assemble_article_context(hits, other + chunks)

        self.assertIsNotNone(result)
        self.assertEqual(result["article_chunk_count"], 4)
        self.assertEqual(result["hits"][4:], [hits[1], hits[3]])
        ids = [hit["chunk_id"] for hit in result["hits"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[:4], [chunk["chunk_id"] for chunk in chunks[:4]])
        self.assertEqual(result["chunk_count"], 6)
        self.assertEqual(result["characters"], sum(len(hit["text"]) for hit in result["hits"]))
        self.assertEqual((hits, chunks, other), original)

    def test_cropped_base_chunks_keep_their_original_text_as_supplementary_evidence(self):
        _, chunks = article_fixture()
        hits = [scored(chunks[2]), scored(chunks[0], 0.6),
                scored(chunks[3], 0.5), scored(chunks[1], 0.4),
                scored(chunks[3], 0.5)]
        original = deepcopy(hits)
        result = article_context.assemble_article_context(hits, chunks)

        self.assertIsNotNone(result)
        count = result["article_chunk_count"]
        self.assertEqual(count, 4)
        self.assertEqual(result["hits"][count:], [hits[1], hits[2]])
        self.assertNotIn("Preceding material", result["hits"][0]["text"])
        self.assertNotIn("Following duties", result["hits"][3]["text"])
        self.assertIn("Preceding material", result["hits"][count]["text"])
        self.assertIn("Following duties", result["hits"][count + 1]["text"])
        self.assertEqual(result["hits"][count]["chunk_id"], result["hits"][0]["chunk_id"])
        self.assertEqual(result["hits"][count + 1]["chunk_id"], result["hits"][3]["chunk_id"])
        self.assertEqual(hits, original)

    def test_the_highest_hit_alone_selects_the_governing_article(self):
        _, chunks = article_fixture()
        _, other = chunks_for([
            "Article 63\nOther rules\n", "A different governing rule.\n",
            "Article 64\nLater rules.\n",
        ], source="other-act.pdf", document_id="other-act")
        hits = [scored(chunks[2]), scored(other[1], 0.6)]
        result = article_context.assemble_article_context(hits, other + chunks)
        self.assertIsNotNone(result)
        self.assertEqual(result["heading"], "Article 17")
        self.assertTrue(all(hit["source"] == "fictional-act.pdf"
                            for hit in result["hits"][:result["article_chunk_count"]]))
        self.assertEqual(result["hits"][-1], hits[1])

    def test_a_lower_ranked_article_cannot_rescue_a_top_hit_without_a_governing_article(self):
        _, chunks = article_fixture()
        _, other = chunks_for(["General guidance mentions Article 17 in passing.\n"],
                              source="guidance.pdf", document_id="guidance")
        hits = [scored(other[0]), scored(chunks[2], 0.6)]
        self.assertIsNone(article_context.assemble_article_context(hits, chunks + other))

    def test_full_top_hit_cannot_straddle_the_start_or_end_of_an_article(self):
        _, chunks = article_fixture()
        for index in (0, 3):
            with self.subTest(index=index):
                self.assertIsNone(article_context.assemble_article_context(
                    [scored(chunks[index])], chunks))

    def test_structural_headings_end_the_governing_article(self):
        for boundary in ("Article 18", "CHAPTER III", "SECTION 2", "ANNEX I"):
            with self.subTest(boundary=boundary):
                document, chunks = chunks_for([
                    "Article 17\nRecords\n",
                    "Operators keep records.\n",
                    "Conditions apply to every allowance.\n" + boundary + "\nUNRELATED CONTENT.\n",
                ])
                result = article_context.assemble_article_context([scored(chunks[1])], chunks)
                self.assertIsNotNone(result)
                selected = result["hits"][:result["article_chunk_count"]]
                self.assertTrue(all(hit["excerpt_end_char"] <= document.index(boundary)
                                    for hit in selected))
                self.assertNotIn("UNRELATED CONTENT", "".join(hit["text"] for hit in selected))

    def test_annex_and_section_content_are_not_misidentified_as_articles(self):
        for heading in ("ANNEX III", "SECTION 2", "CHAPTER I"):
            with self.subTest(heading=heading):
                _, chunks = chunks_for([
                    "Article 17\nEarlier duties.\n" + heading + "\n",
                    "Content governed by the intervening structural heading.\n",
                    "Article 18\nFollowing duties.\n",
                ])
                self.assertIsNone(article_context.assemble_article_context([scored(chunks[1])], chunks))

    def test_missing_heading_ending_boundary_middle_text_or_pages_return_none(self):
        _, chunks = article_fixture()
        missing_page = deepcopy(chunks)
        missing_page[1]["pages"] = [3]
        missing_page[2]["pages"] = [4]
        missing_page[3]["pages"] = [5]
        for available in (chunks[1:], chunks[:3], [chunks[0], *chunks[2:]], missing_page):
            with self.subTest(available=[chunk["chunk_id"] for chunk in available]):
                top = next(chunk for chunk in available if chunk["chunk_id"] == chunks[2]["chunk_id"])
                self.assertIsNone(article_context.assemble_article_context([scored(top)], available))

    def test_top_hit_must_match_the_saved_authoritative_chunk(self):
        _, chunks = article_fixture()
        changes = (
            ("chunk_id", "absent"), ("source", "foreign.pdf"),
            ("document_id", "foreign"), ("text", "Invented authoritative rule."),
            ("start_char", chunks[2]["start_char"] + 1),
            ("end_char", chunks[2]["end_char"] - 1), ("pages", [999]),
        )
        for key, value in changes:
            with self.subTest(key=key):
                hit = scored(chunks[2])
                hit[key] = value
                self.assertIsNone(article_context.assemble_article_context([hit], chunks))

    def test_invalid_saved_metadata_cannot_authorize_expansion(self):
        _, chunks = article_fixture()
        changes = (("start_char", -1), ("end_char", True), ("pages", []),
                   ("pages", [0]), ("text", None))
        for key, value in changes:
            with self.subTest(key=key):
                available = deepcopy(chunks)
                available[1][key] = value
                self.assertIsNone(article_context.assemble_article_context([scored(chunks[2])], available))
        self.assertIsNone(article_context.assemble_article_context(
            [scored(chunks[2])], chunks + [deepcopy(chunks[1])]))

    def test_exact_limits_include_retained_base_evidence_and_overruns_fall_back(self):
        _, chunks = article_fixture()
        hits = [scored(chunks[2]), scored(chunks[4], 0.4)]
        normal = article_context.assemble_article_context(hits, chunks)
        self.assertIsNotNone(normal)
        exact = article_context.assemble_article_context(
            hits, chunks, max_chars=normal["characters"], max_chunks=normal["chunk_count"])
        self.assertIsNotNone(exact)
        for limits in ({"max_chars": normal["characters"] - 1},
                       {"max_chunks": normal["chunk_count"] - 1}):
            with self.subTest(limits=limits):
                self.assertIsNone(article_context.assemble_article_context(hits, chunks, **limits))

    def test_repeated_article_labels_are_ambiguous(self):
        _, chunks = chunks_for([
            "Article 17\nFirst occurrence\n", "The highest-ranked rule.\n",
            "Article 18\nIntervening rule.\n",
            "Article 17\nSecond occurrence\n", "A conflicting copy.\n",
            "Article 18\nClosing rule.\n",
        ])
        self.assertIsNone(article_context.assemble_article_context([scored(chunks[1])], chunks))

    def test_empty_or_malformed_input_does_not_load_the_corpus(self):
        with patch.object(article_context, "load_chunks",
                          side_effect=AssertionError("unexpected corpus access")):
            for hits in ([], [{}]):
                with self.subTest(hits=hits):
                    self.assertIsNone(article_context.expand_article_context(hits, Path("/unused")))
        self.assertIsNone(article_context.assemble_article_context([], []))

    def test_adapter_loads_saved_chunks_once_and_records_elapsed_time(self):
        _, chunks = article_fixture()
        data_dir = Path("/unused/data")
        with patch.object(article_context, "load_chunks", return_value=(chunks, "unused-checksum")) as load:
            result = article_context.expand_article_context([scored(chunks[2])], data_dir)
        load.assert_called_once_with(data_dir)
        self.assertIsNotNone(result)
        self.assertTrue(result["complete"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
