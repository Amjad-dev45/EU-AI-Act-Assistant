"""Offline evidence-window checks for complete article and annex list answers."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag import section_expansion


def chunks_for(parts, *, source="act.pdf", document_id="act", overlap=0):
    """Keep synthetic text, offsets, and page provenance internally consistent."""
    document = "".join(parts)
    chunks = []
    end = 0
    for index, part in enumerate(parts):
        start = max(0, end - overlap) if index else 0
        end += len(part)
        chunks.append({
            "chunk_id": f"{source}:{start}:{end}",
            "document_id": document_id,
            "source": source,
            "pages": [index + 1] if not overlap or not index else [index, index + 1],
            "start_char": start, "end_char": end,
            "text": document[start:end], "oversized": False,
        })
    return document, chunks


def article_fixture(*, overlap=0):
    return chunks_for([
        "Preceding definition.\nArticle 5\nProhibited practices\n(a) First item.\n",
        "(b) Second item.\n(c) Third item.\n",
        "(d) Fourth item.\nArticle 6\nFollowing obligations.\n",
    ], overlap=overlap)


def scored(chunk):
    return {**deepcopy(chunk), "score": 0.42, "rerank_score": 3.5,
            "bm25_score": 4.2, "rrf_score": 0.03}


class SectionExpansionTests(unittest.TestCase):
    def test_only_explicit_single_section_list_requests_activate_expansion(self):
        requests = {
            "List the prohibited practices in Article 5.": "Article 5",
            "Enumerate the areas in Annex III.": "Annex III",
            "Which practices are prohibited by Article 5?": "Article 5",
            "What are the high-risk areas listed in Annex III?": "Annex III",
            "What are the areas of high-risk AI systems listed in Annex III?": "Annex III",
            "What AI practices are prohibited under Article 5?": "Article 5",
            "List Article 5's rules and summarise Article 5.": "Article 5",
        }
        for question, expected in requests.items():
            with self.subTest(question=question):
                self.assertEqual(section_expansion.requested_section(question), expected)
        for question in (
            "What does Article 4 say?", "How does Article 99 apply?",
            "From what date does Article 5 apply?", "List all prohibited practices.",
            "List the rules from Article 5 and Article 6.",
            "Which obligations appear in Article 5 and Annex III?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(section_expansion.requested_section(question))

    def test_unrequested_or_disabled_expansion_never_reads_the_corpus(self):
        _, chunks = article_fixture()
        hits = [scored(chunks[0])]
        original = deepcopy(hits)
        with patch.object(section_expansion, "load_chunks",
                          side_effect=AssertionError("unexpected corpus access")):
            for question, enabled in (("What does Article 4 say?", True),
                                      ("List Article 5 practices.", False)):
                result = section_expansion.expand_context(
                    question, hits, Path("/unused"), enabled=enabled)
                self.assertFalse(result["applied"])
                self.assertIsNone(result["complete"])
                self.assertEqual(result["hits"], original)
                self.assertIsNot(result["hits"], hits)
        self.assertEqual(hits, original)

    def test_article_expansion_is_complete_cropped_deduplicated_and_citable(self):
        document, chunks = article_fixture(overlap=8)
        hits = [scored(chunks[0])]
        original = deepcopy((hits, chunks))
        result = section_expansion.expand_section(
            "List the practices in Article 5.", hits, list(reversed(chunks)))

        self.assertTrue(result["applied"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["heading"], "Article 5")
        selected = result["hits"]
        self.assertEqual([hit["chunk_id"] for hit in selected],
                         [chunk["chunk_id"] for chunk in chunks])
        selected_positions = []
        for hit, original_chunk in zip(selected, chunks):
            start, end = hit["excerpt_start_char"], hit["excerpt_end_char"]
            self.assertEqual(hit["text"], document[start:end])
            selected_positions.extend(range(start, end))
            self.assertEqual(hit["section_heading"], "Article 5")
            for key in ("chunk_id", "pages", "source", "document_id",
                        "start_char", "end_char"):
                self.assertEqual(hit[key], original_chunk[key])
        self.assertEqual(len(selected_positions), len(set(selected_positions)))
        expected_positions = set(range(document.index("Article 5"),
                                       document.index("Article 6")))
        self.assertTrue(all(document[position].isspace()
                            for position in expected_positions - set(selected_positions)))
        self.assertTrue(set(selected_positions).issubset(expected_positions))
        for key in ("score", "rerank_score", "bm25_score", "rrf_score"):
            self.assertEqual(selected[0][key], hits[0][key])
            self.assertNotIn(key, selected[1])
        self.assertEqual(result["characters"], sum(len(hit["text"]) for hit in selected))
        self.assertEqual(result["chunk_count"], len(selected))
        self.assertEqual((hits, chunks), original)

    def test_annex_continues_through_internal_headings_until_the_next_annex(self):
        document, chunks = chunks_for([
            "Earlier content. ANNEX III\nHigh-risk areas\n1. Biometrics.\n",
            "SECTION 1\n2. Infrastructure.\nArticle 2\n3. Education.\n",
            "4. Employment.\nANNEX IV\nTechnical documentation.\n",
        ])
        result = section_expansion.expand_section(
            "List the areas in Annex III.", [scored(chunks[0])], chunks)
        self.assertTrue(result["complete"])
        combined = "\n".join(hit["text"] for hit in result["hits"])
        self.assertIn("4. Employment.", combined)
        self.assertIn("SECTION 1", combined)
        self.assertNotIn("Earlier content", combined)
        self.assertNotIn("Technical documentation", combined)
        self.assertTrue(all(hit["excerpt_end_char"] <= document.index("ANNEX IV")
                            for hit in result["hits"]))

    def test_article_stops_before_each_kind_of_following_legal_heading(self):
        for following in ("Article 6", "CHAPTER III", "SECTION 2", "ANNEX I"):
            with self.subTest(following=following):
                _, chunks = chunks_for([
                    "Article 5\nRules\n(a) First practice.\n",
                    "(b) Last practice.\n" + following + "\nUNRELATED CONTENT.\n",
                ])
                result = section_expansion.expand_section(
                    "List the practices in Article 5.", [scored(chunks[0])], chunks)
                self.assertTrue(result["complete"])
                combined = "\n".join(hit["text"] for hit in result["hits"])
                self.assertIn("Last practice", combined)
                self.assertNotIn("UNRELATED CONTENT", combined)

    def test_inline_references_and_unretrieved_headings_cannot_anchor_a_section(self):
        _, chunks = article_fixture()
        _, reference_chunks = chunks_for([
            "The rules referred to in Article 5 apply to these systems.\n",
        ], source="reference.pdf", document_id="reference")
        for hits in ([scored(chunks[1])], [scored(reference_chunks[0])]):
            with self.subTest(hits=hits):
                result = section_expansion.expand_section(
                    "List Article 5 practices.", hits, chunks + reference_chunks)
                self.assertFalse(result["complete"])
                self.assertFalse(result["applied"])
                self.assertEqual(result["hits"], [])

    def test_forged_or_stale_hit_metadata_cannot_select_authoritative_content(self):
        _, chunks = article_fixture()
        for key, value in (("chunk_id", "missing-id"), ("source", "other.pdf"),
                           ("document_id", "other-document"),
                           ("text", "Article 5\nINVENTED TEXT\n")):
            with self.subTest(key=key):
                hit = scored(chunks[0])
                hit[key] = value
                result = section_expansion.expand_section(
                    "List Article 5 practices.", [hit], chunks)
                self.assertFalse(result["complete"])
                self.assertFalse(result["applied"])
                self.assertEqual(result["hits"], [])

    def test_documents_are_isolated_and_multiple_retrieved_anchors_are_ambiguous(self):
        _, chunks = article_fixture()
        _, other_chunks = chunks_for([
            "Article 5\nForeign rules.\n", "Foreign continuation.\nArticle 6\nNext.\n",
        ], source="other.pdf", document_id="other")
        all_chunks = other_chunks + list(reversed(chunks))
        result = section_expansion.expand_section(
            "List Article 5 practices.", [scored(chunks[0])], all_chunks)
        self.assertTrue(result["complete"])
        self.assertTrue(all(hit["source"] == "act.pdf" for hit in result["hits"]))
        ambiguous = section_expansion.expand_section(
            "List Article 5 practices.", [scored(chunks[0]), scored(other_chunks[0])],
            all_chunks)
        self.assertFalse(ambiguous["complete"])
        self.assertEqual(ambiguous["reason"], "ambiguous_section")
        self.assertEqual(ambiguous["hits"], [])

    def test_unknown_boundary_missing_middle_chunk_and_missing_pages_fail_closed(self):
        _, chunks = article_fixture()
        cases = [(chunks[:2], "missing_boundary"),
                 ([chunks[0], chunks[2]], "missing_context")]
        missing_page = deepcopy(chunks)
        missing_page[1]["pages"] = [3]
        missing_page[2]["pages"] = [4]
        cases.append((missing_page, "missing_context"))
        for available, reason in cases:
            with self.subTest(reason=reason, available=available):
                result = section_expansion.expand_section(
                    "List Article 5 practices.", [scored(available[0])], available)
                self.assertFalse(result["complete"])
                self.assertFalse(result["applied"])
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["hits"], [])

    def test_missing_page_before_a_separate_closing_heading_is_incomplete(self):
        _, chunks = chunks_for([
            "Article 5\nRules\n(a) First practice.\n",
            "(b) Last available practice.\n",
            "Article 6\nNext article.\n",
        ])
        chunks[2]["pages"] = [4]
        result = section_expansion.expand_section(
            "List Article 5 practices.", [scored(chunks[0])], chunks)
        self.assertFalse(result["complete"])
        self.assertEqual(result["reason"], "missing_context")
        self.assertEqual(result["hits"], [])

    def test_exact_character_and_chunk_limits_succeed_but_overruns_return_no_partial_list(self):
        _, chunks = article_fixture()
        question, hits = "List Article 5 practices.", [scored(chunks[0])]
        normal = section_expansion.expand_section(question, hits, chunks)
        self.assertTrue(normal["complete"])
        self.assertEqual(normal["max_characters"], 20_000)
        self.assertEqual(normal["max_chunks"], 24)
        exact = section_expansion.expand_section(
            question, hits, chunks, max_chars=normal["characters"],
            max_chunks=normal["chunk_count"])
        self.assertTrue(exact["complete"])
        for limits in ({"max_chars": normal["characters"] - 1},
                       {"max_chunks": normal["chunk_count"] - 1}):
            with self.subTest(limits=limits):
                limited = section_expansion.expand_section(question, hits, chunks, **limits)
                self.assertFalse(limited["complete"])
                self.assertFalse(limited["applied"])
                self.assertEqual(limited["reason"], "context_limit")
                self.assertEqual(limited["hits"], [])

    def test_context_adapter_loads_the_saved_chunks_once_for_an_explicit_list(self):
        _, chunks = article_fixture()
        data_dir = Path("/unused/data")
        with patch.object(section_expansion, "load_chunks",
                          return_value=(chunks, "unused-checksum")) as load:
            result = section_expansion.expand_context(
                "List Article 5 practices.", [scored(chunks[0])], data_dir)
        load.assert_called_once_with(data_dir)
        self.assertTrue(result["complete"])
        self.assertEqual(result["chunk_count"], 3)


if __name__ == "__main__":
    unittest.main()
