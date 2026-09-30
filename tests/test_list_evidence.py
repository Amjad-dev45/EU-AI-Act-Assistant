"""Offline checks for complete, source-cited lettered Article evidence.

All passages below are synthetic, so the checks do not require a saved corpus,
network access, or model credentials.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag.list_evidence import group_article_items


ARTICLE = (
    "Article 47\nProhibited activities\n"
    "1. The following activities are prohibited:\n"
    "(a) Manipulating a person's choices;\n"
    "(b) Exploiting a person's vulnerability;\n"
    "(c) Assessing people using either of these grounds:\n"
    "(i) an unrelated social context;\n"
    "(ii) treatment disproportionate to the observed conduct;\n"
    "(d) Predicting an offence from personal traits alone;\n"
    "(e) Expanding a database through untargeted collection;\n"
    "(f) Inferring emotions at work; (g) Categorising sensitive attributes;\n"
    "(h) Identifying people remotely, subject to these conditions:\n"
    "(i) A first limited purpose;\n"
    "(ii) A second limited purpose;\n"
    "(iii) A third limited purpose.\n"
    "2. A competent authority must assess the conditions for point (h).\n"
    "3. An authorisation is subject to additional safeguards.\n"
)


def make_hits(text, cuts=(), *, base=700, heading="Article 47"):
    """Represent adjacent excerpts using explicit source character positions."""
    boundaries = [0, *cuts, len(text)]
    hits = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        hits.append({
            "chunk_id": f"synthetic:{base + left}:{base + right}",
            "document_id": "synthetic-act",
            "source": "synthetic-act.pdf",
            "pages": [index + 1],
            "start_char": base + left,
            "end_char": base + right,
            "excerpt_start_char": base + left,
            "excerpt_end_char": base + right,
            "section_heading": heading,
            "text": text[left:right],
        })
    return hits


def main_hits():
    return make_hits(ARTICLE, [
        ARTICLE.index("treatment disproportionate"),
        ARTICLE.index("(g)"),
        ARTICLE.index("(ii) A second"),
    ])


class ArticleListEvidenceTests(unittest.TestCase):
    def test_all_top_level_items_keep_nested_conditions_and_later_notes(self):
        hits = main_hits()
        original = deepcopy(hits)
        result = group_article_items(hits)

        self.assertIsNotNone(result)
        self.assertEqual(result["heading"], "Article 47")
        self.assertIn("1. The following activities are prohibited:", result["intro"])
        self.assertEqual([item["label"] for item in result["items"]], list("abcdefgh"))
        by_label = {item["label"]: item for item in result["items"]}
        boundaries = [ARTICLE.index(f"({label})") for label in "abcdefgh"]
        boundaries.append(ARTICLE.index("2. A competent"))
        for (label, item), left, right in zip(by_label.items(), boundaries, boundaries[1:]):
            self.assertTrue(item["text"].startswith(f"({label})"))
            self.assertEqual(item["text"], ARTICLE[left:right].rstrip())
        self.assertIn("(i) an unrelated social context", by_label["c"]["text"])
        self.assertIn("(ii) treatment disproportionate", by_label["c"]["text"])
        self.assertNotIn("(d)", by_label["c"]["text"])
        for ordinal in ("i", "ii", "iii"):
            self.assertIn(f"({ordinal})", by_label["h"]["text"])
        self.assertNotIn("2. A competent", by_label["h"]["text"])
        self.assertTrue(result["notes"].startswith("2."))
        self.assertIn("3. An authorisation", result["notes"])
        self.assertEqual(hits, original)

    def test_items_cite_every_actual_excerpt_they_span(self):
        result = group_article_items(main_hits())
        self.assertIsNotNone(result)
        citations = {item["label"]: item["citation_ids"] for item in result["items"]}
        self.assertEqual(citations, {
            "a": [1], "b": [1], "c": [1, 2], "d": [2],
            "e": [2], "f": [2], "g": [3], "h": [3, 4],
        })

    def test_out_of_order_excerpts_are_not_verified_expansion_output(self):
        hits = main_hits()
        shuffled = [hits[2], hits[0], hits[3], hits[1]]
        self.assertIsNone(group_article_items(shuffled))

    def test_citations_use_trimmed_excerpts_instead_of_original_chunk_bounds(self):
        hits = main_hits()
        for hit in hits:
            hit["start_char"] = 0
            hit["end_char"] = 700 + len(ARTICLE) + 100
        result = group_article_items(hits)
        self.assertIsNotNone(result)
        by_label = {item["label"]: item for item in result["items"]}
        self.assertEqual(by_label["a"]["citation_ids"], [1])
        self.assertEqual(by_label["g"]["citation_ids"], [3])
        self.assertEqual(by_label["h"]["citation_ids"], [3, 4])

    def test_a_passage_containing_only_the_following_separator_is_not_cited(self):
        boundary = ARTICLE.index("\n2. A competent")
        hits = make_hits(ARTICLE, [boundary])
        result = group_article_items(hits)
        self.assertIsNotNone(result)
        self.assertEqual(result["items"][-1]["label"], "h")
        self.assertEqual(result["items"][-1]["citation_ids"], [1])
        self.assertIn("2. A competent", result["notes"])

    def test_short_separator_gaps_do_not_lose_a_continuing_item(self):
        for separator in (" ", "\n\n"):
            with self.subTest(separator=repr(separator)):
                text = (
                    "Article 9\nExamples\n1. The following are listed:\n"
                    "(a) A continuing" + separator + "first item;\n"
                    "(b) A second item.\n2. Later safeguards.\n"
                )
                left = text.index(separator + "first item")
                right = left + len(separator)
                hits = make_hits(text, [left, right], heading="Article 9")
                del hits[1]
                result = group_article_items(hits)
                self.assertIsNotNone(result)
                self.assertEqual([item["label"] for item in result["items"]], ["a", "b"])
                self.assertIn("first item", result["items"][0]["text"])
                self.assertEqual(result["items"][0]["citation_ids"], [1, 2])

    def test_missing_middle_text_returns_no_partial_list(self):
        hits = main_hits()
        del hits[1]
        self.assertIsNone(group_article_items(hits))

    def test_nonexpanded_or_malformed_provenance_returns_none(self):
        for key in ("section_heading", "excerpt_start_char", "excerpt_end_char",
                    "source", "document_id", "start_char", "end_char"):
            with self.subTest(missing=key):
                hits = main_hits()
                del hits[1][key]
                self.assertIsNone(group_article_items(hits))
        changes = (
            ("excerpt_start_char", "700"),
            ("excerpt_end_char", True),
            ("excerpt_end_char", 999999),
            ("start_char", 999999),
            ("end_char", 0),
            ("source", ""),
            ("document_id", ""),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                hits = main_hits()
                hits[0][key] = value
                self.assertIsNone(group_article_items(hits))

    def test_different_documents_or_article_headings_are_never_combined(self):
        for key, value in (("source", "other.pdf"),
                           ("document_id", "other-document"),
                           ("section_heading", "Article 48")):
            with self.subTest(key=key):
                hits = main_hits()
                hits[-1][key] = value
                self.assertIsNone(group_article_items(hits))

    def test_annex_and_unlabelled_passages_are_left_for_the_existing_path(self):
        annex = ARTICLE.replace("Article 47", "ANNEX III", 1)
        self.assertIsNone(group_article_items(make_hits(annex, heading="Annex III")))
        self.assertIsNone(group_article_items([]))
        self.assertIsNone(group_article_items(make_hits(
            "Article 47\nRules\n1. A prose paragraph without a lettered list.\n"
            "2. Further prose.\n")))

    def test_incomplete_or_nonsequential_lettered_enumerations_are_unsupported(self):
        for listing in (
            "(b) A second item;\n(c) A third item.",
            "(a) A first item;\n(c) A third item.",
            "(a) A first item;\n(b) A second item;\n(b) Another second item.",
        ):
            with self.subTest(listing=listing):
                text = f"Article 47\nRules\n1. These activities are prohibited:\n{listing}\n2. Safeguards.\n"
                self.assertIsNone(group_article_items(make_hits(text)))

    def test_paragraph_one_must_be_present(self):
        text = ARTICLE.replace("1. The following", "The following", 1)
        self.assertIsNone(group_article_items(make_hits(text)))

    def test_verified_complete_article_can_end_with_its_only_numbered_paragraph(self):
        text = ARTICLE[:ARTICLE.index("2. A competent")]
        result = group_article_items(make_hits(text))
        self.assertIsNotNone(result)
        self.assertEqual([item["label"] for item in result["items"]], list("abcdefgh"))
        self.assertIn("(iii) A third limited purpose.", result["items"][-1]["text"])
        self.assertEqual(result["notes"], "")

    def test_inline_cross_references_do_not_create_items_or_close_the_list(self):
        text = (
            "Article 47\nRules\n1. The following activities are listed:\n"
            "(a) An activity subject to point (b) and paragraph 2. of this Article;\n"
            "(b) A second activity.\n2. Further conditions.\n"
        )
        result = group_article_items(make_hits(text))
        self.assertIsNotNone(result)
        self.assertEqual([item["label"] for item in result["items"]], ["a", "b"])
        self.assertIn("point (b) and paragraph 2.", result["items"][0]["text"])
        self.assertIn("A second activity", result["items"][1]["text"])


if __name__ == "__main__":
    unittest.main()
