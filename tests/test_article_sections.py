"""Offline regressions for article boundaries and their citation metadata.

These checks verify the context sent to the model, not live answer correctness.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag import answering


# A short excerpt of the retrieved Q24 passage, including its joined page boundary.
DEFINITION = (
    "‘downstream provider’ means a provider of an AI system, including a "
    "general-purpose AI system, which integrates an AI model, regardless of "
    "whether the AI model is provided by themselves and vertically integrated "
    "or provided by another entity based on contractual relations. "
)
ARTICLE_4 = (
    "Article 4\n\nAI literacy\n\nProviders and deployers of AI systems shall take "
    "measures to ensure, to their best extent, a sufficient level of AI literacy "
    "of their staff and other persons dealing with the operation and use of AI "
    "systems on their behalf, taking into account their technical knowledge, "
    "experience, education and training and the context the AI systems are to "
    "be used in, and considering the persons or groups of persons on whom the "
    "AI systems are to be used."
)


class ArticleSectionTests(unittest.TestCase):
    def assert_source_preserved(self, text, sections):
        self.assertEqual("".join(section["text"] for section in sections), text)

    def test_multiple_headings_keep_each_rule_with_its_own_article(self):
        first = "Article 4\nAI literacy\nProviders shall train staff.\n\n"
        second = "Article 5\nProhibited practices\nThe following practices are prohibited."
        sections = answering.article_sections(first + second)
        self.assertEqual(sections, [
            {"heading": "Article 4", "text": first},
            {"heading": "Article 5", "text": second},
        ])
        self.assert_source_preserved(first + second, sections)

    def test_page_join_does_not_assign_preceding_definition_to_article_four(self):
        source = DEFINITION + ARTICLE_4
        sections = answering.article_sections(source)
        self.assertEqual(sections, [
            {"heading": None, "text": DEFINITION},
            {"heading": "Article 4", "text": ARTICLE_4},
        ])
        self.assertNotIn("downstream provider", sections[1]["text"])
        self.assertNotIn("Providers and deployers", sections[0]["text"])
        self.assert_source_preserved(source, sections)

    def test_inline_citations_to_other_laws_are_not_headings(self):
        source = (
            "The exclusion follows Article 4(2) TEU and national rules.\n"
            "‘profiling’ means profiling as defined in Article 4, point (4) "
            "of Regulation (EU) 2016/679;\n"
            "See Article 4 for the literacy obligation."
        )
        sections = answering.article_sections(source)
        self.assertEqual(sections, [{"heading": None, "text": source}])
        self.assert_source_preserved(source, sections)

    def test_citation_starting_a_line_is_not_an_article_heading(self):
        source = "National security follows\nArticle 4(2) TEU.\nArticle 4, point (4), defines profiling."
        self.assertEqual(answering.article_sections(source),
                         [{"heading": None, "text": source}])

    def test_heading_free_passage_stays_unknown_including_surrounding_whitespace(self):
        source = "\n  A passage from an article whose heading is not in this chunk.  \n"
        sections = answering.article_sections(source)
        self.assertEqual(sections, [{"heading": None, "text": source}])
        self.assert_source_preserved(source, sections)

    def test_leading_whitespace_before_a_heading_is_preserved(self):
        source = "\n\n" + ARTICLE_4
        sections = answering.article_sections(source)
        self.assert_source_preserved(source, sections)
        self.assertEqual(sections[-1]["heading"], "Article 4")
        self.assertEqual(sections[-1]["text"], ARTICLE_4)

    def test_empty_passage_has_no_sections(self):
        self.assertEqual(answering.article_sections(""), [])

    def test_preparing_evidence_copies_metadata_without_mutating_retrieval(self):
        hits = [{"chunk_id": "law:264629:265371", "source": "law.pdf",
                 "pages": [50, 51], "text": DEFINITION + ARTICLE_4,
                 "score": 0.2593, "rerank_score": 1.4456}]
        before = deepcopy(hits)
        evidence = answering.prepare_evidence(hits)
        self.assertEqual(hits, before)
        self.assertEqual(evidence[0]["id"], 1)
        self.assertEqual(evidence[0]["chunk_id"], hits[0]["chunk_id"])
        self.assertEqual(evidence[0]["source"], "law.pdf")
        self.assertEqual(evidence[0]["pages"], [50, 51])
        self.assertNotIn("text", evidence[0])
        self.assert_source_preserved(hits[0]["text"], evidence[0]["sections"])
        evidence[0]["pages"].append(999)
        self.assertEqual(hits, before)

    def test_model_receives_article_separation_and_citations_keep_original_pages(self):
        hits = [{"chunk_id": "law:264629:265371", "source": "law.pdf",
                 "pages": [50, 51], "text": DEFINITION + ARTICLE_4}]
        before = deepcopy(hits)
        payload = {"answerable": True, "statements": [{
            "text": "Providers and deployers must take measures to ensure sufficient AI literacy.",
            "citation_ids": [1],
        }]}
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content=json.dumps(payload)))],
        )
        create = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(answering, "create_client", side_effect=AssertionError("No API calls")):
            result = answering.generate_answer("What does Article 4 say?", hits, client=client)

        create.assert_called_once()
        messages = create.call_args.kwargs["messages"]
        context_message = next(message["content"] for message in messages
                               if message["role"] == "user" and message["content"].startswith("<context>"))
        context = json.loads(context_message.removeprefix("<context>\n").removesuffix("\n</context>"))
        self.assertEqual(context[0]["sections"], [
            {"heading": None, "text": DEFINITION},
            {"heading": "Article 4", "text": ARTICLE_4},
        ])
        self.assertEqual(context[0]["pages"], [50, 51])
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], [{
            "id": 1, "chunk_id": "law:264629:265371", "source": "law.pdf", "pages": [50, 51],
        }])
        self.assertEqual(hits, before)


if __name__ == "__main__":
    unittest.main()
