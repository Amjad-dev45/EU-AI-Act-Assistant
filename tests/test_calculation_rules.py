"""Source-only checks for optional higher/lower calculation-rule quotations."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag.calculation_rules import select_calculation_rules


ARTICLE = (
    "Article 72\nAdministrative calculations\n"
    "1. Authorities impose proportionate administrative fines.\n"
    "2. Companies remain responsible for complying with their obligations.\n"
    "3. Large companies are subject to administrative fines of up to 17 units or 7 % "
    "of turnover, whichever is higher, for the following failures:\n"
    "(a) failure to retain records;\n"
    "(b) failure to permit an audit.\n"
    "4. Large companies are subject to administrative fines of up to 11 units or 5 % "
    "of turnover, whichever is higher, for supplying inaccurate information.\n"
    "5. Authorities must provide notice and an opportunity to respond.\n"
    "6. The decision shall explain the applicable amount and percentage.\n"
    "7. In the case of charities, including start-ups, the administrative fines "
    "referred to in paragraphs 3 and 4 shall not exceed the percentage or amount "
    "referred to in those paragraphs, whichever is lower, provided that:\n"
    "(a) the entity satisfies both conditions:\n"
    "(i) registered charitable status;\n"
    "(ii) no distribution of profits;\n"
    "(b) the entity submits its annual report.\n"
    "8. Authorities shall publish annual enforcement statistics.\n"
)


def make_hits(text, cuts=(), *, heading="Article 72", base=400):
    boundaries = [0, *cuts, len(text)]
    result = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        result.append({
            "chunk_id": f"synthetic:{base + left}:{base + right}",
            "document_id": "synthetic-rules",
            "source": "synthetic-rules.pdf",
            "pages": [index + 1],
            "start_char": base + left,
            "end_char": base + right,
            "excerpt_start_char": base + left,
            "excerpt_end_char": base + right,
            "section_heading": heading,
            "text": text[left:right],
        })
    return result


def rule_hits():
    return make_hits(ARTICLE, [
        ARTICLE.index("(a) failure"),
        ARTICLE.index("4. Large companies"),
        ARTICLE.index("referred to in those paragraphs"),
        ARTICLE.index("\n8. Authorities"),
    ])


def paragraph(number):
    left = ARTICLE.index(f"{number}. ")
    right = ARTICLE.index(f"{number + 1}. ")
    return ARTICLE[left:right].rstrip()


class CalculationRuleTests(unittest.TestCase):
    def test_comparison_returns_complete_source_paragraphs_with_qualification_first(self):
        hits = rule_hits()
        original = deepcopy(hits)
        result = select_calculation_rules("Compare the calculation rules for the two classes.", hits)

        self.assertIsNotNone(result)
        self.assertEqual(result["heading"], "Article 72")
        self.assertEqual([item["label"] for item in result["items"]], ["7", "3", "4"])
        self.assertEqual([item["text"] for item in result["items"]],
                         [paragraph(7), paragraph(3), paragraph(4)])
        self.assertEqual([item["citation_ids"] for item in result["items"]],
                         [[3, 4], [1, 2], [3]])
        self.assertEqual(hits, original)

    def test_nested_conditions_remain_in_the_whole_numbered_rule(self):
        result = select_calculation_rules("How are charities fined?", rule_hits())
        self.assertIsNotNone(result)
        lower, higher, _ = result["items"]
        for text in ("provided that:", "(a) the entity", "both conditions",
                     "(i) registered charitable status", "(ii) no distribution of profits",
                     "(b) the entity submits its annual report"):
            self.assertIn(text, lower["text"])
        self.assertIn("(a) failure to retain records", higher["text"])
        self.assertIn("(b) failure to permit an audit", higher["text"])
        self.assertNotIn("8. Authorities", lower["text"])

    def test_comparison_wording_activates_without_naming_an_exception_class(self):
        questions = (
            "Compare the calculation rules.",
            "What is the comparison between these rules?",
            "How are the rules compared?",
            "What differs in one rule versus the other?",
            "Explain the first rule vs the second.",
            "What is the difference in calculation?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNotNone(select_calculation_rules(question, rule_hits()))

    def test_source_derived_scope_accepts_generic_plural_and_hyphen_variants(self):
        for actor in ("charity", "charities", "start-up", "start-ups", "startup", "startups"):
            with self.subTest(actor=actor):
                result = select_calculation_rules(f"Explain fine rules for {actor}.", rule_hits())
                self.assertIsNotNone(result)
                self.assertEqual(result["items"][0]["label"], "7")

    def test_ordinary_maximum_fine_questions_do_not_select_the_exception(self):
        for question in (
            "What is the maximum fine?",
            "What is the highest fine for companies?",
            "How much is the largest administrative fine?",
            "Which penalties apply to large companies?",
            "What is the fine for inaccurate information?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(select_calculation_rules(question, rule_hits()))

    def test_different_article_numbers_actor_classes_and_amounts_are_source_derived(self):
        text = ARTICLE.replace("Article 72", "Article 41").replace("charities", "cooperatives")
        text = text.replace("start-ups", "micro-enterprises").replace("17 units", "23 credits")
        result = select_calculation_rules("What fine applies to a cooperative?",
                                          make_hits(text, heading="Article 41"))
        self.assertIsNotNone(result)
        self.assertEqual(result["heading"], "Article 41")
        self.assertIn("cooperatives", result["items"][0]["text"])
        self.assertIn("23 credits", result["items"][1]["text"])
        self.assertIsNone(select_calculation_rules("What fine applies to a charity?",
                                                  make_hits(text, heading="Article 41")))

    def test_both_opposing_rules_must_be_present(self):
        for text in (
            ARTICLE.replace("whichever is lower", "subject to ordinary calculation"),
            ARTICLE.replace("whichever is higher", "subject to ordinary calculation"),
            "Article 72\nRules\n1. Authorities may impose a proportionate fine.\n",
        ):
            with self.subTest(text=text[-80:]):
                self.assertIsNone(select_calculation_rules("Compare the calculation rules.", make_hits(text)))

    def test_citations_refer_to_actual_excerpts_and_ignore_only_separator_overlap(self):
        hits = rule_hits()
        for hit in hits:
            hit["start_char"] = 0
            hit["end_char"] = 400 + len(ARTICLE) + 100
        result = select_calculation_rules("Compare these calculation rules.", hits)
        self.assertIsNotNone(result)
        by_label = {item["label"]: item for item in result["items"]}
        self.assertEqual(by_label["7"]["citation_ids"], [3, 4])
        self.assertNotIn(5, by_label["7"]["citation_ids"])
        self.assertEqual(by_label["3"]["citation_ids"], [1, 2])

    def test_missing_middle_or_out_of_order_excerpts_are_unsupported(self):
        hits = rule_hits()
        for selected in ([hits[0], *hits[2:]], [hits[1], hits[0], *hits[2:]]):
            with self.subTest(ids=[hit["chunk_id"] for hit in selected]):
                self.assertIsNone(select_calculation_rules("Compare the calculations.", selected))

    def test_mixed_provenance_and_unverified_hits_cannot_authorize_quotations(self):
        changes = (("section_heading", "Article 73"), ("source", "other.pdf"),
                   ("document_id", "other"), ("excerpt_end_char", True),
                   ("excerpt_end_char", 999999))
        for key, value in changes:
            with self.subTest(key=key):
                hits = rule_hits()
                hits[1][key] = value
                self.assertIsNone(select_calculation_rules("Compare the calculations.", hits))
        hits = rule_hits()
        del hits[0]["excerpt_start_char"]
        self.assertIsNone(select_calculation_rules("Compare the calculations.", hits))

    def test_annex_empty_and_unrelated_questions_leave_the_existing_path(self):
        annex = ARTICLE.replace("Article 72", "ANNEX II", 1)
        self.assertIsNone(select_calculation_rules("Compare the calculation rules.",
                                                  make_hits(annex, heading="Annex II")))
        self.assertIsNone(select_calculation_rules("Compare the calculation rules.", []))
        self.assertIsNone(select_calculation_rules("How should ordinary records be retained?", rule_hits()))


if __name__ == "__main__":
    unittest.main()
