"""Teams table conversion preserves evidence without sending a real message."""

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.agent import teams
from services.agent.teams_format import answer_blocks


PROVIDER = "A natural or legal person that develops or places an AI system on the market under its own name or trademark."
DEPLOYER = "A natural or legal person using an AI system under its authority, excluding personal non-professional use."
EXAMPLE = ("<table><tr><th>Term</th><th>Definition</th></tr>"
           f"<tr><td>Provider</td><td>{PROVIDER}</td></tr>"
           f"<tr><td>Deployer</td><td>{DEPLOYER}</td></tr></table> [1] [2]")


def rows(blocks):
    return [[column["items"][0]["text"] for column in block["columns"]]
            for block in blocks if block["type"] == "ColumnSet"]


class TeamsFormatTests(unittest.TestCase):
    def test_reported_table_keeps_complete_definitions_citations_and_headers(self):
        blocks = answer_blocks(EXAMPLE)
        self.assertEqual(rows(blocks), [["Term", "Definition"], ["Provider", PROVIDER], ["Deployer", DEPLOYER]])
        self.assertEqual(blocks[-1]["text"], "[1] [2]")
        self.assertTrue(all(cell["items"][0]["weight"] == "Bolder" for cell in blocks[0]["columns"]))
        self.assertNotIn("<table", json.dumps(blocks))

    def test_mixed_content_entities_multiple_tables_and_fences(self):
        blocks = answer_blocks("Before.\n```html\n<table><tr><td>R&amp;D</td><td>€7 million &lt; €35 million</td></tr></table>\n```\n"
                               "Between [1].<table><tr><td>Second</td></tr></table>After [2].")
        self.assertEqual(rows(blocks), [["R&D", "€7 million < €35 million"], ["Second"]])
        self.assertEqual([block["text"] for block in blocks if block["type"] == "TextBlock"],
                         ["Before.", "Between [1].", "After [2]."])

    def test_plain_markdown_and_ordinary_code_quotes_remain_verbatim(self):
        for text in ("AI literacy **is required**. [1]\n\n```\nExact source wording.\n```", "7% of €1 billion = €70 million."):
            with self.subTest(text=text):
                self.assertEqual(answer_blocks(text), [{"type": "TextBlock", "text": text, "wrap": True}])

    def test_table_filters_scripts_styles_and_unsafe_links_without_losing_text(self):
        blocks = answer_blocks('<html><head><style>BAD_CSS</style></head><body><table onclick="BAD_EVENT">'
                               '<tr><td><b>Provider</b><br>Official definition</td>'
                               '<td><a href="javascript:BAD_URL">Keep this label</a>'
                               '<script>BAD_SCRIPT</script><img src="https://example.com/remote" alt="Image description">'
                               '<a href="https://example.com/article(4)">Article 4</a></td></tr></table></body></html>')
        rendered = json.dumps(blocks)
        self.assertNotIn("BAD_", rendered)
        self.assertNotIn("remote", rendered)
        self.assertIn("**Provider**\nOfficial definition", rows(blocks)[0])
        self.assertIn("Keep this labelImage description[Article 4](https://example.com/article%284%29)", rows(blocks)[0])

    def test_merged_and_nested_tables_refused_instead_of_changing_their_meaning(self):
        for text in ('<table><tr><td colspan="2">Merged</td></tr></table>',
                     '<table><tr><td rowspan="2">Merged</td></tr></table>',
                     '<table><tr><td><table><tr><td>Nested</td></tr></table></td></tr></table>'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "cannot be shared"):
                answer_blocks(text)

    def test_unfinished_simple_table_retains_last_cell(self):
        self.assertEqual(rows(answer_blocks("<table><tr><th>Term<th>Definition<tr><td>Provider<td>Definition")),
                         [["Term", "Definition"], ["Provider", "Definition"]])

    def test_prepared_card_uses_table_columns_and_keeps_sources_without_posting(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
            "web_hook": "https://example.com/test-only", "AGENT_LOG_DIR": directory,
        }), patch.object(teams, "_post") as post:
            result = {"status": "answered", "mock": False, "answer": EXAMPLE,
                      "citations": [{"id": number, "source": "OJ_L_202401689_EN_TXT.pdf", "pages": [46],
                                     "text": "INTERNAL_SOURCE_TEXT"} for number in (1, 2)],
                      "trace": [{"private": "INTERNAL_TRACE"}]}
            draft = teams.prepare_share("Please make a table.", result)
            self.assertTrue(draft["available"])
            db = sqlite3.connect(Path(directory) / "teams_shares.sqlite3")
            try:
                payload = db.execute("SELECT payload FROM shares WHERE token = ?", (draft["token"],)).fetchone()[0]
            finally:
                db.close()
            card = json.loads(payload)["attachments"][0]["content"]
            self.assertEqual(card["version"], "1.2")
            self.assertEqual(rows(card["body"]), [["Term", "Definition"], ["Provider", PROVIDER], ["Deployer", DEPLOYER]])
            self.assertIn("[1] OJ_L_202401689_EN_TXT.pdf — pages 46\n\n[2] OJ_L_202401689_EN_TXT.pdf — pages 46",
                          [block.get("text") for block in card["body"]])
            self.assertNotIn("INTERNAL_", payload)
            post.assert_not_called()

    def test_large_table_refused_without_truncating_evidence(self):
        text = "<table>" + "<tr><td>" + "€" * 10000 + "</td></tr></table>"
        with self.assertRaisesRegex(ValueError, "too long"):
            teams._card("Question", text, [])


if __name__ == "__main__":
    unittest.main()
