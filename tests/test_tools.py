"""Offline tests for the four standalone tools; no paid or external calls."""

from contextlib import asynccontextmanager
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.messaging import SlackSender
from app.tools import CorpusTools, TOOL_SCHEMAS, calculator


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock(chunks=[{"source": "b.pdf"}, {"source": "a.pdf"}, {"source": "a.pdf"}])
        self.sender = Mock(configured=True, channel="C123", mode="mcp")
        self.sender.send.return_value = {"status": "sent"}
        self.tools = CorpusTools(self.retriever, sender=self.sender)

    def test_search_preserves_pages_and_stable_ids(self):
        a = {"source": "a.pdf", "chunk_id": 0, "pages": [1, 2], "text": "amount 100", "score": .9}
        b = {"source": "b.pdf", "chunk_id": 1, "pages": [3], "text": "exception", "score": .8}
        self.retriever.retrieve.side_effect = [{"passages": [a]}, {"passages": [b, a]}, {"passages": []}]
        self.assertEqual(self.tools.search_corpus("amount")["passages"][0]["pages"], [1, 2])
        self.assertEqual([p["citation_id"] for p in self.tools.search_corpus("exception")["passages"]], [2, 1])
        self.assertEqual(len(self.tools.passages), 2)
        self.assertEqual(self.tools.search_corpus("missing")["passages"], [])

    def test_list_documents(self):
        self.assertEqual(self.tools.list_documents(), {"documents": ["a.pdf", "b.pdf"]})
        self.retriever.retrieve.assert_not_called()

    def test_calculator(self):
        self.assertEqual(calculator("10000 * 0.15 + 250"), {"value": 1750})
        for expression in ["__import__('os')", "2 ** 999999", "'a' * 100000", "True + 1", "1e999", "1 << 4"]:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculator(expression)

    def test_dispatch_and_recoverable_errors(self):
        self.assertEqual(json.loads(self.tools.run_tool("calculator", '{"expression":"2 + 2"}')), {"value": 4})
        for name, arguments in [("unknown", "{}"), ("calculator", "{"), ("calculator", "[]"),
                ("calculator", '{"expression":"1/0"}'), ("calculator", '{"expression":NaN}'),
                ("approve_message", '{"approved":true}')]:
            self.assertIn("error", json.loads(self.tools.run_tool(name, arguments)))
        self.assertEqual({t["function"]["name"] for t in TOOL_SCHEMAS},
                         {"search_corpus", "calculator", "list_documents", "send_message"})

    def test_send_requires_approval_and_is_one_shot(self):
        proposal = self.tools.send_message("Exact answer [1]\n[1] a.pdf, pages 1, 2")
        self.sender.send.assert_not_called()
        self.assertEqual(proposal["status"], "approval_required")
        self.assertEqual(proposal["channel"], "C123")
        other = CorpusTools(self.retriever, sender=self.sender)
        self.assertEqual(other.approve_message(proposal["proposal_id"], True)["status"], "not_pending")
        self.assertEqual(self.tools.approve_message(proposal["proposal_id"], True)["status"], "sent")
        self.sender.send.assert_called_once_with(proposal["text"])
        self.assertEqual(self.tools.approve_message(proposal["proposal_id"], True)["status"], "not_pending")

    def test_rejection_and_delivery_failure(self):
        proposal = self.tools.send_message("hello")
        self.assertEqual(self.tools.approve_message(proposal["proposal_id"], False)["status"], "rejected")
        self.sender.send.assert_not_called()
        proposal = self.tools.send_message("hello")
        self.sender.send.side_effect = TimeoutError("secret token")
        result = self.tools.approve_message(proposal["proposal_id"], True)
        self.assertEqual(result["status"], "delivery_unknown")
        self.assertNotIn("secret token", json.dumps(result))
        self.tools.approve_message(proposal["proposal_id"], True)
        self.sender.send.assert_called_once()

    def test_missing_configuration(self):
        self.sender.configured = False
        self.assertEqual(self.tools.send_message("hello")["status"], "not_configured")
        self.sender.send.assert_not_called()

    def test_bot_rejection(self):
        with patch("requests.post") as post:
            post.return_value.json.return_value = {"ok": False}
            with self.assertRaises(RuntimeError):
                SlackSender("bot", "C123", "dummy").send("hello")
            self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_mcp_discovery_and_send(self):
        session = Mock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=NS(tools=[NS(name="slack_send_message", inputSchema={
            "type": "object", "properties": {"channel_id": {"type": "string"}, "message": {"type": "string"}},
            "required": ["channel_id", "message"], "additionalProperties": False})]))
        session.call_tool = AsyncMock(return_value=NS(isError=False))
        @asynccontextmanager
        async def transport(*args, **kwargs):
            yield None, None, None
        @asynccontextmanager
        async def connect(*args, **kwargs):
            yield session
        with patch("mcp.client.streamable_http.streamable_http_client", side_effect=transport) as stream, \
             patch("mcp.ClientSession", side_effect=connect):
            self.assertEqual(SlackSender("mcp", "C123", "dummy").send("approved text")["status"], "sent")
        self.assertEqual(stream.call_args.args[0], "https://mcp.slack.com/mcp")
        session.call_tool.assert_awaited_once_with("slack_send_message", {"channel_id": "C123", "message": "approved text"})


if __name__ == "__main__":
    unittest.main()
