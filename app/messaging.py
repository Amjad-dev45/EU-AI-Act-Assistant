"""Human-approved Slack delivery. The agent never receives this transport."""

import asyncio
from dataclasses import dataclass
import os
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True)
class SlackSender:
    mode: str
    channel: str
    token: str

    @classmethod
    def from_env(cls):
        mode = os.getenv("SLACK_TRANSPORT", "mcp")
        key = "SLACK_MCP_TOKEN" if mode == "mcp" else "SLACK_BOT_TOKEN"
        return cls(mode, os.getenv("SLACK_CHANNEL_ID", ""), os.getenv(key, ""))

    @property
    def configured(self):
        return self.mode in {"mcp", "bot"} and bool(self.channel and self.token)

    def send(self, text):
        if not self.configured:
            raise ValueError("Configure Slack transport, channel and token in .env")
        if self.mode == "mcp":
            return asyncio.run(self._send_mcp(text))
        import requests
        response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer " + self.token},
            json={"channel": self.channel, "text": text, "mrkdwn": False,
                  "unfurl_links": False, "unfurl_media": False}, timeout=30,
            allow_redirects=False)
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            raise RuntimeError("Slack rejected the message")
        return {"channel": result.get("channel"), "ts": result.get("ts")}

    async def _send_mcp(self, text):
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from jsonschema import validate
        async with asyncio.timeout(45):
            async with httpx.AsyncClient(headers={"Authorization": "Bearer " + self.token},
                                         timeout=30, follow_redirects=False) as http:
                async with streamable_http_client("https://mcp.slack.com/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        listing = await session.list_tools()
                        tool = next((t for t in listing.tools if t.name == "slack_send_message"), None)
                        if tool is None:
                            raise RuntimeError("Slack MCP did not expose slack_send_message")
                        arguments = {"channel_id": self.channel, "message": text}
                        validate(arguments, tool.inputSchema)
                        result = await session.call_tool(tool.name, arguments)
                        if result.isError:
                            raise RuntimeError("Slack MCP rejected the message")
                        return {"status": "sent", "transport": "mcp"}


class MessageTool:
    """Store exact proposals until host-side human approval; no model can approve.

    Keep one instance per user session. Proposals are consumed before delivery,
    because a failed response can still mean Slack received the message.
    """
    def __init__(self, sender):
        self.sender = sender
        self._pending = {}
        self._lock = Lock()

    def propose(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 5000:
            raise ValueError("Message must contain 1 to 5000 characters")
        if not self.sender.configured:
            return {"status": "not_configured", "message": "Configure Slack channel and credentials in .env."}
        proposal_id = str(uuid4())
        with self._lock:
            self._pending[proposal_id] = text
        return {"status": "approval_required", "proposal_id": proposal_id,
                "channel": self.sender.channel, "transport": self.sender.mode, "text": text}

    def decide(self, proposal_id, approved):
        with self._lock:
            text = self._pending.pop(proposal_id, None)
        if text is None:
            return {"status": "not_pending"}
        if approved is not True:
            return {"status": "rejected"}
        try:
            return {"status": "sent", "receipt": self.sender.send(text)}
        except Exception:
            return {"status": "delivery_unknown",
                    "message": "Check Slack before trying again. No automatic retry was made."}
