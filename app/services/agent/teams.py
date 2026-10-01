"""Teams sharing: save an answer draft, then send only after a UI button click.

The model cannot call this endpoint or choose its destination. Only an opaque
draft token crosses the UI boundary; the webhook stays in the Agent environment.
"""

import json
import os
from pathlib import Path
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import uuid

from contracts.errors import ApiError
from .teams_format import answer_blocks

MAX_CARD_BYTES = 28_000
TIMEOUT_S = 15
MESSAGES = {
    "accepted": "Submitted to Teams. Check the channel for delivery.",
    "rejected": "Teams rejected the message. Check the webhook/workflow configuration before generating a new answer to share.",
    "unknown": "Delivery could not be confirmed. Check the Teams channel; this answer will not be sent again automatically.",
}


def _webhook():
    """Validate configuration without exposing the URL in errors or logs."""
    value = os.environ.get("web_hook", "").strip()
    try:
        parsed = urllib.parse.urlsplit(value)
        valid = (parsed.scheme == "https" and parsed.hostname
                 and not parsed.username and not parsed.password and not parsed.fragment)
        if valid:
            return value
    except ValueError:
        pass
    raise ValueError("Set a valid HTTPS web_hook in .env and restart the Agent.")


def _card(question, answer, citations):
    """Preserve answer content; render HTML tables with native card columns."""
    body = [
        {"type": "TextBlock", "text": "Challenge Week · EU AI Act", "weight": "Bolder", "size": "Large", "wrap": True},
        {"type": "TextBlock", "text": "Question", "weight": "Bolder", "separator": True},
        {"type": "TextBlock", "text": question, "wrap": True},
        {"type": "TextBlock", "text": "Answer", "weight": "Bolder", "separator": True},
        *answer_blocks(answer),
    ]
    sources = []
    for item in citations:
        pages = ", ".join(str(page) for page in item.get("pages", []) if type(page) is int and page > 0)
        sources.append(f"[{item.get('id', '?')}] {item.get('source', 'Source')}" + (f" — pages {pages}" if pages else ""))
    if sources:
        body.extend([
            {"type": "TextBlock", "text": "Sources", "weight": "Bolder", "separator": True},
            {"type": "TextBlock", "text": "\n\n".join(sources), "wrap": True},
        ])
    body.append({"type": "TextBlock", "text": "AI-generated answer · Shared by user approval", "isSubtle": True, "wrap": True})
    payload = {"type": "message", "attachments": [{
        "contentType": "application/vnd.microsoft.card.adaptive", "contentUrl": None,
        "content": {"$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard", "version": "1.2", "body": body},
    }]}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_CARD_BYTES:
        raise ValueError("This answer is too long for a Teams card. Ask for a shorter answer; nothing was sent.")
    return encoded.decode("utf-8")


def _database():
    directory = Path(os.environ.get("AGENT_LOG_DIR", Path(__file__).resolve().parents[3] / "logs" / "agent_runs"))
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "teams_shares.sqlite3", timeout=5)
    db.execute("CREATE TABLE IF NOT EXISTS shares (token TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    return db


def prepare_share(question, result):
    """Persist a bounded draft after an answer. This NEVER contacts Teams."""
    if result.get("status") != "answered" or result.get("mock") is not False:
        return {"available": False, "message": "Teams sharing is available after a successful answer."}
    try:
        _webhook()
        payload = _card(question, result["answer"], result["citations"])
        token = uuid.uuid4().hex
        db = _database()
        try:
            with db:
                db.execute("INSERT INTO shares (token, payload, status) VALUES (?, ?, 'ready')", (token, payload))
        finally:
            db.close()
    except ValueError as error:
        return {"available": False, "message": str(error)}
    except Exception:
        return {"available": False, "message": "Could not prepare Teams sharing. Check the Agent's writable log folder."}
    return {"available": True, "token": token,
            "message": "Click Send to Teams to share the latest question, answer, and sources with the configured channel."}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # never forward the secret URL/payload to a redirected host


def _post(webhook, payload):
    """One bounded POST, no automatic retries and no raw provider error output."""
    request = urllib.request.Request(webhook, data=payload.encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=TIMEOUT_S) as response:
            status, raw = response.status, response.read(4096).strip()
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        return "rejected" if 300 <= status < 500 and status != 408 else "unknown"
    except Exception:
        # A timeout/disconnection can occur AFTER Teams accepted the message.
        return "unknown"
    # Connectors return '1'; Workflows generally return an empty 202. A 200
    # error body is not evidence of acceptance, so never report it as success.
    if status in (200, 201, 202, 204) and raw in (b"", b"1"):
        return "accepted"
    return "unknown"


def send_share(body):
    """Approve an immutable draft; atomically claim it before any network call."""
    if (not isinstance(body, dict) or body.get("approved") is not True
            or not isinstance(body.get("token"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", body["token"])):
        raise ApiError(400, "invalid_request", "Use the Send to Teams button to approve a prepared answer.")
    # Even a caller with the token cannot replace the text or destination.
    if set(body) - {"token", "approved", "request_id"}:
        raise ApiError(400, "invalid_request", "Only a prepared answer can be shared.")
    try:
        webhook = _webhook()
    except ValueError:
        raise ApiError(503, "teams_not_configured", "Set a valid HTTPS web_hook in .env and restart the Agent.") from None
    db = _database()
    try:
        with db:
            row = db.execute("SELECT payload, status FROM shares WHERE token = ?", (body["token"],)).fetchone()
            if row is None:
                raise ApiError(404, "share_not_found", "This answer is no longer available to share. Generate a new answer.")
            claimed = db.execute("UPDATE shares SET status = 'sending' WHERE token = ? AND status = 'ready'", (body["token"],)).rowcount
        if not claimed:
            current = db.execute("SELECT status FROM shares WHERE token = ?", (body["token"],)).fetchone()[0]
            outcome = current if current in MESSAGES else "unknown"
            return 200, {"status": outcome, "message": MESSAGES[outcome], "duplicate": True}
        outcome = _post(webhook, row[0])
        with db:
            db.execute("UPDATE shares SET status = ? WHERE token = ?", (outcome, body["token"]))
        return 200, {"status": outcome, "message": MESSAGES[outcome], "duplicate": False}
    finally:
        db.close()
