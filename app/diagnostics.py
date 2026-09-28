"""Detailed local debugging output with credential redaction."""

import html
import logging
import os
import re
import traceback
from uuid import uuid4


def redact(text):
    secrets = [value for name, value in os.environ.items()
               if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I) and len(value) >= 6]
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(Bearer\s+)[^\s'\"<>]+", r"\1[REDACTED]", text)
    return re.sub(r"(https?://)[^/\s@]+:[^/\s@]+@", r"\1[REDACTED]@", text)


def error_details(exc):
    """Include chained exceptions and stack locations, never captured locals."""
    error_id = getattr(exc, "rag_request_id", None) or str(uuid4())
    summary = redact(f"{type(exc).__name__}: {exc}")
    stack = redact("".join(traceback.format_exception(exc)))
    logging.getLogger(__name__).error("Request error %s\n%s", error_id, stack)
    panel = (
        "<details open style='border:1px solid #b44;padding:12px;border-radius:6px'>"
        "<summary><strong>Error details</strong></summary>"
        f"<p>Error ID: <code>{html.escape(error_id)}</code></p>"
        f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere'>{html.escape(stack)}</pre>"
        "</details>"
    )
    return {"error_id": error_id, "summary": summary, "html": panel}
