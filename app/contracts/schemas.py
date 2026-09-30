"""Request validation + tiny builders. Pure functions, no I/O.

Contracts are documented by the JSON files in `contracts/examples/`.
"""
import re
import uuid

from .errors import ApiError, INVALID_REQUEST

DEMO_TOOLS = ("rag", "calculator", "internet", "quote_exact")
MAX_MESSAGE_CHARS = 2000
MAX_K = 10
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def new_request_id(prefix: str = "req") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _bad(message: str) -> ApiError:
    return ApiError(400, INVALID_REQUEST, message)


def _require_object(body) -> dict:
    if not isinstance(body, dict):
        raise _bad("Request body must be a JSON object.")
    return body


def validate_request_id(body: dict) -> str:
    rid = body.get("request_id")
    if not isinstance(rid, str) or not _REQUEST_ID_RE.match(rid):
        raise _bad("`request_id` is required: 1-64 chars of letters, digits, _ . : -")
    return rid


def _text(body: dict, field: str, required: bool = True, max_chars: int = MAX_MESSAGE_CHARS):
    value = body.get(field)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise _bad(f"`{field}` must be a non-empty string.")
    if len(value) > max_chars:
        raise _bad(f"`{field}` is longer than {max_chars} characters.")
    return value


def injection_test_flag(body):
    value = body.get("injection_test", False)
    if type(value) is not bool:
        raise _bad("`injection_test` must be a boolean.")
    return value


def validate_chat(body) -> dict:
    body = _require_object(body)
    validate_request_id(body)
    _text(body, "message")
    history = body.get("history", [])
    if not isinstance(history, list) or len(history) > 6:
        raise _bad("`history` must contain at most 6 messages.")
    clean = []
    for message in history:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise _bad("History roles must be user or assistant.")
        content = _text(message, "content", max_chars=12000)
        clean.append({"role": message["role"], "content": content})
    if sum(len(message["content"]) for message in clean) > 24000:
        raise _bad("History is longer than 24000 characters; start a new chat.")
    return {"request_id": body["request_id"], "message": body["message"], "history": clean, "injection_test": injection_test_flag(body)}


def validate_search(body) -> dict:
    body = _require_object(body)
    validate_request_id(body)
    _text(body, "query")
    injection_test_flag(body)
    k = body.get("k", 3)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise _bad(f"`k` must be an integer between 1 and {MAX_K}.")
    body["k"] = k
    return body


def validate_quote(body) -> dict:
    body = _require_object(body)
    validate_request_id(body)
    _text(body, "document_id", max_chars=128)
    page = body.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise _bad("`page` must be an integer >= 1.")
    _text(body, "phrase", required=False)  # optional exact phrase
    return body


def validate_service_event(body) -> dict:
    body = _require_object(body)
    validate_request_id(body)
    _text(body, "service", max_chars=32)
    _text(body, "endpoint", max_chars=64)
    for field in ("status_code", "duration_ms"):
        v = body.get(field)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise _bad(f"`{field}` must be a non-negative number.")
    return body


def zero_usage() -> dict:
    """Initialize usage before any model or tool calls."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "external_cost_usd": 0.0}


def validate_ping(body) -> dict:
    """Connectivity test: {request_id, message, forward?}."""
    body = _require_object(body)
    validate_request_id(body)
    _text(body, "message", max_chars=200)
    body["forward"] = bool(body.get("forward", False))
    return body
