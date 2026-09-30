"""Bounded chat state and Agent callbacks; no Gradio or model dependency."""

from html import escape
import json
import os

from .retrieval_trace import format_retrieval
from .tool_calls import format_tool_calls
from .context_usage import format_context
from .rag_demo import format_rag_demo
from contracts.errors import ApiError
from contracts.httpkit import call_json
from contracts.schemas import new_request_id

AGENT_URL = os.environ.get("AGENT_URL", "http://agent:8001").rstrip("/")
TIMEOUT_S = float(os.environ.get("UI_TIMEOUT_S", "300"))
MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 24000
MAX_DISPLAY_MESSAGES = 40
MAX_REQUEST_BYTES = 128 * 1024


def bounded_history(messages):
    """Keep recent complete exchanges within the Agent's count/text limits."""
    pairs = []
    for index in range(0, len(messages) - 1, 2):
        pair = messages[index:index + 2]
        if (all(isinstance(item, dict) and isinstance(item.get("content"), str) for item in pair)
                and pair[0].get("role") == "user" and pair[1].get("role") == "assistant"):
            pairs.append([{"role": item["role"], "content": item["content"][:12000]}
                          for item in pair])
    selected, characters = [], 0
    for pair in reversed(pairs):
        size = sum(len(item["content"]) for item in pair)
        if len(selected) + 2 > MAX_HISTORY_MESSAGES or characters + size > MAX_HISTORY_CHARS:
            break
        selected = pair + selected
        characters += size
    return selected


def _state(value):
    value = value if isinstance(value, dict) else {}
    history = value.get("history", [])
    display = value.get("display", [])
    return {"history": bounded_history(history if isinstance(history, list) else []),
            "display": list(display[-MAX_DISPLAY_MESSAGES:]) if isinstance(display, list) else []}


def _failure(message, state, feedback):
    # Keep the failed question editable and leave successful conversation intact.
    return message, state["display"], state, feedback, "", "", {}, "", format_context(), format_rag_demo([])


def _sources(citations):
    lines = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        pages = item.get("pages", [])
        pages = [str(page) for page in pages if type(page) is int and page > 0] if isinstance(pages, list) else []
        source = escape(str(item.get("source", "Source")))
        number = escape(str(item.get("id", "?")))
        lines.append(f"- [{number}] {source}" + (f" — pages {', '.join(pages)}" if pages else ""))
    return "\n".join(lines) or "No sources cited for this answer."


def _public_details(body):
    """Show the answer contract, excluding arbitrary internal service metadata."""
    return {key: body[key] for key in ("request_id", "status", "answer", "citations", "trace",
                                      "usage", "steps", "mock", "context_window") if key in body}


def submit_chat(message, session, injection_test=False):
    """Send bounded conversation to the Agent and return all ten UI outputs."""
    state = _state(session)
    if not isinstance(message, str) or not message.strip():
        return _failure(message or "", state, "Enter a question first.")
    if len(message) > MAX_MESSAGE_CHARS:
        return _failure(message, state, "Keep your question within 2,000 characters.")
    question = message.strip()
    request_id = new_request_id("chat")
    payload = {"request_id": request_id, "message": question, "history": state["history"]}
    if injection_test:
        payload["injection_test"] = True
    # The transport escapes Unicode when serializing JSON. Long non-ASCII
    # histories can hit its byte limit before reaching the character limit.
    while payload["history"] and len(json.dumps(payload).encode("utf-8")) > MAX_REQUEST_BYTES:
        payload["history"] = payload["history"][2:]
    try:
        status, body = call_json("POST", f"{AGENT_URL}/v1/chat", payload, TIMEOUT_S, "agent")
    except ApiError as error:
        feedback = ("The answer timed out. Your question is still here; try again."
                    if error.status == 504 else "The assistant is unavailable. Your question is still here; try again.")
        return _failure(message, state, feedback)
    except Exception:
        # Do not expose a traceback, provider response, or service configuration.
        return _failure(message, state, "Could not complete the request. Your question is still here; try again.")
    if isinstance(body, dict) and body.get("status") == "error" and body.get("mock") is False:
        failed = _failure(message, state, "The assistant could not finish this request. Your question is still here; try again.")
        trace = body.get("trace", [])
        summary = format_tool_calls(trace)
        return *failed[:5], summary, _public_details(body), format_retrieval(trace), format_context(body), format_rag_demo(trace)
    if (status != 200 or not isinstance(body, dict) or body.get("mock") is not False
            or not isinstance(body.get("answer"), str) or not body["answer"].strip()
            or not isinstance(body.get("citations"), list)
            or not isinstance(body.get("trace"), list) or not isinstance(body.get("usage"), dict)):
        return _failure(message, state, "The assistant could not return an answer. Your question is still here; try again.")

    answer = body["answer"]
    sources = _sources(body["citations"])
    display_answer = answer + ("\n\n**Sources**\n\n" + sources if body["citations"] else "")
    history = bounded_history(state["history"] + [{"role": "user", "content": question},
                                                   {"role": "assistant", "content": answer}])
    display = (state["display"] + [{"role": "user", "content": question},
                                   {"role": "assistant", "content": display_answer}])[-MAX_DISPLAY_MESSAGES:]
    details = _public_details(body)
    feedback = "Answer ready."
    tokens = body["usage"].get("total_tokens")
    if type(tokens) is int and tokens >= 0:
        feedback += f" {tokens:,} tokens."
    return "", display, {"history": history, "display": display}, feedback, sources, format_tool_calls(body["trace"]), details, format_retrieval(body["trace"]), format_context(body), format_rag_demo(body["trace"])


def new_chat():
    """Clear messages, follow-up context, sources and the previous trajectory."""
    return "", [], {"history": [], "display": []}, "New chat started.", "", "", {}, "", format_context(), format_rag_demo([])


def ping():
    """Keep the old connectivity check available without exposing internal URLs."""
    try:
        status, body = call_json("POST", f"{AGENT_URL}/v1/ping",
                                 {"request_id": new_request_id("ping"), "message": "hello", "forward": True},
                                 min(TIMEOUT_S, 15), "agent")
    except Exception:
        return "Connection unavailable. Check that the services are running."
    return "UI → Agent → RAG connected." if status == 200 and body.get("ok") else "Connection check failed."
