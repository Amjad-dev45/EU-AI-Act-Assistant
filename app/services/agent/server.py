"""PLACEHOLDER (Person 3): Agent API. Add /v1/chat and /v1/metrics next to /v1/ping."""
import os
import time

from contracts.errors import ApiError, BAD_GATEWAY
from contracts.httpkit import call_json, serve_forever
from contracts.schemas import validate_ping

RAG_URL = os.environ.get("RAG_URL", "http://rag:8002")
RAG_TIMEOUT_S = float(os.environ.get("RAG_TIMEOUT_S", "3"))


def health(_body):
    return 200, {"status": "ok", "service": "agent"}  # must not depend on RAG


def ping(body):
    """Connectivity test. With forward=true the Agent also pings RAG over HTTP."""
    req = validate_ping(body)
    rid = req["request_id"]
    chain = [{"service": "agent", "status": "ok"}]
    if req["forward"]:
        started = time.perf_counter()
        try:
            status, rag = call_json("POST", f"{RAG_URL}/v1/ping",
                                    {"request_id": rid, "message": req["message"]}, RAG_TIMEOUT_S, "rag")
        except ApiError as e:
            chain.append({"service": "rag", "status": "error", "error": e.code})
            raise ApiError(e.status, e.code, e.message, extra={"ok": False, "chain": chain})
        ms = round((time.perf_counter() - started) * 1000, 2)
        if status != 200:
            chain.append({"service": "rag", "status": "error", "error": "bad_gateway"})
            raise ApiError(502, BAD_GATEWAY, f"`rag` answered HTTP {status}.", extra={"ok": False, "chain": chain})
        chain.append({"service": "rag", "status": "ok", "duration_ms": ms})
    return 200, {"request_id": rid, "ok": True, "echo": req["message"], "chain": chain}


ROUTES = {
    ("GET", "/health"): health,
    ("POST", "/v1/ping"): ping,  # connectivity test
}


def main() -> None:
    serve_forever("agent", ROUTES, "0.0.0.0", int(os.environ.get("AGENT_PORT", "8001")))
