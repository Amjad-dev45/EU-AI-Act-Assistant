"""PERSON 2: RAG API business logic (transport lives in server.py).

Imports Person 1's quote module. Each request reports its own service-side timing
to the Agent's monitor (best effort: RAG never depends on the Agent to answer).
"""
import os
import time

from contracts.errors import ApiError
from contracts.httpkit import fire_and_forget
from contracts.schemas import validate_quote, validate_search

from . import store
from .quote import find_quote
from .search import search_chunks

AGENT_URL = os.environ.get("AGENT_URL", "http://agent:8001")


def _report(request_id: str, endpoint: str, status: int, started: float) -> None:
    fire_and_forget(f"{AGENT_URL}/v1/internal/service-events", {
        "request_id": request_id, "service": "rag", "endpoint": endpoint,
        "status_code": status, "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    })


def _timed(endpoint: str, body, work):
    started = time.perf_counter()
    rid = body.get("request_id") if isinstance(body, dict) else None
    try:
        result = work()
    except ApiError as e:
        if rid:
            _report(rid, endpoint, e.status, started)
        raise
    _report(rid, endpoint, 200, started)
    return 200, result


def health(_body):
    # Measures only this service: no calls to other containers.
    return 200, {"status": "ok", "service": "rag"}


def search(body):
    def work():
        req = validate_search(body)
        source, chunks = store.load_chunks()
        return {"request_id": req["request_id"], "chunk_source": source,
                "hits": search_chunks(chunks, req["query"], req["k"])}
    return _timed("/v1/search", body, work)


def quote(body):
    def work():
        req = validate_quote(body)
        found = find_quote(store.load_source(), req["document_id"], req["page"], req.get("phrase"))
        return {"request_id": req["request_id"], **found}
    return _timed("/v1/quote", body, work)


def ping(body):
    """Connectivity test: echo the message back. No data access."""
    from contracts.schemas import validate_ping
    req = validate_ping(body)
    return 200, {"request_id": req["request_id"], "service": "rag", "echo": req["message"]}
