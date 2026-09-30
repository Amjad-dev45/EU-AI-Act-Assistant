"""RAG service entry: routes only."""
import os

from contracts.httpkit import serve_forever

from . import handlers

ROUTES = {
    ("GET", "/health"): handlers.health,
    ("POST", "/v1/search"): handlers.search,
    ("POST", "/v1/quote"): handlers.quote,
    ("POST", "/v1/ping"): handlers.ping,  # connectivity test
}


def main() -> None:
    serve_forever("rag", ROUTES, "0.0.0.0", int(os.environ.get("RAG_PORT", "8002")))
