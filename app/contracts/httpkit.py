"""Transport only: a tiny JSON HTTP server and client on the standard library.

Business logic lives elsewhere; handlers passed in here take a parsed body
and return `(status, dict)`, or raise `ApiError`.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .errors import (
    ApiError, BAD_GATEWAY, DEPENDENCY_TIMEOUT, DEPENDENCY_UNAVAILABLE, INTERNAL_ERROR,
    INVALID_REQUEST, METHOD_NOT_ALLOWED, NOT_FOUND, PAYLOAD_TOO_LARGE,
)
from .schemas import new_request_id

MAX_BODY_BYTES = 64 * 1024


def log(service: str, message: str) -> None:
    print(f"[{service}] {message}", file=sys.stderr, flush=True)


# --- Server -----------------------------------------------------------------
def make_server(service: str, routes: dict, host: str, port: int) -> ThreadingHTTPServer:
    """routes: {("GET"|"POST", "/path"): handler(body_or_None) -> (status, dict)}"""
    paths = {p for (_, p) in routes}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 15  # bounded socket time per connection

        def log_message(self, fmt, *args):  # silence default noisy logging
            pass

        def _send(self, status: int, payload: dict, request_id: str | None) -> None:
            payload.setdefault("request_id", request_id or new_request_id("srv"))
            payload.setdefault("mock", True)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, method: str) -> None:
            started = time.perf_counter()
            request_id, status = None, 500
            try:
                path = self.path.split("?", 1)[0]
                if path not in paths:
                    raise ApiError(404, NOT_FOUND, f"Unknown endpoint {path}.")
                handler = routes.get((method, path))
                if handler is None:
                    raise ApiError(405, METHOD_NOT_ALLOWED, f"{method} not allowed on {path}.")
                body = None
                if method == "POST":
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_BODY_BYTES:
                        raise ApiError(413, PAYLOAD_TOO_LARGE, "Request body too large.")
                    raw = self.rfile.read(length) if length else b""
                    try:
                        body = json.loads(raw or b"null")
                    except ValueError:
                        raise ApiError(400, INVALID_REQUEST, "Body is not valid JSON.")
                    if isinstance(body, dict):
                        request_id = body.get("request_id") if isinstance(body.get("request_id"), str) else None
                status, payload = handler(body)
                if isinstance(payload, dict) and payload.get("request_id"):
                    request_id = payload["request_id"]
            except ApiError as e:
                status = e.status
                payload = {**e.extra, "error": {"code": e.code, "message": e.message}}
            except Exception as e:  # never leak a stack trace to the caller
                log(service, f"unhandled error: {e!r}")
                status = 500
                payload = {"error": {"code": INTERNAL_ERROR, "message": "Unexpected server error."}}
            self._send(status, payload, request_id)
            ms = (time.perf_counter() - started) * 1000
            log(service, f"{method} {self.path} -> {status} {ms:.1f}ms request_id={request_id}")

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_forever(service: str, routes: dict, host: str, port: int) -> None:
    server = make_server(service, routes, host, port)
    log(service, f"listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# --- Client -----------------------------------------------------------------
def call_json(method: str, url: str, payload: dict | None, timeout: float, dependency: str) -> tuple[int, dict]:
    """Bounded-timeout JSON call. Returns (status, body) for any HTTP response.

    Connection failures become ApiError(503/504) naming the `dependency`, so a
    downed service is reported promptly instead of being hidden.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except TimeoutError:
        raise ApiError(504, DEPENDENCY_TIMEOUT, f"`{dependency}` did not answer within {timeout:g}s.")
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, TimeoutError):
            raise ApiError(504, DEPENDENCY_TIMEOUT, f"`{dependency}` did not answer within {timeout:g}s.")
        raise ApiError(503, DEPENDENCY_UNAVAILABLE, f"`{dependency}` is unreachable ({reason}).")
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        raise ApiError(502, BAD_GATEWAY, f"`{dependency}` returned a non-JSON response (HTTP {status}).")
    return status, body


def fire_and_forget(url: str, payload: dict, timeout: float = 1.0) -> None:
    """Best-effort POST in a background thread; failures are only logged."""
    def _run():
        try:
            call_json("POST", url, payload, timeout, "agent")
        except ApiError as e:
            log("metrics", f"could not report to agent: {e.code}")
    threading.Thread(target=_run, daemon=True).start()
