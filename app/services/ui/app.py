"""PLACEHOLDER (Person 1): Gradio UI. The browser only talks to this server;
this server calls the Agent by service name (AGENT_URL)."""
import os
import time

import gradio as gr

from contracts.errors import ApiError
from contracts.httpkit import call_json
from contracts.schemas import new_request_id

AGENT_URL = os.environ.get("AGENT_URL", "http://agent:8001")
# Must be larger than the Agent's own RAG timeout, so the Agent can report first.
TIMEOUT_S = float(os.environ.get("UI_TIMEOUT_S", "8"))


def ping(message: str, forward: bool):
    """Send a message UI -> Agent (-> RAG) and show every hop."""
    rid = new_request_id("ping")
    started = time.perf_counter()
    try:
        status, body = call_json("POST", f"{AGENT_URL}/v1/ping",
                                 {"request_id": rid, "message": message or "hello", "forward": forward},
                                 TIMEOUT_S, "agent")
    except ApiError as e:  # Agent itself is unreachable / too slow
        status, body = e.status, {"ok": False, "error": {"code": e.code, "message": e.message},
                                  "chain": [{"service": "agent", "status": "error", "error": e.code}]}
    ms = round((time.perf_counter() - started) * 1000, 2)
    body["chain"] = [{"service": "ui", "status": "ok"}] + body.get("chain", [])
    body["round_trip_ms"] = ms
    ok = status == 200 and body.get("ok")
    hops = " → ".join(f"{h['service']} {'✅' if h['status'] == 'ok' else '❌'}" for h in body["chain"])
    summary = f"{'✅ OK' if ok else '❌ FAILED'} in {ms} ms\n\n{hops}"
    if not ok and "error" in body:
        summary += f"\n\n`{body['error']['code']}`: {body['error']['message']}"
    return summary, body


def main() -> None:
    with gr.Blocks(title="Scaffold UI") as demo:
        gr.Markdown(f"# Scaffold UI\nContainer connectivity test. Agent at `{AGENT_URL}`.")
        msg = gr.Textbox(label="Message", value="hello from the UI")
        with gr.Row():
            b1 = gr.Button("Ping Agent (UI → Agent)")
            b2 = gr.Button("Ping Agent → RAG (UI → Agent → RAG)", variant="primary")
        summary = gr.Markdown()
        raw = gr.JSON(label="Raw response")
        b1.click(lambda m: ping(m, False), msg, [summary, raw], api_name="ping_agent")
        b2.click(lambda m: ping(m, True), msg, [summary, raw], api_name="ping_chain")
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", "7860")))
