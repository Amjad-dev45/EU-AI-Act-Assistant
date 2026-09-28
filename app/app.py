import base64
import html
import io
import os
import sys
from pathlib import Path
from uuid import uuid4
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

REPO_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_DIR))
from app.retrieval import Retriever
from app.answering import Answerer
from app.highlight import render_page
from app.diagnostics import error_details

load_dotenv(REPO_DIR / ".env")
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(REPO_DIR / "data")))
if not DATA_DIR.is_absolute():
    DATA_DIR = REPO_DIR / DATA_DIR
DATA_DIR = DATA_DIR.resolve()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
retriever = Retriever(DATA_DIR, client)
answerer = Answerer(retriever, client)


def to_data_uri(image):
    """Turn a PIL image into text that can be embedded in the page (no file needed)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_sources_html(citations):
    parts = ["<h3>Sources</h3>"]
    for c in citations:
        pages = ", ".join(str(p) for p in c["pages"])
        title = f"[{c['citation_id']}] {c['source']}, page {pages}"
        body = ""
        for page in c["pages"]:
            try:
                source_path = (DATA_DIR / "sources" / c["source"]).resolve()
                if not source_path.is_relative_to((DATA_DIR / "sources").resolve()):
                    raise ValueError("Source is outside the sources folder")
                image = render_page(page, c["text"], pdf_path=str(source_path))
                body += (f"<p><b>Page {page}</b></p>"
                         f"<img src='{to_data_uri(image)}' style='max-width:100%; border-radius:6px;'>")
            except Exception as e:
                print("highlight failed:", e)
        parts.append(
            "<details style='margin-bottom:10px; padding:8px; border:1px solid #555; border-radius:6px;'>"
            f"<summary style='cursor:pointer; font-weight:600;'>{html.escape(title)}</summary>"
            f"{body}</details>"
        )
    return "".join(parts)


def session_summary(conversation_id):
    try:
        return answerer.monitor.summary(conversation_id, kind="interactive")
    except Exception:
        return {"error": "Session statistics could not be loaded."}


def ask(question, conversation_id, evaluate):
    try:
        result = answerer.answer(question, conversation_id=conversation_id, evaluate=evaluate)
    except Exception as e:
        details = error_details(e)
        return ("⚠️ Request failed. See Error details below.", details["html"],
                {"error_id": details["error_id"], "error": details["summary"]},
                session_summary(conversation_id))

    answer_text = result["answer"]
    if result["status"] == "insufficient_evidence":
        answer_text = "⚠️ " + answer_text
    elif result["status"] == "needs_clarification":
        answer_text = "❓ " + answer_text

    sources_html = build_sources_html(result["citations"]) if result["citations"] else ""
    stats = result["telemetry"]
    request_stats = {"request_id": stats["request_id"], "saved": result["telemetry_persisted"],
                     "usage": stats["usage"], "estimated_cost": stats["cost"],
                     "timings_ms": stats["timings_ms"], "metrics": stats["metrics"],
                     "semantic_evaluation": stats["semantic_evaluation"]}
    return answer_text, sources_html, request_stats, session_summary(conversation_id)


with gr.Blocks(title="EU AI Act Assistant") as demo:
    gr.Markdown("# EU AI Act Assistant")
    question = gr.Textbox(label="Your question")
    conversation = gr.State(value=lambda: str(uuid4()))
    evaluate = gr.Checkbox(value=False, label="Evaluate answer quality (one additional paid model call)")
    button = gr.Button("Ask")
    answer = gr.Markdown()
    sources = gr.HTML()
    with gr.Accordion("Usage and answer quality", open=False):
        gr.Markdown("Quality scores are model estimates. Valid citation labels alone do not prove support. "
                    "Session totals group requests; they do not add conversation memory.")
        request_stats = gr.JSON(label="This request")
        session_stats = gr.JSON(label="This session")
        reset = gr.Button("Start new session")

    gr.Examples(
        examples=[
            "What is the maximum fine for using a prohibited AI practice?",
            "Does a chatbot have to tell users that they are talking to an AI?",
            "What's the weather in Paris today?",
            "What are the obligations?",
        ],
        inputs=question,
    )

    inputs = [question, conversation, evaluate]
    outputs = [answer, sources, request_stats, session_stats]
    button.click(ask, inputs=inputs, outputs=outputs, concurrency_limit=1, concurrency_id="rag")
    question.submit(ask, inputs=inputs, outputs=outputs, concurrency_limit=1, concurrency_id="rag")
    reset.click(lambda: (str(uuid4()), "", "", "", {}, {}),
                outputs=[conversation, question, answer, sources, request_stats, session_stats],
                concurrency_limit=1, concurrency_id="rag")

if __name__ == "__main__":
    demo.launch()
