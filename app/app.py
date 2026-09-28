import base64
import html
import io
import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import Retriever
from answering import Answerer
from highlight import render_page

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
retriever = Retriever("data", client)
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
                image = render_page(page, c["text"])
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


def ask(question):
    if not question.strip():
        return "Please type a question.", ""

    try:
        result = answerer.answer(question)
    except Exception as e:
        return f"⚠️ Something went wrong: {e}", ""

    answer_text = result["answer"]
    if result["status"] == "insufficient_evidence":
        answer_text = "⚠️ " + answer_text
    elif result["status"] == "needs_clarification":
        answer_text = "❓ " + answer_text

    sources_html = build_sources_html(result["citations"]) if result["citations"] else ""
    return answer_text, sources_html


with gr.Blocks(title="EU AI Act Assistant") as demo:
    gr.Markdown("# EU AI Act Assistant")
    question = gr.Textbox(label="Your question")
    button = gr.Button("Ask")
    answer = gr.Markdown()
    sources = gr.HTML()

    gr.Examples(
        examples=[
            "What is the maximum fine for using a prohibited AI practice?",
            "Does a chatbot have to tell users that they are talking to an AI?",
            "What's the weather in Paris today?",
            "What are the obligations?",
        ],
        inputs=question,
    )

    button.click(ask, inputs=question, outputs=[answer, sources])
    question.submit(ask, inputs=question, outputs=[answer, sources])

demo.launch()