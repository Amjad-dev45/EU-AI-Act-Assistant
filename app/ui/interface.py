"""Build the Gradio interface and connect its events to an injected RAG backend.

This module does not load credentials, create API clients, or own conversation memory.
"""

from uuid import uuid4

import gradio as gr

from .diagnostics import error_details
from .sources import build_sources_html


class GradioUI:
    """Own UI callbacks; the supplied Answerer owns the RAG pipeline."""

    def __init__(self, answerer, data_dir):
        self.answerer = answerer
        self.data_dir = data_dir

    # Read statistics without letting a reporting failure hide the answer.
    def session_summary(self, conversation_id):
        try:
            return self.answerer.monitor.summary(conversation_id, kind="interactive")
        except Exception:
            return {"error": "Session statistics could not be loaded."}

    # Handle a submitted question and convert backend results to display values.
    def ask(self, question, conversation_id, evaluate):
        try:
            result = self.answerer.answer(
                question, conversation_id=conversation_id, evaluate=evaluate,
            )
        except Exception as exc:
            details = error_details(exc)
            return (
                "⚠️ Request failed. See Error details below.", details["html"],
                {"error_id": details["error_id"], "error": details["summary"]},
                self.session_summary(conversation_id),
            )

        answer_text = result["answer"]
        if result["status"] == "insufficient_evidence":
            answer_text = "⚠️ " + answer_text
        elif result["status"] == "needs_clarification":
            answer_text = "❓ " + answer_text

        sources_html = (
            build_sources_html(result["citations"], self.data_dir)
            if result["citations"] else ""
        )
        stats = result["telemetry"]
        request_stats = {
            "request_id": stats["request_id"],
            "saved": result["telemetry_persisted"],
            "usage": stats["usage"],
            "estimated_cost": stats["cost"],
            "timings_ms": stats["timings_ms"],
            "metrics": stats["metrics"],
            "reranking": stats["metadata"].get("reranking"),
            "retrieval_candidates": stats.get("retrieval_candidates", []),
            "final_passages": stats["retrieval"],
            "semantic_evaluation": stats["semantic_evaluation"],
        }
        return answer_text, sources_html, request_stats, self.session_summary(conversation_id)

    # Declare components and event wiring without launching a server.
    def build(self):
        with gr.Blocks(title="EU AI Act Assistant") as demo:
            gr.Markdown("# EU AI Act Assistant")
            question = gr.Textbox(label="Your question")
            self.conversation = gr.State(value=lambda: str(uuid4()))
            evaluate = gr.Checkbox(
                value=False,
                label="Evaluate answer quality (one additional paid model call)",
            )
            button = gr.Button("Ask")
            answer = gr.Markdown()
            sources = gr.HTML()
            with gr.Accordion("Usage and answer quality", open=False):
                gr.Markdown(
                    "Quality scores are model estimates. Valid citation labels alone "
                    "do not prove support. Session totals group requests; "
                    "they do not add conversation memory."
                )
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

            # Session IDs remain isolated; serialize submissions and resets.
            inputs = [question, self.conversation, evaluate]
            outputs = [answer, sources, request_stats, session_stats]
            button.click(self.ask, inputs=inputs, outputs=outputs,
                         concurrency_limit=1, concurrency_id="rag")
            question.submit(self.ask, inputs=inputs, outputs=outputs,
                            concurrency_limit=1, concurrency_id="rag")
            reset.click(
                lambda: (str(uuid4()), "", "", "", {}, {}),
                outputs=[self.conversation, question, answer, sources, request_stats, session_stats],
                concurrency_limit=1, concurrency_id="rag",
            )
        return demo
