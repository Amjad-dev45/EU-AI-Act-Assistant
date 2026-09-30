"""Gradio chat: browser → UI → Agent; the UI never calls RAG or a model directly."""

import os

from .chat import new_chat, ping, submit_chat
from .tool_calls import TOOL_CSS, format_tool_calls
from .context_usage import CONTEXT_CSS, format_context
from .rag_demo import RAG_CSS, format_rag_demo
from .demo_guide import DEMO_GUIDE, DEMO_QUESTIONS


def create_app():
    """Build the screen without starting a server or making service calls."""
    import gradio as gr

    with gr.Blocks(title="EU AI Act Assistant") as demo:
        gr.Markdown("# EU AI Act Assistant\nAsk about the corpus, then ask follow-up questions.")
        state = gr.State({"history": [], "display": []})
        with gr.Row():
            with gr.Column(scale=2):
                conversation = gr.Chatbot(label="Conversation", height=520,
                                          buttons=["copy"], sanitize_html=True)
                context_bar = gr.HTML(format_context(), css_template=CONTEXT_CSS,
                                      elem_id="context-usage", js_on_load=None)
                message = gr.Textbox(label="Your question", lines=2,
                                     placeholder="What obligations does Article 4 introduce?")
                injection_test = gr.Checkbox(value=False, label="Injection test (demo only)",
                                             info="Adds a temporary malicious source to FAISS + BM25 for this request; the saved corpus stays unchanged.")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("New chat")
                status = gr.Markdown("Remembers up to three recent exchanges in this tab.")
            with gr.Column(scale=1):
                gr.Markdown("### Sources")
                sources = gr.Markdown("Sources will appear here.")
                gr.Markdown("### Tool calls")
                trajectory = gr.HTML(format_tool_calls([]), css_template=TOOL_CSS,
                                     elem_id="tool-calls", js_on_load=None)
                with gr.Accordion("Full results and usage", open=False):
                    details = gr.JSON(label="Latest response")

        with gr.Tabs():
            with gr.Tab("RAG walkthrough"):
                rag_walkthrough = gr.HTML(format_rag_demo([]), css_template=RAG_CSS,
                                          elem_id="rag-walkthrough", js_on_load=None)
            with gr.Tab("Demo guide"):
                gr.Examples(examples=DEMO_QUESTIONS, inputs=message,
                            label="Existing benchmark questions — select, then Send")
                gr.Markdown(DEMO_GUIDE)

        with gr.Accordion("Full retrieval details", open=False):
            retrieval_details = gr.Markdown()

        outputs = [message, conversation, state, status, sources, trajectory, details, retrieval_details, context_bar, rag_walkthrough]
        # One shared event prevents enter/click double submission. Reset uses the
        # same queue so an in-flight answer cannot repopulate a cleared chat.
        gr.on([message.submit, send.click], submit_chat, [message, state, injection_test], outputs,
              trigger_mode="once", concurrency_limit=1, concurrency_id="chat",
              api_name="chat", show_progress="full")
        clear.click(lambda: (*new_chat(), False), None, [*outputs, injection_test], trigger_mode="once",
                    concurrency_limit=1, concurrency_id="chat", api_name="new_chat")

        with gr.Accordion("Connection check", open=False):
            connection = gr.Button("Check connection")
            connection_status = gr.Markdown()
            connection.click(ping, None, connection_status, api_name="connection_check")

    return demo.queue(default_concurrency_limit=1)


def main():
    create_app().launch(server_name="0.0.0.0",
                        server_port=int(os.environ.get("UI_PORT", "7860")))


if __name__ == "__main__":
    main()
