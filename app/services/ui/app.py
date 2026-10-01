"""Gradio chat: browser → UI → Agent; the UI never calls RAG or a model directly."""

import os
from contracts.schemas import MAX_HISTORY_MESSAGES

from .chat import ping
from .editing import edit_with_sources
from .security_notice import notify_injection
from .sharing import DEFAULT_MESSAGE, send_to_teams
from .source_viewer import (submit_with_sources, clear_with_sources, open_source,
                            SOURCE_LAYOUT_CSS, PREVIEW_CSS, PREVIEW_JS)
from .tool_calls import TOOL_CSS, format_tool_calls
from .retrieval_trace import RETRIEVAL_CSS, format_retrieval
from .context_usage import CONTEXT_CSS, CONTEXT_LAYOUT_CSS, format_context
from .rag_demo import RAG_CSS, format_rag_demo
from .demo_guide import DEMO_GUIDE, DEMO_QUESTIONS
from .demo_cases import COPY_CSS, COPY_JS, format_demo_cases


def create_app():
    """Build the screen without starting a server or making service calls."""
    import gradio as gr

    with gr.Blocks(title="EU AI Act Assistant") as demo:
        gr.Markdown("# EU AI Act Assistant\nAsk about the corpus, then ask follow-up questions.")
        state = gr.State({"history": [], "display": []})
        share_state = gr.State({})
        source_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=2):
                # Group removes the layout gap between the chat and its meter.
                with gr.Group(elem_id="conversation-context"):
                    conversation = gr.Chatbot(label="Conversation", height=520,
                                              buttons=["copy"], editable="user", sanitize_html=True,
                                              elem_id="conversation")
                    context_bar = gr.HTML(format_context(), css_template=CONTEXT_CSS,
                                          elem_id="context-usage", js_on_load=None,
                                          padding=False,
                                          head=f"<style>{CONTEXT_LAYOUT_CSS}{SOURCE_LAYOUT_CSS}</style>")
                message = gr.Textbox(label="Your question", show_label=False, lines=2,
                                     elem_id="question-input",
                                     placeholder="What obligations does Article 4 introduce?")
                injection_test = gr.Checkbox(value=False, label="Injection test (demo only)",
                                             info="Adds a temporary malicious source to FAISS + BM25 for this request; the saved corpus stays unchanged.")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("New chat")
                status = gr.Markdown(f"Remembers up to {MAX_HISTORY_MESSAGES} previous messages (user and assistant). Edit a question with its pencil to replace it and remove later messages.",
                                     elem_id="chat-status")
                share_button = gr.Button("Send to Teams", interactive=False, size="sm")
                share_status = gr.Markdown(DEFAULT_MESSAGE, elem_id="teams-share-status")
            with gr.Column(scale=1):
                gr.Markdown("### Sources")
                sources = gr.Radio(choices=[], label="Click a cited page to preview its PDF",
                                   interactive=True, elem_id="source-links")
                source_preview = gr.HTML('', css_template=PREVIEW_CSS, js_on_load=PREVIEW_JS,
                                         elem_id='source-preview')
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
                gr.HTML(format_demo_cases(), css_template=COPY_CSS, js_on_load=COPY_JS,
                        elem_id="demo-copy-cases")
                with gr.Accordion("Original benchmark examples", open=False):
                    gr.Examples(examples=DEMO_QUESTIONS, inputs=message,
                                label="Existing benchmark questions — select, then Send")
                gr.Markdown(DEMO_GUIDE)

        with gr.Accordion("Full retrieval details", open=False):
            retrieval_details = gr.HTML(format_retrieval([]), css_template=RETRIEVAL_CSS,
                                        elem_id="retrieval-details", js_on_load=None)

        outputs = [message, conversation, state, status, sources, trajectory, details, retrieval_details, context_bar, rag_walkthrough]
        # One shared event prevents enter/click double submission. Reset uses the
        # same queue so an in-flight answer cannot repopulate a cleared chat.
        share_outputs = [share_state, share_button, share_status]
        preview_outputs = [source_state, source_preview]
        def submit_message(text, session, injection):
            result = submit_with_sources(text, session, injection)
            notify_injection(result[6])
            return result

        gr.on([message.submit, send.click], submit_message, [message, state, injection_test], [*outputs, *share_outputs, *preview_outputs],
              trigger_mode="once", concurrency_limit=1, concurrency_id="chat",
              api_name="chat", show_progress="full")

        def edit_message(session, injection, event: gr.EditData):
            for result in edit_with_sources(session, injection, event.index,
                                            event.previous_value, event.value):
                notify_injection(result[6])
                yield result

        conversation.edit(edit_message, [state, injection_test],
                          [*outputs, *share_outputs, *preview_outputs], trigger_mode="once",
                          concurrency_limit=1, concurrency_id="chat", api_name=False,
                          show_progress="minimal")
        gr.on([clear.click, conversation.clear], clear_with_sources, None, [*outputs, *share_outputs, injection_test, *preview_outputs], trigger_mode="once",
                    concurrency_limit=1, concurrency_id="chat", api_name="new_chat")
        def share_answer(session, response):
            result = send_to_teams(session, response)
            # Use the same outcome as the persistent status; submission does
            # not guarantee that the Teams workflow delivered the message.
            popup = gr.Info if result[1].get("value") == "Submitted to Teams" else gr.Warning
            popup(result[2], title="Teams sharing", duration=8)
            return result

        share_button.click(share_answer, [share_state, details], share_outputs, trigger_mode="once",
                           concurrency_limit=1, concurrency_id="chat", api_name=False,
                           show_progress="minimal")
        sources.input(open_source, [sources, source_state, details], source_preview,
                      concurrency_limit=1, concurrency_id='chat', trigger_mode='always_last',
                      api_name=False, show_progress='minimal')

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
