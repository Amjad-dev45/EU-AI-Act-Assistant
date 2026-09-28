import gradio as gr

def fake_backend(question):
    # Stands in for your teammates' function. It returns the 3 things you agreed on.
    if "weather" in question.lower():
        return {
            "answer": "I don't have information about this.",
            "answerable": False,
            "sources": [],
        }
    return {
        "answer": "Up to EUR 35 000 000 or 7 % of worldwide turnover (Article 99).",
        "answerable": True,
        "sources": [
            {"source": "OJ_L_202401689_EN_TXT.pdf", "page": 115, "text": "Non-compliance with the prohibition of the AI practices..."}
        ],
    }

def ask(question):
    if not question.strip():
        return "Please type a question.", ""

    result = fake_backend(question)

    answer_text = result["answer"]
    if not result["answerable"]:
        answer_text = "⚠️ " + answer_text

    sources_text = ""
    if result["sources"]:
        sources_text = "### Sources\n\n"
        for s in result["sources"]:
            sources_text += f"**{s['source']}, page {s['page']}**\n\n> {s['text']}\n\n"

    return answer_text, sources_text

with gr.Blocks(title="EU AI Act Assistant") as demo:
    gr.Markdown("# EU AI Act Assistant")
    question = gr.Textbox(label="Your question")
    button = gr.Button("Ask")
    answer = gr.Markdown()
    sources = gr.Markdown()

    gr.Examples(
        examples=[
            "What is the maximum fine for using a prohibited AI practice?",
            "Does a chatbot have to tell users that they are talking to an AI?",
            "What's the weather in Paris today?",
        ],
        inputs=question,
    )

    button.click(ask, inputs=question, outputs=[answer, sources])
    question.submit(ask, inputs=question, outputs=[answer, sources])

demo.launch()