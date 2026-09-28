"""Launch the existing web UI with ``python app/app.py`` from the repository.

Importing this entry point does not load .env, create a client, or open the index.
"""

import sys
from pathlib import Path

# Support direct execution while keeping package imports unambiguous.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def create_app():
    """Load local settings, construct the backend, then build the Gradio UI."""
    from dotenv import load_dotenv
    from openai import OpenAI

    from app.settings import REPO_DIR, data_directory
    from app.rag.retrieval import Retriever
    from app.rag.answering import Answerer
    from app.ui.interface import GradioUI

    # Load credentials only when the application is explicitly constructed.
    load_dotenv(REPO_DIR / ".env")
    data_dir = data_directory()
    client = OpenAI()
    answerer = Answerer(Retriever(data_dir, client), client)
    return GradioUI(answerer, data_dir).build()


if __name__ == "__main__":
    create_app().launch()
