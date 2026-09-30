"""Shared OpenAI client configuration for embeddings and answer generation."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def create_client():
    """Read the project key only when an API call is needed; never print it."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in the project's .env file.")
    return OpenAI(timeout=60, max_retries=2)
