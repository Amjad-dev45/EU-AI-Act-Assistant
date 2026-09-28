"""Repository paths and environment overrides shared by UI, RAG, and monitoring."""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]


def configured_path(variable, default):
    """Resolve a configured path relative to the repository, not the shell's cwd."""
    path = Path(os.getenv(variable) or default).expanduser()
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def data_directory():
    """Read RAG_DATA_DIR after the caller has loaded its .env file."""
    return configured_path("RAG_DATA_DIR", "data")
