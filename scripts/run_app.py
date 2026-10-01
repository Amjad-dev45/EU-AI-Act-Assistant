"""Start Compose using the same .env parser as the working local RAG scripts.

Usage: uv run scripts/run_app.py [up -d --build | logs -f | down]
The original .env is never rewritten or included in a Docker image.
"""
import os
from pathlib import Path
import subprocess
import sys

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"OPENAI_API_KEY", "web_hook", "SCAFFOLD_UI_PORT", "UI_TIMEOUT_S", "RAG_TIMEOUT_S",
            "RAG_MIN_RERANK_SCORE", "LOCAL_UID", "LOCAL_GID"}


def main():
    # Only application settings reach Compose; never print their values.
    environment = os.environ.copy()
    for key, value in dotenv_values(ROOT / ".env", interpolate=False).items():
        if key in SETTINGS and value is not None:
            environment.setdefault(key, value)
    arguments = sys.argv[1:] or ["up", "-d", "--build"]
    # Disable Compose's stricter implicit .env parser while retaining its YAML
    # interpolation from the already-loaded environment.
    return subprocess.run(["docker", "compose", "--env-file", os.devnull, *arguments],
                          cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
