"""Where the RAG service reads its data.

Source text (for exact quotes) always comes from the committed synthetic fixture.
Chunks come from the prepared demo artifacts if present, else the committed fixture.
"""
import json
import os
from pathlib import Path

from contracts.httpkit import log

FIXTURE_DIR = Path(os.environ.get("RAG_FIXTURE_DIR", Path(__file__).parent / "fixtures"))
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "/data/scaffold-demo"))


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_source() -> dict:
    return _read_json(FIXTURE_DIR / "demo_source.json")


def load_chunks() -> tuple[str, list[dict]]:
    """Returns ("prepared" | "fixture", chunks). Re-read per request: files are tiny
    and this lets `prepare` output take effect without restarting the service."""
    prepared, manifest = DATA_DIR / "chunks.json", DATA_DIR / "manifest.json"
    if prepared.is_file() and manifest.is_file():
        try:
            chunks = _read_json(prepared)["chunks"]
            if isinstance(chunks, list) and chunks:
                return "prepared", chunks
        except (OSError, ValueError, KeyError) as e:
            log("rag", f"prepared artifacts unreadable ({e!r}); using fixture chunks")
    return "fixture", _read_json(FIXTURE_DIR / "demo_chunks.json")["chunks"]
