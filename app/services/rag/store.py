"""Read the prepared corpus used by the production retrieval pipeline."""

import json
import os
from pathlib import Path

from contracts.errors import ApiError, DEPENDENCY_UNAVAILABLE


DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
# Kept for the separately invoked legacy scaffold preparation job. Production
# handlers never read this directory or fall back to its sample data.
FIXTURE_DIR = Path(os.environ.get("RAG_FIXTURE_DIR", Path(__file__).parent / "fixtures"))


def load_source() -> list[dict]:
    """Return extracted pages; missing production data never selects a fixture."""
    try:
        with (DATA_DIR / "corpus.json").open(encoding="utf-8") as source:
            pages = json.load(source)
        if not isinstance(pages, list) or not pages:
            raise ValueError("Empty or invalid corpus")
        seen = set()
        for page in pages:
            if (not isinstance(page, dict)
                    or not isinstance(page.get("source"), str) or not page["source"].strip()
                    or type(page.get("page")) is not int or page["page"] < 1
                    or not isinstance(page.get("text"), str)):
                raise ValueError("Invalid corpus page")
            identity = (page["source"], page["page"])
            if identity in seen:
                raise ValueError("Duplicate corpus page")
            seen.add(identity)
        return pages
    except (OSError, ValueError, TypeError):
        raise ApiError(503, DEPENDENCY_UNAVAILABLE,
                       "The extracted corpus is unavailable or invalid. Prepare corpus.json in RAG_DATA_DIR.") from None
