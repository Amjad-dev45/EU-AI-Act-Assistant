"""Dummy processing job (temporary container, same image as RAG).

Reads the synthetic fixture and writes MOCK chunks + a manifest to the demo folder.
No PDF extraction, no embeddings, no fake .faiss/.npy files.
Later replacement: PERSON 1 (extract/chunk) and PERSON 2 (embed/index) fill this in.
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

from . import store
from .chunking import chunk_documents


def _write_atomic(path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> None:
    source_path = store.FIXTURE_DIR / "demo_source.json"
    raw = source_path.read_bytes()
    chunks = chunk_documents(json.loads(raw))
    store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_atomic(store.DATA_DIR / "chunks.json", {"mock": True, "chunks": chunks})
    _write_atomic(store.DATA_DIR / "manifest.json", {
        "mock": True,
        "kind": "scaffold-demo",
        "warning": "MOCK artifacts from a synthetic fixture. No embeddings and no vector index exist.",
        "source": source_path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_count": len(chunks),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"[prepare] wrote {len(chunks)} mock chunks to {store.DATA_DIR}", flush=True)
