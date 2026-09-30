"""Build a saved cosine-similarity index and retrieve chunks with their sources."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np

from .embeddings import DIMENSIONS, MODEL, embed_texts


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


# Preserve chunk order: FAISS row i must always refer to chunk i.
def load_chunks(data_dir: Path) -> tuple[list[dict], str]:
    path = Path(data_dir) / "chunks.json"
    if not path.is_file():
        raise ValueError(f"Missing {path}; run the chunking step first.")
    raw = path.read_bytes()
    try:
        chunks = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}; regenerate chunks.json.") from exc
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a nonempty list of chunks.")
    seen = set()
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {position} must be an object.")
        for field in ("chunk_id", "text", "source", "document_id"):
            if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                raise ValueError(f"Chunk {position} needs a nonempty {field} string.")
        if chunk["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
        seen.add(chunk["chunk_id"])
        pages = chunk.get("pages")
        if not isinstance(pages, list) or not pages or any(
            type(page) is not int or page <= 0 for page in pages
        ):
            raise ValueError(f"Chunk {position} needs a list of positive page numbers.")
    return chunks, _sha256(raw)


# Reject stale or mismatched artifacts before embedding any query.
def load_index(data_dir: Path):
    data_dir = Path(data_dir)
    chunks, chunks_hash = load_chunks(data_dir)
    required = ("my_index.faiss", "index_meta.json")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing index artifacts: {', '.join(missing)}; run the index builder (use --rebuild for partial output).")
    try:
        meta = json.loads((data_dir / "index_meta.json").read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid index_meta.json; rebuild with --rebuild.") from exc
    expected = {"model": MODEL, "dimensions": DIMENSIONS,
                "chunk_count": len(chunks), "chunks_sha256": chunks_hash}
    if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Index metadata does not match chunks or embedding settings; rebuild with --rebuild.")
    index_path = data_dir / "my_index.faiss"
    if meta.get("index_sha256") != _sha256(index_path.read_bytes()):
        raise ValueError("FAISS index checksum mismatch; rebuild with --rebuild.")
    try:
        index = faiss.read_index(str(index_path))
    except RuntimeError as exc:
        raise ValueError("Cannot read FAISS index; rebuild with --rebuild.") from exc
    if (not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or index.d != DIMENSIONS or index.ntotal != len(chunks)):
        raise ValueError("FAISS index type, dimensions, or chunk count is invalid; rebuild with --rebuild.")
    return index, chunks, meta


# Embed once, normalize for cosine similarity, and publish metadata last.
def build_index(data_dir: Path, client=None, rebuild=False) -> dict:
    data_dir = Path(data_dir)
    chunks, chunks_hash = load_chunks(data_dir)
    artifacts = ("embeddings.npy", "my_index.faiss", "index_meta.json")
    if not rebuild and any((data_dir / name).exists() for name in artifacts):
        _, _, meta = load_index(data_dir)
        return {**meta, "reused": True, "embedding_tokens": 0}
    vectors, token_count = embed_texts([chunk["text"] for chunk in chunks], data_dir, client=client)
    vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if vectors.shape != (len(chunks), DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError("Embedding response has invalid dimensions or nonfinite values; previous index preserved.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embedding response contains a zero vector; previous index preserved.")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(vectors)
    if load_chunks(data_dir)[1] != chunks_hash:
        raise ValueError("chunks.json changed during embedding; run the builder again.")
    meta = {"model": MODEL, "dimensions": DIMENSIONS, "chunk_count": len(chunks),
            "chunks_sha256": chunks_hash, "embedding_tokens": int(token_count)}
    with TemporaryDirectory(prefix=".index-", dir=data_dir) as temporary:
        staging = Path(temporary)
        np.save(staging / "embeddings.npy", vectors, allow_pickle=False)
        faiss.write_index(index, str(staging / "my_index.faiss"))
        meta["index_sha256"] = _sha256((staging / "my_index.faiss").read_bytes())
        (staging / "index_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        for name in artifacts:  # Publish metadata last, after both binary artifacts.
            (staging / name).replace(data_dir / name)
    return {**meta, "reused": False}


# Search only; this does not generate an answer or rebuild the index.
def search_index(index, chunks: list[dict], query_vector, k=5) -> list[dict]:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")
    if index.ntotal != len(chunks) or index.ntotal == 0:
        raise ValueError("Index and chunks must have the same nonzero count.")
    query = np.array(query_vector, dtype=np.float32, order="C", copy=True)
    if query.shape not in ((index.d,), (1, index.d)) or not np.isfinite(query).all():
        raise ValueError(f"Query embedding must contain {index.d} finite values.")
    query = query.reshape(1, index.d)
    if np.linalg.norm(query) == 0:
        raise ValueError("Query embedding cannot be a zero vector.")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, min(k, index.ntotal))
    return [{**chunks[int(position)], "score": float(score)}
            for position, score in zip(indices[0], scores[0])]
