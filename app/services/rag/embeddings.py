"""OpenAI embeddings, cached by model and exact text to avoid repeat API calls."""

import hashlib
from pathlib import Path
import tempfile

import numpy as np

from .client import create_client

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 50


def checked_vectors(values, count):
    vectors = np.asarray(values, dtype="float32")
    if (vectors.shape != (count, DIMENSIONS) or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError(f"Expected {count} finite, nonzero embeddings of size {DIMENSIONS}.")
    return np.ascontiguousarray(vectors)


# Publish each cache file only after it is complete, so interruptions can resume.
def save_vector(path, vector):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, vector, allow_pickle=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def embed_texts(texts, data_dir, client=None, allow_api=True):
    """Return vectors in input order and tokens used by new API calls this run."""
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be nonempty strings.")
    cache_dir = Path(data_dir) / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = [hashlib.sha256(f"{MODEL}\0{DIMENSIONS}\0{text}".encode()).hexdigest() for text in texts]
    unique = dict(zip(keys, texts))
    cached, missing = {}, []
    for key in unique:
        path = cache_dir / f"{key}.npy"
        if path.exists():
            cached[key] = checked_vectors([np.load(path, allow_pickle=False)], 1)[0]
        else:
            missing.append(key)

    # Deduplicate identical texts, then save every successful batch immediately.
    tokens = 0
    if missing and not allow_api:
        raise ValueError(f"{len(missing)} embedding(s) missing from cache; cached-only mode makes no API calls.")
    if missing and client is None:
        client = create_client()
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        response = client.embeddings.create(
            model=MODEL, dimensions=DIMENSIONS, encoding_format="float",
            input=[unique[key] for key in batch],
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError("Embedding response did not match the requested batch.")
        vectors = checked_vectors([item.embedding for item in items], len(batch))
        tokens += response.usage.total_tokens
        for key, vector in zip(batch, vectors):
            save_vector(cache_dir / f"{key}.npy", vector)
            cached[key] = vector
        if len(texts) > 1:
            print(f"Embedded {start + len(batch)}/{len(missing)} new texts", flush=True)
    return np.stack([cached[key] for key in keys]), tokens
