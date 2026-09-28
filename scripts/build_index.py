"""Embed chunks and build a reusable FAISS search index."""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "text-embedding-3-small"
BATCH_SIZE = 50


# Cache writes are atomic so interrupted runs can reuse completed batches.
def save_json(path, value):
    """Write through a temporary file to avoid partial cache files."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)



# Command-line inputs, validation, execution, and output.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="Generate missing embeddings and build the index")
    # Parse options before opening files or making API requests.
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    raw = (data_dir / "chunks.json").read_bytes()
    chunks = json.loads(raw)
    if not chunks:
        parser.error("chunks.json is empty")
    texts = [chunk["text"] for chunk in chunks]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        parser.error("Every chunk must contain nonempty text")

    # A changed chunk file gets a different cache directory.
    fingerprint = hashlib.sha256(MODEL.encode() + b"\0" + raw).hexdigest()
    cache_dir = data_dir / "embedding_cache" / fingerprint
    encoding = tiktoken.encoding_for_model(MODEL)
    token_counts = [len(encoding.encode(text, disallowed_special=())) for text in texts]
    if max(token_counts) > 8191:
        parser.error("A chunk exceeds the supported token limit")
    pending_tokens = sum(
        sum(token_counts[start:start + BATCH_SIZE])
        for start in range(0, len(chunks), BATCH_SIZE)
        if not (cache_dir / f"{start:06d}.json").exists()
    )
    print(f"Chunks: {len(chunks)}")
    print(f"Total input tokens: {sum(token_counts):,}")
    print(f"Tokens without a cached batch: {pending_tokens:,}")
    # Preview exits before credentials or an API client are loaded.
    if not args.run:
        print("Preview only. Add --run to generate embeddings.")
        return

    # Load the repository's .env, regardless of the working directory.
    repo_dir = Path(__file__).resolve().parents[1]
    # Load credentials only for the requested runtime operation.
    load_dotenv(repo_dir / ".env")
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = None
    embeddings = []
    billed_tokens = 0
    # Reuse matching cached batches; embed only batches missing from disk.
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        cache_file = cache_dir / f"{start:06d}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            if client is None:
                client = OpenAI()
            response = client.embeddings.create(
                model=MODEL, input=batch, encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("Unexpected embedding response order")
            cached = {
                "vectors": [item.embedding for item in ordered],
                "tokens": response.usage.total_tokens,
            }
            save_json(cache_file, cached)
            billed_tokens += cached["tokens"]
        vectors = np.asarray(cached["vectors"], dtype="float32")
        if (vectors.shape != (len(batch), 1536)
                or not np.isfinite(vectors).all()
                or np.any(np.linalg.norm(vectors, axis=1) == 0)):
            raise ValueError(f"Invalid embedding batch: {cache_file}")
        embeddings.extend(cached["vectors"])
        print(f"Ready: {len(embeddings)}/{len(chunks)}")

    # Normalize vectors and persist the matching index plus integrity metadata.
    vectors = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    # Keep vectors so the index can be rebuilt without API calls.
    vectors_file = data_dir / "embeddings.npy"
    with vectors_file.with_suffix(".tmp").open("wb") as handle:
        np.save(handle, vectors)
    vectors_file.with_suffix(".tmp").replace(vectors_file)
    index_file = data_dir / "my_index.faiss"
    temporary_index = data_dir / "my_index.faiss.tmp"
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(index_file)
    save_json(data_dir / "index_metadata.json", {
        "model": MODEL,
        "fingerprint": fingerprint,
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "chunk_count": len(chunks),
        "dimensions": vectors.shape[1],
        "similarity": "cosine",
    })
    print(f"Input tokens billed during this run: {billed_tokens:,}")
    print(f"Saved index: {index_file}")


# Running this file explicitly invokes the command.
if __name__ == "__main__":
    main()
