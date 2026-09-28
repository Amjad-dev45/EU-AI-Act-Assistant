"""Retrieve relevant passages from the saved FAISS index."""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import tiktoken

MODEL = "text-embedding-3-small"


class Retriever:
    def __init__(self, data_dir, client):
        self.client = client
        data_dir = Path(data_dir)
        raw = (data_dir / "chunks.json").read_bytes()
        self.chunks = json.loads(raw)
        metadata = json.loads((data_dir / "index_metadata.json").read_text())
        if hashlib.sha256(raw).hexdigest() != metadata["chunks_sha256"]:
            raise ValueError("Chunks have changed since indexing")
        if metadata["model"] != MODEL:
            raise ValueError("Index uses a different embedding model")
        self.index = faiss.read_index(str(data_dir / "my_index.faiss"))
        if not (self.index.ntotal == len(self.chunks) == metadata["chunk_count"]):
            raise ValueError("Index and chunk counts do not match")
        if self.index.d != metadata["dimensions"]:
            raise ValueError("Index dimensions do not match metadata")
        self.encoding = tiktoken.encoding_for_model(MODEL)

    def retrieve(self, question, k=5):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        tokens = self.encoding.encode(question, disallowed_special=())
        if len(tokens) > 8191:
            raise ValueError("Question is too long")
        response = self.client.embeddings.create(
            model=MODEL, input=[question], encoding_format="float",
        )
        vector = np.asarray([response.data[0].embedding], dtype="float32")
        if (vector.shape != (1, self.index.d)
                or not np.isfinite(vector).all()
                or np.linalg.norm(vector) == 0):
            raise ValueError("Invalid question embedding")
        faiss.normalize_L2(vector)
        scores, positions = self.index.search(vector, min(k, self.index.ntotal))
        passages = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            passages.append({
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "pages": chunk["pages"],
                "score": float(score),
            })
        return {
            "question": question,
            "passages": passages,
            "embedding_tokens": response.usage.total_tokens,
        }
