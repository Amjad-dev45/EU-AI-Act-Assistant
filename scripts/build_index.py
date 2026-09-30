"""Build the saved FAISS index: uv run scripts/build_index.py."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag.vector_index import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="Replace index artifacts; reuse cached embeddings for unchanged text")
    args = parser.parse_args()
    result = build_index(ROOT / "data", rebuild=args.rebuild)
    print("Reused existing index." if result["reused"] else "Built and saved index.")
    print(f"Chunks: {result['chunk_count']}; dimensions: {result['dimensions']}")
    print(f"Model: {result['model']}")
    print(f"New embedding tokens this run: {result['embedding_tokens']}")
    print(f"Saved artifacts: {ROOT / 'data'}")


if __name__ == "__main__":
    main()
