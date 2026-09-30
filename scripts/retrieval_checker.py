"""Inspect retrieval: uv run scripts/retrieval_checker.py --question-number 1."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag.retrieval import retrieve
from evaluation.metrics import first_expected_rank, retrieval_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="A question you wrote for the corpus")
    parser.add_argument("--question-number", type=int, help="Use a 1-based entry from data/questions.json")
    parser.add_argument("-k", type=int, default=5, help="Number of chunks to display (default: 5)")
    parser.add_argument("--candidates", type=int, default=20, help="Candidates per search and after fusion (default: 20)")
    parser.add_argument("--no-hybrid", action="store_true", help="Rerank FAISS results without BM25 keyword search")
    parser.add_argument("--no-rerank", action="store_true", help="Use the vector-only baseline")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k must be positive.")
    if args.candidates < 1 or (not args.no_rerank and args.candidates < args.k):
        parser.error("--candidates must be positive and at least -k when reranking.")
    if bool(args.question) == (args.question_number is not None):
        parser.error("Supply either a question or --question-number.")

    # Student-authored questions stay untouched; their source/page pairs are ground truth.
    question, expected = args.question, None
    path = ROOT / "data/questions.json"
    questions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if args.question_number is not None:
        if not 1 <= args.question_number <= len(questions):
            parser.error(f"Choose a question number from 1 to {len(questions)}.")
        entry = questions[args.question_number - 1]
        question, expected = entry["question"], retrieval_labels(entry)
    else:
        for entry in questions:
            if entry["question"] == question:
                expected = retrieval_labels(entry)
                break
    if not question.strip():
        parser.error("Question must not be blank.")

    result = retrieve(question, ROOT / "data", args.k, args.candidates,
                      use_rerank=not args.no_rerank, use_hybrid=not args.no_hybrid)
    hits = result["hits"]
    mode = "Vector-only baseline"
    if result["reranked"]:
        search = "hybrid (FAISS + BM25 → RRF)" if result["hybrid"] else "vector"
        mode = f"Reranked {result['candidate_count']} {search} candidates"
    print(f"Question: {question}\n{mode}; showing {len(hits)} results.")
    for rank, hit in enumerate(hits, 1):
        pages = ", ".join(map(str, hit["pages"]))
        score = f"cosine {hit['score']:.4f}"
        if "bm25_score" in hit:
            score += f" | BM25 {hit['bm25_score']:.4f}"
        if "rrf_score" in hit:
            score += f" | RRF {hit['rrf_score']:.4f}"
        if "rerank_score" in hit:
            score += f" | reranker {hit['rerank_score']:.4f}"
        print(f"\n{rank}. {hit['source']} | pages {pages} | {score}")
        print(f"Chunk: {hit['chunk_id']}\n{hit['text']}")
    if expected:
        found = {(hit["source"], page) for hit in hits for page in hit["pages"]}
        print("\nExpected pages:")
        for item in expected:
            status = "HIT" if (item["source"], item["page"]) in found else "MISS"
            print(f"  {status}: {item['source']} page {item['page']}")
        rank = first_expected_rank(hits, expected)
        print(f"First expected-page rank: {rank or 'not in results'}")
        if result["reranked"]:
            before = first_expected_rank(result["vector_hits"], expected)
            print(f"Vector-only rank before reranking: {before or 'not in results'}")
        print(f"Reciprocal rank for this question: {1 / rank if rank else 0:.3f}")
    else:
        print("\nRetrieval metrics not applicable: refusal case or no expected pages. Review results manually.")
    print(f"\nTime: {result['elapsed_seconds']:.2f}s; new embedding tokens: {result['embedding_tokens']}")
    if result["reranked"]:
        print("First run includes model loading. Reranker scores are relevance scores, not probabilities.")


if __name__ == "__main__":
    main()
