"""Show retrieved passages and compare their pages with an evaluation question."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
from app.rag.retrieval import Retriever


def check_expected_pages(passages, expected):
    """Check page coverage, not answer correctness or passage-level evidence."""
    if not expected:
        return None
    retrieved = {(p["source"], page) for p in passages for page in p["pages"]}
    checks = [dict(ref, found=(ref["source"], ref["page"]) in retrieved)
              for ref in expected]
    return {
        "pages": checks,
        "found": sum(ref["found"] for ref in checks),
        "total": len(checks),
        "all_found": all(ref["found"] for ref in checks),
    }



# Command-line inputs, validation, execution, and output.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--question", help="Your question text")
    choice.add_argument("--question-id", type=int,
                        help="1-based position in questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", action="store_true",
                        help="Print structured JSON instead of readable text")
    # Parse options before opening files or making API requests.
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    expected = None
    if args.question_id is not None:
        questions = json.loads((args.data_dir / "questions.json").read_text(encoding="utf-8"))
        if not 1 <= args.question_id <= len(questions):
            parser.error(f"--question-id must be between 1 and {len(questions)}")
        entry = questions[args.question_id - 1]
        question = entry["question"]
        expected = entry.get("expected_chunks")
    else:
        question = args.question
    if not question.strip():
        parser.error("Question cannot be empty")
    # Load credentials only for the requested runtime operation.
    load_dotenv(REPO_DIR / ".env")
    retriever = Retriever(args.data_dir, OpenAI())
    result = retriever.retrieve(question, args.k)
    result["expected_page_coverage"] = check_expected_pages(result["passages"], expected)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"Question: {question}")
    for rank, passage in enumerate(result["passages"], 1):
        pages = ", ".join(str(page) for page in passage["pages"])
        print(f"\n{rank}. Chunk {passage['chunk_id']} | score {passage['score']:.4f}")
        print(f"Source: {passage['source']} | pages {pages}")
        print(passage["text"])
    coverage = result["expected_page_coverage"]
    if coverage is None:
        print("\nExpected-page check: N/A (no expected pages supplied)")
    else:
        print(f"\nExpected-page coverage: {coverage['found']}/{coverage['total']}")
        for ref in coverage["pages"]:
            status = "FOUND" if ref["found"] else "MISSED"
            print(f"  {status}: {ref['source']} page {ref['page']}")
        print("Page coverage does not establish answer correctness.")
    print(f"Embedding input tokens: {result['embedding_tokens']}")


# Running this file explicitly invokes the command.
if __name__ == "__main__":
    main()
