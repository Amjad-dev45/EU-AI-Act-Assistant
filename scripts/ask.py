"""Run the RAG text pipeline without a web interface."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
from app.answering import Answerer
from app.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--question")
    choices.add_argument("--question-id", type=int, help="1-based position in questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    question = args.question
    if args.question_id is not None:
        questions = json.loads((args.data_dir / "questions.json").read_text(encoding="utf-8"))
        if not 1 <= args.question_id <= len(questions):
            parser.error(f"--question-id must be between 1 and {len(questions)}")
        question = questions[args.question_id - 1]["question"]
    if not question.strip():
        parser.error("Question cannot be empty")
    load_dotenv(REPO_DIR / ".env")
    client = OpenAI()
    answerer = Answerer(Retriever(args.data_dir, client), client)
    result = answerer.answer(question, k=args.k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    for citation in result["citations"]:
        pages = ", ".join(str(p) for p in citation["pages"])
        print(f"[{citation['citation_id']}] {citation['source']} — pages {pages}")
    print(f"\nStatus: {result['status']}")
    print(f"Token usage: {json.dumps(result['usage'])}")


if __name__ == "__main__":
    main()
