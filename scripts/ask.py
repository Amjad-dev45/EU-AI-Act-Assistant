"""Run the RAG text pipeline without a web interface."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
from app.rag.answering import Answerer
from app.rag.retrieval import Retriever



# Command-line inputs, validation, execution, and output.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--question")
    choices.add_argument("--question-id", type=int, help="1-based position in questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--conversation-id", help="Reuse an ID to group costs across questions")
    parser.add_argument("--judge", action="store_true", help="Additional paid semantic evaluation call")
    # Parse options before opening files or making API requests.
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
    # Load credentials only for the requested runtime operation.
    load_dotenv(REPO_DIR / ".env")
    client = OpenAI()
    answerer = Answerer(Retriever(args.data_dir, client), client)
    result = answerer.answer(question, k=args.k, conversation_id=args.conversation_id, evaluate=args.judge)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    for citation in result["citations"]:
        pages = ", ".join(str(p) for p in citation["pages"])
        print(f"[{citation['citation_id']}] {citation['source']} — pages {pages}")
    print(f"\nStatus: {result['status']}")
    print(f"Token usage: {json.dumps(result['usage'])}")
    print(f"Request ID: {result['telemetry']['request_id']}")
    print(f"Conversation ID: {result['telemetry']['conversation_id']}")
    print(f"Estimated cost: {json.dumps(result['telemetry']['cost'])}")
    print(f"Timings (ms): {json.dumps(result['telemetry']['timings_ms'])}")
    if not result["telemetry_persisted"]:
        print("WARNING: request statistics could not be saved.", file=sys.stderr)


# Running this file explicitly invokes the command.
if __name__ == "__main__":
    main()
