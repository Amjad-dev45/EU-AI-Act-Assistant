"""Check grounded answers: uv run scripts/answer_checker.py --question-number 24."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from openai import OpenAIError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.rag.answering import answer_question


def select_questions(args, parser):
    """Select student-written text without exposing evaluation labels to generation."""
    if args.question is not None:
        selected = [(None, args.question)]
    else:
        try:
            entries = json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("Cannot read data/questions.json as JSON.")
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            parser.error("data/questions.json must contain a list of question objects.")
        if args.priority:
            selected = [(number, entry.get("question")) for number, entry in enumerate(entries, 1)
                        if entry.get("priority") is True]
            if not selected:
                parser.error("No questions have priority set to true.")
        else:
            if not 1 <= args.question_number <= len(entries):
                parser.error(f"Choose a question number from 1 to {len(entries)}.")
            selected = [(args.question_number, entries[args.question_number - 1].get("question"))]
    if any(not isinstance(question, str) or not question.strip() for _, question in selected):
        parser.error("Each selected question must be a nonblank string.")
    return selected


def print_result(result, path):
    """Show citations and actual usage; relevance scores are not confidence values."""
    print(f"\nQuestion: {result['question']}\nStatus: {result['status']}")
    if result.get("answer_mode") == "source_quotes":
        print("Answer mode: exact source calculation rules (no chat request)")
    if result["status"] == "error":
        print("Answer generation failed; inspect the saved run's error status.")
    else:
        print(result["answer"])
    context = result.get("answer_context", {})
    if context.get("requested"):
        state = "complete" if context["complete"] else "incomplete"
        print(f"Section context: {context['heading']} | {state} | {context['chunk_count']} chunks")
    for citation in result["citations"]:
        pages = ", ".join(map(str, citation["pages"]))
        print(f"[{citation['id']}] {citation['source']} | pages {pages}")
    score = result["top_rerank_score"]
    print("Top reranker score: " + ("unavailable" if score is None else f"{score:.4f}")
          + " (relevance, not probability)")
    usage = result["usage"]
    chat = "unknown" if usage is None else ", ".join(
        f"{label}: {usage.get(key) if usage.get(key) is not None else 'unknown'}"
        for label, key in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"),
                           ("total", "total_tokens")))
    print(f"New embedding tokens: {result['retrieval']['embedding_tokens']}; chat tokens: {chat}")
    print(f"Time: {result['elapsed_seconds']:.2f}s; saved: {path}")
    print("Review the answer and citations manually; answer accuracy has not been measured.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("question", nargs="?", help="A question about the corpus")
    selection.add_argument("--question-number", type=int, help="1-based entry in data/questions.json")
    selection.add_argument("--priority", action="store_true", help="Answer entries with priority: true")
    parser.add_argument("--no-expansion", action="store_true", help="Use only the original five retrieved passages")
    args = parser.parse_args(argv)
    questions = select_questions(args, parser)
    # Serial requests share the cached reranker; each completed attempt has its own local record.
    for number, question in questions:
        try:
            options = {"use_expansion": False} if args.no_expansion else {}
            result = answer_question(question, ROOT / "data", **options)
        except (OpenAIError, ValueError) as error:
            print(f"Request failed ({type(error).__name__}); check local data/API configuration and try again.",
                  file=sys.stderr)
            return 1
        result = {**result, "question_number": number}
        directory = ROOT / "data/answer_runs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print_result(result, path)
        if result["status"] == "error":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
