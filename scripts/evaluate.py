"""Evaluate the existing Answerer on labelled questions; preview by default."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
from app.evaluation import metric_averages, validate_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, help="Defaults to data-dir/questions.json")
    parser.add_argument("--output", type=Path, help="Defaults to logs/evaluations/<run-id>.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge", action="store_true", help="Paid semantic/reference judge per answer")
    parser.add_argument("--run", action="store_true", help="Make paid embedding and chat API calls")
    args = parser.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--k and --limit must be positive")
    raw = (args.questions or args.data_dir / "questions.json").read_bytes()
    labels = json.loads(raw)
    validate_labels(labels)
    if args.limit:
        labels = labels[:args.limit]
    if not args.run:
        print(f"Validated {len(labels)} questions. Per question: 1 embedding + 1 generation"
              + (" + 1 optional judge call." if args.judge else " call."))
        print("Preview only. Add --run to execute; --judge enables additional semantic scoring.")
        return
    # Imports/client initialization are deferred so preview needs only the standard library.
    from dotenv import load_dotenv
    from openai import OpenAI
    from app.answering import Answerer
    from app.retrieval import Retriever
    load_dotenv(REPO_DIR / ".env")
    client = OpenAI()
    answerer = Answerer(Retriever(args.data_dir, client), client)
    run_id = str(uuid4())
    fingerprint = hashlib.sha256(raw).hexdigest()
    output = args.output or REPO_DIR / "logs/evaluations" / f"{run_id}.json"
    rows = []
    def save_report():
        report = {
            "run_id": run_id, "dataset_sha256": fingerprint, "k": args.k,
            "judge_enabled": args.judge, "planned_questions": len(labels), "completed_questions": len(rows),
            "rows": rows, "metrics": metric_averages(rows),
            "by_category": {category: metric_averages([r for r in rows if r["category"] == category])
                            for category in sorted({r["category"] for r in rows})},
            "by_difficulty": {difficulty: metric_averages([r for r in rows if r["difficulty"] == difficulty])
                              for difficulty in sorted({r["difficulty"] for r in rows})},
            "monitoring": answerer.monitor.summary(run_id, kind="evaluation"),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output)
    for number, label in enumerate(labels, 1):
        row = {"question_number": number, "question": label["question"],
               "category": label.get("category", "unspecified"),
               "difficulty": label.get("difficulty", "unspecified")}
        try:
            result = answerer.answer(label["question"], k=args.k, conversation_id=run_id,
                evaluate=args.judge, label=label, kind="evaluation",
                metadata={"dataset_sha256": fingerprint, "question_number": number})
            row.update({"status": result["status"], "answer": result["answer"],
                        "expected_answer": label.get("expected_answer"),
                        "metrics": dict(result["labelled_evaluation"]),
                        "telemetry": result["telemetry"], "telemetry_persisted": result["telemetry_persisted"]})
            judge = result["telemetry"]["semantic_evaluation"]
            for metric in ("context_relevance", "answer_relevance", "faithfulness", "answer_supported", "reference_correctness"):
                row["metrics"]["judge_" + metric] = judge.get(metric)
        except Exception as exc:
            row.update({"status": "error", "error_type": type(exc).__name__, "metrics": {}})
        rows.append(row)
        save_report()  # A partial run remains reviewable after interruption.
        print(f"{number}/{len(labels)}: {row['status']}")
    print(f"Report: {output.resolve()}")
    if any(row["status"] == "error" or row.get("telemetry", {}).get("semantic_evaluation", {}).get("status") == "error"
           or row.get("telemetry_persisted") is False for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
