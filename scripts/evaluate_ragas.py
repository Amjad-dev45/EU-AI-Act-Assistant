"""Judge saved answers: uv run --project evaluation scripts/evaluate_ragas.py --question-number 24."""
import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from evaluation.ragas_inputs import load_samples, read_json
from evaluation.ragas_judge import judge_sample, make_scorers, summarize


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


async def evaluate(samples, args, report, path):
    # Disable RAGAS telemetry. Only judge/embedding requests go to the configured provider.
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    import httpx
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in the project .env.")
    usage = {"chat_requests": 0, "embedding_requests": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "embedding_tokens": 0}

    async def count_usage(response):
        # Record only numeric usage, never HTTP headers, credentials, or prompts.
        await response.aread()
        if not response.is_success:
            return
        try:
            body = response.json()
            used = body.get("usage", {})
            if response.request.url.path.endswith("/embeddings"):
                usage["embedding_requests"] += 1
                usage["embedding_tokens"] += used.get("total_tokens", 0)
            elif response.request.url.path.endswith("/chat/completions"):
                usage["chat_requests"] += 1
                usage["prompt_tokens"] += used.get("prompt_tokens", 0)
                usage["completion_tokens"] += used.get("completion_tokens", 0)
        except (ValueError, TypeError):
            pass

    report.update(ragas_version=importlib.metadata.version("ragas"), usage=usage)
    async with httpx.AsyncClient(event_hooks={"response": [count_usage]}) as http:
        async with AsyncOpenAI(http_client=http, timeout=60, max_retries=0) as client:
            scorers = make_scorers(client, args.model, args.embedding_model)
            for sample in samples:
                print(f"Judging question {sample['question_number']}...", flush=True)
                row = await judge_sample(sample, scorers, args.timeout)
                report["rows"].append(row)
                report["summary"] = summarize(report["rows"])
                save(report, path)  # Keep completed rows if a later judge call fails.
                print(json.dumps({"question_number": row["question_number"], "scores": row["scores"],
                                  "errors": row["errors"], "skip_reason": row["skip_reason"]}), flush=True)
    report["complete"] = True
    save(report, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question-number", type=int, action="append", help="Repeat to select several questions")
    group.add_argument("--priority", action="store_true", help="Saved priority answers")
    group.add_argument("--all", action="store_true", help="All saved benchmark answers; makes paid judge calls")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect saved inputs without API calls or RAGAS")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, help="New report path (existing files are never overwritten)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds per metric")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("Timeout must be positive.")
    try:
        questions = read_json(args.data_dir / "questions.json")
        numbers = args.question_number or [n for n, q in enumerate(questions, 1) if args.all or q.get("priority")]
        samples = load_samples(args.data_dir, list(dict.fromkeys(numbers)))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = args.output or args.data_dir / "ragas_runs" / (run_id + ".json")
        if path.exists():
            raise ValueError("Choose a new output path; existing reports are preserved.")
        report = {"run_id": run_id, "complete": False, "selected_questions": numbers,
                  "judge_model": args.model, "embedding_model": args.embedding_model,
                  "scope": "Saved standalone RAG answers, not final multi-tool agent responses.",
                  "context_scope": {"faithfulness": "saved answer_context.hits (including expansion)",
                                    "context_precision_and_recall": "saved retrieval.hits (ranked, before expansion)"},
                  "limitations": "LLM judge estimates, not measured accuracy. Context recall is claim-based, not page Recall@k. Refusal/ambiguity and incomplete answers require manual review; skipped/errors are excluded from means. Saved runs may use older code; see per-row provenance. Judging incurs separate API charges.",
                  "rows": []}
        if args.prepare_only:
            report.update(mode="prepare_only", rows=samples)
            save(report, path)
            print(f"Prepared {len(samples)} samples; no API calls. Saved: {path}")
        else:
            # Missing dependencies fail before any remote calls and give the exact command.
            try:
                importlib.metadata.version("ragas")
            except importlib.metadata.PackageNotFoundError:
                parser.error("Use uv run --project evaluation scripts/evaluate_ragas.py with the same options.")
            asyncio.run(evaluate(samples, args, report, path))
            print(f"Saved: {path}")
            print(json.dumps(report["summary"], indent=2))
            if report["summary"]["rows_with_errors"]:
                return 1
    except (OSError, ValueError) as error:
        print(f"Evaluation could not run ({type(error).__name__}). Check saved inputs and API configuration.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
