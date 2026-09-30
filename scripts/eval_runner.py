"""Compare retrieval across questions.json: uv run scripts/eval_runner.py."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from evaluation.runner import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Check only the first N questions")
    parser.add_argument("--candidates", type=int, default=20,
                        help="Candidates per retriever and after fusion, before reranking (default: 20)")
    parser.add_argument("--cached-only", action="store_true", help="Fail on missing embeddings; never call OpenAI")
    parser.add_argument("--output", type=Path, help="Result JSON path (default: data/eval_results.json; eval_smoke.json with --limit)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.candidates < 5:
        parser.error("--candidates must be at least 5.")
    report = evaluate(ROOT / "data", args.output, args.limit, args.candidates, args.cached_only)

    print("\nMode          Recall@1  Recall@3  Recall@5  MRR@5    Mean seconds")
    for variant in ("vector", "reranked", "hybrid"):
        stats = report["summary"][variant]["overall"]
        scores = [stats[key] for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5")]
        values = "  ".join(f"{value:7.3f}" if value is not None else "    n/a" for value in scores)
        print(f"{variant:12} {values}  {stats['mean_latency_seconds']:.3f}")
    print(f"\nScored: {stats['scored_questions']}; unscored: {stats['unscored_questions']}")
    for title, key in (("Vector → reranked", "comparison"), ("Reranked → hybrid", "hybrid_comparison")):
        print(f"\n{title}")
        for metric, changes in report[key].items():
            print(f"{metric}: improved questions {changes['improvements']}; regressed questions {changes['regressions']}")
    print(f"New embedding tokens: {report['new_embedding_tokens']}")
    default_name = "eval_smoke.json" if args.limit is not None else "eval_results.json"
    print(f"Saved: {args.output or ROOT / 'data' / default_name}")
    print("Recall uses any expected-page hit. No answers or refusal decisions were evaluated.")


if __name__ == "__main__":
    main()
