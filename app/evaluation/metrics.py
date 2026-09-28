"""Validate evaluation labels and calculate deterministic retrieval/answer metrics.

Page matches and lexical overlap are proxies, not semantic correctness scores.
"""

from collections import Counter
import math
import re

# Compare expected evidence and behavior with actual retrieval and answer output.
def labelled_metrics(result, label):
    """Page matches are coarse proxies; exact chunk metrics require chunk labels."""
    passages = result["retrieved_passages"]
    expected = {(r["source"], r["page"]) for r in (label.get("expected_chunks") or [])}
    pairs = [{(p["source"], page) for page in p["pages"]} for p in passages]
    observed = set().union(*pairs) if pairs else set()
    hits = [bool(p & expected) for p in pairs]
    metrics = {
        "expected_page_recall": len(expected & observed) / len(expected) if expected else None,
        "all_expected_pages_found": expected <= observed if expected else None,
        "page_hit_precision_at_k": sum(hits) / len(hits) if expected and hits else (0.0 if expected else None),
        "page_hit_mrr": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0) if expected else None,
    }
    expected_ids = set((label.get("expected_chunk_ids") or []))
    ids = [p["chunk_id"] for p in passages]
    # Deduplicate IDs before scoring, so duplicates cannot inflate precision/DCG.
    ids = list(dict.fromkeys(ids))
    relevant = [chunk_id in expected_ids for chunk_id in ids]
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(relevant, 1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(ids), len(expected_ids)) + 1))
    metrics.update({
        "chunk_recall_at_k": len(set(ids) & expected_ids) / len(expected_ids) if expected_ids else None,
        "chunk_precision_at_k": sum(relevant) / len(ids) if expected_ids and ids else (0.0 if expected_ids else None),
        "chunk_mrr": next((1 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0) if expected_ids else None,
        "chunk_ndcg_at_k": dcg / ideal if ideal else (0.0 if expected_ids else None),
    })
    expected_status = label.get("expected_status")
    if expected_status is None:
        expected_status = {"out-of-scope": "insufficient_evidence",
                           "ambiguous": "needs_clarification"}.get(label.get("category"))
    metrics["expected_status_match"] = result["status"] == expected_status if expected_status else None
    reference = label.get("expected_answer")
    metrics.update({"answer_exact_match": None, "answer_token_f1": None})
    if isinstance(reference, str) and reference.strip():
        def tokens(text):
            return re.findall(r"\w+", re.sub(r"\[\d+\]", "", text).casefold())
        actual, target = tokens(result["answer"]), tokens(reference)
        overlap = sum((Counter(actual) & Counter(target)).values())
        metrics["answer_exact_match"] = actual == target
        metrics["answer_token_f1"] = 2 * overlap / (len(actual) + len(target)) if actual or target else 1.0
    return metrics


# Validate before any paid API calls; course null labels mean not applicable.
def validate_labels(labels):
    if not isinstance(labels, list) or not labels:
        raise ValueError("Evaluation dataset must be a nonempty JSON list")
    for i, label in enumerate(labels, 1):
        if not isinstance(label, dict) or not isinstance(label.get("question"), str) or not label["question"].strip():
            raise ValueError(f"Question {i} needs a nonempty question string")
        for key in ("category", "difficulty"):
            if key in label and not isinstance(label[key], str):
                raise ValueError(f"Question {i}: {key} must be a string")
        if label.get("expected_answer") is not None and not isinstance(label["expected_answer"], str):
            raise ValueError(f"Question {i}: expected_answer must be a string or null")
        if "expected_status" in label and label["expected_status"] not in {
                "answered", "insufficient_evidence", "needs_clarification"}:
            raise ValueError(f"Question {i}: invalid expected_status")
        for key in ("expected_chunks", "expected_chunk_ids"):
            if label.get(key) is not None and not isinstance(label[key], list):
                raise ValueError(f"Question {i}: {key} must be a list")
        for ref in (label.get("expected_chunks") or []):
            if (not isinstance(ref, dict) or not isinstance(ref.get("source"), str)
                    or type(ref.get("page")) is not int or ref["page"] < 1):
                raise ValueError(f"Question {i}: expected_chunks needs source/page objects")
        if any(type(x) not in (int, str) for x in (label.get("expected_chunk_ids") or [])):
            raise ValueError(f"Question {i}: expected_chunk_ids must contain strings or integers")


# Average only applicable numeric metrics and report their sample counts.
def metric_averages(rows):
    """Return each metric's mean and number of applicable samples."""
    metrics = [row.get("metrics", {}) for row in rows]
    names = sorted({name for result in metrics for name in result})
    summary = {}
    for name in names:
        values = [result[name] for result in metrics
                  if isinstance(result.get(name), (int, float))]
        summary[name] = {
            "mean": sum(values) / len(values) if values else None,
            "samples": len(values),
        }
    return summary
