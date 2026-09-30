"""Page-level retrieval scores; these do not measure answer correctness."""

from collections import defaultdict


# Refusal cases may carry contextual references, but have no retrievable answer.
def retrieval_labels(question):
    if (question.get("category") in {"out-of-scope", "ambiguous"}
            or question.get("sub_type") == "on_topic_unanswerable"):
        return []
    return question.get("expected_chunks") or []


# Expected labels name pages, so any page in a cross-page chunk can match.
def first_expected_rank(hits: list[dict], expected_chunks: list[dict]) -> int | None:
    expected = {(chunk["source"], chunk["page"]) for chunk in expected_chunks or []}
    for rank, hit in enumerate(hits, start=1):
        if any((hit["source"], page) in expected for page in hit["pages"]):
            return rank
    return None


# Unlabeled questions contribute latency, but never count as retrieval misses.
def summarize(rows: list[dict], variant: str) -> dict:
    labeled = [row for row in rows if retrieval_labels(row)]
    ranks = [first_expected_rank(row[variant]["hits"][:5], retrieval_labels(row))
             for row in labeled]
    count = len(labeled)
    result = {
        "total_questions": len(rows),
        "scored_questions": count,
        "unscored_questions": len(rows) - count,
    }
    for k in (1, 3, 5):
        result[f"recall_at_{k}"] = (
            sum(rank is not None and rank <= k for rank in ranks) / count
            if count else None
        )
    result["mrr_at_5"] = (
        sum(1 / rank if rank is not None else 0 for rank in ranks) / count
        if count else None
    )
    result["mean_latency_seconds"] = (
        sum(row[variant]["elapsed_seconds"] for row in rows) / len(rows)
        if rows else None
    )
    return result


# Use the same denominators and scoring rules within every category/difficulty.
def breakdown(rows: list[dict], variant: str, field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {name: summarize(group, variant) for name, group in sorted(groups.items())}


# Compare the same labeled questions; a top-five miss is worse than rank five.
def compare_results(rows: list[dict], baseline="vector", candidate="reranked") -> dict:
    changes = {name: {"regressions": [], "improvements": []}
               for name in ("recall_at_1", "recall_at_3", "recall_at_5", "rank")}
    for row in rows:
        expected = retrieval_labels(row)
        if not expected:
            continue
        before = first_expected_rank(row[baseline]["hits"][:5], expected)
        after = first_expected_rank(row[candidate]["hits"][:5], expected)
        before, after = before or 6, after or 6
        number = row["question_number"]
        for k in (1, 3, 5):
            if before <= k < after:
                changes[f"recall_at_{k}"]["regressions"].append(number)
            elif after <= k < before:
                changes[f"recall_at_{k}"]["improvements"].append(number)
        if after > before:
            changes["rank"]["regressions"].append(number)
        elif after < before:
            changes["rank"]["improvements"].append(number)
    return changes
