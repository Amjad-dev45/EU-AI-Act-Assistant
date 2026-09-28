"""Persist request records in SQLite and build read-only aggregate reports."""

from contextlib import closing
import json
import logging
import math
from pathlib import Path
import sqlite3
from statistics import mean

from app.settings import configured_path
from .tracing import RequestTrace, STAGES
from .usage import Pricing, usage_summary, cost_summary

# Open a separate connection for every operation; UI workers share no cursor.
class Monitor:
    """One connection per operation, transaction per request; safe across UI workers."""
    def __init__(self, path=None, pricing=None):
        self.path = Path(path) if path is not None else configured_path("RAG_METRICS_DB", "logs/requests.sqlite3")
        self.pricing = pricing or Pricing()

    def trace(self, question, conversation_id=None, kind="interactive", metadata=None):
        return RequestTrace(question, self.pricing, conversation_id, kind, metadata)

    def save(self, trace):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    started_at TEXT NOT NULL, kind TEXT NOT NULL, record TEXT NOT NULL)""")
                connection.execute("CREATE INDEX IF NOT EXISTS requests_conversation ON requests(conversation_id)")
                r = trace.record
                connection.execute("INSERT INTO requests VALUES (?, ?, ?, ?, ?)",
                                   (r["request_id"], r["conversation_id"], r["started_at"], r["kind"],
                                    json.dumps(r, ensure_ascii=False, allow_nan=False)))
            return True
        except (OSError, sqlite3.Error, ValueError):
            logging.getLogger(__name__).error("Telemetry persistence failed for request %s", trace.record["request_id"])
            return False

    def records(self, conversation_id=None, kind=None):
        if not self.path.exists():
            return []
        query, parameters = "SELECT record FROM requests WHERE 1=1", []
        for column, value in (("conversation_id", conversation_id), ("kind", kind)):
            if value is not None:
                query += f" AND {column} = ?"
                parameters.append(value)
        query += " ORDER BY started_at"
        # Read-only mode also prevents an empty DB being created by a report.
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as connection:
            if not connection.execute("SELECT name FROM sqlite_master WHERE name='requests'").fetchone():
                return []
            return [json.loads(row[0]) for row in connection.execute(query, parameters)]

    def summary(self, conversation_id=None, kind=None):
        return summarize(self.records(conversation_id, kind))


# Report unknown totals explicitly instead of silently counting them as zero.
def summarize(records):
    count = len(records)
    costs = [r["cost"]["total_usd"] for r in records if r["cost"]["complete"]]
    tokens = [r["usage"]["total_tokens"] for r in records if r["usage"]["complete"]]
    latencies = sorted(r["timings_ms"]["total_latency_ms"] for r in records)
    statuses = {status: sum(r["status"] == status for r in records)
                for status in sorted({r["status"] for r in records})}
    def average(values):
        return mean(values) if values else None
    def percentile(fraction):
        return latencies[max(0, math.ceil(count * fraction) - 1)] if count else None
    return {
        "requests": count, "statuses": statuses,
        "error_rate": statuses.get("error", 0) / count if count else None,
        "total_cost_usd": sum(costs) if len(costs) == count else None,
        "known_total_cost_usd": sum(r["cost"]["known_total_usd"] for r in records),
        "average_cost_per_question_usd": average(costs) if len(costs) == count else None,
        "requests_with_unknown_cost": count - len(costs),
        "total_tokens": sum(tokens) if len(tokens) == count else None,
        "known_total_tokens": sum(r["usage"]["known_tokens"]["total_tokens"] for r in records),
        "average_tokens_per_question": average(tokens) if len(tokens) == count else None,
        "requests_with_unknown_usage": count - len(tokens),
        "mean_latency_ms": average(latencies), "p50_latency_ms": percentile(.5),
        "p95_latency_ms": percentile(.95),
        "mean_stage_timings_ms": {
            name + "_time_ms": average([r["timings_ms"][name + "_time_ms"] for r in records
                                       if r["timings_ms"].get(name + "_time_ms") is not None])
            for name in STAGES},
        "mean_metrics": {name: average([r["metrics"][name] for r in records
                                        if isinstance(r["metrics"].get(name), (int, float))])
                         for name in ("retrieved_chunks", "retrieved_documents", "cited_documents",
                                      "mean_similarity", "citation_validity")},
        "semantic_evaluation_errors": sum(r["semantic_evaluation"]["status"] == "error" for r in records),
        "mean_judge_scores": {
            name: average([r["semantic_evaluation"][name] for r in records
                           if r["semantic_evaluation"].get(name) is not None])
            for name in ("context_relevance", "answer_relevance", "faithfulness", "answer_supported")},
        "judge_evaluated_requests": sum(r["semantic_evaluation"]["status"] == "completed" for r in records),
        "usage_by_operation": usage_summary([c for r in records for c in r["api_calls"]])["by_operation"],
        "cost_by_operation_usd": cost_summary([c for r in records for c in r["api_calls"]])["by_operation_usd"],
    }
