"""Local request telemetry. No prompts, passages or credentials in API call logs."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from statistics import mean
from time import perf_counter
from uuid import uuid4

REPO_DIR = Path(__file__).resolve().parents[1]
STAGES = (
    "query_preprocessing", "embedding_generation", "retrieval", "reranking",
    "context_building", "llm_generation", "evaluation",
)


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def provider_usage(response, operation):
    """Unknown usage stays unknown; token estimates are never substituted."""
    usage = field(response, "usage")
    if usage is None:
        return None
    input_tokens = field(usage, "prompt_tokens")
    total = field(usage, "total_tokens")
    output = 0 if operation == "embedding" else field(usage, "completion_tokens")
    if input_tokens is None and operation == "embedding":
        input_tokens = total
    details = field(usage, "prompt_tokens_details")
    cached = field(details, "cached_tokens", 0) or 0
    if any(type(n) is not int or n < 0 for n in (input_tokens, output, total, cached)):
        return None
    if cached > input_tokens or total != input_tokens + output:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output,
            "total_tokens": total, "cached_input_tokens": cached,
            "source": "provider"}


class Pricing:
    def __init__(self, path=None):
        path = path or os.getenv("RAG_PRICING_FILE") or REPO_DIR / "config/pricing.json"
        self.config = json.loads(Path(path).read_text(encoding="utf-8"))
        if self.config["currency"] != "USD":
            raise ValueError("Pricing currency must be USD")
        for rates in self.config["models"].values():
            for name in ("input_per_million", "output_per_million"):
                if name not in rates:
                    raise ValueError(f"Missing price: {name}")
            for rate in rates.values():
                if isinstance(rate, bool) or not math.isfinite(float(rate)) or float(rate) < 0:
                    raise ValueError("Prices must be finite, nonnegative numbers")

    def estimate(self, model, usage):
        key = self.config.get("aliases", {}).get(model, model)
        rates = self.config["models"].get(key)
        if rates is None or usage is None:
            return None
        cached = usage["cached_input_tokens"]
        # If a model reports cache usage without a cache price, don't invent a price.
        if cached and "cached_input_per_million" not in rates:
            return None
        divisor = Decimal(1_000_000)
        ordinary = usage["input_tokens"] - cached
        input_cost = (ordinary * Decimal(str(rates["input_per_million"]))
                      + cached * Decimal(str(rates.get("cached_input_per_million", 0)))) / divisor
        output_cost = usage["output_tokens"] * Decimal(str(rates["output_per_million"])) / divisor
        return {"input_usd": float(input_cost), "output_usd": float(output_cost),
                "total_usd": float(input_cost + output_cost), "rates": rates,
                "pricing_model": key, "pricing_version": self.config["version"]}


def usage_summary(calls):
    def group(selected):
        complete = all(c["usage"] is not None for c in selected)
        known = {key: sum(c["usage"][key] for c in selected if c["usage"] is not None)
                 for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")}
        return {**{k: v if complete else None for k, v in known.items()},
                "known_tokens": known, "complete": complete, "calls": len(selected)}

    additional = sorted({c["operation"] for c in calls} - {"main_llm", "embedding"})
    return {**group(calls),
            "main_llm": group([c for c in calls if c["operation"] == "main_llm"]),
            "embedding": group([c for c in calls if c["operation"] == "embedding"]),
            "additional_llm": group([c for c in calls if c["operation"] in additional]),
            "by_operation": {op: group([c for c in calls if c["operation"] == op])
                             for op in sorted({c["operation"] for c in calls})}}


def cost_summary(calls):
    def subtotal(selected, part="total_usd"):
        if any(c["cost"] is None for c in selected):
            return None
        return float(sum((Decimal(str(c["cost"][part])) for c in selected), Decimal(0)))

    main = [c for c in calls if c["operation"] == "main_llm"]
    extras = [c for c in calls if c["operation"] not in {"main_llm", "embedding"}]
    return {"currency": "USD", "estimated": True,
            "llm_input_usd": subtotal(main, "input_usd"),
            "llm_output_usd": subtotal(main, "output_usd"),
            "embedding_usd": subtotal([c for c in calls if c["operation"] == "embedding"]),
            "evaluation_usd": subtotal([c for c in calls if c["operation"] == "evaluation"]),
            "reranking_usd": subtotal([c for c in calls if c["operation"] == "reranking"]),
            "additional_llm_usd": subtotal(extras),
            "by_operation_usd": {op: subtotal([c for c in calls if c["operation"] == op])
                                 for op in sorted({c["operation"] for c in calls})},
            "total_usd": subtotal(calls),
            "known_total_usd": subtotal([c for c in calls if c["cost"] is not None]),
            "complete": all(c["cost"] is not None for c in calls)}


def source_metadata(passage, rank):
    source = passage["source"]
    return {"document_id": passage.get("document_id") or hashlib.sha256(source.encode()).hexdigest(),
            "chunk_id": passage["chunk_id"], "source": source,
            "source_reference": passage.get("source_url") or passage.get("source_reference") or source,
            "pages": passage.get("pages", []), "score": passage.get("score"),
            "rank": rank, "citation_id": passage.get("citation_id")}


class RequestTrace:
    def __init__(self, question, pricing, conversation_id=None, kind="interactive", metadata=None):
        self.started = perf_counter()
        self.pricing = pricing
        self.record = {
            "schema_version": 1, "request_id": str(uuid4()),
            "conversation_id": conversation_id or str(uuid4()),
            "started_at": datetime.now(timezone.utc).isoformat(), "kind": kind,
            "metadata": metadata or {}, "original_query": question,
            "search_query": None, "rewritten_query": None, "status": "running",
            "timings_ms": {name + "_time_ms": None for name in STAGES},
            "api_calls": [], "retrieval": [], "context_chunk_ids": [],
            "context_document_ids": [], "cited_chunk_ids": [], "cited_document_ids": [],
            "metrics": {}, "semantic_evaluation": {"status": "disabled"},
        }

    @contextmanager
    def stage(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            key = name + "_time_ms"
            timings = self.record["timings_ms"]
            timings[key] = (timings.get(key) or 0) + (perf_counter() - start) * 1000

    def api_call(self, stage, operation, model, invoke):
        call = {"operation": operation, "requested_model": model,
                "model": model, "usage": None, "cost": None, "status": "error"}
        self.record["api_calls"].append(call)
        start = perf_counter()
        try:
            with self.stage(stage):
                response = invoke()
            call["model"] = field(response, "model") or model
            call["response_id"] = field(response, "id")
            call["usage"] = provider_usage(response, operation)
            call["cost"] = self.pricing.estimate(call["model"], call["usage"])
            call["status"] = "success"
            return response
        except Exception as exc:
            call["error_type"] = type(exc).__name__
            raise
        finally:
            call["latency_ms"] = (perf_counter() - start) * 1000

    def retrieved(self, passages):
        self.record["retrieval"] = [source_metadata(p, rank) for rank, p in enumerate(passages, 1)]
        self.record["metrics"].update({
            "retrieved_chunks": len(passages),
            "retrieved_documents": len({p["document_id"] for p in self.record["retrieval"]}),
            "mean_similarity": mean(p["score"] for p in passages) if passages else None,
        })

    def finish(self, status, error=None):
        self.record["status"] = status
        if error is not None:
            # Exception text may contain prompts, keys or provider payloads.
            self.record["error_type"] = type(error).__name__
        self.record["timings_ms"]["total_latency_ms"] = (perf_counter() - self.started) * 1000
        self.record["usage"] = usage_summary(self.record["api_calls"])
        self.record["cost"] = cost_summary(self.record["api_calls"])
        return self.record


class Monitor:
    """One connection per operation, transaction per request; safe across UI workers."""
    def __init__(self, path=None, pricing=None):
        self.path = Path(path or os.getenv("RAG_METRICS_DB") or REPO_DIR / "logs/requests.sqlite3")
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
