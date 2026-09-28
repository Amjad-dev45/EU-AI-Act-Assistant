"""Capture one request's stages, provider calls, source metadata, and outcome."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from statistics import mean
from time import perf_counter
from uuid import uuid4

from .usage import field, provider_usage, usage_summary, cost_summary

# Named stages are stable keys in saved records and existing dashboards.
STAGES = (
    "query_preprocessing", "embedding_generation", "retrieval", "reranking",
    "context_building", "llm_generation", "evaluation",
)

# Save identifiers and scores, not full retrieved passage text.
def source_metadata(passage, rank):
    source = passage["source"]
    return {"document_id": passage.get("document_id") or hashlib.sha256(source.encode()).hexdigest(),
            "chunk_id": passage["chunk_id"], "source": source,
            "source_reference": passage.get("source_url") or passage.get("source_reference") or source,
            "pages": passage.get("pages", []), "score": passage.get("score"),
            "rank": rank, "citation_id": passage.get("citation_id"),
            "retrieval_rank": passage.get("retrieval_rank"),
            "rerank_score": passage.get("rerank_score")}


# Each instance belongs to exactly one request.
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
