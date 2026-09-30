"""Compare vector, reranked, and hybrid retrieval on the same questions and index."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

from services.rag.embeddings import embed_texts
from services.rag.hybrid import BM25Index, RRF_K, hybrid_candidates
from services.rag.reranking import MODEL, REVISION, load_model, rerank
from services.rag.vector_index import load_index, search_index
from .metrics import breakdown, compare_results, first_expected_rank, retrieval_labels, summarize


# Validate student-written labels before making any embedding API calls.
def load_questions(path, chunks):
    raw = Path(path).read_bytes()
    questions = json.loads(raw)
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions.json must contain a nonempty list.")
    available = {(chunk["source"], page) for chunk in chunks for page in chunk["pages"]}
    for number, question in enumerate(questions, 1):
        if not isinstance(question, dict) or not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"Question {number} needs nonempty question text.")
        expected = question.get("expected_chunks", [])
        if expected is None:
            expected = []  # Unanswerable questions may use null instead of an empty list.
        if not isinstance(expected, list):
            raise ValueError(f"Question {number}: expected_chunks must be a list.")
        for item in expected:
            if (not isinstance(item, dict) or not isinstance(item.get("source"), str)
                    or type(item.get("page")) is not int or item["page"] < 1):
                raise ValueError(f"Question {number} has an invalid source/page label.")
            if (item["source"], item["page"]) not in available:
                raise ValueError(f"Question {number}: expected source/page is absent from chunks: {item}")
    return questions, hashlib.sha256(raw).hexdigest()


def save_report(path, report):
    """Replace a JSON report only after a complete temporary file is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# Keep reports compact; chunk IDs resolve to the original text in chunks.json.
def variant_result(hits, expected, seconds):
    rank = first_expected_rank(hits, expected)
    return {
        "hits": [{key: hit[key] for key in ("chunk_id", "source", "pages", "score", "rerank_score",
                                          "bm25_score", "rrf_score") if key in hit}
                 for hit in hits],
        "first_expected_rank": rank,
        "reciprocal_rank": (1 / rank if rank else 0) if expected else None,
        "elapsed_seconds": seconds,
        "failure_reason": "retrieval_miss" if expected and rank is None else None,
    }


def evaluate(data_dir, output=None, limit=None, candidate_k=20, cached_only=False, progress=print):
    """Load once, embed each question once, and compare three top-five result lists."""
    if type(candidate_k) is not int or candidate_k < 5:
        raise ValueError("Candidate count must be at least 5.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Limit must be positive.")
    data_dir = Path(data_dir)
    default_name = "eval_smoke.json" if limit is not None else "eval_results.json"
    output = Path(output) if output is not None else data_dir / default_name
    protected = {data_dir / name for name in ("questions.json", "chunks.json", "corpus.json", "sample.json", "index_meta.json")}
    if output.suffix != ".json":
        raise ValueError("Evaluation output must be a .json file.")
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Evaluation output must not overwrite an existing input or index file.")
    started = time.perf_counter()
    index, chunks, index_meta = load_index(data_dir)
    index_load_seconds = time.perf_counter() - started
    questions, questions_hash = load_questions(data_dir / "questions.json", chunks)
    selected = questions[:limit] if limit else questions
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint = output.parent / "evaluation_runs" / f"{run_id}.json"
    report = {
        "schema_version": 1, "mode": "retrieval-only", "status": "running", "run_id": run_id,
        "config": {"final_k": 5, "candidate_k": candidate_k, "reranker": MODEL,
                   "reranker_revision": REVISION, "cached_only": cached_only,
                   "lexical_retriever": "BM25Okapi", "fusion": "reciprocal_rank_fusion",
                   "keyword_tokenizer": "Lowercase words, stopword removal, and combined numbered article/annex/section references.",
                   "rrf_k": RRF_K,
                   "hybrid_pipeline": "FAISS and BM25 each retrieve candidate_k; RRF keeps candidate_k; reranker keeps 5.",
                   "history_used_for_retrieval": False},
        "inputs": {"questions_sha256": questions_hash, "index_metadata": index_meta},
        "total_questions": len(questions), "selected_questions": len(selected),
        "index_load_seconds": index_load_seconds, "new_embedding_tokens": 0, "rows": [],
        "scoring_revision": 2,
        "latency_definition": "Question embedding/cache lookup plus search and optional reranking; excludes startup and report writes.",
    }
    save_report(checkpoint, report)
    try:
        loading = time.perf_counter()
        lexical_index = BM25Index(chunks)
        report["lexical_load_seconds"] = time.perf_counter() - loading
        progress("Loading reranker once; model loading is excluded from per-question latency.")
        loading = time.perf_counter()
        model = load_model()
        report["model_load_seconds"] = time.perf_counter() - loading

        for number, question in enumerate(selected, 1):
            text, expected = question["question"], retrieval_labels(question)
            # All variants share this embedding time and vector; usage is counted once.
            embedding_start = time.perf_counter()
            vectors, tokens = embed_texts([text], data_dir, allow_api=not cached_only)
            embedding_seconds = time.perf_counter() - embedding_start
            report["new_embedding_tokens"] += tokens

            search_start = time.perf_counter()
            vector_hits = search_index(index, chunks, vectors[0], 5)
            vector_seconds = time.perf_counter() - search_start
            # Reranked and hybrid reuse one vector candidate search and its timing.
            candidate_start = time.perf_counter()
            candidates = search_index(index, chunks, vectors[0], candidate_k)
            candidate_seconds = time.perf_counter() - candidate_start
            rerank_start = time.perf_counter()
            reranked_hits = rerank(text, candidates, 5, model=model)
            reranked_seconds = time.perf_counter() - rerank_start

            # Time hybrid independently so its latency excludes the other reranker call.
            hybrid_start = time.perf_counter()
            fused = hybrid_candidates(index, vectors[0], text, candidates, lexical_index, k=candidate_k)
            hybrid_hits = rerank(text, fused, 5, model=model)
            hybrid_seconds = time.perf_counter() - hybrid_start

            row = {
                **question, "question_number": number, "expected_chunks": question.get("expected_chunks") or [],
                "retrieval_scored": bool(expected),
                "query": text, "rewritten_query": None, "generated_answer": None, "answer_correct": None,
                "embedding_seconds": embedding_seconds, "new_embedding_tokens": tokens,
                "vector": variant_result(vector_hits, expected, embedding_seconds + vector_seconds),
                "reranked": variant_result(reranked_hits, expected,
                                           embedding_seconds + candidate_seconds + reranked_seconds),
                "hybrid": variant_result(hybrid_hits, expected,
                                         embedding_seconds + candidate_seconds + hybrid_seconds),
            }
            report["rows"].append(row)
            save_report(checkpoint, report)
            status = (f"vector rank={row['vector']['first_expected_rank']}; "
                      f"reranked rank={row['reranked']['first_expected_rank']}; "
                      f"hybrid rank={row['hybrid']['first_expected_rank']}") if expected else "unscored for retrieval: refusal case or no expected pages"
            progress(f"[{number}/{len(selected)}] {status}")

        report["summary"] = {
            variant: {"overall": summarize(report["rows"], variant),
                      "by_category": breakdown(report["rows"], variant, "category"),
                      "by_difficulty": breakdown(report["rows"], variant, "difficulty"),
                      "by_sub_type": breakdown(report["rows"], variant, "sub_type")}
            for variant in ("vector", "reranked", "hybrid")
        }
        report["comparison"] = compare_results(report["rows"])
        report["hybrid_comparison"] = compare_results(report["rows"], "reranked", "hybrid")
        report["status"] = "complete"
        report["wall_seconds"] = time.perf_counter() - started
        save_report(checkpoint, report)
        save_report(output, report)
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__  # Keep provider messages/credentials out of reports.
        save_report(checkpoint, report)
        progress(f"Incomplete run saved to {checkpoint}; previous completed output is unchanged.")
        raise
    return report
