"""Readable execution evidence for the UI; no inferred private reasoning."""
from html import escape
import re


def plain(value):
    # Disable source-provided HTML and Markdown links/images in diagnostics.
    return re.sub(r'([\\`*_{}\[\]()#+.!|>-])', r'\\\1', escape(str(value))).replace('\n', ' ')


def number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def format_retrieval(trace):
    sections = []
    for entry in trace if isinstance(trace, list) else []:
        data = entry.get("diagnostics") if isinstance(entry, dict) else None
        if not isinstance(data, dict):
            continue
        retrieval = data.get("retrieval", {})
        detail = retrieval.get("trace", {})
        if not detail:
            continue
        rows = [f"### Search {len(sections) + 1}", f"**Query:** {plain(detail.get('query', ''))}",
                plain(detail.get("policy_explanation", ""))]
        if entry.get("cached"):
            rows.append("**Cached result:** these stages were reused, not executed again.")
        rows.append("**BM25 query terms:** " + plain(", ".join(detail.get("query_terms", [])) or "none"))
        if detail.get("numbered_references"):
            rows.append("**Numbered references matched by the tokenizer:** " + plain(", ".join(detail["numbered_references"])))
        stages = {stage["name"]: stage for stage in detail.get("stages", [])}
        rows.append(" → ".join(f"{plain(stage['label'])}: {stage['count']}" for stage in stages.values() if stage["enabled"]))
        # Before/after ranks refer to the same candidate pool; a dash means absent.
        table_start = len(rows) + 1
        rows.extend(["**Selected chunks: ranks before and after reranking**",
                     "| Source / pages | FAISS rank | BM25 rank | RRF rank | Reranked | Cosine | BM25 | Reranker |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for hit in stages.get("selected", {}).get("results", []):
            source = plain(hit.get("source", "")) + " / " + ", ".join(map(str, hit.get("pages", [])))
            ranks = [str(hit.get(key) or "—") for key in ("faiss_rank", "bm25_rank", "fusion_rank", "rerank_rank")]
            scores = [number(hit.get(key)) for key in ("score", "bm25_score", "rerank_score")]
            rows.append("| " + " | ".join([source, *ranks, *scores]) + " |")
        table_end = len(rows)
        rows.append(plain(detail.get("score_note", "")))
        for name in ("faiss", "bm25", "selected"):
            stage = stages.get(name, {})
            if not stage.get("enabled"):
                continue
            rows.append(f"**{plain(stage['label'])}: first five passages**")
            for hit in stage.get("results", [])[:5]:
                rows.append(f"{hit['rank']}. {plain(hit.get('source', ''))}, pages {plain(hit.get('pages', []))}: "
                            + plain(hit.get("text", "")[:200]) + "…")
        context, answer = data.get("context", {}), data.get("answer", {})
        rows.append(f"**Answer context:** {plain(context.get('chunk_count', 0))} passages; "
                    f"{plain(context.get('heading') or context.get('mode') or 'ranked passages')}; "
                    f"expanded: {plain(context.get('applied', False))}; "
                    f"complete section: {plain(context.get('complete'))}.")
        rows.append(f"**Relevance gate:** top score {number(retrieval.get('top_rerank_score'))}; "
                    f"minimum {number(retrieval.get('min_rerank_score'))}. "
                    f"Answer status: {plain(answer.get('status', 'unknown'))}; "
                    f"refusal reason: {plain(answer.get('refusal_reason') or 'none')}.")
        rows.append("**Stage timings (ms):** " + "; ".join(f"{plain(key)}: {value}" for key, value in detail.get("timings_ms", {}).items()))
        rows.append("Full candidate text and scores are available in **Full results and usage → trace → diagnostics**.")
        sections.append("\n\n".join(rows[:table_start]) + "\n\n"
                        + "\n".join(rows[table_start:table_end]) + "\n\n"
                        + "\n\n".join(rows[table_end:]))
    return "\n\n---\n\n".join(sections) or "No retrieval stages recorded for this message."
