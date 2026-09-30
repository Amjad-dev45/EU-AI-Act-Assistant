"""A RAG walkthrough populated only from the latest recorded search results."""
from html import escape
from .retrieval_trace import number

RAG_CSS = """
.rd-empty { padding: 22px; border: 1px dashed var(--border-color-primary); border-radius: 12px; color: var(--body-text-color); }
.rd-run { padding: 18px; border: 1px solid var(--border-color-primary); border-radius: 12px; margin: 12px 0; color: var(--body-text-color); }
.rd-query { font-size: 17px; font-weight: 600; margin: 0 0 8px; }
.rd-note { font-size: 12px; line-height: 1.65; margin: 8px 0; }
.rd-flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 16px 0; }
.rd-step { border: 1px solid var(--border-color-primary); border-top: 3px solid #2563eb; border-radius: 9px; padding: 12px; background: var(--background-fill-secondary); }
.rd-step b { font-size: 24px; display: block; margin: 5px 0; font-variant-numeric: tabular-nums; }
.rd-step span { font-size: 12px; }
.rd-step:nth-child(2) { border-top-color: #0891b2; }
.rd-step:nth-child(3) { border-top-color: #7c3aed; }
.rd-step:nth-child(4) { border-top-color: #b45309; }
.rd-step:nth-child(5) { border-top-color: #047857; }
.rd-candidates { display: flex; flex-wrap: wrap; gap: 8px; }
.rd-candidates > details { flex: 1 1 200px; min-width: 0; padding: 9px 12px; border: 1px solid var(--border-color-primary); border-radius: 8px; }
.rd-run summary { cursor: pointer; font-size: 13px; padding: 5px 0; }
.rd-run summary:focus-visible { outline: 2px solid #2563eb; outline-offset: 3px; }
.rd-hit { padding: 10px; margin: 6px 0; border-radius: 7px; background: var(--background-fill-secondary); }
.rd-hit p { font-size: 12px; line-height: 1.65; margin: 5px 0; }
.rd-source { font-weight: 600; overflow-wrap: anywhere; }
.rd-score { font-variant-numeric: tabular-nums; }
.rd-passage { white-space: pre-wrap; max-height: 260px; overflow: auto; overflow-wrap: anywhere; }
.rd-results { max-height: 520px; overflow: auto; }
.rd-selected { margin: 14px 0; }
.rd-selected h4 { margin-bottom: 10px; }
@media (max-width: 650px) { .rd-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); } .rd-run { padding: 12px; } }
"""


def text(value):
    return escape(str(value), quote=True)


def _hit(hit, selected=False):
    pages = ", ".join(map(str, hit.get("pages", [])))
    rank = hit.get("selected_rank") if selected else hit.get("rank")
    before = hit.get("fusion_rank") or hit.get("faiss_rank") or "—"
    after = hit.get("rerank_rank") or "—"
    title = f"#{rank} · {hit.get('source', 'Source')} · pages {pages}"
    if selected:
        title += f" · rank {before} → {after}"
    scores = " · ".join(f"{label} {number(hit.get(key))}" for label, key in
                        (("Cosine", "score"), ("BM25", "bm25_score"), ("RRF", "rrf_score"), ("Reranker", "rerank_score")) if key in hit)
    return ('<details class="rd-hit"><summary class="rd-source">' + text(title) + '</summary>'
            '<p class="rd-score">' + text(scores) + '</p><p class="rd-passage">' + text(hit.get("text", "")) + '</p></details>')


def format_rag_demo(trace):
    runs = []
    for call in trace if isinstance(trace, list) else []:
        if not isinstance(call, dict):
            continue
        data = call.get("diagnostics") or {}
        retrieval = data.get("retrieval", {})
        detail = retrieval.get("trace", {})
        if not detail:
            continue
        stages = {s['name']: s for s in detail.get('stages', [])}
        html = '<section class="rd-run"><p class="rd-query">' + text(detail.get("query", "")) + '</p>'
        if call.get("cached"):
            html += '<p class="rd-note"><strong>Cached search:</strong> showing the earlier execution.</p>'
        if detail.get("injection_test"):
            html += '<p class="rd-note"><strong>INJECTION TEST ON:</strong> a temporary attack chunk was added to both search indexes. The saved corpus is unchanged.</p>'
        index = detail.get("index", {})
        if index:
            html += '<p class="rd-note">Search index · ' + text(index.get("chunk_count", "—")) + ' chunks · '
            html += text(index.get("embedding_model", "—")) + ' · ' + text(index.get("dimensions", "—")) + ' dimensions'
            html += ' · largest indexed chunk: ' + text(index.get("largest_chunk_characters", "—")) + ' characters</p>'
        html += '<p class="rd-note">' + text(detail.get("policy_explanation", "")) + '</p><div class="rd-flow">'
        for name, label, description in (("faiss", "1 · FAISS", "Semantic candidates"),
                ("bm25", "2 · BM25", "Keyword candidates"), ("rrf", "3 · Rank fusion", "Merged candidate shortlist"),
                ("reranker", "4 · Cross-encoder", "Candidates scored again"), ("selected", "5 · Top results", "Kept after reranking")):
            stage = stages.get(name, {})
            count = stage.get("count", "—") if stage.get("enabled") else "Off"
            html += f'<div class="rd-step"><span>{label}</span><b>{text(count)}</b><span>{description}</span></div>'
        html += '</div><p class="rd-note"><strong>BM25 terms:</strong> ' + text(", ".join(detail.get("query_terms", [])) or "none") + '</p>'
        html += '<div class="rd-selected"><h4>Selected evidence · before → after reranking</h4>'
        for hit in stages.get("selected", {}).get("results", []):
            html += _hit(hit, selected=True)
        html += '</div><div class="rd-candidates">'
        for name in ("faiss", "bm25", "rrf", "reranker"):
            stage = stages.get(name, {})
            if not stage.get("enabled"):
                continue
            html += '<details><summary>' + text(stage.get("label", name)) + ' · ' + text(stage.get("count", 0)) + '</summary><div class="rd-results">'
            html += ''.join(_hit(h) for h in stage.get("results", [])) + '</div></details>'
        html += '</div>'
        context = data.get("context", {})
        security = context.get("security", {})
        blocked = security.get("blocked", [])
        if blocked:
            html += '<p class="rd-note"><strong>Security guard:</strong> ' + str(len(blocked)) + ' suspicious passage(s) excluded from answer context.</p>'
            for item in blocked:
                html += '<p class="rd-note">' + text(item.get("source", "")) + ' · ' + text(', '.join(item.get("reasons", []))) + '</p>'
        elif security.get("screened"):
            html += '<p class="rd-note">Source screening: no command-pattern matches. This does not guarantee that a passage is safe.</p>'
        answer = data.get("answer", {})
        html += '<p class="rd-note"><strong>Answer generation:</strong> ' + text(context.get("chunk_count", "—")) + ' context passages'
        if context.get("applied"):
            html += ' after section expansion (' + text(context.get("heading") or "section context") + ')'
        html += '. Extra section passages are separate from the ranked top results. Status: ' + text(answer.get("status", "unknown")) + '.</p>'
        html += '<p class="rd-note"><strong>Refusal gate:</strong> top reranker score ' + number(retrieval.get("top_rerank_score"))
        html += ' / minimum ' + number(retrieval.get("min_rerank_score")) + '. ' + text(detail.get("score_note", "")) + '</p>'
        timing = detail.get("timings_ms", {})
        html += '<p class="rd-note">' + ' · '.join(text(k.replace('_', ' ')) + ': ' + text(v) + ' ms' for k, v in timing.items()) + '</p></section>'
        runs.append(html)
    return ''.join(runs) or '<div class="rd-empty">Ask a corpus question to see the live retrieval pipeline, reranking changes, and five selected passages here. A message that uses no RAG tool has no retrieval results.</div>'
