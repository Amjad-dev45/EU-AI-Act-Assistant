"""Expandable cards for recorded retrieval stages, scores, and exact JSON."""

from html import escape

from .json_view import json_content, json_view
from .tool_calls import TOOL_CSS, trace_card

RETRIEVAL_CSS = TOOL_CSS + """
.rt-stages { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 10px; margin: 12px 0; align-items: start; }
.rt-stages .tc-body { padding-top: 10px; }
.rt-table { overflow: auto; margin: 8px 0; }
.rt-table table { width: 100%; border-collapse: collapse; font-size: 11px; white-space: nowrap; }
.rt-table th, .rt-table td { padding: 7px; text-align: right; border: 1px solid var(--border-color-primary, #dce1e8); }
.rt-table th:first-child, .rt-table td:first-child { text-align: left; }
.rt-table th { font-weight: 600; }
"""

STAGE_COLORS = {"faiss": "search", "bm25": "keyword", "rrf": "quote",
                "reranker": "reranker", "selected": "calculator"}


def text(value):
    return escape(str(value), quote=True)


def number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _selected_table(hits):
    headers = ("Source / pages", "FAISS rank", "BM25 rank", "RRF rank", "Reranked", "Cosine", "BM25", "Reranker")
    rows = []
    for hit in hits:
        source = str(hit.get("source", "")) + " / " + ", ".join(map(str, hit.get("pages", [])))
        ranks = [str(hit.get(key) or "—") for key in ("faiss_rank", "bm25_rank", "fusion_rank", "rerank_rank")]
        scores = [number(hit.get(key)) for key in ("score", "bm25_score", "rerank_score")]
        rows.append('<tr>' + ''.join('<td>' + text(value) + '</td>' for value in (source, *ranks, *scores)) + '</tr>')
    return ('<section class="tc-section"><h4>Selected chunks · ranks before and after reranking</h4>'
            '<div class="rt-table"><table><thead><tr>' + ''.join('<th>' + label + '</th>' for label in headers)
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>')


def format_retrieval(trace):
    """Share the tool-card design; expose every candidate without generating data."""
    cards = []
    for entry in trace if isinstance(trace, list) else []:
        data = entry.get("diagnostics") if isinstance(entry, dict) else None
        if not isinstance(data, dict):
            continue
        retrieval = data.get("retrieval", {})
        detail = retrieval.get("trace", {})
        if not isinstance(detail, dict) or not detail:
            continue
        stages = [stage for stage in detail.get("stages", []) if isinstance(stage, dict)]
        body = '<p class="tc-muted">' + text(detail.get("policy_explanation", "")) + '</p>'
        if entry.get("cached"):
            body += '<p class="tc-muted"><strong>Cached result:</strong> these stages were reused, not executed again.</p>'
        body += '<p class="tc-muted"><strong>BM25 terms:</strong> ' + text(", ".join(detail.get("query_terms", [])) or "none") + '</p>'
        if detail.get("numbered_references"):
            body += '<p class="tc-muted"><strong>Numbered references:</strong> ' + text(", ".join(detail["numbered_references"])) + '</p>'
        body += '<div class="rt-stages">'
        for stage in stages:
            enabled = stage.get("enabled")
            count = stage.get("count", 0)
            preview = f'{count} result' + ('s' if count != 1 else '') + ' · expand to inspect JSON' if enabled else 'Disabled for this search'
            body += trace_card(stage.get("label", stage.get("name", "Stage")), preview,
                               json_content(stage), color=STAGE_COLORS.get(stage.get("name"), "other"))
        body += '</div>'
        selected = next((s for s in stages if s.get("name") == "selected"), {})
        body += _selected_table(selected.get("results", []))
        body += '<p class="tc-muted">' + text(detail.get("score_note", "")) + '</p>'
        context, answer = data.get("context", {}), data.get("answer", {})
        body += '<p class="tc-muted"><strong>Answer context:</strong> ' + text(context.get("chunk_count", 0)) + ' passages · '
        body += text(context.get("heading") or context.get("mode") or "ranked passages")
        body += ' · expanded: ' + text(context.get("applied", False)) + ' · complete section: ' + text(context.get("complete")) + '</p>'
        body += '<p class="tc-muted"><strong>Relevance gate:</strong> top score ' + number(retrieval.get("top_rerank_score"))
        body += ' · minimum ' + number(retrieval.get("min_rerank_score")) + ' · answer: ' + text(answer.get("status", "unknown"))
        body += ' · refusal reason: ' + text(answer.get("refusal_reason") or "none") + '</p>'
        body += '<p class="tc-muted"><strong>Stage timings (ms):</strong> ' + text(
            '; '.join(f'{key}: {value}' for key, value in detail.get("timings_ms", {}).items()) or 'Not recorded') + '</p>'
        body += json_view(data, "Complete retrieval record · JSON")
        badges = '<span class="tc-status">Cached</span>' if entry.get("cached") else ''
        cards.append(trace_card(f'Search {len(cards) + 1}', detail.get("query", ""), body,
                                color="search", meta='Observed stages · expand to inspect', badges=badges))
    return '<div class="tc-list">' + ''.join(cards) + '</div>' if cards else '<p class="tc-empty">No retrieval stages recorded for this message.</p>'
