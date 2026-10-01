"""Visible, numbered RAG substeps built only from recorded execution data."""

from html import escape

from .json_view import json_content
from .security_notice import security_event_description

STEPS_CSS = """
.es-pipeline { display: grid; gap: 5px; margin: 12px 0 0; min-width: 0; }
.es-note { margin: 2px 0 4px; font-size: 10px; line-height: 1.5; color: var(--body-text-color, #475569); }
.es-step { --tc-accent: #475569; margin: 0; border: 1px solid var(--border-color-primary, #dce1e8); border-left: 3px solid var(--tc-accent); border-radius: 7px; overflow: hidden; color: var(--body-text-color, #1f2937); background: var(--block-background-fill, #fff); }
.es-query, .es-faiss { --tc-accent: #2563eb; }
.es-bm25 { --tc-accent: #0891b2; }
.es-rrf, .es-context { --tc-accent: #7c3aed; }
.es-reranker { --tc-accent: #b45309; }
.es-selected, .es-answer { --tc-accent: #047857; }
.es-security { --tc-accent: #b91c1c; }
.es-summary { display: flex; align-items: center; gap: 6px; padding: 7px 8px; cursor: pointer; list-style: none; background: color-mix(in srgb, var(--tc-accent) 5%, transparent); font-size: 11px; }
.es-summary::-webkit-details-marker { display: none; }
.es-summary:focus-visible { outline: 2px solid var(--tc-accent); outline-offset: -2px; }
.es-number { flex: none; font-size: 10px; font-variant-numeric: tabular-nums; opacity: .75; }
.es-title { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.es-count { flex: none; font-size: 10px; opacity: .8; }
.es-step[open] > .es-summary .tc-arrow { transform: rotate(90deg); }
.es-body { padding: 8px; border-top: 1px solid var(--border-color-primary, #dce1e8); }
.es-description { margin: 0 0 8px; font-size: 11px; line-height: 1.6; overflow-wrap: anywhere; white-space: pre-wrap; }
"""

STAGES = {
    "faiss": ("FAISS · semantic search", "Retrieved passages by similarity to the query."),
    "bm25": ("BM25 · keyword search", "Matched the query's words and numbered references."),
    "rrf": ("RRF · combine rankings", "Combined the semantic and keyword rankings."),
    "reranker": ("Rerank · cross-encoder", "Scored the candidates again for relevance to this query."),
    "selected": ("Select top passages", "Retained the highest-ranked passages for the answer pipeline."),
}


def format_execution_steps(data, call_number, *, arguments=None, cached=False, security_events=None):
    """Substep numbers belong to their tool call, not additional model turns."""
    data = data if isinstance(data, dict) else {}
    security_events = security_events or []
    retrieval = data.get("retrieval", {})
    trace = retrieval.get("trace", {}) if isinstance(retrieval, dict) else {}
    trace = trace if isinstance(trace, dict) else {}
    if not trace and not security_events:
        return ''
    steps = []

    def add(name, label, description, payload, count=''):
        index = f'{call_number}.{len(steps) + 1}'
        badge = ('Reused · ' if cached and count else 'Reused' if cached else '') + str(count)
        steps.append(f'<details class="es-step es-{name}"><summary class="es-summary">'
                     '<span class="es-number">' + index + '</span><span class="es-title">' + escape(label)
                     + '</span><span class="es-count">' + escape(badge) + '</span>'
                     '<span class="tc-arrow" aria-hidden="true"></span></summary><div class="es-body">'
                     '<p class="es-description">' + escape(description) + '</p>'
                     + json_content(payload) + '</div></details>')

    def add_security(stages):
        for event in security_events:
            if event['stage'] in stages:
                add('security', event['label'], security_event_description(event), event, 'Injection blocked')

    query = trace.get("query")
    if isinstance(query, str):
        add('query', 'Query sent to retrieval', query,
            {"query": query, "tool_arguments": arguments or {}, "query_terms": trace.get("query_terms", [])})
    # Preserve the recorded order and omit disabled stages rather than implying
    # that every search must have run BM25 or a reranker.
    for stage in trace.get("stages", []):
        if not isinstance(stage, dict) or not stage.get("enabled"):
            continue
        name = stage.get("name")
        if name not in STAGES:
            continue
        label, description = STAGES[name]
        if name == 'bm25':
            description += ' Terms: ' + ', '.join(map(str, trace.get('query_terms', [])))
        count = stage.get('count')
        badge = f'{count} result' + ('s' if count != 1 else '') if type(count) is int else ''
        add(name, label, description, stage, badge)
    add_security({'retrieval_screening'})
    context = data.get("context", {})
    if isinstance(context, dict) and context.get("applied"):
        add('context', 'Expand section context', str(context.get('heading') or 'Added section passages.'),
            context, f'{context.get("chunk_count", 0)} passages')
    add_security({'context_screening', 'generation_request', 'generation_screening'})
    answer = data.get("answer", {})
    if isinstance(answer, dict) and answer.get("status"):
        # Some answers copy verified source lists without a model call. Never
        # label that as LLM generation, or a refusal as a successful answer.
        description = 'Recorded RAG answer outcome. The agent may combine it with other tools.'
        if answer.get('answer_mode'):
            description += ' Mode: ' + str(answer['answer_mode'])
        add('answer', 'RAG answer', description, answer, str(answer['status']))
    add_security({'agent_evidence', 'unknown', 'unknown_request'})
    note = ('Cached call: these recorded substeps were reused, not executed again.' if cached
            else f'Inside tool call {call_number} · recorded substeps')
    return '<div class="es-pipeline"><p class="es-note">' + escape(note) + '</p>' + ''.join(steps) + '</div>'
