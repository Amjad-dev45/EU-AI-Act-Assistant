"""Expandable tool cards built from observed calls, with all payload text escaped."""
from html import escape
import json
import math
from pathlib import Path
from .json_view import JSON_CSS, json_content, json_view
from .execution_steps import STEPS_CSS, format_execution_steps
from .security_notice import tool_security_events

TOOL_CSS = Path(__file__).with_name("tool_calls.css").read_text(encoding="utf-8") + JSON_CSS + STEPS_CSS
# Fixed colors and icons identify tools; status is also written out in text.
TOOLS = {
    "search_corpus": ("search", "RAG search", '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>'),
    "calculator": ("calculator", "Calculator", '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 11h2m4 0h2m-8 4h2m4 0h2m-8 3h2m4 0h2"/>'),
    "quote_exact": ("quote", "Exact quote", '<path d="M10 6H4v7h6V6Zm10 0h-6v7h6V6ZM10 13c0 4-2 5-4 6m14-6c0 4-2 5-4 6"/>'),
    "lookup_definition": ("definition", "Definition lookup", '<path d="M3 4h7l2 2 2-2h7v15h-7l-2 2-2-2H3V4Zm9 2v15"/>'),
    "get_application_dates": ("dates", "Application dates", '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 2v6m10-6v6M3 11h18m-13 4h3m3 0h3"/>'),
}


def _text(value):
    return escape(str(value), quote=True)


def trace_card(label, preview, body, *, color="other", meta="", badges="", icon=""):
    """Shared disclosure card for tool calls and observed retrieval stages."""
    icon = icon or '<path d="M8 4H4v16h4m8-16h4v16h-4M10 9l-3 3 3 3m4-6 3 3-3 3"/>'
    svg = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + icon + '</svg>'
    summary = (f'<summary class="tc-summary"><span class="tc-icon">{svg}</span><span class="tc-heading">'
               f'<span class="tc-title">{_text(label)}</span><span class="tc-preview">{_text(preview)}</span>'
               + (f'<span class="tc-meta">{_text(meta)}</span>' if meta else '') + '</span>'
               '<span class="tc-badges">' + badges + '</span><span class="tc-arrow" aria-hidden="true"></span></summary>')
    return f'<details class="tc-card tc-{color}">' + summary + '<div class="tc-body">' + body + '</div></details>'


def _arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _retrieval(data):
    """Show actual stage counts; retain complete scores/passages in a nested disclosure."""
    retrieval = data.get("retrieval", {})
    trace = retrieval.get("trace", {})
    if not isinstance(trace, dict) or not trace:
        return ""
    labels = {"faiss": "FAISS", "bm25": "BM25", "rrf": "RRF", "reranker": "Reranked", "selected": "Selected"}
    stages = [s for s in trace.get("stages", []) if isinstance(s, dict) and s.get("enabled")]
    chips = ''.join(f'<span class="tc-stage">{_text(labels.get(s.get("name"), s.get("label", "Stage")))} '
                    f'<b>{_text(s.get("count", 0))}</b></span>' for s in stages)
    terms = ", ".join(str(t) for t in trace.get("query_terms", []))
    content = '<section class="tc-section"><h4>Retrieval</h4><div class="tc-stages">' + chips + '</div>'
    if terms:
        content += '<p class="tc-muted"><strong>BM25 terms:</strong> ' + _text(terms) + '</p>'
    context = data.get("context", {})
    if isinstance(context, dict) and context.get("applied"):
        content += '<p class="tc-muted">Answer context: ' + _text(context.get("heading") or "expanded section")
        content += ' · ' + _text(context.get("chunk_count", 0)) + ' passages</p>'
    content += json_view(data, "Full retrieval details · JSON") + '</section>'
    return content


def _result(output):
    if not isinstance(output, dict):
        return '<p class="tc-result">' + _text(output if output is not None else "No output recorded.") + '</p>'
    if output.get("error"):
        return '<p class="tc-error">' + _text(output.get("message") or output["error"]) + '</p>'
    if "result" in output and not isinstance(output["result"], (dict, list)):
        return '<p class="tc-answer tc-number">' + _text(output["result"]) + '</p>'
    grounded = output.get("grounded_answer", {})
    answer = grounded.get("answer") if isinstance(grounded, dict) else None
    answer = answer or output.get("quote") or output.get("definition") or output.get('message')
    if not answer and output.get('rules'):
        answer = '\n\n'.join(row['text'] for row in output['rules'])
    content = '<p class="tc-answer">' + _text(answer) + '</p>' if answer else ''
    evidence = output.get("evidence", output.get("hits", []))
    if isinstance(evidence, list) and evidence:
        content += '<div class="tc-evidence">'
        for hit in evidence:
            if not isinstance(hit, dict):
                continue
            pages = hit.get("pages", [])
            pages = ", ".join(str(p) for p in pages) if isinstance(pages, list) else str(pages)
            content += '<p>' + _text(hit.get("source") or hit.get("document_id") or "Source")
            content += ' · pages ' + _text(pages) + '</p>'
        content += '</div>'
    return content or '<p class="tc-muted">See the recorded output below.</p>'


def format_tool_calls(trace):
    """Native details/summary supports mouse and keyboard without custom JavaScript."""
    cards = []
    for entry in trace if isinstance(trace, list) else []:
        if not isinstance(entry, dict) or not (entry.get("type") == "tool" or "tool" in entry):
            continue
        name = str(entry.get("name", entry.get("tool", "tool")))
        color, label, icon = TOOLS.get(name, ("other", name, '<path d="m8 5-6 7 6 7m8-14 6 7-6 7"/>'))
        args = _arguments(entry.get("arguments", {}))
        preview = (args.get("query") or args.get("expression") or args.get("phrase") or args.get("term") or args.get("topic") or args.get("document_id")
                   or "View inputs and results") if isinstance(args, dict) else args
        output = entry.get("output")
        # Duplicate calls retain the original result without claiming another execution.
        cached = bool(entry.get("cached"))
        if isinstance(output, dict) and output.get("cached") is True and "result" in output:
            output, cached = output["result"], True
        failed = isinstance(output, dict) and bool(output.get("error"))
        refused = isinstance(output, dict) and isinstance(output.get("grounded_answer"), dict) and output["grounded_answer"].get("status") == "refused"
        no_match = isinstance(output, dict) and output.get('found') is False
        status = "Failed" if failed else "No match" if no_match else "No answer" if refused else "Complete" if output is not None else "No result"
        badges = '<span class="tc-status' + (' tc-failed' if failed else '') + '">' + status + '</span>'
        security_events = tool_security_events(entry)
        if security_events:
            badges += '<span class="tc-status tc-failed">Injection blocked</span>'
        if cached:
            badges += '<span class="tc-status">Cached</span>'
        duration = entry.get("duration_ms")
        timing = ''
        if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
            timing = f'{duration / 1000:.1f}s' if duration >= 1000 else f'{duration:.0f}ms'
        body = '<section class="tc-section"><h4>Input JSON</h4>' + json_content(args) + '</section>'
        body += '<section class="tc-section"><h4>Result</h4>' + _result(output) + '</section>'
        data = entry.get("diagnostics")
        if isinstance(data, dict):
            body += _retrieval(data)
        if cached:
            body += '<p class="tc-muted">Reused the earlier result; this tool was not run again.</p>'
        body += json_view(output, "Output JSON · " + name)
        call_number = len(cards) + 1
        # Keep every substep inside its parent disclosure so collapsed calls
        # occupy only one card, even when a turn runs several searches.
        if name in {'search_corpus', 'lookup_definition'} or security_events:
            body = format_execution_steps(data, call_number, arguments=args, cached=cached,
                                          security_events=security_events) + body
        card = trace_card(f'{call_number}. {label}', preview, body, color=color, badges=badges, icon=icon,
                          meta='Tool call' + (f' · {timing}' if timing else ''))
        cards.append(card)
    security = [e for e in trace if isinstance(e, dict) and e.get("type") == "security"] if isinstance(trace, list) else []
    if security and not cards:
        return '<p class="tc-empty"><strong>Request blocked before tools ran.</strong><br>Step: input check. An explicit instruction override was detected. No model or tool call was made.</p>'
    scope = any(isinstance(e, dict) and e.get('type') == 'scope'
                and e.get('action') == 'no_supported_evidence' for e in trace) if isinstance(trace, list) else False
    if scope and not cards:
        return '<p class="tc-empty"><strong>No supported answer.</strong><br>The agent made no tool calls; the scope check prevented an answer without source evidence.</p>'
    if not cards:
        return '<p class="tc-empty">No tool calls for this message.</p>'
    return '<div class="tc-list">' + ''.join(cards) + '</div>'
