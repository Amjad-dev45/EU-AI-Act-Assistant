"""Accessible bars for actual token usage in the latest model requests."""
from html import escape
from contracts.model_usage import context_snapshot

# These target Gradio's outer wrappers, beyond the scoped HTML template.
CONTEXT_LAYOUT_CSS = """
#context-usage .html-container { padding: 0 !important; }
#conversation-context #conversation { border-width: 1px; border-color: var(--border-color-primary, #ddd); }
"""

CONTEXT_CSS = """
padding: 0; margin: 0;
.cu-panel { padding: 6px 10px; background: var(--block-background-fill, #fff); color: var(--body-text-color, #334155); }
.cu-meter { display: flex; align-items: center; gap: 8px; min-width: 0; font-size: 11px; line-height: 18px; }
.cu-title { flex: none; font-weight: 600; white-space: nowrap; }
.cu-name { flex: none; white-space: nowrap; opacity: .8; }
.cu-count { flex: none; white-space: nowrap; font-variant-numeric: tabular-nums; }
.cu-track { flex: 1; min-width: 16px; height: 6px; border-radius: 6px; overflow: hidden; background: var(--background-fill-secondary, #e2e8f0); }
.cu-fill { display: block; height: 100%; border-radius: 9px; background: #2563eb; }
.cu-rag .cu-fill { background: #7c3aed; }
.cu-high .cu-fill { background: #b45309; }
.cu-details { margin-top: 2px; font-size: 10px; line-height: 14px; }
.cu-details summary { cursor: pointer; width: fit-content; opacity: .8; }
.cu-details summary:focus-visible { outline: 2px solid #2563eb; outline-offset: 2px; }
.cu-details .cu-meter { margin-top: 6px; }
.cu-note { font-size: 10px; line-height: 1.4; margin: 5px 0 0; opacity: .8; }
@media (max-width: 600px) {
  .cu-panel { padding: 5px 7px; }
  .cu-meter { gap: 5px; font-size: 10px; }
}
"""


def _meter(snapshot, label, style="", title=""):
    used, limit = snapshot.get("used_tokens"), snapshot.get("limit_tokens")
    known = type(used) is int and used >= 0 and type(limit) is int and limit > 0
    if known:
        percent = used / limit * 100
        description = f"{used:,} / {limit:,} tokens"
        bar = (f'<div class="cu-track" role="meter" aria-label="{escape(label)}" aria-valuemin="0" '
               f'aria-valuemax="{limit}" aria-valuenow="{min(used, limit)}" aria-valuetext="{description} · {percent:.1f}%">'
               f'<span class="cu-fill" style="width: {min(percent, 100):.4f}%"></span></div>')
        if percent >= 80:
            style += " cu-high"
        note = f"{snapshot['prompt_tokens']:,} input + {snapshot['completion_tokens']:,} output tokens · {percent:.1f}% of the context window"
    else:
        description = f"— / {limit:,} tokens" if type(limit) is int else "Usage unavailable"
        bar = '<div class="cu-track" aria-hidden="true"></div>'
        note = "Measured after a model response; no estimate shown."
    heading = f'<strong class="cu-title">{escape(title)}</strong>' if title else ''
    meter = (f'<div class="cu-meter {style}">{heading}<span class="cu-name">{escape(label)}</span>'
             + bar + f'<span class="cu-count">{escape(description)}</span></div>')
    return meter, f"{label}: {note}"


def format_context(body=None):
    context = (body or {}).get("context_window", {})
    agent = context.get("agent") or context_snapshot(None)
    html, agent_note = _meter(agent, "Agent", title="Context window")
    notes = [agent_note]
    rag = context.get("rag")
    rag_meter = ''
    if isinstance(rag, dict) and rag.get("model"):
        rag_meter, rag_note = _meter(rag, "RAG answer", "cu-rag")
        notes.append(rag_note)
    history = context.get("history_messages_sent")
    note = "Latest request per model, not cumulative tokens. Agent and RAG use separate contexts. History is capped at 3 recent exchanges."
    if type(history) is int:
        note += f" Sent {history} previous messages on this turn."
    notes.append(note)
    # One line stays visible; RAG's separate meter and explanations expand below.
    details = '<details class="cu-details"><summary>Usage details</summary>' + rag_meter
    details += ''.join('<p class="cu-note">' + escape(text) + '</p>' for text in notes) + '</details>'
    return '<section class="cu-panel" aria-label="Context window">' + html + details + '</section>'
