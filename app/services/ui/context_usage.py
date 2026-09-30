"""Accessible bars for actual token usage in the latest model requests."""
from html import escape
from contracts.model_usage import context_snapshot

CONTEXT_CSS = """
.cu-panel { padding: 12px 14px; border: 1px solid var(--border-color-primary, #ddd); border-radius: 10px; }
.cu-meter + .cu-meter { margin-top: 12px; }
.cu-label { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 12px; font-size: 12px; color: var(--body-text-color); }
.cu-label strong { font-weight: 600; }
.cu-track { height: 9px; border-radius: 9px; overflow: hidden; background: var(--background-fill-secondary, #e2e8f0); margin-top: 7px; }
.cu-fill { display: block; height: 100%; border-radius: 9px; background: #2563eb; }
.cu-rag .cu-fill { background: #7c3aed; }
.cu-high .cu-fill { background: #b45309; }
.cu-note { font-size: 11px; margin: 6px 0 0; color: var(--body-text-color); opacity: .8; }
"""


def _meter(snapshot, label, style=""):
    used, limit = snapshot.get("used_tokens"), snapshot.get("limit_tokens")
    known = type(used) is int and used >= 0 and type(limit) is int and limit > 0
    if known:
        percent = used / limit * 100
        description = f"{used:,} / {limit:,} tokens · {percent:.1f}%"
        bar = (f'<div class="cu-track" role="meter" aria-label="{escape(label)}" aria-valuemin="0" '
               f'aria-valuemax="{limit}" aria-valuenow="{min(used, limit)}" aria-valuetext="{description}">'
               f'<span class="cu-fill" style="width: {min(percent, 100):.4f}%"></span></div>')
        if percent >= 80:
            style += " cu-high"
        note = f"{snapshot['prompt_tokens']:,} input + {snapshot['completion_tokens']:,} output tokens"
    else:
        description = f"— / {limit:,} tokens" if type(limit) is int else "Usage unavailable"
        bar = '<div class="cu-track" aria-hidden="true"></div>'
        note = "Measured after a model response; no estimate shown."
    return (f'<div class="cu-meter {style}"><div class="cu-label"><strong>{escape(label)}</strong>'
            f'<span>{escape(description)}</span></div>{bar}<p class="cu-note">{escape(note)}</p></div>')


def format_context(body=None):
    context = (body or {}).get("context_window", {})
    agent = context.get("agent") or context_snapshot(None)
    html = _meter(agent, "Context window · latest agent request")
    rag = context.get("rag")
    if isinstance(rag, dict) and rag.get("model"):
        html += _meter(rag, "RAG answer · separate context", "cu-rag")
    history = context.get("history_messages_sent")
    note = "Per-request usage, not cumulative tokens. History is capped at 3 recent exchanges."
    if type(history) is int:
        note += f" Sent {history} previous messages on this turn."
    return '<div class="cu-panel">' + html + '<p class="cu-note">' + escape(note) + '</p></div>'
