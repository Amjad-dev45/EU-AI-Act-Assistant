"""Read-only JSON inspection with native expandable objects and escaped values."""

from html import escape
import json

JSON_CSS = """
.ji-view { margin-top: 10px; border: 1px solid var(--border-color-primary, #dce1e8); border-radius: 8px; overflow: hidden; }
.ji-view > summary { padding: 9px 10px; cursor: pointer; font-size: 12px; font-weight: 600; background: var(--background-fill-secondary, #f1f5f9); }
.ji-content { min-width: 0; color: var(--body-text-color, #1f2937); }
.ji-tree, .ji-raw pre { margin: 0; padding: 10px; max-height: 360px; overflow: auto; background: var(--background-fill-secondary, #f1f5f9); font: 11px/1.7 ui-monospace, SFMono-Regular, Consolas, monospace; }
.ji-line, .ji-node > summary, .ji-fallback { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
.ji-node { margin: 0; }
.ji-node > summary { padding: 3px 0; cursor: pointer; }
.ji-node > summary:hover { background: color-mix(in srgb, var(--body-text-color, #1f2937) 5%, transparent); }
.ji-branch { padding-left: 12px; margin-left: 4px; border-left: 1px solid var(--border-color-primary, #dce1e8); }
.ji-line { padding: 2px 0; }
.ji-key { font-weight: 600; }
.ji-shape { opacity: .65; font-weight: normal; }
.ji-string { color: color-mix(in srgb, #16a34a 55%, var(--body-text-color, #1f2937)); }
.ji-number { color: color-mix(in srgb, #2563eb 55%, var(--body-text-color, #1f2937)); }
.ji-literal { color: color-mix(in srgb, #a855f7 55%, var(--body-text-color, #1f2937)); }
.ji-raw { padding: 5px 10px; border-top: 1px solid var(--border-color-primary, #dce1e8); }
.ji-raw > summary { cursor: pointer; font-size: 11px; padding: 4px 0; }
.ji-raw pre, .ji-fallback { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
.ji-content summary:focus-visible, .ji-view > summary:focus-visible { outline: 2px solid var(--tc-accent, #2563eb); outline-offset: -2px; }
"""


def _dump(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2), quote=True)


def _node(value, label=None, depth=0):
    key = '' if label is None else '<span class="ji-key">' + escape(str(label), quote=True) + '</span>: '
    if isinstance(value, (dict, list)) and value:
        # Deep payloads remain complete in a plain JSON block instead of adding
        # an impractical number of nested indentation levels.
        if depth >= 8:
            return '<div class="ji-line">' + key + '<pre class="ji-fallback">' + _dump(value) + '</pre></div>'
        marker, unit = ('{ }', 'field') if isinstance(value, dict) else ('[ ]', 'item')
        shape = f'{marker} · {len(value)} {unit}' + ('s' if len(value) != 1 else '')
        pairs = ((json.dumps(str(k), ensure_ascii=False), v) for k, v in value.items()) if isinstance(value, dict) else enumerate(value)
        children = ''.join(_node(v, k, depth + 1) for k, v in pairs)
        return ('<details class="ji-node"><summary>' + key + '<span class="ji-shape">'
                + shape + '</span></summary><div class="ji-branch">' + children + '</div></details>')
    style = 'string' if isinstance(value, str) else 'literal' if value is None or isinstance(value, bool) else 'number'
    return '<div class="ji-line">' + key + f'<span class="ji-{style}">' + _dump(value) + '</span></div>'


def json_content(value):
    """A field tree plus the exact, complete JSON for inspection/copying."""
    if isinstance(value, dict) and value:
        tree = ''.join(_node(v, json.dumps(str(k), ensure_ascii=False)) for k, v in value.items())
    else:
        tree = _node(value)
    return ('<div class="ji-content"><div class="ji-tree">' + tree + '</div>'
            '<details class="ji-raw"><summary>Raw JSON</summary><pre>' + _dump(value) + '</pre></details></div>')


def json_view(value, label="Inspect JSON"):
    return '<details class="ji-view"><summary>' + escape(label) + '</summary>' + json_content(value) + '</details>'
