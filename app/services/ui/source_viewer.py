"""Clickable citations and on-demand PDF previews, separate from model context."""
import base64
from html import escape

from contracts.httpkit import call_json
from contracts.schemas import new_request_id
from .chat import AGENT_URL
from .sharing import submit_with_sharing, clear_with_sharing

SOURCE_LAYOUT_CSS = """
#source-links .wrap { display: flex; flex-direction: column; gap: 5px; }
#source-links label { width: 100%; box-sizing: border-box; overflow-wrap: anywhere; cursor: pointer; }
#source-links input[type=radio] { flex-shrink: 0; }
"""
PREVIEW_CSS = """
.pv-box { border: 1px solid var(--border-color-primary); border-radius: 9px; overflow: hidden; }
.pv-box summary { cursor: pointer; padding: 9px; font-size: 12px; font-weight: 600; }
.pv-box img { display: block; width: 100%; height: auto; background: white; }
.pv-open { border: 0; padding: 0; cursor: zoom-in; width: 100%; background: white; }
.pv-note { font-size: 11px; line-height: 1.5; padding: 8px; margin: 0; }
.pv-excerpt { white-space: pre-wrap; font-size: 12px; padding: 8px; max-height: 180px; overflow: auto; }
.pv-dialog { position: fixed; top: 50%; left: 50%; right: auto; bottom: auto; transform: translate(-50%, -50%); margin: 0 !important; height: fit-content; box-sizing: border-box; width: min(1000px, 94vw); max-height: 92vh; overflow: auto; border: 0; border-radius: 10px; padding: 12px; background: var(--background-fill-primary, white); color: var(--body-text-color, black); }
.pv-dialog::backdrop { background: rgb(0 0 0 / .65); }
.pv-dialog img { width: 100%; height: auto; }
.pv-close { float: right; padding: 6px 12px; cursor: pointer; border: 1px solid #aaa; border-radius: 6px; }
"""
PREVIEW_JS = """
element.addEventListener('click', (event) => {
  const dialog = element.querySelector('dialog');
  if (!dialog) return;
  if (event.target.closest('[data-preview-open]')) dialog.showModal();
  if (event.target.closest('[data-preview-close]') || event.target === dialog) dialog.close();
});
"""


def source_options(response):
    """Resolve citation IDs to the actual recorded passages, including cached calls."""
    evidence = {}
    for call in response.get('trace', []):
        if call.get('type') != 'tool':
            continue
        output = call.get('output', {})
        if not isinstance(output, dict):
            continue
        if output.get('cached'):
            output = output.get('result', {})
        for row in output.get('evidence', []):
            evidence[(row.get('id'), row.get('chunk_id'))] = row
    choices, items = [], {}
    for citation in response.get('citations', []):
        document = citation.get('document_id') or citation.get('source')
        row = evidence.get((citation.get('id'), citation.get('chunk_id')), {})
        # Metadata must agree before any passage is used for highlighting.
        same = (row.get('source') == citation.get('source') and row.get('pages') == citation.get('pages'))
        excerpt = row.get('text', '')[:12000] if same else ''
        for page in citation.get('pages', []):
            key = f"{response.get('request_id')}:{citation.get('id')}:{page}"
            choices.append((f"[{citation.get('id')}] {citation.get('source')} · page {page}", key))
            items[key] = {'request_id': response.get('request_id'), 'document_id': document,
                          'page': page, 'excerpt': excerpt}
    return choices, items


def _with_sources(outputs):
    values = list(outputs)
    choices, items = source_options(values[6])
    values[4] = {'__type__': 'update', 'choices': choices, 'value': None}
    return *values, items, ''


def submit_with_sources(message, session, injection_test=False):
    return _with_sources(submit_with_sharing(message, session, injection_test))


def clear_with_sources():
    return _with_sources(clear_with_sharing())


def open_source(choice, items, displayed_response):
    """Only a source from the displayed answer can initiate a preview request."""
    item = items.get(choice) if isinstance(items, dict) else None
    if not item or item['request_id'] != (displayed_response or {}).get('request_id'):
        return '<p>Select a source from the latest answer.</p>'
    payload = {**item, 'request_id': new_request_id('preview')}
    try:
        status, result = call_json('POST', f'{AGENT_URL}/v1/source-preview', payload, 25, 'agent')
        if status != 200:
            return '<p>The original PDF preview is unavailable. Check that its file is in data/sources.</p>'
        encoded = result['image_base64']
        if (result.get('source') != item['document_id'] or result.get('page') != item['page']
                or not isinstance(encoded, str) or len(encoded) > 8_000_000
                or not base64.b64decode(encoded, validate=True).startswith(b'\x89PNG\r\n\x1a\n')):
            raise ValueError('Invalid PDF preview')
    except Exception:
        return '<p>Could not load this PDF preview. You can try clicking the source again.</p>'
    title = escape(f"{result['source']} · page {result['page']}")
    image = f'<img src="data:image/png;base64,{encoded}" alt="{title} — source excerpt">'
    return ('<details class="pv-box" open><summary>' + title + '</summary>'
            '<button class="pv-open" data-preview-open aria-label="Enlarge source preview">' + image + '</button>'
            '<p class="pv-note">' + escape(result.get('message', '')) + ' Click the image to enlarge.</p>'
            '<details><summary>Extracted passage</summary><div class="pv-excerpt">' + escape(item['excerpt']) + '</div></details></details>'
            '<dialog class="pv-dialog"><button class="pv-close" data-preview-close>Close</button><p>' + title + '</p>' + image + '</dialog>')
