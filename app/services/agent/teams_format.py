"""Render simple HTML answer tables as native Adaptive Card 1.2 columns."""

from html.parser import HTMLParser
import re
from urllib.parse import urlsplit, quote


def _text(value, **style):
    return {"type": "TextBlock", "text": value, "wrap": True, **style}


def _table(rows):
    """Use columns supported by our existing card version; preserve cell order."""
    if not rows:
        return []
    count = max(len(row) for row in rows)
    widths = [min(100, max(16, max((len(row[i][0]) for row in rows if i < len(row)), default=0)))
              for i in range(count)]
    result = []
    for number, row in enumerate(rows):
        columns = []
        for i in range(count):
            value, header = row[i] if i < len(row) else ("", False)
            columns.append({"type": "Column", "width": widths[i], "items": [
                _text(value or " ", weight="Bolder" if header else "Default")]})
        result.append({"type": "ColumnSet", "columns": columns, "spacing": "Small",
                       "separator": number > 0})
    return result


class _AnswerTableParser(HTMLParser):
    """Read content and basic inline formatting; never forward HTML or CSS."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.text, self.rows = [], [], []
        self.in_table = False
        self.row = self.cell = None
        self.header = False
        self.hidden = []
        self.links = []

    def append(self, text):
        if self.cell is not None:
            self.cell.append(text)
        else:
            self.text.append(text)

    def flush_text(self):
        text = ''.join(self.text).strip()
        if text:
            self.blocks.append(_text(text))
        self.text = []

    def close_cell(self):
        if self.cell is not None:
            self.row.append((''.join(self.cell).strip(), self.header))
            self.cell = None

    def close_row(self):
        self.close_cell()
        if self.row:
            self.rows.append(self.row)
        self.row = None

    def close_table(self):
        self.close_row()
        self.flush_text()  # Captions precede the rows.
        self.blocks.extend(_table(self.rows))
        self.rows, self.in_table = [], False

    def handle_starttag(self, tag, attributes):
        if tag in {'script', 'style', 'head', 'iframe'}:
            self.hidden.append(tag)
            return
        if self.hidden:
            return
        attrs = dict(attributes)
        if tag == 'table':
            if self.in_table:
                raise ValueError("Nested HTML tables cannot be shared. Ask for a simple table.")
            self.flush_text()
            self.in_table = True
        elif tag == 'tr' and self.in_table:
            self.close_row()
            self.row = []
        elif tag in {'th', 'td'} and self.in_table:
            if any(attrs.get(name, '1') != '1' for name in ('colspan', 'rowspan')):
                raise ValueError("Merged HTML cells cannot be shared. Ask for a table without merged cells.")
            if self.row is None:
                self.row = []
            self.close_cell()
            self.cell, self.header = [], tag == 'th'
        elif tag == 'br':
            self.append('\n')
        elif tag in {'p', 'div', 'section', 'blockquote', 'h1', 'h2', 'h3'}:
            self.append('\n\n')
        elif tag in {'strong', 'b'}:
            self.append('**')
        elif tag in {'em', 'i'}:
            self.append('_')
        elif tag == 'li':
            self.append('\n- ')
        elif tag == 'a':
            url = attrs.get('href') or ''
            try:
                valid = urlsplit(url).scheme in {'https', 'http'} and not re.search(r'\s', url)
            except ValueError:
                valid = False
            self.links.append(url if valid else '')
            if valid:
                self.append('[')
        elif tag == 'img' and attrs.get('alt'):
            self.append(attrs['alt'])

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag in {'th', 'td'} and self.in_table:
            self.close_cell()
        elif tag == 'tr' and self.in_table:
            self.close_row()
        elif tag == 'table' and self.in_table:
            self.close_table()
        elif tag in {'strong', 'b'}:
            self.append('**')
        elif tag in {'em', 'i'}:
            self.append('_')
        elif tag in {'p', 'div', 'section', 'blockquote', 'li', 'h1', 'h2', 'h3'}:
            self.append('\n\n')
        elif tag == 'a' and self.links:
            url = self.links.pop()
            if url:
                self.append('](' + quote(url, safe=":/?#[]@!$&'*+=;%,") + ')')

    def handle_data(self, data):
        if not self.hidden:
            self.append(data)


def answer_blocks(answer):
    """Plain/Markdown answers stay verbatim; HTML tables become readable rows."""
    if not re.search(r'<table\b', answer, re.I):
        return [_text(answer)]
    # Models sometimes wrap an HTML result in a code fence. Remove only fences
    # around an entire HTML table; leave ordinary quoted code untouched.
    answer = re.sub(r'```(?:html)?\s*\n\s*(<table\b[\s\S]*?</table>)\s*```', r'\1', answer, flags=re.I)
    parser = _AnswerTableParser()
    parser.feed(answer)
    parser.close()
    if parser.in_table:
        parser.close_table()
    parser.flush_text()
    return parser.blocks or [_text("The HTML table contains no text.")]
