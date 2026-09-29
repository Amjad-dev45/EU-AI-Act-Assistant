"""Standalone tools for a teammate's agent loop; no UI or model calls here."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import operator
from pathlib import Path

from .messaging import MessageTool, SlackSender


def schema(name, description, properties=None):
    properties = properties or {}
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}


TOOL_SCHEMAS = [
    schema("search_corpus", "Locate relevant evidence when its page is unknown. Returns source pages and citation IDs. "
           "Use read_pages to inspect known pages or surrounding context.",
           {"query": {"type": "string"}}),
    schema("read_pages", "Read known PDF pages from a listed corpus document, without semantic search. "
           "Use to expand search results. Page numbers are 1-based PDF positions, not printed page labels. "
           "Read at most 3 pages per call; long pages are explicitly marked truncated.",
           {"source": {"type": "string"}, "start_page": {"type": "integer", "minimum": 1},
            "end_page": {"type": "integer", "minimum": 1}}),
    schema("calculator", "Calculate arithmetic using +, -, *, /, //, %, and parentheses.",
           {"expression": {"type": "string"}}),
    schema("list_documents", "List available source documents."),
    schema("today", "Get the current UTC date and time for questions relative to today. "
           "Does not retrieve rules or determine whether a legal obligation applies."),
    schema("send_message", "Propose an exact message to the configured Slack channel for human approval. "
           "This does not send until a human confirms through the host application.",
           {"text": {"type": "string"}}),
]

MAX_READ_PAGES = 3
MAX_PAGE_CHARACTERS = 12000


def today():
    """Use explicit UTC so answers do not depend on the server's local timezone."""
    now = datetime.now(timezone.utc)
    return {"date": now.date().isoformat(), "datetime": now.isoformat(), "timezone": "UTC"}


def calculator(expression):
    """Restrict simpleeval to short numeric arithmetic; disallow power and calls."""
    from simpleeval import SimpleEval
    if not isinstance(expression, str) or not 1 <= len(expression) <= 200:
        raise ValueError("Use an expression of 1 to 200 characters")
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Only numeric arithmetic is allowed")
        if isinstance(node, ast.Constant) and (
                type(node.value) not in (int, float) or not math.isfinite(node.value)
                or abs(node.value) > 1e15):
            raise ValueError("Numbers must be finite and at most 1e15")
    evaluator = SimpleEval(operators={ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.UAdd: operator.pos, ast.USub: operator.neg},
        names={}, functions={})
    result = evaluator.eval(expression)
    if not math.isfinite(result) or abs(result) > 1e100:
        raise ValueError("Result is too large")
    return {"value": result}



class CorpusTools:
    """Create one instance per question/session; reuse it across that run's searches.

    run_tool is the model-facing dispatcher. approve_message is deliberately
    excluded: the host calls it only after a person reviews and confirms a proposal.
    """
    def __init__(self, retriever, sender=None, k=5, trace=None):
        if type(k) is not int or not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        self.retriever, self.k, self.trace = retriever, k, trace
        self.passages = []
        self._by_id = {}
        self._messages = MessageTool(sender if sender is not None else SlackSender.from_env())

    def search_corpus(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Search query must contain 1 to 4000 characters")
        retrieved = self.retriever.retrieve(query, k=self.k, trace=self.trace)
        selected = []
        for passage in retrieved["passages"]:
            key = (passage["source"], passage["chunk_id"])
            if key not in self._by_id:
                item = {**passage, "citation_id": len(self.passages) + 1}
                self._by_id[key] = item
                self.passages.append(item)
            selected.append(self._by_id[key])
        return {"passages": selected, "hint": "Rephrase if evidence is missing or irrelevant."}

    def list_documents(self):
        return {"documents": sorted({c["source"] for c in self.retriever.chunks})}

    def read_pages(self, source, start_page, end_page):
        """Read bounded PDF evidence; allow only corpus sources within sources/."""
        import pymupdf
        if not isinstance(source, str) or source not in self.list_documents()["documents"]:
            raise ValueError("Choose a source returned by list_documents")
        if (type(start_page) is not int or type(end_page) is not int or
                start_page < 1 or end_page < start_page or end_page - start_page >= MAX_READ_PAGES):
            raise ValueError("Use an inclusive range of 1 to 3 pages, starting at page 1 or higher")
        root = (Path(self.retriever.data_dir) / "sources").resolve()
        path = (root / source).resolve()
        if Path(source).is_absolute() or not path.is_relative_to(root) or path.suffix.lower() != ".pdf":
            raise ValueError("Source must be a PDF within the corpus sources folder")
        selected = []
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            if end_page > document.page_count:
                raise ValueError(f"Document has only {document.page_count} pages")
            for number in range(start_page, end_page + 1):
                text = document[number - 1].get_text("text", sort=True)
                selected.append({"source": source, "pages": [number],
                    "chunk_id": f"pdf-page:{number}",
                    "document_id": hashlib.sha256(source.encode()).hexdigest(),
                    "source_reference": str(path), "text": text[:MAX_PAGE_CHARACTERS],
                    "score": None, "evidence_type": "pdf_page",
                    "truncated": len(text) > MAX_PAGE_CHARACTERS,
                    "extracted_characters": len(text),
                    "text_status": "extracted" if text.strip() else "no_extractable_text"})
            page_count = document.page_count
        # Register only after all requested pages have been read successfully.
        for item in selected:
            key = ("pdf_page", source, item["pages"][0])
            if key not in self._by_id:
                item["citation_id"] = len(self.passages) + 1
                self._by_id[key] = item
                self.passages.append(item)
        passages = [self._by_id[("pdf_page", source, n)] for n in range(start_page, end_page + 1)]
        return {"source": source, "page_count": page_count, "passages": passages,
                "truncated": any(p["truncated"] for p in passages),
                "hint": "Pages use 1-based PDF positions. Empty text may need OCR; this tool does not perform OCR."}

    def send_message(self, text):
        return self._messages.propose(text)

    def approve_message(self, proposal_id, approved):
        """Host-only entry point. Never expose this method as an LLM tool."""
        return self._messages.decide(proposal_id, approved)

    def run_tool(self, name, arguments_json):
        """Return JSON, including recoverable errors, to the calling agent loop."""
        functions = {"search_corpus": self.search_corpus, "calculator": calculator,
                     "list_documents": self.list_documents, "send_message": self.send_message,
                     "read_pages": self.read_pages, "today": today}
        try:
            if name not in functions:
                raise ValueError("Unknown tool")
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            result = functions[name](**arguments)
            return json.dumps(result, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            # Do not expose provider errors that could contain credentials.
            return json.dumps({"error": type(exc).__name__,
                               "hint": "Check tool arguments or rephrase and try again."})
