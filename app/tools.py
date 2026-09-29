"""Standalone tools for a teammate's agent loop; no UI or model calls here."""

import ast
import json
import math
import operator

from .messaging import MessageTool, SlackSender


def schema(name, description, properties=None):
    properties = properties or {}
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}


TOOL_SCHEMAS = [
    schema("search_corpus", "Search the document corpus for evidence with source pages and citation IDs.",
           {"query": {"type": "string"}}),
    schema("calculator", "Calculate arithmetic using +, -, *, /, //, %, and parentheses.",
           {"expression": {"type": "string"}}),
    schema("list_documents", "List available source documents."),
    schema("send_message", "Propose an exact message to the configured Slack channel for human approval. "
           "This does not send until a human confirms through the host application.",
           {"text": {"type": "string"}}),
]


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

    def send_message(self, text):
        return self._messages.propose(text)

    def approve_message(self, proposal_id, approved):
        """Host-only entry point. Never expose this method as an LLM tool."""
        return self._messages.decide(proposal_id, approved)

    def run_tool(self, name, arguments_json):
        """Return JSON, including recoverable errors, to the calling agent loop."""
        functions = {"search_corpus": self.search_corpus, "calculator": calculator,
                     "list_documents": self.list_documents, "send_message": self.send_message}
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
