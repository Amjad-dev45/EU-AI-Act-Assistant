"""Five bounded read-only tools; corpus access stays behind the RAG HTTP API."""
import ast
import hashlib
import json
import math
import operator
import re
import time

from simpleeval import SimpleEval
from contracts.injection_guard import source_risks, SourceInjectionError


def function(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
            "strict": True, "parameters": {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}}}


TOOLS = [
    function("search_corpus", "Search the AI Act corpus for legal facts. Returns actual passages and a grounded answer that can be selected unchanged.",
             {"query": {"type": "string"}}),
    function("calculator", "Evaluate bounded numeric arithmetic, including percentages written as division by 100.",
             {"expression": {"type": "string"}}),
    function("quote_exact", "Verify and copy literal source wording, not a generated summary. First obtain the actual source filename and page using search_corpus, unless the user supplied them. Never guess document IDs or pages. Copy a substantive phrase from evidence; when a chunk spans pages, try its other cited page if the phrase is not found. A null phrase returns the full page for inspection.",
             {"document_id": {"type": "string"}, "page": {"type": "integer"},
              "phrase": {"type": ["string", "null"]}}),
    function("lookup_definition", "Find the definition of any term in the supplied corpus, with source/page citations. Checks structured definitions, then searches the entire corpus using hybrid retrieval and reranking, with no article restriction. Prefer this for definitions; for comparisons call once per term. Pass only the term, not a question. A mere mention is insufficient: no supported definition returns found=false, and ambiguous definitions require clarification.",
             {"term": {"type": "string"}}),
    function("get_application_dates", "Look up application dates and exceptions in the uploaded Act, without guessing from semantic matches. Choose topic: general, ai_literacy, prohibited_practices, gpai, governance, penalties, gpai_fines, conformity_bodies, confidentiality, high_risk_products, high_risk, transparency, legacy_gpai, legacy_high_risk, large_scale_it. For models already on the market use legacy_gpai; preserve all conditions in returned clauses. Other deadlines use search_corpus.",
             {"topic": {"type": "string"}}),
]


def calculate(expression):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("Use an arithmetic expression of at most 200 characters.")
    tree = ast.parse(expression.strip(), mode="eval")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
              ast.Pow: operator.pow}
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.UAdd, ast.USub, *binary)
    nodes = list(ast.walk(tree))
    if len(nodes) > 50 or any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("Only bounded numeric arithmetic is supported.")
    def bounded(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e18:
            raise ValueError("Arithmetic exceeds the supported numeric range.")
        return value
    for node in nodes:
        if isinstance(node, ast.Constant):
            bounded(node.value)
    def wrap(operation, power=False):
        def apply(a, b):
            if power and abs(b) > 12:
                raise ValueError("Exponents must be between -12 and 12.")
            return bounded(operation(a, b))
        return apply
    operations = {kind: wrap(operation, kind is ast.Pow) for kind, operation in binary.items()}
    operations.update({ast.UAdd: lambda a: bounded(a), ast.USub: lambda a: bounded(-a)})
    result = SimpleEval(operators=operations, names={}, functions={}).eval(expression.strip())
    return {"expression": expression, "result": bounded(result)}


class ToolSession:
    def __init__(self, request_id, transport, rag_url, deadline, *, injection_test=False):
        self.request_id, self.transport = request_id, transport
        self.rag_url, self.deadline = rag_url.rstrip("/"), deadline
        self.evidence, self.grounded, self.cache = {}, {}, {}
        self.usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        self.injection_test = injection_test
        self.embedding_tokens = 0
        self.last_diagnostics, self.diagnostics_cache = None, {}

    def add_evidence(self, hit):
        text, pages = hit.get("text"), hit.get("pages")
        document = hit.get("document_id") or hit.get("source")
        source = hit.get("source") or document
        if (not isinstance(text, str) or not text.strip() or not isinstance(pages, list) or not pages
                or any(type(page) is not int or page < 1 for page in pages)
                or any(not isinstance(value, str) or not value for value in (document, source, hit.get("chunk_id")))):
            raise ValueError("RAG returned evidence without valid source metadata.")
        risks = source_risks(text)
        if risks:
            raise SourceInjectionError("The source contains instructions aimed at the assistant and was excluded.",
                blocked={"source": source, "pages": pages, "chunk_id": hit["chunk_id"],
                         "reasons": risks, "stage": "agent_evidence"})
        row = {"text": text, "pages": pages, "document_id": document,
               "source": source, "chunk_id": hit["chunk_id"]}
        for number, old in self.evidence.items():
            if all(old[key] == value for key, value in row.items()):
                return old
        row["id"] = len(self.evidence) + 1
        self.evidence[row["id"]] = row
        return row

    def http(self, path, arguments):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Agent deadline reached.")
        status, result = self.transport("POST", self.rag_url + path,
                {"request_id": self.request_id, **arguments}, min(90, remaining), "rag")
        if status != 200:
            raise ValueError(f"RAG returned HTTP {status}.")
        if not isinstance(result, dict):
            raise ValueError("RAG returned an invalid response.")
        return result

    def search(self, arguments):
        query = arguments.get("query")
        if set(arguments) != {"query"} or not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Search needs one nonempty query of at most 2000 characters.")
        result = self.http("/v1/answer", {"query": query, **({"injection_test": True} if self.injection_test else {})})
        return self.read_answer(result)

    def read_answer(self, result):
        """Share evidence validation, citation remapping and cost/trace accounting."""
        self.last_diagnostics = {
            "retrieval": result.get("retrieval", {}), "context": result.get("context", {}),
            "answer": {key: result.get("grounded_answer", {}).get(key) for key in
                       ("status", "answer_mode", "refusal_reason", "top_rerank_score", "usage", "model", "security")},
        }
        self.embedding_tokens += max(0, int(result.get("retrieval", {}).get("embedding_tokens") or 0))
        hits = result.get("hits", result.get("context", {}).get("hits", []))
        if not isinstance(hits, list) or len(hits) > 64:
            raise ValueError("RAG returned an invalid evidence list.")
        rows = [self.add_evidence(hit) for hit in hits if isinstance(hit, dict) and hit.get("text", "").strip()]
        output = {"evidence": rows}
        grounded = result.get("grounded_answer")
        if isinstance(grounded, dict):
            usage = grounded.get("usage", {})
            for key in self.usage:
                self.usage[key] += max(0, int(usage.get(key) or 0))
            if grounded.get("status") in ("answered", "refused") and isinstance(grounded.get("answer"), str):
                local, citations = {}, []
                for citation in grounded.get("citations", []):
                    number = citation.get("id")
                    if type(number) is not int or not 1 <= number <= len(rows):
                        raise ValueError("Grounded answer cites unseen evidence.")
                    row = rows[number - 1]
                    if any(citation.get(key) != row[key] for key in ("chunk_id", "source", "pages")):
                        raise ValueError("Grounded citation metadata does not match its passage.")
                    local[number] = row["id"]
                    citations.append({key: row[key] for key in ("id", "chunk_id", "document_id", "source", "pages")})
                def marker(match):
                    number = int(match[1])
                    if number not in local:
                        raise ValueError("Grounded answer contains an unknown citation marker.")
                    return f"[{local[number]}]"
                answer = re.sub(r"\[(\d+)\]", marker, grounded["answer"])
                if grounded["status"] == "answered" and not citations:
                    raise ValueError("A grounded factual answer needs citations.")
                token = f"rag-{len(self.grounded) + 1}"
                self.grounded[token] = {"answer": answer, "citations": citations, "status": grounded["status"]}
                output["grounded_answer"] = {"id": token, **self.grounded[token]}
        if not rows:
            output['retrieval_hint'] = ('No usable source evidence was returned. For a quotation request, '
                'retry search_corpus with the requested content, keeping its topic and qualifications '
                'but omitting quote-formatting instructions. Do not guess document IDs or pages.')
        return output

    def quote(self, arguments):
        if set(arguments) != {"document_id", "page", "phrase"}:
            raise ValueError("Quote needs document_id, page and nullable phrase.")
        document, page, phrase = arguments["document_id"], arguments["page"], arguments["phrase"]
        if (not isinstance(document, str) or not document or len(document) > 128
                or type(page) is not int or page < 1
                or (phrase is not None and (not isinstance(phrase, str) or not phrase or len(phrase) > 2000))):
            raise ValueError("Invalid exact-quote arguments.")
        known = [row for row in self.evidence.values() if row['document_id'] == document]
        try:
            result = self.http("/v1/quote", arguments)
        except ValueError as exc:
            if str(exc) != 'RAG returned HTTP 404.':
                raise
            other_pages = sorted({p for row in self.evidence.values()
                                  if row['document_id'] == document and page in row['pages']
                                  for p in row['pages'] if p != page})
            if other_pages:
                hint = f" This retrieved chunk also covers pages {other_pages}; try those pages."
            elif any(page in row['pages'] for row in known):
                hint = " Inspect the retrieved page with phrase=null and copy its literal text."
            else:
                hint = " Call search_corpus to obtain the actual source filename and page before retrying. Do not guess other pages."
                if self.evidence and not known:
                    documents = sorted({row['document_id'] for row in self.evidence.values()})
                    hint += f" Use an exact filename from source evidence: {documents}."
            raise ValueError(f"Exact quote was not found on page {page}." + hint) from None
        text, citation = result.get("text"), result.get("citation", {})
        if not result.get("found") or not isinstance(text, str) or not text:
            raise ValueError("Exact quote was not found.")
        if citation.get("document_id") != document or citation.get("pages") != [page]:
            raise ValueError("Quote provenance does not match the requested page.")
        if phrase is not None and text != phrase:
            raise ValueError("The returned phrase is not exact.")
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        row = self.add_evidence({"text": text, "document_id": document, "source": document,
                                "pages": [page], "chunk_id": f"quote:{document}:{page}:{digest}"})
        return {"found": True, "quote": text, "evidence": [row]}

    def lookup(self, name, arguments):
        """Register direct source clauses as evidence using the normal citation IDs."""
        field, route = ('term', '/v1/definition') if name == 'lookup_definition' else ('topic', '/v1/application-dates')
        if (set(arguments) != {field} or not isinstance(arguments[field], str)
                or not arguments[field].strip() or len(arguments[field]) > 200):
            raise ValueError(f'{name} needs one nonempty {field} of at most 200 characters.')
        result = self.http(route, arguments)
        if name == 'lookup_definition' and result.get('mode') == 'corpus_search':
            output = self.read_answer(result)
            grounded = output.get('grounded_answer', {})
            return {**output, 'term': result.get('term'), 'mode': 'corpus_search',
                    'found': grounded.get('status') == 'answered', 'message': result.get('message')}
        hits = result.get('hits', [])
        if not isinstance(hits, list) or len(hits) > 8:
            raise ValueError('Lookup returned invalid evidence.')
        rows = [self.add_evidence(hit) for hit in hits]
        if result.get('found') is True and not rows:
            raise ValueError('Lookup claimed success without source evidence.')
        return {**{k:v for k,v in result.items() if k not in {'hits', 'request_id', 'mock', 'duration_ms'}},
                'evidence': rows}

    def execute(self, name, arguments):
        self.last_diagnostics = None
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if key in self.cache:
            self.last_diagnostics = self.diagnostics_cache.get(key)
            return {"cached": True, "result": self.cache[key]}, True
        # Failed calls also count: a duplicate must not repeat network I/O.
        self.cache[key] = {"error": "previous_call_failed", "message": "The previous identical call failed; do not repeat it."}
        if name == "search_corpus":
            result = self.search(arguments)
        elif name == "calculator":
            if set(arguments) != {"expression"}:
                raise ValueError("Calculator needs an expression.")
            result = calculate(arguments["expression"])
        elif name == "quote_exact":
            result = self.quote(arguments)
        elif name in {'lookup_definition', 'get_application_dates'}:
            result = self.lookup(name, arguments)
        else:
            raise ValueError("Unknown tool.")
        self.cache[key] = result
        self.diagnostics_cache[key] = self.last_diagnostics
        return result, False
