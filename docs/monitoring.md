# RAG evaluation and monitoring

The existing pipeline remains `Retriever → Answerer → Gradio/CLI`. Instrumentation is
implemented in `app/monitoring/` (usage, tracing, storage); evaluation is in
`app/evaluation/` (local metrics and the optional judge).
There are no new monitoring services, databases to deploy, or RAG frameworks.

## Which metrics mean what?

| Metric | During ordinary use? | Labels required? | Meaning / limitation |
|---|---|---|---|
| Tokens, costs, timings, errors | Always | No | Measured per request/API call; costs are estimates |
| Retrieved/context/cited chunk and document counts | Always | No | Distinct document IDs versus individual chunks |
| Similarity and mean similarity | Always | No | FAISS cosine similarity, **not** confidence or semantic relevance |
| Citation validity | Always when labels are cited | No | Fraction of distinct cited labels that exist in the context; not factual support |
| Context relevance | Optional paid judge | No | Mean judge-assigned relevance (0–1) over all retrieved passages |
| Answer relevance | Optional paid judge | No | Judge estimate (0–1) of how well the answer addresses the question |
| Faithfulness / groundedness | Optional paid judge | No | Fraction of judge-extracted factual claims supported by their cited passages |
| Answer supported | Optional paid judge | No | True if all extracted claims are supported, false if any are unsupported |
| Expected-page recall / all pages found | Offline evaluation | Source/page labels | Expected pages covered by at least one retrieved chunk |
| Page-hit precision@k / MRR | Offline evaluation | Source/page labels | Coarse passage relevance proxy; a page match may miss the actual evidence |
| Chunk precision@k, recall@k, MRR, nDCG@k | Offline evaluation | Exact relevant chunk IDs | Binary relevance ranking metrics; labels should identify all relevant chunks |
| Exact match / token F1 | Offline evaluation | Expected answer | Lexical comparison, not semantic correctness; valid paraphrases can score poorly |
| Expected-status match | Offline evaluation | Expected behavior/category | Correct refusal, clarification, or answered status |
| Reference correctness | Optional paid offline judge | Expected answer | Semantic comparison against a labelled reference, also an estimate |

Missing labels and disabled/failed evaluations produce `null` or an explicit status,
not a zero score. Pure refusals/clarification with no factual claims have `null`
faithfulness and support. A refusal can still be judged for relevance; whether it is
the *correct* refusal requires a reference expectation or manual review.

The model judge sees the question, answer and retrieved passages. It is instructed
to ignore instructions in those inputs. Outputs are structured and validated: each
passage must be scored once, scores must lie in [0,1], and a supported claim must
reference an actually cited passage. Judge failure does not discard the chatbot's
answer. These checks do not establish that the judge is right: calibrate with human
review, especially for legal questions. Using the same model family for generation
and judging can introduce correlated errors.

“Used” cannot be observed inside a language model. The system logs which chunks were
**provided in context** and which were **explicitly cited**; cited-document counts
are the observable proxy for usage. Citations alone are never reported as proof of
support. FAISS candidates are saved in `retrieval_candidates`; the final reranked chunks
are saved in `retrieval` and enter context. Both retain cosine similarity; final
chunks also carry `rerank_score` and their original `retrieval_rank`.

## Request record and accounting

Each record has a unique request ID, conversation ID, UTC start time, outcome,
original query, actual search query, optional rewritten query, and corpus fingerprint.
The application strips query whitespace and reranks retrieved passages locally.
It does not rewrite queries: `rewritten_query` stays `null`. Reranking configuration
and candidate counts are recorded in `metadata.reranking`.

`api_calls` records operation, requested/returned model, response ID, latency,
provider-reported usage, pricing snapshot and estimated cost. Usage is captured
immediately after the response, before validating generated answers or judge JSON.
Thus a truncated answer or invalid citation still contributes its billable usage.
API failures retain prior successful calls and mark the failed call's usage unknown.
Exception types, rather than raw provider exception payloads, are persisted.

`usage` includes input, output, total and cached input tokens, separated into:

- `main_llm`: answer generation;
- `embedding`: question embeddings (input only);
- `additional_llm`: evaluation and any instrumented future rewrite/reranking/summary calls;
- `by_operation`: each named operation separately.

Input totals include embedding tokens. `total_tokens` sums all observed call totals;
cached tokens are a subset of input tokens and are not added again. No local tokenizer
estimate replaces provider-reported usage. If an attempted call has missing usage,
request totals are `null`, `complete` is false, and `known_tokens` retains the subtotal.

`cost` separates main LLM input/output, embeddings, evaluation, reranking, other
LLM calls and totals. `additional_llm_usd` includes evaluation/reranking; those are
breakdowns, not extra amounts to add to the total. No-call components are zero.
Unpriced models or missing usage yield `null`, with a `known_total_usd` subtotal.

All prices live in `config/pricing.json` (USD per million tokens, standard service).
The default rates were checked on 2026-09-28 against the official
[chat model](https://developers.openai.com/api/docs/models/gpt-4o-mini) and
[embedding model](https://developers.openai.com/api/docs/models/text-embedding-3-small)
pages. Cached input is discounted separately. Explicit snapshot aliases are in the
same configuration; unknown models never silently inherit another model's price.
Update this file and restart the app to change pricing, or set `RAG_PRICING_FILE`.
Historical records retain their applied rates and version; they are not repriced.

These are estimated costs for observable calls, not a provider invoice. Hidden SDK
retries, interrupted processes, calls made elsewhere, taxes, custom pricing, or
provider usage not returned to the app may differ from the provider dashboard.
One-time corpus indexing in `scripts/build_index.py` is not charged to each question
and remains separately reported by that script.

## Timings and aggregates

Timers use `time.perf_counter()` and store milliseconds for preprocessing,
embedding API generation, FAISS retrieval (including vector preparation and result
mapping), local reranking, context building, LLM generation, and optional evaluation.
The first reranking timing includes lazy model loading/download. Disabled reranking
has a `null` timing. Its API cost is zero; local compute is not priced.

`total_latency_ms` covers the Answerer call through validation and optional evaluation,
including orchestration overhead. It excludes SQLite persistence, PDF/image rendering,
Gradio queue wait, and browser transport. It is a **backend pipeline latency**, not
full browser response time. Stage timings need not sum exactly to the total.

The monitor CLI reports totals and per-question averages, p50/p95 latency using
nearest-rank percentiles, average stage timings, error/status counts, source counts,
quality estimates, and operation-level costs/tokens. Every recorded attempt counts
in the denominator, including validation errors and failed requests. If any request's
cost/usage is unknown, the corresponding whole-group total/average is `null` and
known subtotals plus missing counts are shown. Empty groups have zero totals and
`null` averages. Quality averages include only evaluated samples, with the number of
completed judge evaluations reported separately.

The Gradio session ID is initialized separately per browser session, kept in `gr.State`,
and never shared globally. It groups costs without adding conversational memory.
The CLI can reuse `--conversation-id`. Global monitoring is a local CLI operation;
the UI displays only its own session's statistics.

## Labelled evaluation

Use the existing `questions.json` format, for example (illustrative labels only):

```json
[
  {
    "question": "What is the stated amount?",
    "category": "factual",
    "difficulty": "easy",
    "expected_answer": "100 euros.",
    "expected_chunks": [{"source": "example.pdf", "page": 2}],
    "expected_chunk_ids": [7, 8],
    "expected_status": "answered"
  },
  {
    "question": "What is the weather today?",
    "category": "out-of-scope",
    "expected_status": "insufficient_evidence"
  }
]
```

`expected_chunks` retains the course's name but contains **source/page pairs**.
`expected_chunk_ids` is optional and must use the exact type/value of chunk IDs in
the current index; rechunking can invalidate those labels. Null expected answers/evidence from the course dataset are treated as not applicable.
Repeated source/page and
chunk IDs are deduplicated. Labels must belong to the evaluated corpus version.
`out-of-scope` and `ambiguous` infer refusal and clarification expectations when an
explicit `expected_status` is absent. Other categories need explicit behavior labels
for status accuracy. Reference-answer matching removes numeric citation labels,
case-folds and compares word tokens; it does not interpret negation or paraphrases.

Run `scripts/evaluate.py --data-dir data` to validate without spending. Add `--run`
to call the existing Answerer for each question, and `--judge` for semantic estimates
and reference correctness. `--questions`, `--limit`, `--k`, and `--output` are optional.
Reports are written atomically after each question, with a latest copy at
`data/eval_results.json`, and include dataset/corpus hashes,
answers, retrieval metadata, timings, usage, costs, and category/difficulty scorecards
with metric-specific sample counts. There is no invented overall correctness score.
Errors remain in the report; the runner continues to the next question and exits
nonzero if a pipeline/judge/persistence error occurred. Low quality scores are findings,
not execution errors. Compare saved reports using the same dataset and corpus.

## Storage and extension

`logs/requests.sqlite3` is local and gitignored. Set `RAG_METRICS_DB` to relocate it.
SQLite uses a separate connection and transaction per operation with a busy timeout,
so concurrent workers do not share cursors or overwrite records. Requests persist
across restarts. Logging failures are emitted to the server log and flagged in the
returned result/UI; the answer still returns. Totals cannot include records that
could not be persisted. Interrupted requests may not reach final persistence.

Stored retrieval entries contain stable document IDs (explicit ID or SHA-256 of source
name), chunk IDs, source/file/URL references, page numbers, score, rank and citation ID.
The log stores query and answer text, but **does not store retrieved passage text or
full documents**, system prompts, API keys, or raw API responses. Judge explanations
and claim summaries may quote short answer fragments. Retain/share this local log
accordingly. No automatic retention policy or external telemetry exporter is enabled.

Future additional model calls should go through:

```python
response = trace.api_call(
    "reranking", "reranking", model,
    lambda: client.chat.completions.create(...),
)
```

This adds provider usage, cost and timings automatically; add that model's rate to
the central pricing file. Pure local stages use `with trace.stage("reranking"):`.
If adding rewriting, populate `rewritten_query` and `search_query`. If changing context
selection, record only the actual included chunk/document IDs before generation.
Non-token-priced reranking services would need their own cost adapter; none is used
by this app.

The course-named `scripts/eval_runner.py` command delegates to the same implementation as `scripts/evaluate.py`; both support the same flags. See [architecture](architecture.md) for module ownership.
