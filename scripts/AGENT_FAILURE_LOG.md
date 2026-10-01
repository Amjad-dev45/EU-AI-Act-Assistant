# Agent integration checks

## 2026-09-30: preserve the question at the RAG boundary

Initial live test: UI callback → Agent HTTP → RAG HTTP, with the five existing
priority questions. Q2, Q31 and Q32 failed answer inspection. The first wrong
step was shortening each self-contained question into keywords. This changed the
retrieved fine regime for Q2 and lost the explicit list intent for Q31/Q32.
Q13 and Q26 retained the expected answers. The evidence is saved in
`logs/agent_priority_initial_query_rewrite.json` and the corresponding full traces.

Category: **query intent lost** (3 of 5 priority questions). Prompt guidance alone improved Q2 and Q32 but left Q31 incomplete: the model
still shortened its query. That attempt is preserved in
`logs/agent_priority_prompt_only.json`. The final integration enforces the
original user wording on the first search when no history is present; the trace
records the effective arguments while retaining the model proposal. Follow-ups
may form a standalone question from history; unsuccessful searches may still be
rephrased within the existing six-step cap. Retrieval and answer prompts are
unchanged. Final priority rerun is recorded in
`logs/agent_priority_integration.json`; final review follows below.

These are integration checks over existing student questions. No agent scenario
questions or expected answers were generated or added to `data/questions.json`.
This is not the full agent evaluation or the required agent-vs-RAG comparison.

Final check: all five priority questions pass provisional assistant review. The
first effective search matched each full original question. All 183 offline tests
pass. The final run used 35078 chat tokens and zero embedding tokens; earlier
failed attempts are retained separately and are not included in that total.
Calculator, exact-quote, conversation, and Docker smoke checks follow after this
priority gate. No additional full 44-question answer run is needed for this wiring change.

## Direct replies and deployed services

A real Docker greeting exposed an empty final reply. Bounded validation feedback
alone repeated that blank reply up to the existing six-step cap. The first wrong
instruction was allowing an empty answer when selecting a stored RAG answer;
the model over-applied it to direct replies. The prompt now requests a short
acknowledgement for stored answers and actual nonempty text for direct replies.
Invalid replies remain hidden and recovery stays within the same cap.

The five priority questions were rerun after that prompt change. All five pass
provisional assistant review; source questions and expected answers are unchanged.
The final offline suite has 185 passing tests. Traces retain both the model's
proposed query and the actual first-query wording used by the RAG tool.

Docker startup also found a stray non-assignment line in the existing .env.
`scripts/run_app.py` uses the existing dotenv parser and forwards only application
settings to Compose, leaving .env and its key unchanged. Agent/RAG run with the
host UID/GID to keep newly written cache files and logs readable locally.

The real Docker UI -> Agent -> FAISS/reranker/RAG path successfully answered Q2.
UI greeting/calculator/follow-up/exact-quote results are saved separately in
`logs/docker_ui_smoke.json`; full traces remain under `logs/agent_runs/`.

Final Docker checks pass: greeting without tools, calculator (2,450,000),
contextual follow-up division (1,225,000), exact quote from the real corpus,
connection check, and New chat reset. The indexed-RAG Docker check also passes.
One further parsing correction accepts the model's literal string "null" as the
absent grounded-answer ID; it cannot name a real rag-N answer. All source and
citation checks remain in place. The unchanged final priority prompt passed all
five questions; the final parser has 186 passing offline tests, including this
case. Reports: `logs/docker_ui_smoke.json`, `logs/docker_rag_smoke.json`,
`logs/agent_priority_review.json`, and `logs/agent_integration_summary.json`.

## Scope regression: unsupported direct answers

Observed failure: existing Q39, “How do I train a neural network in PyTorch?”,
returned a full tutorial with status `answered` and no tool calls. The UI carried
six history messages. Source transcript: `logs/agent_runs/1790798104209000603-e538d3d9.json`.
Category: **scope violation / unsupported direct generation** (one observed failure).
The agent prompt described its corpus role but did not explicitly reject unrelated
requests, and final-answer validation accepted any uncited answer before retrieval.

Fix: explicit scope instructions plus a final code boundary. Without validated
source citations or a stored grounded answer, only whole-message conversational
requests or rendered successful calculator results are allowed. Other outputs
become a fixed refusal, including attempted tutorials labeled as refusals. Scope
decisions are logged separately from injection detection. Unsupported model text
is retained in the full trace but is not inserted into the chat answer/history.
No topic blacklist, extra classifier call, or forced retrieval was introduced.

Validation: 212 offline tests pass, including unsupported direct generation with
and without history, factual answers without current evidence, failed searches/
calculations, normal conversation, calculator follow-ups, and supported answers.
Five priority live answers (Q2/Q13/Q26/Q31/Q32) pass provisional assistant inspection
against existing student references. Then ten live checks passed: Q39 with the
original six-message history; Q38–Q41 with empty history; greeting; capabilities;
calculator; arithmetic follow-up; and exact quote. Source reports are
`logs/agent_scope_priority.json` and `logs/agent_scope_scope.json`. These are targeted
regression checks, not an overall answer-accuracy measurement. No student dataset
entries, retrieval settings, or RAG answer prompts changed.

## Quotation completion and page provenance

Three user traces exposed a compound-task bug:
- `1790797938288898086-d292f40e.json`: quote_exact succeeded on page 51, but selecting
  a stored RAG answer discarded the quote and returned only the summary.
- `1790798609743382392-5965b40d.json`: the model chose page 50 from a chunk spanning
  pages 50–51; lookup failed, and the stored-answer shortcut hid the limitation.
- `1790798647135669709-b31e885d.json`: the model skipped quote_exact entirely and
  reused a generated quotation with an incorrect inline page-50 claim.

Categories: **incomplete multi-tool task** and **unverified quotation/provenance**.
Fix: require a successful quote tool result before completing a quoted-source
request, block the summary-only shortcut, and append the verified literal text
and exact page in application code. A 404 response now suggests the other pages
of the matching retrieved chunk. Recovery uses the existing step cap; failed
verification produces a visible limitation. Rejected intermediate final replies
are never saved as the result if later recovery fails.

Validation: all 220 offline tests pass. Five priority live answers pass provisional
review against existing references, followed by three successful replays of the
actual failed requests/history and one successful short-phrase test. Two replays
explicitly retried page 51 after a page-50 miss; every returned quote matched the
real page text exactly and appeared in the final answer with a page-51 citation.
The replays took 6, 5, 4, and 2 model steps respectively. Reports are
`logs/agent_quote_priority.json` and `logs/agent_quote_regression.json`.
These are targeted integration checks, not measured overall accuracy. Retrieval,
RAG generation prompts, source data, and student questions were not changed.


## Quotation formatting polluted retrieval (2026-10-01)

The prompt “Explain Article 4 and include a verified exact quotation.” failed in
`1790869847180471711-687a5c08.json` (fresh) and
`1790869814039172234-7f5cb07a.json` (with history). The full formatting request
received a top reranker score of -6.759, below the unchanged -5 relevance gate.
The agent then guessed AI_Act/ai_act filenames and arbitrary pages.

Fix: remove only a trailing quote-presentation clause from the effective search
query, retaining the substantive question and requiring quote_exact for the
original request. Empty retrieval and unknown quote locations now give explicit
content-search recovery guidance. Actual source filenames and page metadata must
be obtained from evidence or explicitly supplied by the user; the strict quote
endpoint, relevance threshold and six-step cap are unchanged.

Validation: 254 offline tests pass. Both original requests replay successfully in
five model steps, including the original four-message history. Each run recovered
from page 50 to page 51 using the retrieved cross-page evidence; its quotation was
checked as an exact substring of corpus page 51 and appears in the final answer.
Saved traces: `logs/agent_runs/1790870176216638907-c1b3c8a4.json` and
`logs/agent_runs/1790870188616023942-95e200c1.json`. The separate content-query probe
returned Article 4 first with score 1.040. These are focused regression checks,
not an overall accuracy claim; model-dependent recovery can still use extra steps.


## Contextual examples and summaries rejected (2026-10-01)

The memory demo's “Give me an example of each.” and “Summarize that difference
in one sentence.” were refused despite the provider/deployer conversation being
present in the five-message history. Original traces:
`1790873115282078707-69221912.json` and
`1790873123103066572-b32182db.json`. The model attempted an answer from history
without new evidence, so the scope guard correctly blocked it. Live regression
checks also exposed stale history citation IDs and a search for examples in the
source, instead of the underlying definitions needed to illustrate the concepts.

Fix: contextual follow-ups receive explicit guidance to resolve their subjects
and retrieve fresh evidence for every compared term. Illustrative examples use
generic actors and an explicit hypothetical label; requests for documented or
real-world examples still require evidence of those examples. Summaries retain
the requested length. A skipped retrieval or stale citation gets one bounded
evidence-recovery attempt within the existing six-step budget. Failed retrieval
or repeated unsupported answers still lead to refusal; citation membership and
scope checks remain active. No extra classifier call or history expansion.

Validation: 257 offline tests pass, including skipped-tool recovery, stale citation
recovery, refusal when tools are still skipped, and failed-retrieval refusal. Both
original requests replay successfully with their five-message history. The full
four-prompt memory sequence also passes: provider definition, comparison, labelled
hypothetical examples, then a one-sentence summary. Every contextual response
retrieves both definitions and returns current citations in three model steps;
the initial provider question takes two. The final request sends five preceding
individual messages. Traces under `logs/agent_runs/`:

- Original examples replay: `1790873625590621097-1e7a544c.json`
- Original summary replay: `1790873628156608992-00a2fcb5.json`
- Full sequence: `1790873630447691921-1b088d90.json`,
  `1790873633363936771-166f8450.json`,
  `1790873636808553804-1cc654ab.json`,
  `1790873642077354383-4d423721.json`

These are targeted regression checks. Citation membership does not prove every
generated statement is factually complete; overall answer accuracy is separate.
