# Retrieval failure log

## 2026-09-30: add cross-encoder reranking

Checked the existing first question in `data/questions.json`, with its expected
source `OJ_L_202401689_EN_TXT.pdf`, page 1. Both versions used the same saved FAISS
index and cached query embedding; no new OpenAI calls were made.

| Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|---|---|---|---|---|
| On what date was the AI Act (Regulation (EU) 2024/1689) adopted? | Factual | HIT in top 5 before and after | Expected page 1 was rank 4; repeated references ranked above it | Retrieve 20 vector candidates, rerank with `cross-encoder/ms-marco-MiniLM-L6-v2`, keep 5 | Improved to rank 2; still not rank 1 |

| Measure for this question | Vector only | With reranking |
|---|---:|---:|
| Expected-page rank | 4 | 2 |
| Hit@1 | 0 | 0 |
| Hit@3 | 0 | 1 |
| Hit@5 | 1 | 1 |
| Reciprocal rank | 0.25 | 0.50 |

The first returned passage still contains the adoption date. The metrics above
use the question's labelled page, not a judgement that other passages are wrong.
They describe one question, not dataset-wide recall or answer accuracy.

Model revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU, raw logit scores,
maximum 512 tokens per question/passage pair. Model loading took about 14 seconds;
the subsequent warm retrieval took about 1.2 seconds on this laptop. These are
single-run timings, not a benchmark.

Validation at this stage: all 14 offline tests passed. This single-question check
was followed by the full comparison below.

## 2026-09-30: full evaluation and scoring correction

Run `20260930T164552358452Z` completed all 44 questions. Questions 35 and 36 are
`on_topic_unanswerable` and carry contextual references. The evaluator initially
treated those references as answer labels, contrary to the toolkit's exclusion
rule. Scoring revision 2 excludes them from recall/MRR and regression counts;
it preserves the question file and all retrieved hits. Refusal correctness is
still unevaluated because this runner performs retrieval only.

The corrected denominator is 35 scored / 9 unscored. Recall@5 rises from 27/35
(77.14%) to 29/35 (82.86%), and MRR@5 from 0.5357 to 0.7643. Questions 17, 18,
and 22 improve from a top-five miss to a hit. Question 36 is no longer incorrectly
listed as a retrieval regression.

| Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|---|---|---|---|---|
| 26: Does a chatbot have to tell users that they are talking to an AI? | stress-test / yes_no | Vector HIT at 4; reranked MISS for labelled page 82 | Reranking places a passage mentioning chatbots above the transparency passage; page 33 remains relevant, but labelled page 82 drops out | No targeted retrieval change yet | No; investigate before the next retrieval change |
| 35 and 36: on-topic unanswerable questions | Evaluation | Not applicable | Contextual references were counted as answer labels | Exclude refusal cases from retrieval scoring in runner, metrics, and checker | Yes; recomputed stored scores without API calls |

The vector hit on page 82 for question 26 covers Article 50(2), whereas its
expected answer concerns Article 50(1). A page hit alone therefore does not prove
the correct passage was found. Review the evidence before changing retrieval or
student-authored labels.

Validation after the scoring correction: all 26 offline tests pass. The corrected
scorecard was derived from stored hits without re-embedding or running retrieval.

## 2026-09-30: add BM25 and Reciprocal Rank Fusion

The pipeline now retrieves 20 FAISS and 20 BM25 candidates from the same saved
chunks, fuses their ranks with RRF (constant 60), and reranks the best 20 down to
5. Cosine, BM25, RRF, and cross-encoder scores remain distinct. No corpus or
question changes, index rebuild, new embeddings, or OpenAI calls were needed.

An initial full run with separate word/number tokens reached 30/35 Recall@5
(85.71%), recovering question 33 but still missing Article 4. The tokenizer now
adds generic joined terms for numbered articles, sections, and annexes, alongside
the original words. This distinguishes Article 4 from arbitrary paragraphs
numbered 4 without any question-specific page or answer rules.

Final full run: `20260930T170533848794Z`; 44 questions evaluated, 35 scored, 9 excluded
from retrieval scoring; zero new embedding tokens. Vector-only and reranked
results match the earlier baseline. Hybrid reaches 31/35 Recall@5 (88.57%) and
MRR@5 0.7881, up from 29/35 (82.86%) and 0.7643 with reranking alone.

| Question | Category | Retrieval OK? | Problem / result | Fix applied | Fixed? |
|---|---|---|---|---|---|
| 24: What does Article 4 say? | stress-test / exact_term | Hybrid HIT at 1; both earlier modes MISS | Separate article/number tokens retrieved unrelated numbered paragraphs | Joined numbered-reference tokens + BM25/RRF; returned text includes Article 4's AI literacy provision on pages 50–51 | Yes for retrieval |
| 33: How long can testing outside a sandbox last? | stress-test / edge_of_chunk | Hybrid HIT at 3; reranked MISS | Vector candidates missed the duration passage | BM25 adds the passage on pages 92–93 containing the six-month limit and possible extension | Yes for retrieval |
| 16: From what date are Article 5 practices prohibited? | cross-reference / same_document | Rank 1 → 2; still HIT@3/5 | A passage about annual review of Article 5 prohibitions now ranks above the labelled page | Logged regression; no targeted rule applied | No; date evidence still needs manual review |
| 26: Does a chatbot have to disclose it is AI? | stress-test / yes_no | Still MISS for labelled page 82 | Existing reranking regression persists | BM25 did not restore the expected page in the top five | No |

No new Recall@3 or Recall@5 regressions from reranked → hybrid. Question 16
regresses at Recall@1; question 24 improves at Recall@1. Hybrid's remaining
labelled top-five misses are questions 10, 13, 23, and 26. Multi-page questions
still use the required any-page hit metric: a hit does not prove all evidence
needed for an answer is present.

Validation: all 40 offline tests pass. The full comparison uses the cached real
cross-encoder and stored question embeddings. This development-set measurement
does not establish answer accuracy, refusal quality, or unseen-query performance.

## 2026-09-30: grounded answer generation (live checks pending)

Added `scripts/answer_checker.py` and `app/services/rag/answering.py`. The generator
uses the unchanged hybrid top-five retrieval and `gpt-4o-mini`, returns cited
statements or a missing-evidence refusal, and rejects unknown citation IDs.
Python maps IDs back to the source/page metadata of the passages actually sent.
The prompt delimits retrieved text and says document instructions are not to be
followed; this is prompt hardening, not evidence that injection resistance has
been measured.

Offline checks cover citation mapping, bounded context, malformed/truncated
output, separate refusal/error outcomes, token usage, CLI selection, and record
saving. An additional SDK check uses a local mock HTTP transport to verify request
serialization and response parsing without network access. No live generation
requests were made, so answer accuracy and refusal quality remain unmeasured.

Next manual checks: `uv run scripts/answer_checker.py --priority` (current IDs 2,
13, 26, 31, 32), then question 35 for an on-topic unanswerable case. Review against
the student-authored answers; resolve priority failures before a full answer run.
Results are saved individually in `data/answer_runs/`. Existing retrieval misses
and the question 16 rank regression remain open; generation does not fix them.

## 2026-09-30: article-boundary attribution failure and targeted fix

The first live answer to question 24 (`20260930T171859741659Z`) was wrong despite
valid citation IDs and successful page retrieval. The top chunk contains the
end of a definition followed by `Article 4`, its heading, and the AI-literacy
obligation. The model attributed the preceding definition to Article 4, then
cited another incomplete definition on page 49 instead of stating the obligation.
This demonstrates why citation membership and retrieval hits cannot establish
answer correctness.

Prompt `article-sections-v2` presents visible article headings as separate
sections. Leading text has an unknown heading; no prior article is inferred.
The prompt says headings apply forward only and asks for the responsible parties,
required action, and qualifications. Raw text, citations, saved chunks, embeddings,
and FAISS are unchanged. The parser recognizes all 113 article numbers in the
current chunks, including headings joined to a preceding page's sentence.

The same five retrieved passages were replayed for a live question-24 check.
The new answer states that providers and deployers must take measures, to their
best extent, to ensure sufficient AI literacy of relevant staff and people acting
on their behalf, accounting for the qualifications in the provision. It cites the
original chunk on pages 50–51 and no longer claims the preceding definition is
part of Article 4.

| Question | Result of inspection against existing expected answer | Follow-up |
|---|---|---|
| 24: Article 4 | Correct obligation and qualifications, citation [1] | Targeted failure corrected in this live run |
| 2: maximum fine | Matches in substance, page 115 | No issue found in this check |
| 13: general application date | Matches in substance using page 44 | Labelled page 123 still missing from retrieval |
| 26: chatbot disclosure | Matches in substance using page 33 | Labelled page 82 still missing from retrieval |
| 31: Annex III areas | Incomplete: only two of eight expected areas | Retrieve the rest of the enumeration before claiming a complete answer |
| 32: prohibited practices | Incomplete: only (a) and part of (b) | Retrieve remaining items and qualifications; prompt alone did not solve coverage |
| 35: names of designated systemic-risk models | Correct missing-evidence refusal | No issue found in this check |

These are targeted observations, not a dataset-wide accuracy estimate. The five
priority questions do not all pass, so no full answer evaluation was run. Earlier
priority-generation outputs were not available for a before/after comparison;
questions 31 and 32 are current failures, not proven regressions from this patch.

Validation: all 73 offline tests pass, including nine evidence-boundary tests.
Seven live chat requests used 11,613 chat tokens in total and no embedding tokens.
Individual responses and usage are in `data/answer_runs/`; the inspection summary
is `data/answer_regression.json`. The original failing answer is preserved.

## 2026-09-30: complete-section context for list questions

Added `app/services/rag/section_expansion.py` after ranked retrieval. An explicit
list request naming one article or annex can expand from a retrieved heading to
the next section boundary in the same document. Stored offsets remove overlapping
characters; source/page provenance is preserved. Missing or ambiguous headings,
missing boundaries or context, and sections over 20,000 characters or 24 excerpts
produce an incomplete-section refusal before generation. The saved 1,000-character
chunks, embedding cache, and FAISS index are unchanged.

Question 31 now receives Annex III across 10 excerpts (7,390 characters, pages
127–129); question 32 receives Article 5 across 13 excerpts (11,145 characters,
pages 51–53). Expansion is separate from the original ranked five results and is
not counted as improved Recall@5. `--no-expansion` retains the five-passage baseline.

The first expanded generations covered all eight items but weakened qualifications
and citation support. Generic list guidance improved scope preservation, but
applying that guidance to every question regressed question 35 from a refusal to
an irrelevant account of designation criteria. The final version applies list
guidance only to expanded sections, restoring the existing prompt elsewhere.
All attempted outputs are retained; only results matching the final code hashes
are used for the final targeted review.

| Question | Final live review | Status |
|---|---|---|
| 2: prohibited-practice fine | Matches expected amount and turnover condition | Pass |
| 13: general application date | Matches expected date, citing a recital | Pass |
| 26: interacting with AI | Matches disclosure rule and obviousness exception | Pass |
| 31: Annex III areas | All eight areas now present | Pass |
| 32: prohibited practices | All eight practices present, but biometric-dataset exception is conflated, some harm qualifications and supporting excerpt IDs are omitted | Fail: generation fidelity |
| 24: Article 4 | Correct literacy obligation; no preceding-definition attribution | Pass |
| 35: official model designations | Refuses, as required by the corpus | Pass |

The priority gate is 4/5, so the whole-set answer run remains deferred. This is a
targeted assistant inspection against the existing expectations and source text,
not an automated or overall answer-accuracy measurement. The next failure to fix
is preserving Article 5 qualifications and complete supporting citations during
generation; lack of section context is no longer the cause.

Validation: 89 offline tests pass, including section boundaries, missing pages,
overlap removal, source isolation, context budgets, and citation availability.
Fresh retrieval run `20260930T173450235134Z` covers all 44 questions, uses zero new
embedding tokens, and has identical ranked hit IDs for all three pipelines.
Hybrid Recall@1/3/5 remains 71.43% / 85.71% / 88.57%; MRR@5 remains 0.7881.
The section and answer checks replay those saved results without retrieving or
embedding again. Replay generation timings exclude retrieval time.

Details: `data/section_eval_results.json`, `data/section_answer_regression.json`,
and individual `data/answer_runs/` files. Historical v2 reviews remain unchanged.

## 2026-09-30: source items and complete citation mapping

The user's question-32 run `20260930T174520580591Z` confirmed that full section
context alone did not preserve legal qualifications. It merged (a) and (b),
broadened social-scoring language, conflated the two biometric-dataset exclusions,
and added duplicate inline citation numbers. Repeating general prompt advice had
not reliably fixed these problems.

`list_evidence.py` now reconstructs the verified section and groups a leading
lettered enumeration in its first numbered paragraph. Nested Roman clauses stay
with the parent item; page-joined markers such as `; (g)` are retained. The parser
uses source text and offsets, not question numbers, answer labels, or an Article-5
special case. Unsupported layouts keep the previous generation path.

`list_answers.py` derives a required response entry for each actual source label.
Rendering keeps source order and adds every excerpt citation overlapping that
item, including conditions crossing chunk boundaries. Missing entries, blank
answers, malformed refusals, and fabricated extra citation IDs fail validation.
This validates coverage and provenance, not the truth of arbitrary model prose.
The item prompt requests preservation of scope, alternatives, conditions, and
exceptions; the output limit is 2,400 tokens for this path. Ordinary-question and
Annex prompts are unchanged.

Live v4 Q32 (`20260930T175128151616Z-q32.json`) passes inspection: all eight
practices are separate, material qualifications and distinct exceptions are
retained, and citations cover the full source items. The answer is longer because
it closely follows the source rather than dropping conditions. Questions 2, 13,
26, and 31 also pass, so the mandatory full answer run proceeded. Questions 24
and 35 retain their correct obligation and refusal behavior.

All 44 questions were generated with the unchanged saved retrieval results.
Provisional assistant review: 27 pass, 17 fail; all five priority questions pass.
No regression was found among the previously verified passes. Most questions did
not have a prior generated-answer baseline, so newly observed failures are not
automatically regressions caused by this change.

Remaining flagged questions: 7, 10, 11, 14, 15, 16, 17, 20, 21, 28, 29, 30, 34,
37, 42, 43, 44. Failures include missing context for cross-references, omitted
qualifications, incorrect dates and SME-fine handling, and failure to clarify
ambiguous questions. The Q11/Q17 completeness judgments and Q44 ambiguity rubric
need student adjudication. These labels are not a calibrated automatic judge or
student-approved accuracy measurement. Individual reasons are stored with each
question in the reports; the original questions and expected answers are intact.

Validation: 109 offline tests pass. The full live check used 75,118 prompt and
4,467 completion tokens (79,585 total), with no new embedding tokens. Generation
latency excludes replayed retrieval. Chunks, index, dependencies, and prior answer
files were preserved. No new retrieval-recall claim is made for this change.

Reports: `data/list_answer_regression.json`, `data/eval_results.json`, its archive
under `data/evaluation_runs/`, and the individual `data/answer_runs/` records.

## 2026-09-30: applicable Article context and calculation rules

Questions 14 and 34 mishandled the SME fine rule: an applicable exception was
ignored or absent from the short context. Optional expansion now supplies the
complete governing Article and retains supplementary retrieved passages, with
limits of 20,000 characters and 24 excerpts. Ambiguous boundaries or incomplete
source coverage fall back to the original passages. Explicit list handling is
unchanged.

Prompt-only trials still confused higher and lower rules. The final path detects
source-derived comparison rules and matching exception classes, then returns
whole source paragraphs with their citations. It does not hardcode question IDs,
Article numbers, SME amounts, or expected answers. Q14/Q34 now cite Article 99's
specific lower rule before the general higher rules, with zero chat requests.
The general fine question Q2 retains its correct generated answer.

The full run exposed a separator-only tail excerpt; filtering it before combining
Article context fixed the validation error. A regression test covers this case.
Thirteen previously completed answers had identical prepared contexts and
unchanged generation code after that fix. Their original code hashes remain
intact, with explicit equivalence metadata, avoiding unnecessary repeated calls.

Validation: 139 offline tests pass; all five priority questions pass; all 44
answers were checked. Provisional assistant review marks 29 pass and 15 fail,
improving Q14/Q34 with no regressions among previous passes. Remaining questions
7, 10, 11, 15, 16, 17, 20, 21, 28, 29, 30, 37, 42, 43, and 44 are backlog items.
Q7/Q11/Q17 completeness and Q44 ambiguity judgments need student adjudication.
These labels are not calibrated or student-approved accuracy measurements.

Final selected checks used 90,802 prompt and 4,205 completion tokens (95,007
total), excluding earlier trial outputs retained in the raw history. No new
embedding tokens were used. Generation timings exclude replayed retrieval;
retrieval metrics are unchanged. Corpus, chunks, index, questions, dependencies,
and prior answer files are preserved. Nothing was committed.

Reports: `data/applicability_answer_regression.json`,
`data/article_context_coverage.json`, `data/eval_results.json`, its archive under
`data/evaluation_runs/`, and the individual `data/answer_runs/` records.


## 2026-09-30: Retrieval observability and optional RAGAS

Added observed FAISS/BM25/RRF/reranker stages to the agent trace and UI, including
query terms, candidate text, score scales, rank changes, section context, threshold,
and timings. Diagnostics never enter the model's tool messages. Prompts and tool
schemas are unchanged. All 44 questions retain the exact same ordered top-five
hybrid results; verification made zero embedding API calls. 194 offline tests pass.

Added a separate uv environment for RAGAS 0.4.3. Its community dependency is pinned
below 0.4 because newer releases remove an import required by this RAGAS version.
The app's existing pyproject/lockfile and Docker dependencies are unchanged.
RAGAS evaluates saved standalone RAG answers, not the final agent. Refusal and
ambiguity cases need manual review; undefined/error scores never become zero.

One real Q24 judge run completed all five metrics: faithfulness 1.0, answer
relevancy 0.4949, context precision approximately 1.0, context recall 1.0, factual
correctness F1 0.57. This is a smoke test, not overall accuracy or a pass label.
Faithfulness uses saved generation evidence; context metrics use original ranked
retrieval. Full results, usage, and saved-answer provenance: data/ragas_runs/.

## 2026-09-30: RAG demo UI and context usage

Added color-coded tool disclosures, a RAG walkthrough with recorded stage counts
and before/after ranks, and a requirements/demo guide using existing benchmark
questions. The guide explicitly leaves missing calibration, injection tests,
follow-up benchmarks, and Agent day 2 requirements pending. Historical ablation
scores are labeled separately from live results and answer accuracy.

Context bars use the latest individual agent request's API input/output usage and
separate RAG generation usage. Cumulative billing counters never drive the bars;
unknown usage is displayed as unavailable. Observability fields stay outside model
messages. Prompts, tool schemas, retrieval order, and source data are unchanged.
Validation: 204 offline tests pass; browser interaction checks use local fixtures
and cached traces without API calls.


## 2026-09-30: Indexed prompt-injection exercise and quarantine

Existing prompt boundaries already resisted the tested override; no artificially
weakened prompt was used. Added deterministic checks for explicit user overrides,
source command quarantine before generation, screening after section expansion,
and a guard on tool evidence. Safe top scores drive the relevance gate. The
injection checkbox adds an in-memory FAISS/BM25 fixture and resets on New chat.

A first broad fixture missed the top five on Q24/Q31 and was not scored as a pass.
The focused 980-character fixture was then retrieved at ranks 4, 1, and 2 for
existing Q22/Q23/Q32. Neither the original prompt-only baseline nor the guarded
variant followed its Contact-admin instruction. The full guarded pipeline also
answered all three and excluded the malicious source. Normal question/label files
were not changed. These results cover one attack fixture, not all injection forms.

Regression: 737 real source chunks and 44 benchmark questions produced zero rule
matches. Replaying the same retrieval into old/new code produced identical
model-generation inputs for all 44. System/list/section prompts are unchanged.
204 offline tests cover guard behavior, legitimate injection discussion, no-call
refusal, index immutability, explicit test-flag forwarding, and context accounting.
