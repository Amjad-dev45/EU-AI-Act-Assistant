# EU AI Act Assistant — team-6.7

A retrieval-augmented generation (RAG) assistant for the team's English EU AI Act corpus. It searches saved FAISS vectors, generates answers grounded in retrieved passages, and displays document/page citations with highlighted PDF pages. The Gradio UI and command-line tools share the same backend.

Team: Majd Hammoud, Rabeeh Abou Ismail, and Amjad Bsat.

## Setup

Use Python **3.12** and **uv**, as required by the course. `.python-version` records the Python version; `requirements.txt` pins the direct runtime packages verified for this project. These pins do not lock every transitive dependency.

```bash
uv venv --python 3.12
uv pip install --python .venv -r requirements.txt
```

Create `.env` from `.env.example` only if it does not already exist. Set `OPENAI_API_KEY` locally. Never commit `.env` or share it in an archive. Optional path overrides are documented in `.env.example`; relative environment paths resolve from the repository root.

Place the team's shared data inside this checkout, beside `app/` and `scripts/`:

```text
team-6.7/
├── app/                     # Application code, grouped by purpose
│   ├── app.py               # UI launcher
│   ├── settings.py          # Shared repository paths and overrides
│   ├── rag/                 # Retrieval and grounded answer generation
│   ├── ui/                  # Gradio, source highlighting, error display
│   ├── evaluation/          # Reference metrics and optional model judge
│   └── monitoring/          # Token/cost accounting, timing, SQLite records
├── scripts/                 # Extraction, indexing, checks, and CLI commands
├── config/pricing.json      # Versioned API price estimates
├── tests/                   # Offline automated checks
├── docs/                    # Architecture, requirements, and metric definitions
├── data/                    # Gitignored; shared through Teams
│   ├── sources/OJ_L_202401689_EN_TXT.pdf
│   ├── corpus.json
│   ├── sample.json
│   ├── questions.json
│   ├── chunks.json
│   ├── my_index.faiss
│   └── index_metadata.json
└── logs/                    # Gitignored request DB and evaluation run archives
```

Normal answering needs the matching `chunks.json`, `my_index.faiss`, and `index_metadata.json`; source highlighting also needs the PDF. Evaluation needs `questions.json`. Keep shared index files together. The data folder is inside the project but excluded from Git, matching the course layout. Do not re-embed a corpus that already has a matching shared index.

## Run the UI

Run from the repository root:

```bash
.venv/bin/python app/app.py
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv/Scripts/python.exe` in every command below.

The existing interface includes highlighted source pages, request/session usage, estimated costs, timings, and an optional **Evaluate answer quality** checkbox. The judge adds a paid model call and is off by default. **Start new session** resets the statistics grouping; it does not delete earlier logs. Session IDs are not conversation memory: follow-up rewriting has not been implemented.

Failed requests show redacted diagnostic details for local debugging. The interface still uses the original response fields and citation IDs. See [architecture](docs/architecture.md) for the new Python import paths.

## Use the backend without a UI

```bash
# Retrieve passages and check expected source/page coverage.
.venv/bin/python scripts/retrieval_checker.py --data-dir data --question-id 2

# Generate a cited answer.
.venv/bin/python scripts/ask.py --data-dir data --question-id 2

# Ask your own question and return structured JSON.
.venv/bin/python scripts/ask.py --data-dir data --question "What practices are prohibited?" --json
```

Question IDs are 1-based positions in `questions.json`. Retrieval makes one paid question-embedding call; answering also makes one paid generation call. `--k` controls how many passages are retrieved (default 5). `scripts/ask.py` also accepts `--conversation-id` for monitoring groups and `--judge` for optional semantic evaluation. Similarity scores are not correctness probabilities, and page coverage does not prove that the right evidence or answer was found.

## Prepare data only when necessary

The original extraction script reads the gitignored `data/sources/` folder and writes `corpus.json` and `sample.json`. Run it from the repository root when preparing new source data:

```bash
.venv/bin/python scripts/extract.py
.venv/bin/python scripts/chunk.py --data-dir data
```

Chunking joins consecutive pages within each source and records every contributing page. Defaults: **1,000 characters maximum, 200 characters overlap**, producing 805 chunks for the current corpus. It prefers sentence endings, then whitespace, then a hard cut. This heuristic does not perfectly handle abbreviations, long sentences, or footnotes. Chunk `start_char`/`end_char` offsets refer to joined document text; `pages` supplies all citation pages, while `page` keeps the first page for compatibility. Existing `chunks.json` is not overwritten.

```bash
# Token preview: no embedding API calls; tokenizer may download its vocabulary.
.venv/bin/python scripts/build_index.py --data-dir data

# Paid indexing, only when a matching index does not already exist.
.venv/bin/python scripts/build_index.py --data-dir data --run
```

Indexing uses `text-embedding-3-small`, batches of 50, and a cache keyed by the model and exact chunks file. It stores normalized float32 vectors in `embeddings.npy`, a FAISS `IndexFlatIP` in `my_index.faiss`, and integrity metadata in `index_metadata.json`. A rerun reuses completed batches; interrupted requests or cache writes may still need retries. Run one indexer at a time. Changing chunks requires the corresponding index to be rebuilt. Normal answering does not need the `.npy` file or embedding cache.

## Evaluate quality and monitor usage

```bash
# Validate the course dataset without any paid calls.
.venv/bin/python scripts/eval_runner.py --data-dir data

# Run labelled questions; optional --judge adds semantic/reference scoring.
.venv/bin/python scripts/eval_runner.py --data-dir data --limit 5 --run
.venv/bin/python scripts/eval_runner.py --data-dir data --limit 5 --judge --run

# Read saved statistics without model calls.
.venv/bin/python scripts/monitor.py --kind interactive --recent 5
```

`eval_runner.py` is the course-named entry point; `scripts/evaluate.py` remains supported with the same options. `--limit 5` selects the first five dataset entries, not the five priority questions. Add `--output` to select a report path. Reports are checkpointed after each question under `logs/evaluations/`; the latest report is also saved to `data/eval_results.json`. The input questions are never rewritten.

Metrics include page recall/coverage, page-hit precision and MRR, optional exact-chunk metrics, lexical exact match/token F1, and expected behavior. Missing reference fields remain not applicable (`null`). Exact match/F1 are not semantic accuracy; the optional judge's relevance, faithfulness, and reference correctness are estimates. See [metric definitions and limitations](docs/monitoring.md).

Answer requests, including failures, are recorded locally in `logs/requests.sqlite3`. Queries, answers, usage, source metadata, costs and timings are saved; full retrieved passages, API keys and raw provider payloads are not. Prices are estimates from `config/pricing.json`, not a billing statement. Indexing usage is separately reported by its script. Record observed retrieval/answer failures and fixes in [scripts/FAILURE_LOG.md](scripts/FAILURE_LOG.md).

## Tests and current status

```bash
uv pip install --python .venv -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic documents and mocked provider responses, with no paid API requests. They cover the backend, telemetry, citation checks, errors, session separation, source rendering, null course labels, and command entry points.

Basic live tests previously passed for one factual fine question and one out-of-scope weather question. The full 44-question accuracy evaluation is still pending. This reorganization does not add conversation memory or the later agentic project. See [course requirement review](docs/requirements-check.md) and [module responsibilities](docs/architecture.md).

## Local passage reranking

Reranking is enabled by default for the UI, CLI and evaluation runner. FAISS retrieves
20 candidates; `cross-encoder/ms-marco-MiniLM-L6-v2` scores question/passage pairs on
CPU and passes the best 5 (or requested `--k`) to the answerer. The model loads once
per process and downloads into the normal Hugging Face cache on first use. Internet
access is needed for this first download. No index rebuild or extra API calls are
needed. The first question will take longer. The model uses a 512-token pair limit;
very long questions/passages may be truncated by the model tokenizer.

After pulling, install `requirements.txt` using the setup command above. Existing
`.env` files work unchanged. Optional overrides are in `.env.example`:
`RAG_RERANK_ENABLED=false` restores FAISS-only retrieval, and
`RAG_RERANK_CANDIDATES=20` controls the candidate pool (at least the requested k).
Restart the app after changing these settings. Installation includes PyTorch;
CPU-only machines can install its CPU build using the official PyTorch instructions.

In the UI, expand **Usage and answer quality** to inspect reranking settings,
candidate/final rankings and `reranking_time_ms`. Cosine `score` and cross-encoder
`rerank_score` are different scales; neither is an accuracy percentage.

To measure improvement, evaluate the same questions and k in both modes. These
commands make paid embedding/generation calls; add `--judge` to both for paid
reference-correctness estimates:

```bash
RAG_RERANK_ENABLED=false .venv/bin/python scripts/eval_runner.py --data-dir data --run --output logs/evaluations/baseline.json
RAG_RERANK_ENABLED=true .venv/bin/python scripts/eval_runner.py --data-dir data --run --output logs/evaluations/reranked.json
```

Compare `metrics.expected_page_recall`, `page_hit_mrr`, answer token F1 and optional
`judge_reference_correctness`, including each metric's sample count. Page recall
is an evidence-location proxy, not proof of correct evidence. Full evaluation is
still pending; reranking improvement has not been established on this dataset.
