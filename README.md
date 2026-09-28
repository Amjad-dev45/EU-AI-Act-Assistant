# team-6.7

## Team members
- Majd Hammoud
- Rabeeh Abou Ismail
- Amjad Bsat

## Prepare retrieval chunks

Use Python 3.12. With the course workspace layout, run from this repository:

```bash
uv run --python 3.12 --no-project scripts/chunk.py --data-dir data
```

The data folder is inside this repository but excluded from Git and contains `corpus.json`, `sample.json`, `questions.json`, and `sources/OJ_L_202401689_EN_TXT.pdf`. Supply a different `--data-dir` if needed. Data is shared through Teams, not committed to Git.

The script writes `chunks.json`: up to 1,000 characters per chunk with 200 characters of overlap. It joins consecutive pages within each source document, allowing chunks and overlap to cross page boundaries. It prefers sentence endings in the final quarter of the chunk, falls back to whitespace, and finally uses a hard cut. Sentence detection is a punctuation heuristic; it may split abbreviations and cannot guarantee whole sentences longer than the overlap or chunk limit. Footnotes remain in their extracted positions. Sizes are characters, not tokens. Each chunk records `source`, `pages` (all contributing pages), `page` (first page for compatibility), and `start_char`/`end_char` offsets into the joined document text. Use the full `pages` list for citations and evaluation. Missing or duplicate pages are rejected rather than silently joined. These defaults are a starting point to evaluate with retrieval. Override them using `--chunk-size` and `--overlap`. No external packages or API key are needed for this step.

Existing `chunks.json` is not overwritten: its order must stay aligned with the FAISS vectors once an index exists. Use the indexing script below after reviewing your chunks.


## Embeddings and FAISS index

Create the Python 3.12 environment and install dependencies:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Copy `.env.example` to `.env` if you have not already created it, and set `OPENAI_API_KEY`. Never commit `.env`.

Preview token usage without embedding API calls:

```bash
.venv/bin/python scripts/build_index.py --data-dir data
```

The tokenizer may download its vocabulary the first time. The preview does not load `.env` or create an OpenAI client.

After reviewing the preview, generate embeddings and build the index (paid API calls for uncached batches):

```bash
.venv/bin/python scripts/build_index.py --data-dir data --run
```

The script uses `text-embedding-3-small`, batches 50 chunks per request, and caches completed batches under `embedding_cache/` in the data directory. The cache depends on the model and exact chunks file. A rerun reuses completed batches. A failed request or interrupted cache write can still require retrying a batch. Run only one indexing process at a time.

Outputs are `embeddings.npy`, `my_index.faiss`, and `index_metadata.json`. Keep them with the exact matching `chunks.json`; the metadata records its SHA-256 checksum. FAISS uses normalized vectors with inner-product search, equivalent to cosine similarity. Any later retrieval code must validate the checksum and use each chunk's full `pages` list.

The script reports token usage returned by successful embedding calls; check the provider dashboard for actual total spending, including any retries. Generated data stays excluded from Git and can be shared with teammates through Teams.


## Inspect retrieval

Test one of the team's existing evaluation questions (1-based position in questions.json):

```bash
.venv/bin/python scripts/retrieval_checker.py --data-dir data --question-id 2
```

Or supply your own question:

```bash
.venv/bin/python scripts/retrieval_checker.py --data-dir data --question "What AI practices are prohibited?"
```

Each execution makes one paid question-embedding request; it does not regenerate corpus embeddings or generate an answer. Use `--k 10` to inspect more passages and `--json` for structured output. Similarity scores are not confidence probabilities.

The checker displays source documents, all contributing pages, passage text, and similarity scores. With `--question-id`, it reports which expected source/page pairs were found. A page match measures retrieval coverage, not whether the exact evidence was retrieved or an answer is correct. Questions without expected pages are reported as N/A.

`app/retrieval.py` exposes `Retriever(data_dir, client).retrieve(question, k=5)` for use by the existing UI and later answer-generation code. Reuse the Retriever instance to avoid reloading the index for each question. It checks the saved chunks checksum and index dimensions/counts. Backend credentials must remain in the Python process, never in browser code.


## Generate a cited answer

```bash
.venv/bin/python scripts/ask.py --data-dir data --question-id 2
```

Use `--question "your question"` for free text, `--k` for retrieval count, and `--json` for structured output. Each execution makes a paid question-embedding request and a paid gpt-4o-mini generation request. The output includes the answer, citation labels mapped to source/page metadata, retrieved passages, status, and token usage.

The prompt requires document-grounded answers, clarification for ambiguous questions, and refusal when evidence is insufficient. Citation validation checks reference IDs, not whether the claims are factually supported; test with the evaluation set. There is no conversation memory yet.

The existing Gradio UI uses `Answerer(retriever, client)` and displays cited, highlighted source pages. The same backend serves `scripts/ask.py` and the evaluation runner. There is no conversation memory; session IDs group monitoring statistics only.

## Run the UI (Windows / PowerShell)

From this repository, with Python 3.12 and uv installed:

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
Copy-Item .env.example .env  # only if .env does not already exist
```

Set `OPENAI_API_KEY` in `.env` and point `RAG_DATA_DIR` at the folder with the matching
`chunks.json`, `my_index.faiss`, `index_metadata.json`, and `sources/`.
In this workspace that folder is `data/` inside this repository.

```powershell
.venv/Scripts/python.exe app/app.py
```

On macOS/Linux use `.venv/bin/python` instead. The existing source highlighting and
answer validation remain in place. The UI's **Usage and answer quality** panel shows
request tokens, estimated cost, timings, retrieval/citation counts, and session totals.
**Evaluate answer quality** adds one paid judge call; it is off by default.
**Start new session** resets the session grouping, without deleting earlier logs.

Failed questions show an **Error details** panel with the exception chain, file/line
traceback, and request ID. The same diagnostics are printed in the server terminal.
Configured secret values and common API-key/authorization patterns are redacted;
stack locals are never captured. This verbose view is intended for local debugging.

## Evaluation, monitoring and costs

See [docs/monitoring.md](docs/monitoring.md) for metric definitions, limitations,
dataset format, pricing, storage and extension points.

Every answer request is saved locally to `logs/requests.sqlite3`, including failed
requests and usage already incurred. Retrieved text and full documents are not copied
into the log. Queries, answers and source metadata are logged. `logs/` is gitignored.
Pricing is centralized in [config/pricing.json](config/pricing.json).

```powershell
# Read-only monitoring: no model calls
.venv/Scripts/python.exe scripts/monitor.py --kind interactive --recent 5
.venv/Scripts/python.exe scripts/monitor.py --conversation-id YOUR_SESSION_ID

# Reuse an ID to aggregate CLI requests into one conversation
.venv/Scripts/python.exe scripts/ask.py --data-dir data --question "What practices are prohibited?" --conversation-id demo

# Validate a dataset and preview calls without spending
.venv/Scripts/python.exe scripts/evaluate.py --data-dir data

# Run the existing RAG application on labelled questions
.venv/Scripts/python.exe scripts/evaluate.py --data-dir data --run

# Include the additional semantic/reference judge; start with a small subset
.venv/Scripts/python.exe scripts/evaluate.py --data-dir data --limit 5 --judge --run

# Offline integration tests: synthetic corpus, no paid API requests
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Evaluation reports are checkpointed after each question under `logs/evaluations/`.
They include per-question metrics, category/difficulty averages with sample counts,
errors, costs and latency. Evaluation runs and standalone retrieval checks are tagged
separately from interactive traffic so they can be excluded from usage averages.
