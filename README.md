# team-6.7

## Team members
- Majd Hammoud
- Rabeeh Abou Ismail
- Amjad Bsat

## Prepare retrieval chunks

Use Python 3.12. With the course workspace layout, run from this repository:

```bash
uv run --python 3.12 --no-project scripts/chunk.py --data-dir ../../data
```

The data folder is outside this Git repository and contains `corpus.json`, `sample.json`, `questions.json`, and `sources/OJ_L_202401689_EN_TXT.pdf`. Supply a different `--data-dir` if needed. Data is shared through Teams, not committed to Git.

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
.venv/bin/python scripts/build_index.py --data-dir ../../data
```

The tokenizer may download its vocabulary the first time. The preview does not load `.env` or create an OpenAI client.

After reviewing the preview, generate embeddings and build the index (paid API calls for uncached batches):

```bash
.venv/bin/python scripts/build_index.py --data-dir ../../data --run
```

The script uses `text-embedding-3-small`, batches 50 chunks per request, and caches completed batches under `embedding_cache/` in the data directory. The cache depends on the model and exact chunks file. A rerun reuses completed batches. A failed request or interrupted cache write can still require retrying a batch. Run only one indexing process at a time.

Outputs are `embeddings.npy`, `my_index.faiss`, and `index_metadata.json`. Keep them with the exact matching `chunks.json`; the metadata records its SHA-256 checksum. FAISS uses normalized vectors with inner-product search, equivalent to cosine similarity. Any later retrieval code must validate the checksum and use each chunk's full `pages` list.

The script reports token usage returned by successful embedding calls; check the provider dashboard for actual total spending, including any retries. Generated data stays outside Git and can be shared with teammates through Teams.
