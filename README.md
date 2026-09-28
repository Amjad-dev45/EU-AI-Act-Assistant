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

The script writes `chunks.json`: up to 1,200 characters per chunk with 200 characters of overlap. It joins consecutive pages within each source document, allowing chunks and overlap to cross page boundaries. It prefers sentence endings in the final quarter of the chunk, falls back to whitespace, and finally uses a hard cut. Sentence detection is a punctuation heuristic; it may split abbreviations and cannot guarantee whole sentences longer than the overlap or chunk limit. Footnotes remain in their extracted positions. Sizes are characters, not tokens. Each chunk records `source`, `pages` (all contributing pages), `page` (first page for compatibility), and `start_char`/`end_char` offsets into the joined document text. Use the full `pages` list for citations and evaluation. Missing or duplicate pages are rejected rather than silently joined. These defaults are a starting point to evaluate with retrieval. Override them using `--chunk-size` and `--overlap`. No external packages or API key are needed for this step.

Existing `chunks.json` is not overwritten: its order must stay aligned with the FAISS vectors once an index exists. Embeddings and FAISS indexing are the next step and are not implemented yet.
