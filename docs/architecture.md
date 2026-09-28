# Module responsibilities

The application stays under `app/`, with extraction/evaluation entry points under `scripts/` and a gitignored `data/` directory beside them. The folders describe purpose rather than stages of development.

## Reading order

1. `app/app.py`: startup only; load .env, paths, client, backend, and UI.
2. `app/ui/interface.py`: components, events, and display of backend results.
3. `app/rag/retrieval.py`: input checks, question embedding, FAISS search, and source metadata.
4. `app/rag/answering.py`: context construction, grounded generation, citation checks, and request orchestration.
5. `app/evaluation/metrics.py`: local label validation and deterministic metrics.
6. `app/evaluation/judge.py`: the optional paid model judge and its validation.
7. `app/monitoring/usage.py`: provider usage extraction, price estimates, and call subtotals.
8. `app/monitoring/tracing.py`: one request's stage timers and API-call/source records.
9. `app/monitoring/storage.py`: SQLite persistence and aggregate reports.

`app/ui/sources.py` owns PDF highlighting and citation panels; `app/ui/diagnostics.py` owns redacted error output. `app/settings.py` centralizes repository paths and environment overrides. Importing these modules does not create an API client or load credentials. Monitoring/evaluation package exports preserve their public API while their internals are split.

## Data flow

```text
UI or CLI → Answerer → Retriever → question embedding → FAISS passages
                 ↓
             context + generation → citation validation → result
                 ↓
             optional label metrics / optional model judge
                 ↓
             request trace → SQLite and per-session reporting
```

Tracing is shared with retrieval/generation so failed requests retain any known incurred usage. A storage or optional judge failure does not silently discard a valid answer. These behaviors are preserved by offline regression tests.

## Backend imports

```python
from app.rag.retrieval import Retriever
from app.rag.answering import Answerer
from app.ui.interface import GradioUI
```

The earlier flat `app.retrieval` and `app.answering` paths were moved; update external Python integrations accordingly. `Retriever` and `Answerer` call signatures and result fields are unchanged. The direct launch command `python app/app.py` is unchanged. No HTTP endpoint is introduced.

## Suggested team ownership

| Owner | Files and responsibilities |
|---|---|
| Retrieval/data | `app/rag/retrieval.py`, extraction/chunk/index scripts, retrieval checker |
| Answers/evaluation | `app/rag/answering.py`, `app/evaluation/`, eval scripts and failure log |
| UI/monitoring | `app/ui/`, `app/monitoring/`, UI entry point, monitor CLI |

Coordinate changes to public result fields, pricing, requirements and shared README. Use independent branches; do not run simultaneous index rebuilds on the shared data directory. Future follow-up handling should be a separate small module in `app/rag/`, with UI-owned per-session history, rather than mixed into monitoring's session ID.

## Readability conventions

Start each module with its purpose. Use section comments before meaningful stages and docstrings where behavior or limitations are not obvious. Prefer ordinary loops to nested comprehensions when they are easier to explain. Avoid adding wrappers or abstractions for a single trivial operation. Keep API calls distinguishable from free local operations, and keep generated data out of source control.

Passage reranking lives in `app/rag/reranking.py`: cached local cross-encoder, configuration and scoring. `retrieval.py` records initial candidates and the final ranking. Evaluation measures the final evidence given to the answerer.
