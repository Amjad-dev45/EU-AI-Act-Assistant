# RAG failure log

Record actual retrieval or answer failures here during evaluation. No full quality run has been completed yet; do not interpret an empty answer-failure table as perfect accuracy.

| Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|----------|----------|---------------|---------|-------------|--------|

## Evaluation tooling issue

The dataset preview previously stopped at question 38 because its expected answer and evidence are null. The course format permits this for out-of-scope and ambiguous questions. Validation and scoring now accept null reference fields as not applicable. The input questions were not edited. Offline regression tests cover this case.
