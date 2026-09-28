# Course requirement review

Reviewed against the locally downloaded GenAI_week instructions 1–8 on 2026-09-28. This is a code/layout review, not instructor approval or a claim that every deliverable has been submitted.

| Requirement | State after reorganization |
|---|---|
| Application code in app/; scripts/, README, gitignored data/ beside it | Preserved; internal app subpackages separate responsibilities |
| Extraction script and report at scripts/extract.py and scripts/EXTRACTION_REPORT.md | Preserved |
| PDF under data/sources; corpus/sample JSON | Preserved without editing or rerunning extraction |
| Python 3.12 with uv | .python-version records 3.12; documented uv environment setup |
| Raw Python, no LangChain/LlamaIndex/RAG frameworks | Preserved; no new runtime framework introduced |
| Required chat and embedding models | gpt-4o-mini and text-embedding-3-small unchanged |
| FAISS indexing saved to disk and shared | Preserved; no regenerated vectors or index |
| Web UI, document/page citations, source display | Gradio and highlighted sources preserved; launcher unchanged |
| Out-of-scope handling | Prompt and refusal validation preserved; broad quality evaluation remains pending |
| Dataset questions written by students | Supplied 44-question file untouched; refactor generates no questions |
| Required retrieval checker | scripts/retrieval_checker.py preserved |
| Required failure log | scripts/FAILURE_LOG.md added; records confirmed tooling issue, no fabricated answer failures |
| Recommended evaluation runner and data/eval_results.json | Course-named eval_runner.py added; evaluate.py retained; latest report written under data/ in addition to archived runs |
| Null labels allowed by supplied assignment format | Validator/scorer compatibility fixed and covered by tests |
| Shared $5 team budget and cost tracking | Existing request estimates and caches preserved; no new paid calls during refactor; global budget enforcement not implemented |
| README with purpose, setup, execution | Updated for new module layout, Windows/Linux commands and result limitations |
| .env, .venv/, and data/ ignored | Verified; additional environment variants, logs, caches, test/build artifacts excluded |
| Data submitted through Teams; instructor collaborator and team registration | Cannot confirm from local code; team must check submission/access status |
| Later agentic project | Not implemented by this reorganization |

Runtime packages are the existing OpenAI/FAISS/tokenizer/Gradio/PDF stack. Pillow is retained from the existing UI for image rendering and is also part of Gradio's dependencies; the course's explicit library list names Gradio and PyMuPDF, not Pillow separately. No new runtime library is introduced. The test-only httpx dependency is declared separately because offline SDK tests import it directly.

Current direct dependency versions are pinned to the environment used for the checks. This is not a full transitive lock or a guarantee of identical package resolution on every operating system.

Follow-up memory remains a separate feature. The evaluator now accepts the supplied dataset, but a successful preview or passing unit tests do not establish retrieval accuracy, faithfulness, or completion of the full quality evaluation.
