"""Presentation notes checked against instructions 5–7; no generated eval labels."""
# Reuse existing student-written questions verbatim; clicking only fills the input.
DEMO_QUESTIONS = [
    ["What does Article 4 say?"],
    ["What are the areas of high-risk AI systems listed in Annex III?"],
    ["What AI practices are prohibited under Article 5?"],
    ["Is it legal for a company to use software that detects its employees' moods?"],
    ["How do I train a neural network in PyTorch?"],
]
DEMO_GUIDE = """
### Start with RAG

1. **New chat → “What does Article 4 say?”** (existing Q24, page 51). Open **RAG walkthrough**. Explain the two searches, fusion, cross-encoder, and five retained chunks. Expand a passage to show its text and ranks. In the saved benchmark, vectors missed this page; hybrid retrieval found it at rank 1.
2. **New chat → Annex III question** (Q31, pages 127–129 in the source). Show five ranked results, then the additional section context used for a complete list. Section expansion is an extension; it does not change the reported top-five retrieval metric.
3. **New chat → Article 5 question** (Q32, pages 51–52 for the prohibitions). Show the answer citations alongside the underlying passages.
4. **New chat → PyTorch question** (Q39). Demonstrate refusal and inspect the actual tool trace. A refusal may happen in the agent without a search; do not claim the relevance gate ran unless it appears in the trace.
5. **Injection test:** enable the checkbox and ask the employees' moods question (Q23), or Q22/Q32. The temporary attack source appears in the ranked evidence; the guard marks it as excluded from generation. Expand it to show the malicious instruction. Turn the switch off or start a new chat afterwards. Three recorded tests retrieved the attack at ranks 4/1/2 and resisted it before and after screening; the existing prompt already resisted this particular payload. This does not establish universal protection.
6. Ask a follow-up in the same chat to demonstrate bounded memory. The required student-authored follow-up benchmark questions are still pending.

The copy buttons copy a prompt to your clipboard; paste it into chat and press
**Send**. The original benchmark examples fill the input directly.

### Required vs extra — current project status

| Requirement from the assignment | Status | What to show |
|---|---|---|
| RAG day 1: corpus → chunks → saved FAISS embeddings | Implemented | Index information in the walkthrough; `scripts/chunk.py` and `scripts/build_index.py` |
| RAG day 1: wider retrieval → cross-encoder → top results | Implemented | Actual candidate counts and before/after ranks |
| RAG day 1: BM25 + RRF for exact terms | Required for this corpus; implemented | Article 4 query terms, BM25 candidates, fused results |
| RAG day 1: cited answer + document/page in a web UI | Implemented | Answer, Sources, and expandable passages |
| RAG day 1: bounded memory and out-of-scope handling | Implemented; quality needs evaluation | Same-chat follow-up and an out-of-scope question |
| RAG day 1: retrieval checker and failure log | Implemented | `scripts/retrieval_checker.py`, `scripts/FAILURE_LOG.md` |
| RAG day 2: calibrated threshold + correct/wrong refusals | Partial | Gate exists; threshold is provisional, calibration report pending |
| RAG day 2: switchable injection test, 3 questions, before/after results | Tested on 3 existing questions | Compare the existing untrusted-context prompt with added source quarantine; see `data/injection_runs/` |
| RAG day 2: retrieval/full eval modes, metrics by category | Partial | Retrieval runner and saved full-answer reports exist; unified full-mode CLI pending |
| RAG day 2: 3–4 follow-up benchmark questions | Pending | Student must author the questions and expected answers |
| RAG day 2: ablation table and regression checks | Recorded | Saved comparison below; rerun after retrieval/prompt changes |
| Agent day 1: tool loop, calculator, own tool, complete trace | Implemented | Colored cards: RAG search, calculator, exact quote; six model-step cap |
| Agent day 2: own tools | Implemented | Five read-only tools: search, calculator, quote, definition, application dates |
| Agent day 2: approved messaging, injection/action tests, agent evaluation | Partial / pending | Manual Teams button exists; agent-proposed send-message flow and full evaluation remain |

**Extensions:** sentence-aware cross-page chunking (1,000-character hard cap), section expansion, citation validation, detailed retrieval traces, context bars, and clickable highlighted PDF source previews. RAGAS is separate optional judge tooling, beyond the assignment's library list; it is not a substitute for the required evaluation or measured accuracy. Docker separates UI, agent, and RAG.

### Saved retrieval comparison — not live answer accuracy

September 30 development benchmark: **44 questions, 35 eligible for page retrieval**. Any expected-page hit counts; these are not full multi-page recall scores. Cached question embeddings; startup excluded from latency.

| Cumulative configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Mean latency |
|---|---:|---:|---:|---:|---:|
| FAISS vectors | 37.14% | 68.57% | 77.14% | 0.5357 | 0.006 s |
| + cross-encoder reranking | 71.43% | 80.00% | 82.86% | 0.7643 | 0.439 s |
| + BM25 and RRF | 71.43% | 85.71% | 88.57% | 0.7881 | 0.442 s |

Source: README **Measured ablation**, run `20260930T170533848794Z`. The final configuration keeps hybrid search and reranking. Q26 remains a known retrieval regression introduced by reranking. This comparison does not establish overall answer accuracy.

Requirement sources: `instructions/5_rag_project.md`, `6_evaluation_toolkit.md`, `7_agentic_project.md` in the downloaded course materials. This is a checked presentation guide, not an automatic compliance audit; update it when the implementation changes.
"""
