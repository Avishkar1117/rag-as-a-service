# RAGAS Evaluation — latest

**Run:** 2026-05-21 09:05:22  
**Generation:** `gemma-4-31b-it` · **Judge:** `gemma-4-31b-it` · **top_k:** 8 · **chunk_size:** 1024  
**Questions:** 50 (errors: 5) · **Duration:** 5174.5s · **RAGAS:** 0.3.2

## Aggregate scores

| Metric | Score |
|---|---|
| faithfulness | 1.0000 |
| answer_relevancy | 0.7328 |
| llm_context_precision_with_reference | 0.9413 |
| context_recall | 0.9123 |

## By category

| Category | n | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|---|
| adversarial | 5 | — | 0.0000 | 1.0000 | 1.0000 |
| factual | 35 | 1.0000 | 0.8046 | 0.9060 | 0.9231 |
| multihop | 10 | 1.0000 | 0.9555 | 1.0000 | 0.8333 |

**Adversarial refusal rate:** 0.6000 — fraction of unanswerable questions the model correctly declined to answer.

## Notes

Per-question scores: `ragas_20260521_090522.json` (JSON) and the matching `.csv`.
