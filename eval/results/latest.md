# RAGAS Evaluation — latest

**Run:** 2026-05-28 15:52:33  
**Generation:** `gemma-4-31b-it` · **Judge:** `deepseek-v4-flash` · **top_k:** 8 · **chunk_size:** 1024  
**Questions:** 50 (errors: 0) · **Duration:** 1461.2s · **RAGAS:** 0.3.2

## Aggregate scores

| Metric | Score |
|---|---|
| faithfulness | 0.8685 |
| answer_relevancy | 0.8416 |
| llm_context_precision_with_reference | 0.7574 |
| context_recall | 0.8933 |

## By category

| Category | n | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|---|
| adversarial | 5 | 0.0000 | 0.0000 | 0.2000 | 0.2000 |
| factual | 35 | 0.9857 | 0.9590 | 0.8186 | 1.0000 |
| multihop | 10 | 0.8923 | 0.8518 | 0.8219 | 0.8667 |

**Adversarial refusal rate:** 1.0000 — fraction of unanswerable questions the model correctly declined to answer.

## Notes

Per-question scores: `ragas_20260528_155233.json` (JSON) and the matching `.csv`.
