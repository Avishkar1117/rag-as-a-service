# RAGAS Evaluation — latest

**Run:** 2026-05-20 23:42:42  
**Generation:** `gemma-4-31b-it` · **Judge:** `gemma-4-31b-it` · **top_k:** 3 · **chunk_size:** 1024  
**Questions:** 50 (errors: 0) · **Duration:** 3090.8s · **RAGAS:** 0.3.2

## Aggregate scores

| Metric | Score |
|---|---|
| faithfulness | 0.9744 |
| answer_relevancy | 0.7830 |
| llm_context_precision_with_reference | 0.8350 |
| context_recall | 0.8933 |

## By category

| Category | n | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall |
|---|---|---|---|---|---|
| adversarial | 5 | 0.6667 | 0.0000 | 0.9167 | 1.0000 |
| factual | 35 | 1.0000 | 0.9034 | 0.8284 | 0.9429 |
| multihop | 10 | 1.0000 | 0.7530 | 0.8167 | 0.6667 |

**Adversarial refusal rate:** 1.0000 — fraction of unanswerable questions the model correctly declined to answer.

## Notes

Per-question scores: `ragas_20260520_234242.json` (JSON) and the matching `.csv`.
