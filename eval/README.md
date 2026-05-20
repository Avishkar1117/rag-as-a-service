# Evaluation framework

Test set of curated Q&A pairs over a small PDF corpus, used to score the RAG service with [RAGAS](https://docs.ragas.io/) metrics. See [`PROMPT_TEMPLATE.md`](./PROMPT_TEMPLATE.md) for how to draft new test cases.

## Layout

```
eval/
├── README.md               # this file
├── PROMPT_TEMPLATE.md      # prompt for drafting Q&A pairs in Claude.ai / Gemini
├── dataset.jsonl           # the 50 curated Q&A pairs
├── corpus/                 # source PDFs the questions are grounded in
│   ├── attention.pdf
│   ├── process_mining.pdf
│   └── generic_agent.pdf
└── results/                # RAGAS run outputs (populated by run_ragas.py on Day 10-11)
    └── .gitkeep
```

## Dataset schema

`dataset.jsonl` is one JSON object per line with this shape:

```json
{
  "id": "q001",
  "question": "What is the dimension of the embeddings produced by the base Transformer in 'Attention is All You Need'?",
  "ground_truth_answer": "512 dimensions (d_model = 512).",
  "source_doc": "attention.pdf",
  "source_page": 3,
  "category": "factual",
  "difficulty": "easy"
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Unique identifier, e.g. `q001` — used in RAGAS output tables. |
| `question` | string | The question fed to `/query`. |
| `ground_truth_answer` | string | What the model SHOULD answer. Used by RAGAS for `answer_correctness` and `context_recall`. Be concrete and verifiable. |
| `source_doc` | string | Filename in `corpus/` where the answer lives. |
| `source_page` | int \| null | Page in the PDF. `null` for adversarial questions. |
| `category` | string | `factual` \| `multihop` \| `adversarial` |
| `difficulty` | string | `easy` \| `medium` \| `hard` — informal, for slicing results. |

### Categories

- **factual** (~30): direct lookups. "What is X?" → single chunk likely answers.
- **multihop** (~15): require combining info from multiple sections or pages. "Compare A and B." "What's the relationship between X and Y?"
- **adversarial** (~5): the question has NO answer in the corpus. The model should say "I don't have enough information." If it confidently makes something up, that's a hallucination failure.

## How to add a new test case

1. Open the source PDF, find a question worth asking.
2. Write the ground-truth answer based on the PDF content.
3. Append a JSON line to `dataset.jsonl` matching the schema.
4. Re-run `eval/run_ragas.py` (Day 10-11) to check the score doesn't regress.

## Drafting Q&A pairs with an LLM

See [`PROMPT_TEMPLATE.md`](./PROMPT_TEMPLATE.md). The flow:

1. Open Claude.ai or Gemini.
2. Upload one source PDF.
3. Paste the prompt template.
4. Get ~25 candidate Q&A pairs.
5. Curate: verify each answer against the PDF, drop anything unclear, keep ~15.
6. Repeat per PDF.
7. Hand-write 5 adversarial questions (LLMs are bad at writing genuinely unanswerable ones).
8. Total target: ~45 LLM-drafted + 5 adversarial = 50.

## Running the eval (Day 10-11, not yet implemented)

```bash
uv run python eval/run_ragas.py
```

Produces timestamped JSON + CSV under `results/` plus a Markdown summary at `results/latest.md`.
