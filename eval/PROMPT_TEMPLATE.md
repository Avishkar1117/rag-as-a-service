# Prompt template for drafting Q&A pairs

Use this in Claude.ai or Gemini (whichever you have) to draft candidate Q&A pairs from each PDF. Then **curate** the output before adding to `dataset.jsonl`.

## How to use

1. Open Claude.ai or Gemini in a new chat.
2. Upload **one** source PDF.
3. Paste the prompt below, replacing `<DOC_NAME>` with the corpus filename (`attention.pdf`, `process_mining.pdf`, `generic_agent.pdf`).
4. Save the output to a scratch file.
5. **Curate**: open each Q&A, verify the answer against the PDF, drop anything wrong/ambiguous, keep ~15.
6. Hand-write 5 adversarial questions separately (LLMs are bad at generating genuinely unanswerable ones).
7. Append final entries to `eval/dataset.jsonl`.

## The prompt

```
You are helping me build a test set for evaluating a retrieval-augmented
generation (RAG) system over the attached PDF.

Generate 25 question-answer pairs grounded strictly in this PDF. Output them
as JSON Lines (JSONL), one object per line, matching this schema EXACTLY:

{
  "id": "qXXX",
  "question": "...",
  "ground_truth_answer": "...",
  "source_doc": "<DOC_NAME>",
  "source_page": <int>,
  "category": "factual" | "multihop",
  "difficulty": "easy" | "medium" | "hard"
}

Rules:
- 18 factual questions: direct lookups with concrete answers (numbers, names,
  definitions). These should be answerable from a single page or paragraph.
- 7 multihop questions: require combining info across multiple sections/pages
  or comparing two concepts. The answer should require synthesis.
- `ground_truth_answer` must be CONCRETE and VERIFIABLE — not "explained in
  section 4" but the actual answer text.
- `source_page` is the 1-indexed page where the answer lives. Pick the most
  relevant single page even for multihop questions.
- DO NOT make up content. If the PDF doesn't support a claim, don't ask it.
- DO NOT include "according to the paper" or "in this document" in questions
  — questions should be self-contained.
- Vary the question style: what / why / how / compare / quantify.
- Use sequential ids starting from qXXX (I'll renumber).

Output ONLY the JSONL, no commentary, no markdown fences.
```

## Curation checklist

Before pasting into `dataset.jsonl`, for each candidate Q&A:

- [ ] Open the PDF at `source_page`. Does the answer ACTUALLY appear there?
- [ ] Is the answer **specific and short** (a fact, a number, a comparison)? Or is it vague handwaving? Drop the vague ones.
- [ ] Could a careful reader answer this from the PDF text alone? If they'd need outside knowledge → drop.
- [ ] Is the question phrased clearly without hedging ("maybe", "perhaps")?
- [ ] Does the `category` match the actual cognitive load (don't label a single-page lookup as `multihop`)?

## Writing adversarial questions by hand

5 questions where the answer is **deliberately not in the corpus**. Examples:

- "What is the GDP of Germany?"
- "Who is the CEO of OpenAI?"
- "What temperature does water boil at on Mars?"

For each:
- `category`: `"adversarial"`
- `source_page`: `null`
- `ground_truth_answer`: `"I don't have enough information."` (exact string — RAGAS will compare against this)
- `difficulty`: `"easy"` (correctly refusing is supposed to be easy)

If the model confidently makes up an answer, that's a hallucination failure.

## Target distribution

| | Factual | Multihop | Adversarial | Total |
|---|---|---|---|---|
| attention.pdf | 10 | 5 | — | 15 |
| process_mining.pdf | 10 | 5 | — | 15 |
| generic_agent.pdf | 10 | 5 | — | 15 |
| adversarial (mixed) | — | — | 5 | 5 |
| **Total** | **30** | **15** | **5** | **50** |

## Final renumbering

After all 50 entries are in `dataset.jsonl`, renumber the `id` field from `q001` to `q050` sequentially. Easiest: open the file in VS Code, find/replace.
