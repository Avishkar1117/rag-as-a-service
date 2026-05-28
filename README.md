---
title: RAG-as-a-Service
emoji: "\U0001F50D"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Production RAG API with evaluation pipeline
---

# RAG-as-a-Service

A production-grade Retrieval-Augmented Generation service: upload a PDF, ask
questions, get cited answers. The differentiator versus a notebook prototype is
the rest of the stack, a typed HTTP API, a Redis embedding cache, structured
logging, Sentry, per-query cost tracking, and a RAGAS evaluation harness wired
into CI as a regression guard.

## Live demo

- **Space:** https://huggingface.co/spaces/Avishkar1117/rag-service
- **Demo UI:** https://avishkar1117-rag-service.hf.space/
- **Swagger UI:** https://avishkar1117-rag-service.hf.space/docs

Hosted on a Hugging Face free-tier Space (Docker SDK). The Space sleeps after
inactivity, so the first request may take ~30 s while the container wakes.

## Architecture

```
                       ┌──────────────────────────────────────────┐
   Client ────────────▶│            FastAPI Service               │
   (browser/curl)      │                                          │
                       │  GET  /              demo page           │
                       │  POST /ingest        upload PDF          │
                       │  POST /query         ask question        │
                       │  GET  /health        liveness            │
                       │  GET  /metrics       cost + latency      │
                       │  GET  /docs          OpenAPI / Swagger   │
                       └────────────┬─────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
     ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
     │  Upstash    │         │   Chroma    │         │   Gemini    │
     │   Redis     │         │ (persisted) │         │   API       │
     │  embedding  │         │   vector    │         │  Gemma 4    │
     │   cache     │         │   store     │         │  + OCR      │
     └─────────────┘         └─────────────┘         └─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  Sentry (errors)         │
                       │  Structured JSON logs    │
                       │  CostTracker (USD)       │
                       └──────────────────────────┘

                       ┌──────────────────────────┐
                       │  RAGAS Eval Harness      │  offline + CI
                       │  50 curated Q&A pairs    │
                       │  faithfulness, recall,   │
                       │  precision, relevance    │
                       └──────────────────────────┘
```

## Evaluation results

50 curated Q&A pairs (35 factual, 10 multi-hop, 5 adversarial) over a 3-PDF
corpus, scored by RAGAS on four metrics plus adversarial refusal rate.

**Run config:** `gemma-4-31b-it` (generation), `deepseek-v4-flash` (judge,
DeepSeek API), `top_k=8`, `chunk_size=1024`, `models/gemini-embedding-001`.
The generator and judge come from different model families and providers —
the judge can't recognize its own reasoning patterns in an answer, which
removes the self-bias floor a same-model setup has on faithfulness.

### Headline scores (50 questions, May 28)

| Metric | Full 50 | Answered-only (44) |
|---|---|---|
| faithfulness | 0.8685 | **0.9869** |
| answer_relevancy | 0.8416 | **0.9564** |
| context_precision | 0.7574 | 0.8288 |
| context_recall | 0.8933 | **0.9811** |
| **adversarial refusal rate** | **1.0000** | — |

Both columns are honest, just answering different questions. **Full 50**
is what RAGAS reports — refusals (correctly given on the 5 adversarials +
1 hard multihop) count as zeros on faithfulness/relevancy by design.
**Answered-only** strips those out and asks "when the pipeline does answer,
how well does it answer?"

By category (full-50 aggregates):

| Category | n | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|
| factual | 35 | 0.9857 | 0.9590 | 0.8186 | 1.0000 |
| multihop | 10 | 0.8923 | 0.8518 | 0.8219 | 0.8667 |
| adversarial | 5 | — | — | — | — *(all 5 correctly refused)* |

50/50 questions completed with 0 API errors — the retry layer absorbed
transient Gemini 500/503s cleanly.

### Cross-provider judge vs original same-model baseline

The original `top_k=8` baseline (May 21) used Gemma 4 as both generator and
RAGAS judge. The new run uses DeepSeek as a cross-provider judge against the
same generator:

| Metric | Gemma judge (May 21) | DeepSeek judge (May 28) | Δ |
|---|---|---|---|
| faithfulness | 1.0000 | 0.8685 | −0.1315 |
| answer_relevancy | 0.7328 | 0.8416 | +0.1088 |
| context_precision | 0.9413 | 0.7574 | −0.1839 |
| context_recall | 0.9123 | 0.8933 | −0.0190 |
| **adversarial refusal rate** | **0.6000** | **1.0000** | **+0.4000** |

Two findings worth flagging:

1. **`context_precision` dropped by ~0.18.** Same generator, same retrieval,
   same questions — the dominant change is the judge. That delta is the
   self-bias correction. Gemma judging its own retrieval was inflating
   precision; the DeepSeek number is the honest one.
2. **Adversarial refusal rate jumped from 0.60 to perfect 1.00.** All five
   unanswerable questions are now correctly declined, where the original
   `top_k=8` baseline answered two of them incorrectly.

### Caveats & known limitations

- **q050 — a deterministic conservative refusal.** This multihop question
  (*"What three minimal capabilities does GA argue every agent system must
  implement, and what stage of the execution pipeline does each address?"*,
  `generic_agent.pdf`) was refused both on May 27 and May 28, both times
  with `context_recall ≈ 0.50`. Retrieval reliably returns only half the
  facts needed for the 6-fact answer the question demands, so Gemma's
  prompt-compliant move is to refuse rather than fabricate the missing
  half. A reranker would lift retrieval coverage enough to unlock this case
  (see "What I'd do next" below).
- **Full-50 aggregates conflate two different things.** Refusals (both
  correct adversarials and conservative refusals like q050) score zero
  across faithfulness/relevancy by RAGAS design. That's why the full-50
  faithfulness of 0.8685 is materially lower than the answered-only 0.9869.
  Both numbers are real; the answered-only one is the better signal of
  pipeline quality, the full-50 is the better signal of end-to-end
  coverage.
- **Corpus rebuild during this session refreshed stale OCR chunks.** A
  silent Chroma staleness was suppressing retrieval for two of the three
  corpus PDFs until `--rebuild` was used. The regression eval surfaced it;
  the rebuild fixed it. That same staleness could affect production
  ingestion if a PDF is re-ingested without rebuild — a real operational
  lesson rather than a one-off bug.

Per-question scores: `eval/results/ragas_20260528_155233.json` (and matching
`.csv`). Latest aggregate: `eval/results/latest.md`.

### Evaluation as CI

A 10-question regression guard (`tests/eval/test_regression.py`) re-runs the
RAGAS harness on every pull request and fails if faithfulness drops below
0.85. The workflow is in `.github/workflows/eval.yml`. This is the angle
that makes the service genuinely shippable — any prompt, retrieval, or
model change that quietly regresses quality gets caught before merge. The
framework also caught the stale-Chroma issue mentioned above; that's
exactly the class of regression CI-time eval is for.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Multipart PDF upload. Returns `{document_id, n_chunks}`. |
| `POST /query` | `{question, top_k?, document_ids?}` → `{answer, citations, latency_ms, cost_usd}`. |
| `GET /health` | Liveness - returns `{status: "ok"}`. |
| `GET /metrics` | Rolling cost (today, mean/query), p50/p95 latency, cache hit rate, totals. |
| `GET /docs` | Interactive Swagger UI. |
| `GET /` | Single-page browser demo (upload, ask, see context + cost). |

Example:

```bash
curl -X POST https://avishkar1117-rag-service.hf.space/ingest \
  -F "file=@paper.pdf"
# → {"document_id": "abc123", "n_chunks": 42}

curl -X POST https://avishkar1117-rag-service.hf.space/query \
  -H "content-type: application/json" \
  -d '{"question": "What dataset was used?", "document_ids": ["abc123"]}'
# → {"answer": "...", "citations": [...], "latency_ms": 2150, "cost_usd": 0.0}
```

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LlamaIndex |
| Vector store | Chroma (persistent volume) |
| LLM (generation) | Gemma 4 31B (Gemini API, OpenAI-compatible endpoint) |
| LLM (RAGAS judge) | DeepSeek v4 Flash (DeepSeek API, OpenAI-compatible). Swappable via `JUDGE_PROVIDER` env var; OpenRouter and Gemini configs saved in code as fallbacks. |
| Embeddings | `models/gemini-embedding-001` |
| OCR | PyMuPDF text extraction → Gemini Vision fallback for scans |
| Cache | Upstash Redis - embedding cache (sha256-keyed), optional answer cache |
| Observability | Structured JSON logs, Sentry, in-process CostTracker |
| Eval | RAGAS 0.3.2 |
| Container | Docker (multi-stage), docker-compose |
| Deploy | Hugging Face Spaces (Docker SDK) |
| Tooling | `uv`, ruff, mypy, pytest |
| CI | GitHub Actions - lint + tests + docker build, plus RAGAS regression on PRs |

## Run locally

```bash
cp .env.example .env   # fill in GEMINI_API_KEY (and Sentry/Redis if used)
docker compose up --build
# visit http://localhost:8000/
```

Without Docker:

```bash
uv sync
uv run uvicorn rag_service.main:app --reload
```

## Tech choices and trade-offs

- **Gemma 4 31B over GPT-4o-class models.** The eval and the deployed service
  use the same model so RAGAS results reflect production. Gemma's free-tier
  Gemini quota is generous enough for both. The OpenAI-compatible endpoint
  means RAGAS's built-in `llm_factory` works unmodified, so no extra adapter.
- **Single-collection Chroma store.** All ingested PDFs share one collection
  with a `document_id` metadata filter at query time. Simpler than per-doc
  collections, and the `top_k=8` tuning shows it retrieves cleanly.
- **Embedding cache, not answer cache (by default).** Answers depend on
  retrieved chunks which change as documents are added, caching them
  invites stale results. Embeddings are content-addressed by
  `sha256(text + model_name)` and safe to cache aggressively. Answer cache
  is wired but disabled by default (TTL configurable).
- **Hugging Face Spaces over Render.** Free tier, persistent URL, Docker
  SDK supports the existing image with no rewrite. Cold-start latency
  (~30 s) is the trade-off.
- **Cross-provider RAGAS judge (DeepSeek v4 Flash).** The generator (Gemma
  4 on Gemini) and judge (DeepSeek v4 Flash on DeepSeek) come from
  different model families and different providers. A judge that doesn't
  share weights or training lineage with the generator can't recognize its
  own reasoning patterns in an answer — that eliminates self-bias on
  faithfulness scoring. The original same-model setup (Gemma judging
  Gemma) is preserved in code as `JUDGE_PROVIDER=gemini`, and OpenRouter
  is available as a third option, so swapping judges later is a one-line
  `.env` change with no code edits.

## What I'd do next

1. **Add a reranker** (Cohere or `bge-reranker-base`) between retrieval and
   generation. The current `top_k=8` is a blunt instrument; reranking 20→8
   would lift `context_precision` further and, more importantly, fix the
   q050-style cases where retrieval gets partway to the answer but not all
   the way. q050 is the deterministic test case this would close.
2. **Tighten the refusal prompt.** The current "Answer using ONLY the
   context below. If the context does not contain the answer, say 'I don't
   have enough information.'" is binary. A more nuanced version could let
   the model partially answer with a confidence qualifier — useful for
   hard multihop questions like q050 where it has half the picture.
3. **Wrap the generation call in `with_retry`** (`src/rag_service/retry.py`).
   The embedding cache already uses the rate-limit-aware retry helper, but
   `core/generation.py` calls Gemini directly. During eval, transient
   Gemini 500/503s on the generation path are caught only by a shorter
   eval-script retry loop — the production `/query` endpoint has no
   retries at all. This is a small PR with a real reliability win.
4. **Stream responses** so first-token latency feels closer to ChatGPT.
   Currently the client waits for the full Gemini completion.
5. **Per-tenant API keys + rate limits** before any real users hit it. A
   single shared key in a header is the minimum next step.

## Repository layout

```
src/rag_service/
  api/          # FastAPI routes, Pydantic schemas
  core/         # ingestion, retrieval, generation, pipeline
  cache/        # Redis embedding cache + answer cache
  observability/  # cost_tracker, request_log, logging_setup
  llm/          # Gemma + GitHub Models clients
  static/       # single-page demo
eval/
  dataset.jsonl       # 50 curated Q&A pairs
  run_ragas.py        # eval entrypoint, writes JSON+CSV+latest.md
  ragas_eval.py       # thin RAGAS wrapper (one file, easy to swap)
  results/            # timestamped runs
tests/
  unit/         # core + cache + observability
  eval/         # CI regression guard
.github/workflows/
  ci.yml        # lint + mypy + unit tests + docker build
  eval.yml      # RAGAS regression on PRs
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
