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
the rest of the stack — a typed HTTP API, a Redis embedding cache, structured
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

Both runs use the same generation model (Gemma 4 31B), the same RAGAS judge,
the same `chunk_size=1024`, and the same 50-question dataset (35 factual, 10
multi-hop, 5 adversarial). The only thing that changed is `top_k`.

| Metric | Baseline `top_k=3` | Tuned `top_k=8` | Δ |
|---|---|---|---|
| faithfulness | 0.9744 | **1.0000** | +0.0256 |
| answer_relevancy | 0.7830 | 0.7328 | −0.0502 |
| context_precision | 0.8350 | **0.9413** | +0.1063 |
| context_recall | 0.8933 | **0.9123** | +0.0190 |
| multihop context_recall | 0.6667 | **0.8333** | +0.1666 |
| adversarial refusal rate | 1.0000 | 0.6000 | −0.4000 |

The retrieval win is clearest on **multi-hop questions**, where pulling 8
chunks instead of 3 lets the model see both halves of a two-part answer:
`context_recall` on that slice jumps from 0.67 to 0.83. Faithfulness is
also pinned to 1.0 at `top_k=8` (no hallucinations on the 45 non-adversarial
questions that scored cleanly).

The trade-off is adversarial refusal: at `top_k=8` two of the five
unanswerable questions retrieved enough loosely-related context that the
model attempted an answer instead of declining. This is a known
precision/refusal trade-off with wider retrieval; tightening the refusal
prompt is the next iteration.

Per-question scores: `eval/results/ragas_20260521_090522.json` (and matching
`.csv`). Latest aggregate report: `eval/results/latest.md`.

### Evaluation as CI

A 10-question regression guard (`tests/eval/test_regression.py`) re-runs the
RAGAS harness on every pull request and fails if faithfulness drops below
0.85. The workflow is in `.github/workflows/eval.yml`. This is the angle that
makes the service genuinely shippable — prompt or model changes that quietly
regress quality get caught before they merge.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Multipart PDF upload. Returns `{document_id, n_chunks}`. |
| `POST /query` | `{question, top_k?, document_ids?}` → `{answer, citations, latency_ms, cost_usd}`. |
| `GET /health` | Liveness — returns `{status: "ok"}`. |
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
| LLM | Gemma 4 31B (Gemini API, OpenAI-compatible endpoint) |
| Embeddings | `models/gemini-embedding-001` |
| OCR | PyMuPDF text extraction → Gemini Vision fallback for scans |
| Cache | Upstash Redis — embedding cache (sha256-keyed), optional answer cache |
| Observability | Structured JSON logs, Sentry, in-process CostTracker |
| Eval | RAGAS 0.3.2 |
| Container | Docker (multi-stage), docker-compose |
| Deploy | Hugging Face Spaces (Docker SDK) |
| Tooling | `uv`, ruff, mypy, pytest |
| CI | GitHub Actions — lint + tests + docker build, plus RAGAS regression on PRs |

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
  means RAGAS's built-in `llm_factory` works unmodified — no extra adapter.
- **Single-collection Chroma store.** All ingested PDFs share one collection
  with a `document_id` metadata filter at query time. Simpler than per-doc
  collections, and the `top_k=8` tuning shows it retrieves cleanly.
- **Embedding cache, not answer cache (by default).** Answers depend on
  retrieved chunks which change as documents are added — caching them
  invites stale results. Embeddings are content-addressed by
  `sha256(text + model_name)` and safe to cache aggressively. Answer cache
  is wired but disabled by default (TTL configurable).
- **Hugging Face Spaces over Render.** Free tier, persistent URL, Docker
  SDK supports the existing image with no rewrite. Cold-start latency
  (~30 s) is the trade-off.
- **RAGAS judge = generation model.** Cheaper and avoids dragging in a
  second API. A separate judge would reduce self-bias on faithfulness — a
  fair next step if budget allowed.

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

## What I'd do next

1. **Tighten the refusal prompt** to recover adversarial refusal rate at
   `top_k=8` without giving back the multi-hop recall gains.
2. **Add a reranker** (Cohere or BGE) between retrieval and generation —
   the current `top_k=8` is a blunt instrument; reranking 20→8 would lift
   precision further.
3. **Stream responses** so first-token latency feels closer to ChatGPT.
4. **A second judge model** for RAGAS to reduce self-bias on faithfulness.
5. **Per-tenant API keys + rate limits** before any real users hit it.
