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

Production-grade RAG service: FastAPI + LlamaIndex + Chroma + Gemma 4 31B + Redis cache.

## Endpoints

- `POST /ingest` — upload PDF, get back a `document_id`
- `POST /query` — ask a question against an ingested document
- `GET /health` — liveness
- `GET /metrics` — cache hit rate
- `GET /docs` — Swagger UI

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LlamaIndex |
| Vector store | Chroma (persistent volume) |
| LLM | Gemma 4 31B (via Gemini API) |
| Embeddings | Gemini `embedding-001` |
| OCR | Hybrid: PyMuPDF text extraction → Gemini Vision fallback |
| Cache | Upstash Redis (embedding cache, sha256-keyed) |
| Deploy | Docker, Hugging Face Spaces |

## Run locally

```bash
cp .env.example .env  # fill in keys
docker compose up --build
# visit http://localhost:8000/
```

## Status

In active development.
