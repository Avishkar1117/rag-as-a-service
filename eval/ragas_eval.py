"""Thin wrapper around RAGAS — the only module in eval/ that imports `ragas`.

RAGAS changes its API often (see CLAUDE.md §9). Keeping every RAGAS import and
type confined here means a future version bump, or a swap to another eval
library, touches exactly one file.

The judge LLM runs through Gemini's OpenAI-compatible endpoint, so RAGAS's
built-in `llm_factory` (which speaks the OpenAI protocol) works with the
already-installed `openai` SDK — no extra dependency. Embeddings reuse the same
GeminiEmbedding the service uses in production.
"""

from __future__ import annotations

import os

import ragas
from llama_index.embeddings.gemini import GeminiEmbedding
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from rag_service.config import settings

# Keep eval runs offline and quiet — no anonymous usage telemetry.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

RAGAS_VERSION: str = ragas.__version__

# Gemini's OpenAI-compatible endpoint; lets llm_factory's OpenAI client reach Gemini.
_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# The four metrics from the project brief (CLAUDE.md §6, Day 10-11), in report order:
#   faithfulness        — does the answer follow from the retrieved context?
#   answer relevance    — does the answer actually address the question?
#   context precision   — is the retrieved context on-topic?
#   context recall      — did retrieval surface the ground-truth content?
_METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]
METRIC_NAMES: list[str] = [m.name for m in _METRICS]


def _judge_llm():
    # llm_factory builds an OpenAI client that authenticates via OPENAI_API_KEY.
    # We point that client at Gemini, so the key it reads must be the Gemini key.
    os.environ["OPENAI_API_KEY"] = settings.gemini_api_key
    return llm_factory(model=settings.ragas_judge_model, base_url=_GEMINI_OPENAI_BASE_URL)


def _judge_embeddings() -> LlamaIndexEmbeddingsWrapper:
    underlying = GeminiEmbedding(
        model_name=settings.embedding_model, api_key=settings.gemini_api_key
    )
    return LlamaIndexEmbeddingsWrapper(underlying)


def score(samples: list[dict]):
    """Score samples with the four RAGAS metrics.

    Each sample dict must provide: user_input, retrieved_contexts, response,
    reference. Returns a pandas DataFrame — one row per sample, one column per
    METRIC_NAMES entry (plus the input columns). A NaN cell means RAGAS could
    not score that metric for that sample.
    """
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["user_input"],
                retrieved_contexts=s["retrieved_contexts"],
                response=s["response"],
                reference=s["reference"],
            )
            for s in samples
        ]
    )
    # Conservative concurrency + generous retries: the Gemini free tier rate-limits,
    # and RAGAS fires many judge calls per question.
    run_config = RunConfig(timeout=240, max_retries=10, max_wait=90, max_workers=3)
    result = evaluate(
        dataset=dataset,
        metrics=_METRICS,
        llm=_judge_llm(),
        embeddings=_judge_embeddings(),
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    return result.to_pandas()
