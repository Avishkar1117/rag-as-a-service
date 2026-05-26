"""Thin wrapper around RAGAS — the only module in eval/ that imports `ragas`.

RAGAS changes its API often (see CLAUDE.md §9). Keeping every RAGAS import and
type confined here means a future version bump, or a swap to another eval
library, touches exactly one file.

The judge LLM is provider-pluggable. `settings.judge_provider` selects one of
"deepseek" | "openrouter" | "gemini" and the corresponding {api_key, base_url,
model} triple is read from Settings. All three providers speak the OpenAI
protocol, so RAGAS's built-in `llm_factory` works with the already-installed
`openai` SDK — no extra dependency. To swap judges, change one line in `.env`;
the others stay warm in code.

Embeddings still reuse the same GeminiEmbedding the service uses in
production, so context-recall/precision scoring stays consistent with
production retrieval regardless of which judge is active.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _JudgeProfile:
    """A resolved {api_key, base_url, model} triple for one judge provider."""

    api_key: str
    base_url: str
    model: str
    # The env var name the user is expected to set — used in the error message
    # when the key is missing, so the failure tells you exactly what to fix.
    key_env_var: str


def _resolve_judge_profile() -> _JudgeProfile:
    """Build the active judge profile from `settings.judge_provider`.

    Adding a new provider means: add fields to Settings, add a branch here.
    Nothing else in the code needs to know.
    """
    provider = settings.judge_provider.lower()
    if provider == "deepseek":
        return _JudgeProfile(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_judge_model,
            key_env_var="DEEPSEEK_API_KEY",
        )
    if provider == "openrouter":
        return _JudgeProfile(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_judge_model,
            key_env_var="OPENROUTER_API_KEY",
        )
    if provider == "gemini":
        return _JudgeProfile(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_judge_base_url,
            model=settings.gemini_judge_model,
            key_env_var="GEMINI_API_KEY",
        )
    raise RuntimeError(
        f"unknown JUDGE_PROVIDER={settings.judge_provider!r} — "
        "valid values are 'deepseek', 'openrouter', or 'gemini'"
    )


def judge_model() -> str:
    """The judge model that will be used for the next eval run.

    Exposed so `run_ragas.py` can record it in the report without duplicating
    the provider-resolution logic.
    """
    return _resolve_judge_profile().model


def _judge_llm():
    """Build the RAGAS judge LLM for the currently-selected provider.

    llm_factory builds an OpenAI client that authenticates via OPENAI_API_KEY.
    We point that client at whichever provider `judge_provider` selects. Fail
    loudly here rather than letting the OpenAI client emit a confusing auth
    error mid-eval.
    """
    profile = _resolve_judge_profile()
    if not profile.api_key:
        raise RuntimeError(
            f"{profile.key_env_var} is not set — required for "
            f"JUDGE_PROVIDER={settings.judge_provider!r}. "
            "Add it to .env or export it before running eval."
        )
    os.environ["OPENAI_API_KEY"] = profile.api_key
    return llm_factory(model=profile.model, base_url=profile.base_url)


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
    # Conservative concurrency + generous retries: free tiers rate-limit, and
    # RAGAS fires many judge calls per question.
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
