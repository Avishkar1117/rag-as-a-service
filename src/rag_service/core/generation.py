from __future__ import annotations

from dataclasses import dataclass

from google import genai
from llama_index.core.schema import NodeWithScore

from rag_service.config import settings
from rag_service.retry import with_retry

_PROMPT = """\
Answer using ONLY the context below. \
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class GenerationResult:
    """Output of one generation call, including token usage for cost tracking."""

    answer: str
    citations: list[str]
    model: str
    prompt_tokens: int
    output_tokens: int


def generate(question: str, nodes: list[NodeWithScore]) -> GenerationResult:
    """Call Gemma with retrieved context. Returns answer, citations, and token usage.

    The Gemini API call is wrapped in ``with_retry`` so transient 500/503s and
    free-tier 429 rate limits don't kill a user-facing /query (or an eval
    question) on the first hiccup. The embedding cache uses the same helper;
    this closes the matching gap on the generation path.
    """
    context = "\n\n".join(node.node.get_content() for node in nodes)
    citations = [node.node.metadata.get("document_id", "") for node in nodes]

    client = genai.Client(api_key=settings.gemini_api_key)
    response = with_retry(
        lambda: client.models.generate_content(
            model=settings.gemma_model,
            contents=_PROMPT.format(context=context, question=question),
        ),
        what="gemma generation",
    )

    # usage_metadata is absent on some error/streamed responses; default to 0.
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    return GenerationResult(
        answer=(response.text or "").strip(),
        citations=citations,
        model=settings.gemma_model,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
