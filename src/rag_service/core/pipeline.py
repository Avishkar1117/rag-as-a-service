import time

import structlog

from rag_service.core.generation import generate
from rag_service.core.retrieval import retrieve
from rag_service.observability.cost_tracker import estimate_cost, tracker
from rag_service.observability.request_log import request_id_var

logger = structlog.get_logger(__name__)


def query_pipeline(question: str, document_id: str, top_k: int) -> dict:
    """End-to-end RAG: retrieve → generate. Returns answer, citations, latency, cost.

    Records cost + latency in the metrics tracker and emits one structured
    `query_completed` log event per call.
    """
    start = time.perf_counter()
    nodes = retrieve(document_id, question, top_k)
    result = generate(question, nodes)
    latency_ms = int((time.perf_counter() - start) * 1000)

    cost_usd = estimate_cost(result.model, result.prompt_tokens, result.output_tokens)
    tracker.record(cost_usd, latency_ms)

    logger.info(
        "query_completed",
        request_id=request_id_var.get(),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        n_retrieved=len(nodes),
    )
    return {
        "answer": result.answer,
        "citations": result.citations,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }
