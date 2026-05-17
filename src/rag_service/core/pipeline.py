import time

from rag_service.core.generation import generate
from rag_service.core.retrieval import retrieve


def query_pipeline(question: str, document_id: str, top_k: int) -> dict:
    """End-to-end RAG: retrieve → generate. Returns answer, citations, latency_ms."""
    start = time.perf_counter()
    nodes = retrieve(document_id, question, top_k)
    answer, citations = generate(question, nodes)
    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }
