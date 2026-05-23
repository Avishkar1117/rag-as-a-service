from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rag_service.cache.redis_cache import stats as cache_stats
from rag_service.observability.cost_tracker import tracker as cost_tracker


@pytest.fixture
def client():
    # Patch lifespan setup so tests don't touch real Gemini / GitHub Models.
    with patch("rag_service.main.setup_llamaindex_settings"):
        from rag_service.main import app

        with TestClient(app) as c:
            yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_serves_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "RAG-as-a-Service" in r.text


def test_ingest_rejects_non_pdf(client):
    r = client.post(
        "/ingest",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert r.status_code == 400


def test_ingest_returns_document_id_and_chunks(client):
    with patch("rag_service.api.routes.ingest_document", return_value=7):
        r = client.post(
            "/ingest",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["n_chunks"] == 7
    assert len(data["document_id"]) == 12


def test_query_returns_answer_and_citations(client):
    mock_result = {
        "answer": "Berlin is the capital",
        "citations": ["docX"],
        "latency_ms": 142,
        "cost_usd": 0.0,
    }
    with patch("rag_service.api.routes.query_pipeline", return_value=mock_result):
        r = client.post(
            "/query",
            json={"question": "Capital of Germany?", "document_id": "docX", "top_k": 3},
        )
    assert r.status_code == 200
    assert r.json() == mock_result


def test_query_rejects_empty_question(client):
    r = client.post(
        "/query",
        json={"question": "", "document_id": "docX", "top_k": 3},
    )
    assert r.status_code == 422


def test_query_rejects_top_k_out_of_range(client):
    r = client.post(
        "/query",
        json={"question": "q?", "document_id": "docX", "top_k": 999},
    )
    assert r.status_code == 422


def test_metrics_returns_cache_and_cost_stats(client):
    cache_stats.reset()
    cache_stats.hits = 7
    cache_stats.misses = 3
    cost_tracker.reset()
    cost_tracker.record(cost_usd=0.05, latency_ms=200)

    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["cache_hits"] == 7
    assert data["cache_misses"] == 3
    assert data["cache_hit_rate"] == 0.7
    assert data["n_queries"] == 1
    assert data["total_cost_usd_today"] == 0.05
    assert data["p50_latency_ms"] == 200.0

    cache_stats.reset()
    cost_tracker.reset()
