from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("rag_service.main.setup_llamaindex_settings"):
        from rag_service.main import app

        with TestClient(app) as c:
            yield c


def test_response_includes_request_id_header(client):
    r = client.get("/health")
    assert r.status_code == 200
    rid = r.headers.get("X-Request-ID")
    assert rid is not None
    assert len(rid) == 36  # uuid4 string


def test_each_request_gets_unique_id(client):
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


def test_request_completed_log_has_expected_fields(client):
    with patch("rag_service.observability.request_log.logger") as mock_logger:
        r = client.get("/health")

    assert r.status_code == 200
    mock_logger.info.assert_called_once()
    call_kwargs = mock_logger.info.call_args.kwargs
    assert call_kwargs["method"] == "GET"
    assert call_kwargs["path"] == "/health"
    assert call_kwargs["status"] == 200
    assert "latency_ms" in call_kwargs
    assert "request_id" in call_kwargs
    assert "cache_hits" in call_kwargs
    assert "cache_misses" in call_kwargs


def test_failed_request_logs_error(client):
    from rag_service.main import app

    @app.get("/__boom")
    def boom():
        raise RuntimeError("intentional")

    with patch("rag_service.observability.request_log.logger") as mock_logger:
        with pytest.raises(RuntimeError):
            client.get("/__boom")

    mock_logger.error.assert_called_once()
    call_kwargs = mock_logger.error.call_args.kwargs
    assert call_kwargs["path"] == "/__boom"
    assert "latency_ms" in call_kwargs
