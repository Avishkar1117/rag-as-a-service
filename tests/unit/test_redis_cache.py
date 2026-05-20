import json
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from rag_service.cache.redis_cache import (
    CachingEmbedding,
    _compute_with_retry,
    _embed_cache_key,
    _redact_url,
    get_redis_client,
    stats,
)


class FakeEmbedding(BaseEmbedding):
    _query_calls: list[str] = PrivateAttr(default_factory=list)
    _text_calls: list[str] = PrivateAttr(default_factory=list)

    def _get_query_embedding(self, query: str) -> list[float]:
        self._query_calls.append(query)
        return [0.1, 0.2, 0.3]

    def _get_text_embedding(self, text: str) -> list[float]:
        self._text_calls.append(text)
        return [0.4, 0.5, 0.6]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


@pytest.fixture(autouse=True)
def _reset_stats():
    stats.reset()
    yield
    stats.reset()


def test_cache_miss_calls_underlying_and_stores():
    fake_redis = MagicMock()
    fake_redis.get.return_value = None
    underlying = FakeEmbedding()

    embed = CachingEmbedding(underlying=underlying, redis_client=fake_redis, cache_model_name="m1")
    result = embed._get_query_embedding("hello")

    assert result == [0.1, 0.2, 0.3]
    assert underlying._query_calls == ["hello"]
    fake_redis.set.assert_called_once()
    assert stats.misses == 1
    assert stats.hits == 0


def test_cache_hit_skips_underlying():
    fake_redis = MagicMock()
    fake_redis.get.return_value = json.dumps([9.9, 8.8, 7.7]).encode("utf-8")
    underlying = FakeEmbedding()

    embed = CachingEmbedding(underlying=underlying, redis_client=fake_redis, cache_model_name="m1")
    result = embed._get_query_embedding("hello")

    assert result == [9.9, 8.8, 7.7]
    assert underlying._query_calls == []
    fake_redis.set.assert_not_called()
    assert stats.hits == 1
    assert stats.misses == 0


def test_falls_through_when_redis_client_is_none():
    underlying = FakeEmbedding()
    embed = CachingEmbedding(underlying=underlying, redis_client=None, cache_model_name="m1")

    result = embed._get_text_embedding("doc text")

    assert result == [0.4, 0.5, 0.6]
    assert underlying._text_calls == ["doc text"]
    assert stats.misses == 1


def test_redis_read_error_treated_as_miss():
    fake_redis = MagicMock()
    fake_redis.get.side_effect = RuntimeError("connection lost")
    underlying = FakeEmbedding()

    embed = CachingEmbedding(underlying=underlying, redis_client=fake_redis, cache_model_name="m1")
    result = embed._get_query_embedding("hello")

    assert result == [0.1, 0.2, 0.3]
    assert stats.misses == 1


def test_cache_key_differs_by_model_name():
    assert _embed_cache_key("text", "model-a") != _embed_cache_key("text", "model-b")


def test_redact_url_hides_password():
    redacted = _redact_url("rediss://default:secret123@host.upstash.io:6379")
    assert "secret123" not in redacted
    assert "***" in redacted
    assert "host.upstash.io" in redacted


def test_redact_url_handles_no_credentials():
    assert _redact_url("redis://localhost:6379") == "redis://localhost:6379"


def test_get_redis_client_returns_none_on_connection_failure():
    with patch("rag_service.cache.redis_cache.redis.Redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("nope")
        mock_from_url.return_value = mock_client

        assert get_redis_client() is None


def test_compute_with_retry_recovers_from_rate_limit():
    calls = []

    def flaky(text: str) -> list[float]:
        calls.append(text)
        if len(calls) < 3:
            raise RuntimeError("429 ResourceExhausted: quota exceeded")
        return [1.0, 2.0]

    with patch("rag_service.cache.redis_cache.time.sleep") as mock_sleep:
        result = _compute_with_retry(flaky, "hello")

    assert result == [1.0, 2.0]
    assert len(calls) == 3
    assert mock_sleep.call_count == 2


def test_compute_with_retry_reraises_non_rate_limit_error():
    def boom(text: str) -> list[float]:
        raise ValueError("malformed input")

    with patch("rag_service.cache.redis_cache.time.sleep"):
        with pytest.raises(ValueError, match="malformed input"):
            _compute_with_retry(boom, "hello")
