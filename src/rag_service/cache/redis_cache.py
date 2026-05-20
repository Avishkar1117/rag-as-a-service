import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import redis
from llama_index.core.embeddings import BaseEmbedding
from pydantic import ConfigDict, Field

from rag_service.config import settings

logger = logging.getLogger(__name__)


def _redact_url(url: str) -> str:
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0


stats = CacheStats()


def get_redis_client() -> redis.Redis | None:
    """Connect to Redis. Returns None if unreachable so the cache silently no-ops."""
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        client.ping()
        logger.info("redis connected at %s", _redact_url(settings.redis_url))
        return client
    except Exception as e:
        logger.warning("redis unavailable, embedding cache disabled: %s", e)
        return None


def _embed_cache_key(text: str, model_name: str) -> str:
    h = hashlib.sha256(f"{model_name}::{text}".encode("utf-8")).hexdigest()
    return f"embed:{h}"


_RATE_LIMIT_MARKERS = ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "quota")
_MAX_EMBED_RETRIES = 6


def _is_rate_limit(err: Exception) -> bool:
    return any(marker in str(err) for marker in _RATE_LIMIT_MARKERS)


def _compute_with_retry(compute, text: str) -> list[float]:
    """Run an embedding call, backing off on rate-limit (429) errors.

    The Gemini embedding free tier rate-limits bursts, so a large ingest can
    outrun the per-minute quota. Wait for the window to refresh and retry.
    """
    for attempt in range(_MAX_EMBED_RETRIES):
        try:
            return compute(text)
        except Exception as e:
            if not _is_rate_limit(e) or attempt == _MAX_EMBED_RETRIES - 1:
                raise
            wait = 30 * (attempt + 1)
            logger.warning("embedding rate-limited, wait=%ds attempt=%d", wait, attempt + 1)
            time.sleep(wait)
    raise RuntimeError("unreachable")  # loop always returns or raises above


class CachingEmbedding(BaseEmbedding):
    """Read-through Redis cache wrapping any BaseEmbedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    underlying: BaseEmbedding = Field(...)
    redis_client: Any = Field(default=None)
    cache_ttl: int = Field(default=86400)
    cache_model_name: str = Field(default="default")

    def _read(self, text: str) -> list[float] | None:
        if self.redis_client is None:
            return None
        try:
            raw = self.redis_client.get(_embed_cache_key(text, self.cache_model_name))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("redis read failed, treating as miss: %s", e)
            return None

    def _write(self, text: str, embedding: list[float]) -> None:
        if self.redis_client is None:
            return
        try:
            self.redis_client.set(
                _embed_cache_key(text, self.cache_model_name),
                json.dumps(embedding),
                ex=self.cache_ttl,
            )
        except Exception as e:
            logger.warning("redis write failed: %s", e)

    def _cached(self, text: str, compute) -> list[float]:
        cached = self._read(text)
        if cached is not None:
            stats.hits += 1
            return cached
        stats.misses += 1
        embedding = _compute_with_retry(compute, text)
        self._write(text, embedding)
        return embedding

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._cached(query, self.underlying._get_query_embedding)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._cached(text, self.underlying._get_text_embedding)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
