import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import redis
from llama_index.core.embeddings import BaseEmbedding
from pydantic import ConfigDict, Field

from rag_service.config import settings
from rag_service.retry import with_retry

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


class CachingEmbedding(BaseEmbedding):
    """Read-through Redis cache wrapping any BaseEmbedding.

    Every cache miss runs through ``with_retry`` — the underlying Gemini API
    intermittently rate-limits (429) or returns transient 5xx errors, and one
    un-retried failure would sink a whole multi-page ingest. Text embeddings
    also take a batch path: misses across a batch are embedded in a single
    grouped call instead of one request per chunk.
    """

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
        embedding = with_retry(lambda: compute(text), what="embedding")
        self._write(text, embedding)
        return embedding

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._cached(query, self.underlying._get_query_embedding)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._cached(text, self.underlying._get_text_embedding)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Batch path: serve cache hits, embed all misses in one grouped call.

        LlamaIndex calls this once per ``embed_batch_size`` group while
        indexing, so batching the misses turns N per-chunk requests into one —
        fewer round-trips and fewer chances to hit a transient error.
        """
        cached = [self._read(text) for text in texts]
        miss_idx = [i for i, hit in enumerate(cached) if hit is None]
        stats.hits += len(texts) - len(miss_idx)
        stats.misses += len(miss_idx)
        if not miss_idx:
            return [hit for hit in cached if hit is not None]

        miss_texts = [texts[i] for i in miss_idx]
        computed = with_retry(
            lambda: self.underlying._get_text_embeddings(miss_texts),
            what=f"embedding batch x{len(miss_texts)}",
        )
        filled = dict(zip(miss_idx, computed))

        result: list[list[float]] = []
        for i, hit in enumerate(cached):
            embedding = hit if hit is not None else filled[i]
            if hit is None:
                self._write(texts[i], embedding)
            result.append(embedding)
        return result

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)
