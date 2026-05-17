from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.gemini import GeminiEmbedding

from rag_service.cache.redis_cache import CachingEmbedding, get_redis_client
from rag_service.config import settings

_EMBED_MODEL_NAME = "models/gemini-embedding-001"


def setup_llamaindex_settings() -> None:
    """Configure global LlamaIndex settings. Call once at app startup."""
    underlying = GeminiEmbedding(
        model_name=_EMBED_MODEL_NAME,
        api_key=settings.gemini_api_key,
    )
    LlamaSettings.embed_model = CachingEmbedding(
        underlying=underlying,
        redis_client=get_redis_client(),
        cache_ttl=settings.cache_ttl,
        cache_model_name=_EMBED_MODEL_NAME,
    )
    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap
