import time
import uuid
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rag_service.cache.redis_cache import stats as cache_stats

logger = structlog.get_logger(__name__)

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Logs one JSON event per request with timing + cache hit/miss delta."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

        start = time.perf_counter()
        hits_before = cache_stats.hits
        misses_before = cache_stats.misses

        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                latency_ms=elapsed_ms,
            )
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=elapsed_ms,
            cache_hits=cache_stats.hits - hits_before,
            cache_misses=cache_stats.misses - misses_before,
        )

        response.headers["X-Request-ID"] = request_id
        return response
