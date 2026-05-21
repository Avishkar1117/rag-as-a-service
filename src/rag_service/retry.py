"""Shared retry helper for flaky upstream API calls.

The embedding cache and the OCR step both call Google APIs that intermittently
return 429 (free-tier rate limit) and 500/503 (transient server errors). A
single un-retried failure during a large multi-page ingest fails the whole
document, so every upstream call is wrapped in ``with_retry``: exponential
backoff with full jitter over a handful of attempts.
"""

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Substrings matched against the exception text. 429 / quota is the per-minute
# free-tier limit; 500 / 503 are transient Google outages. Matching on text
# keeps this agnostic to which client raised it (google-genai, grpc, requests).
_RETRYABLE_MARKERS = (
    "429",
    "ResourceExhausted",
    "RESOURCE_EXHAUSTED",
    "quota",
    "500",
    "503",
    "INTERNAL",
    "UNAVAILABLE",
)

_MAX_ATTEMPTS = 6
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 60.0


def is_retryable(err: Exception) -> bool:
    """True for transient upstream failures worth retrying (rate limits, 5xx)."""
    text = str(err)
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter; ``attempt`` is 0-indexed."""
    ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2**attempt)
    return random.uniform(0.0, ceiling)


def with_retry(
    operation: Callable[[], T],
    *,
    what: str,
    max_attempts: int = _MAX_ATTEMPTS,
) -> T:
    """Run ``operation``, retrying transient upstream failures with backoff.

    Non-retryable errors — and the last attempt's error — are re-raised
    unchanged, so genuine bugs still surface instead of being masked.
    """
    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as e:
            if not is_retryable(e) or attempt == max_attempts - 1:
                raise
            wait = _backoff_seconds(attempt)
            logger.warning(
                "%s failed (%s); retrying in %.1fs (attempt %d/%d)",
                what,
                type(e).__name__,
                wait,
                attempt + 1,
                max_attempts - 1,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable: with_retry must return or raise inside the loop")
