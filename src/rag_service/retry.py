"""Shared retry helper for flaky upstream API calls.

The embedding cache and the OCR step both call Google APIs that intermittently
fail in two distinct ways, and each wants a different backoff:

* **Rate limits** (429 / RESOURCE_EXHAUSTED) — the free tier caps requests and
  tokens *per minute*. Once a window is exhausted it only clears when the 60s
  window rolls over, so a short backoff just burns an attempt on a guaranteed
  repeat 429. These wait out a full minute (plus jitter) before retrying.
* **Transient 5xx** (500 / 503 / INTERNAL / UNAVAILABLE) — brief Google-side
  outages that usually clear within seconds. These use exponential backoff.

A single un-retried failure during a large multi-page ingest fails the whole
document, so every upstream call is wrapped in ``with_retry``.
"""

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Substrings matched against the exception text. Matching on text keeps this
# agnostic to which client raised it (google-genai, grpc, requests).
#
# Rate limits clear only when the per-minute quota window rolls over; transient
# 5xx errors usually clear within seconds. The two need different backoff, so
# they are tracked separately and unioned for the "worth retrying?" check.
_RATE_LIMIT_MARKERS = ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "quota")
_TRANSIENT_MARKERS = ("500", "503", "INTERNAL", "UNAVAILABLE")
_RETRYABLE_MARKERS = _RATE_LIMIT_MARKERS + _TRANSIENT_MARKERS

_MAX_ATTEMPTS = 6
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 60.0
# Free-tier quotas are enforced over a rolling 60s window. A rate-limit retry
# waits one full window so the quota has provably reset; the jittered upper
# bound keeps concurrent workers from retrying in lockstep.
_RATE_LIMIT_WINDOW_S = 60.0


def is_retryable(err: Exception) -> bool:
    """True for transient upstream failures worth retrying (rate limits, 5xx)."""
    text = str(err)
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _is_rate_limit(err: Exception) -> bool:
    """True when the failure is a quota / rate-limit error, not a transient 5xx."""
    text = str(err)
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _backoff_seconds(attempt: int, *, rate_limited: bool) -> float:
    """Seconds to wait before the next attempt; ``attempt`` is 0-indexed.

    A rate limit waits out a full per-minute window — short backoff would just
    burn an attempt on a quota that provably has not reset. Transient 5xx use
    exponential backoff with equal jitter: a guaranteed floor of half the
    ceiling, so a late attempt can't collapse to a near-zero wait.
    """
    if rate_limited:
        return random.uniform(_RATE_LIMIT_WINDOW_S, _RATE_LIMIT_WINDOW_S * 1.5)
    ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2**attempt)
    return ceiling / 2 + random.uniform(0.0, ceiling / 2)


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
            wait = _backoff_seconds(attempt, rate_limited=_is_rate_limit(e))
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
