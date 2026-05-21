"""Per-query cost estimation and rolling cost/latency metrics.

The default generation model (Gemma on the Gemini API free tier) has no
per-token charge, so its tracked cost is $0.00. The pricing table still
carries paid models so the methodology holds — and so pointing GEMMA_MODEL
at a billed model immediately produces real cost figures.

Cost covers the generation LLM call only. A query embeds one short question
via a Redis-cached call; that token volume is negligible and not priced.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """List price in USD per 1,000,000 tokens."""

    input_per_1m: float
    output_per_1m: float


# List prices in USD per 1M tokens. Verify against the provider pricing pages
# (ai.google.dev/pricing, openai.com/api/pricing) before relying on these.
MODEL_PRICING: dict[str, ModelPrice] = {
    # Gemma on the Gemini API free tier — no per-token billing.
    "gemma-4-31b-it": ModelPrice(0.0, 0.0),
    # Paid reference models, used if GEMMA_MODEL points at a billed model.
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gpt-4o": ModelPrice(2.50, 10.00),
}


def estimate_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """USD cost of one generation call. Unknown models cost 0.0 (logged)."""
    price = MODEL_PRICING.get(model)
    if price is None:
        logger.warning("no pricing entry for model %r, recording cost=0", model)
        return 0.0
    cost = (prompt_tokens * price.input_per_1m + output_tokens * price.output_per_1m) / 1_000_000
    return round(cost, 6)


@dataclass
class QueryRecord:
    """One query's cost and latency, stamped with when it completed."""

    timestamp: datetime
    cost_usd: float
    latency_ms: int


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile. `pct` in [0, 1]. Empty input -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


class CostTracker:
    """In-memory rolling record of per-query cost and latency.

    Process-local and non-persistent: aggregates reset on restart, which is
    acceptable for a /metrics snapshot. Bounded by `max_records` to cap memory.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._records: deque[QueryRecord] = deque(maxlen=max_records)

    def record(self, cost_usd: float, latency_ms: int) -> None:
        self._records.append(QueryRecord(datetime.now(timezone.utc), cost_usd, latency_ms))

    def snapshot(self) -> dict:
        """Aggregate metrics over the records currently held."""
        records = list(self._records)
        n = len(records)
        if n == 0:
            return {
                "n_queries": 0,
                "total_cost_usd_today": 0.0,
                "mean_cost_usd_per_query": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }
        today = datetime.now(timezone.utc).date()
        costs = [r.cost_usd for r in records]
        latencies = [float(r.latency_ms) for r in records]
        total_today = sum(r.cost_usd for r in records if r.timestamp.date() == today)
        return {
            "n_queries": n,
            "total_cost_usd_today": round(total_today, 6),
            "mean_cost_usd_per_query": round(sum(costs) / n, 6),
            "p50_latency_ms": round(_percentile(latencies, 0.50), 1),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 1),
        }

    def reset(self) -> None:
        self._records.clear()


# Process-wide singleton, mirroring `stats` in cache.redis_cache.
tracker = CostTracker()
