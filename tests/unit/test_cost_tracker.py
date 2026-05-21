from datetime import datetime, timedelta, timezone

from rag_service.observability.cost_tracker import (
    MODEL_PRICING,
    CostTracker,
    QueryRecord,
    _percentile,
    estimate_cost,
)


def test_estimate_cost_paid_model():
    # gpt-4o: $2.50/1M input, $10.00/1M output.
    cost = estimate_cost("gpt-4o", prompt_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 12.5


def test_estimate_cost_partial_tokens():
    cost = estimate_cost("gpt-4o", 2000, 500)
    expected = round((2000 * 2.5 + 500 * 10.0) / 1_000_000, 6)
    assert cost == expected


def test_estimate_cost_free_model_is_zero():
    assert estimate_cost("gemma-4-31b-it", 5000, 5000) == 0.0


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost("some-model-we-do-not-price", 1000, 1000) == 0.0


def test_pricing_table_prices_the_default_model():
    # The configured default generation model must be in the pricing table.
    assert "gemma-4-31b-it" in MODEL_PRICING


def test_percentile_basic():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(values, 0.5) == 30.0
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 1.0) == 50.0


def test_percentile_interpolates():
    # p95 over 0..100 (101 evenly spaced values) lands exactly on 95.0.
    values = [float(i) for i in range(101)]
    assert _percentile(values, 0.95) == 95.0


def test_percentile_empty_is_zero():
    assert _percentile([], 0.5) == 0.0


def test_tracker_snapshot_empty():
    snap = CostTracker().snapshot()
    assert snap == {
        "n_queries": 0,
        "total_cost_usd_today": 0.0,
        "mean_cost_usd_per_query": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
    }


def test_tracker_records_and_aggregates():
    t = CostTracker()
    t.record(cost_usd=0.01, latency_ms=100)
    t.record(cost_usd=0.03, latency_ms=300)
    snap = t.snapshot()
    assert snap["n_queries"] == 2
    assert snap["mean_cost_usd_per_query"] == 0.02
    assert snap["total_cost_usd_today"] == 0.04
    assert snap["p50_latency_ms"] == 200.0


def test_tracker_total_cost_today_excludes_old_records():
    t = CostTracker()
    # Inject a record stamped yesterday, then record one for today.
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    t._records.append(QueryRecord(timestamp=yesterday, cost_usd=5.0, latency_ms=100))
    t.record(cost_usd=0.02, latency_ms=100)
    snap = t.snapshot()
    assert snap["n_queries"] == 2  # both records are still held
    assert snap["total_cost_usd_today"] == 0.02  # only today's cost is summed


def test_tracker_respects_max_records():
    t = CostTracker(max_records=3)
    for i in range(10):
        t.record(cost_usd=0.0, latency_ms=i)
    assert t.snapshot()["n_queries"] == 3


def test_tracker_reset():
    t = CostTracker()
    t.record(0.01, 100)
    t.reset()
    assert t.snapshot()["n_queries"] == 0
