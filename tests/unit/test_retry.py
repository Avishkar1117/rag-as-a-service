from unittest.mock import patch

import pytest

from rag_service.retry import is_retryable, with_retry


def test_is_retryable_detects_rate_limit():
    assert is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert is_retryable(RuntimeError("ResourceExhausted"))


def test_is_retryable_detects_transient_server_errors():
    assert is_retryable(RuntimeError("500 INTERNAL"))
    assert is_retryable(RuntimeError("503 UNAVAILABLE"))


def test_is_retryable_false_for_real_errors():
    assert not is_retryable(ValueError("malformed input"))
    assert not is_retryable(KeyError("missing key"))


def test_with_retry_returns_value_on_success():
    assert with_retry(lambda: 42, what="thing") == 42


def test_with_retry_recovers_from_transient_500():
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("500 INTERNAL")
        return "ok"

    with patch("rag_service.retry.time.sleep") as mock_sleep:
        result = with_retry(flaky, what="embedding")

    assert result == "ok"
    assert len(calls) == 3
    assert mock_sleep.call_count == 2


def test_with_retry_recovers_from_rate_limit():
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("429 ResourceExhausted")
        return "ok"

    with patch("rag_service.retry.time.sleep"):
        assert with_retry(flaky, what="embedding") == "ok"

    assert len(calls) == 2


def test_with_retry_reraises_non_retryable_immediately():
    calls = []

    def boom() -> None:
        calls.append(1)
        raise ValueError("malformed input")

    with patch("rag_service.retry.time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="malformed input"):
            with_retry(boom, what="embedding")

    assert len(calls) == 1  # a genuine error is not retried
    mock_sleep.assert_not_called()


def test_with_retry_gives_up_after_max_attempts():
    calls = []

    def always_unavailable() -> None:
        calls.append(1)
        raise RuntimeError("503 UNAVAILABLE")

    with patch("rag_service.retry.time.sleep"):
        with pytest.raises(RuntimeError, match="503"):
            with_retry(always_unavailable, what="embedding", max_attempts=4)

    assert len(calls) == 4
