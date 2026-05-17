from unittest.mock import patch

from rag_service.observability.sentry_setup import setup_sentry


def test_setup_sentry_noop_when_dsn_empty():
    with patch("rag_service.observability.sentry_setup.sentry_sdk.init") as mock_init:
        setup_sentry("")
    mock_init.assert_not_called()


def test_setup_sentry_initializes_when_dsn_present():
    with patch("rag_service.observability.sentry_setup.sentry_sdk.init") as mock_init:
        setup_sentry("https://abc@sentry.io/123")
    mock_init.assert_called_once()
    kwargs = mock_init.call_args.kwargs
    assert kwargs["dsn"] == "https://abc@sentry.io/123"
    assert kwargs["send_default_pii"] is False
