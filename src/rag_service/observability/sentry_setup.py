import sentry_sdk
import structlog

logger = structlog.get_logger(__name__)


def setup_sentry(dsn: str) -> None:
    """Initialize Sentry. No-op when DSN is empty (e.g. local dev)."""
    if not dsn:
        logger.info("sentry_disabled", reason="empty_dsn")
        return

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.1,
        send_default_pii=False,
    )
    logger.info("sentry_initialized")
