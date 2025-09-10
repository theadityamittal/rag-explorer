"""Warning suppression utilities for cleaner CLI output."""

import logging
import os
import warnings
from contextlib import contextmanager


def suppress_noisy_warnings():
    """Suppress known harmless warnings that clutter CLI output."""

    # Suppress urllib3 OpenSSL warning - this is a system-level issue on macOS
    # The warning is harmless but very noisy
    warnings.filterwarnings(
        "ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+.*"
    )

    # Also suppress by module name
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

    # Suppress all NotOpenSSLWarning specifically
    try:
        import urllib3

        warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
    except (ImportError, AttributeError):
        pass

    # Suppress ChromaDB telemetry warnings - these are due to version incompatibilities
    # and don't affect functionality
    warnings.filterwarnings("ignore", message=".*capture.*takes.*positional argument.*")

    warnings.filterwarnings("ignore", message=".*Failed to send telemetry event.*")

    # Set environment variables to disable ChromaDB telemetry entirely
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_CLIENT_TELEMETRY"] = "False"
    os.environ["CHROMA_CLIENT_TELEMETRY_ENABLED"] = "False"
    os.environ["POSTHOG_DISABLED"] = "True"


def configure_logging():
    """Configure logging to reduce noise in CLI output."""
    # Set ChromaDB logging to ERROR level to reduce noise
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)

    # Suppress other noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Configure root logger to be less verbose
    logging.basicConfig(level=logging.WARNING)


@contextmanager
def quiet_mode():
    """Context manager for completely quiet operation."""
    # Store original stderr
    import sys
    from io import StringIO

    old_stderr = sys.stderr
    sys.stderr = captured_stderr = StringIO()

    try:
        yield
    finally:
        # Restore stderr
        sys.stderr = old_stderr

        # Optionally log captured warnings to debug if needed
        captured = captured_stderr.getvalue()
        if captured and os.getenv("DEBUG_WARNINGS"):
            print(f"Captured warnings: {captured}", file=old_stderr)


def init_clean_cli():
    """Initialize clean CLI environment by suppressing all known noisy warnings."""
    suppress_noisy_warnings()
    configure_logging()
