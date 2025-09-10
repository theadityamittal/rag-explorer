"""Stderr suppression utilities for ChromaDB telemetry noise."""

import os
import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr temporarily."""
    original_stderr = sys.stderr
    try:
        # Redirect stderr to a StringIO buffer
        sys.stderr = StringIO()
        yield
    finally:
        # Restore original stderr
        sys.stderr = original_stderr


@contextmanager
def filter_stderr_lines():
    """Context manager to filter out known noise from stderr."""
    original_stderr = sys.stderr

    class FilteredStderr:
        def __init__(self, original):
            self.original = original

        def write(self, text):
            # Filter out known telemetry error messages
            if any(
                phrase in text
                for phrase in [
                    "Failed to send telemetry event",
                    "capture() takes 1 positional argument",
                    "ClientStartEvent",
                    "ClientCreateCollectionEvent",
                    "CollectionAddEvent",
                    "CollectionQueryEvent",
                ]
            ):
                return  # Don't write these lines
            return self.original.write(text)

        def flush(self):
            return self.original.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

    try:
        sys.stderr = FilteredStderr(original_stderr)
        yield
    finally:
        sys.stderr = original_stderr
