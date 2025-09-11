"""Unified CLI package for Support Deflect Bot."""

from .main import cli
from .ask_session import UnifiedAskSession, BatchAskProcessor
from .output import (
    format_answer,
    format_search_results,
    format_metrics_table,
    format_status_summary,
    format_crawl_results,
    format_provider_validation
)

__all__ = [
    "cli",
    "UnifiedAskSession",
    "BatchAskProcessor",
    "format_answer",
    "format_search_results", 
    "format_metrics_table",
    "format_status_summary",
    "format_crawl_results",
    "format_provider_validation"
]

__version__ = "2.0.0"