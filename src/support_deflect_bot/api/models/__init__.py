"""API models for Support Deflect Bot."""

from .requests import (
    AskRequest,
    SearchRequest,
    IndexRequest,
    CrawlRequest,
    BatchAskRequest,
)

from .responses import (
    Source,
    AskResponse,
    SearchResult,
    SearchResponse,
    ProcessingDetail,
    IndexResponse,
    CrawlResponse,
    HealthResponse,
    BatchAskResponse,
    ErrorResponse,
)

__all__ = [
    # Request models
    "AskRequest",
    "SearchRequest", 
    "IndexRequest",
    "CrawlRequest",
    "BatchAskRequest",
    # Response models
    "Source",
    "AskResponse",
    "SearchResult",
    "SearchResponse",
    "ProcessingDetail",
    "IndexResponse",
    "CrawlResponse", 
    "HealthResponse",
    "BatchAskResponse",
    "ErrorResponse",
]