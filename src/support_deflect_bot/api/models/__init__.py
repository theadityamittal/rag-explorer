"""API models for Support Deflect Bot."""

from .requests import (
    AskRequest,
    SearchRequest, 
    IndexRequest,
    CrawlRequest,
    BatchAskRequest
)
from .responses import (
    AskResponse,
    SearchResponse,
    IndexResponse, 
    CrawlResponse,
    HealthResponse,
    BatchAskResponse,
    Source,
    SearchResult,
    ProcessingDetail
)

__all__ = [
    # Request models
    "AskRequest",
    "SearchRequest", 
    "IndexRequest",
    "CrawlRequest",
    "BatchAskRequest",
    # Response models
    "AskResponse",
    "SearchResponse",
    "IndexResponse", 
    "CrawlResponse",
    "HealthResponse",
    "BatchAskResponse",
    "Source",
    "SearchResult",
    "ProcessingDetail"
]