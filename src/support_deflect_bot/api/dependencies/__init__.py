"""API dependencies for Support Deflect Bot."""

from .engine import (
    get_rag_engine,
    get_query_service,
    get_document_processor,
    get_embedding_service,
    set_rag_engine,
    set_query_service,
    set_document_processor,
    set_embedding_service,
    cleanup_engines,
)

from .validation import (
    validate_domain_filter,
    validate_user_agent,
    validate_crawl_urls,
    validate_file_patterns,
    validate_pagination,
)

from .security import (
    get_api_key,
    verify_api_key,
    rate_limiter,
    check_content_type,
    validate_request_size,
    SecurityHeaders,
)

__all__ = [
    # Engine dependencies
    "get_rag_engine",
    "get_query_service", 
    "get_document_processor",
    "get_embedding_service",
    "set_rag_engine",
    "set_query_service",
    "set_document_processor",
    "set_embedding_service",
    "cleanup_engines",
    # Validation dependencies
    "validate_domain_filter",
    "validate_user_agent",
    "validate_crawl_urls",
    "validate_file_patterns",
    "validate_pagination",
    # Security dependencies
    "get_api_key",
    "verify_api_key",
    "rate_limiter",
    "check_content_type",
    "validate_request_size",
    "SecurityHeaders",
]