"""Dependencies for engine components in the API."""

from functools import lru_cache
from ...engine.rag_engine import UnifiedRAGEngine
from ...engine.document_processor import UnifiedDocumentProcessor
from ...engine.query_service import UnifiedQueryService
from ...engine.embedding_service import UnifiedEmbeddingService
from ...config.manager import ConfigurationManager

@lru_cache()
def get_config_manager() -> ConfigurationManager:
    """Get the configuration manager singleton."""
    return ConfigurationManager()

@lru_cache()
def get_embedding_service() -> UnifiedEmbeddingService:
    """Get the unified embedding service singleton."""
    config_manager = get_config_manager()
    return UnifiedEmbeddingService(config_manager)

@lru_cache()
def get_query_service() -> UnifiedQueryService:
    """Get the unified query service singleton."""
    config_manager = get_config_manager()
    embedding_service = get_embedding_service()
    return UnifiedQueryService(config_manager, embedding_service)

@lru_cache()
def get_document_processor() -> UnifiedDocumentProcessor:
    """Get the unified document processor singleton."""
    config_manager = get_config_manager()
    embedding_service = get_embedding_service()
    return UnifiedDocumentProcessor(config_manager, embedding_service)

@lru_cache()
def get_rag_engine() -> UnifiedRAGEngine:
    """Get the unified RAG engine singleton."""
    config_manager = get_config_manager()
    query_service = get_query_service()
    return UnifiedRAGEngine(config_manager, query_service)