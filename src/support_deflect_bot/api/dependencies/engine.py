"""Engine dependencies for FastAPI application."""

from fastapi import HTTPException, status
from ...engine import UnifiedRAGEngine, UnifiedQueryService, UnifiedDocumentProcessor, UnifiedEmbeddingService

# Global engine instances (will be initialized by main app)
_rag_engine: UnifiedRAGEngine = None
_query_service: UnifiedQueryService = None  
_document_processor: UnifiedDocumentProcessor = None
_embedding_service: UnifiedEmbeddingService = None

def get_rag_engine() -> UnifiedRAGEngine:
    """Dependency to get the RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        # Initialize on first use (lazy initialization)
        try:
            _rag_engine = UnifiedRAGEngine()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"RAG engine not available: {str(e)}"
            )
    return _rag_engine

def get_query_service() -> UnifiedQueryService:
    """Dependency to get the query service instance."""
    global _query_service
    if _query_service is None:
        try:
            _query_service = UnifiedQueryService()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Query service not available: {str(e)}"
            )
    return _query_service

def get_document_processor() -> UnifiedDocumentProcessor:
    """Dependency to get the document processor instance."""
    global _document_processor
    if _document_processor is None:
        try:
            _document_processor = UnifiedDocumentProcessor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Document processor not available: {str(e)}"
            )
    return _document_processor

def get_embedding_service() -> UnifiedEmbeddingService:
    """Dependency to get the embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        try:
            _embedding_service = UnifiedEmbeddingService()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Embedding service not available: {str(e)}"
            )
    return _embedding_service

def set_rag_engine(engine: UnifiedRAGEngine):
    """Set the global RAG engine instance."""
    global _rag_engine
    _rag_engine = engine

def set_query_service(service: UnifiedQueryService):
    """Set the global query service instance."""
    global _query_service
    _query_service = service

def set_document_processor(processor: UnifiedDocumentProcessor):
    """Set the global document processor instance."""
    global _document_processor
    _document_processor = processor

def set_embedding_service(service: UnifiedEmbeddingService):
    """Set the global embedding service instance."""
    global _embedding_service
    _embedding_service = service

def cleanup_engines():
    """Cleanup engine instances."""
    global _rag_engine, _query_service, _document_processor, _embedding_service
    _rag_engine = None
    _query_service = None
    _document_processor = None
    _embedding_service = None