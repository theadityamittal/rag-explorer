"""Shared engine package for Support Deflect Bot."""

from .rag_engine import UnifiedRAGEngine
from .document_processor import UnifiedDocumentProcessor
from .query_service import UnifiedQueryService
from .embedding_service import UnifiedEmbeddingService

__all__ = [
    "UnifiedRAGEngine",
    "UnifiedDocumentProcessor", 
    "UnifiedQueryService",
    "UnifiedEmbeddingService"
]

__version__ = "1.0.0"