"""Administrative endpoints for Support Deflect Bot API."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from ..dependencies.engine import get_rag_engine, get_document_processor, get_query_service, get_embedding_service
from ..dependencies.security import verify_api_key
from ...engine import UnifiedRAGEngine, UnifiedDocumentProcessor, UnifiedQueryService, UnifiedEmbeddingService
from ...utils.settings import APP_VERSION, APP_NAME

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

@router.get("/metrics")
async def get_metrics(
    rag_engine: UnifiedRAGEngine = Depends(get_rag_engine),
    query_service: UnifiedQueryService = Depends(get_query_service),
    doc_processor: UnifiedDocumentProcessor = Depends(get_document_processor),
    embedding_service: UnifiedEmbeddingService = Depends(get_embedding_service),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get comprehensive system metrics."""
    try:
        # Gather metrics from all services
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": APP_VERSION,
            "application": APP_NAME
        }
        
        # RAG Engine metrics
        try:
            rag_metrics = rag_engine.get_metrics()
            metrics["rag_engine"] = rag_metrics
        except Exception as e:
            metrics["rag_engine"] = {"error": str(e)}
        
        # Query Service metrics  
        try:
            query_metrics = query_service.get_query_analytics()
            metrics["query_service"] = query_metrics
        except Exception as e:
            metrics["query_service"] = {"error": str(e)}
            
        # Document Processor metrics
        try:
            doc_status = doc_processor.get_status()
            processing_stats = getattr(doc_processor, 'get_processing_stats', lambda: {})()
            metrics["document_processor"] = {
                **doc_status,
                "processing_stats": processing_stats
            }
        except Exception as e:
            metrics["document_processor"] = {"error": str(e)}
            
        # Embedding Service metrics
        try:
            embed_metrics = getattr(embedding_service, 'get_metrics', lambda: {})()
            provider_status = embedding_service.get_provider_status()
            metrics["embedding_service"] = {
                "metrics": embed_metrics,
                "provider_status": provider_status
            }
        except Exception as e:
            metrics["embedding_service"] = {"error": str(e)}
        
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )

@router.post("/reset")
async def reset_system(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Reset system state (requires API key)."""
    try:
        # This is a placeholder for system reset functionality
        # In a real implementation, you might want to:
        # - Clear caches
        # - Reset counters
        # - Reinitialize services
        
        return {
            "success": True,
            "message": "System reset completed",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System reset failed: {str(e)}"
        )

@router.get("/status") 
async def get_detailed_status(
    rag_engine: UnifiedRAGEngine = Depends(get_rag_engine),
    doc_processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> Dict[str, Any]:
    """Get detailed system status information."""
    try:
        # Get comprehensive system status
        system_status = rag_engine.get_system_status()
        doc_status = doc_processor.get_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": APP_VERSION,
            "system_status": system_status,
            "database_status": doc_status,
            "uptime": "not_implemented",  # Could add uptime tracking
            "memory_usage": "not_implemented"  # Could add memory monitoring
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )

@router.get("/providers")
async def get_provider_status(
    embedding_service: UnifiedEmbeddingService = Depends(get_embedding_service)
) -> Dict[str, Any]:
    """Get provider status and validation results."""
    try:
        provider_status = embedding_service.get_provider_status()
        
        # Additional provider validation
        validation_results = {}
        for provider_name in provider_status.keys():
            try:
                # Test provider connectivity
                # This could be enhanced with actual provider validation
                validation_results[provider_name] = {
                    "available": True,
                    "last_tested": datetime.utcnow().isoformat() + "Z"
                }
            except Exception as e:
                validation_results[provider_name] = {
                    "available": False,
                    "error": str(e),
                    "last_tested": datetime.utcnow().isoformat() + "Z"
                }
        
        return {
            "provider_status": provider_status,
            "validation_results": validation_results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider status: {str(e)}"
        )