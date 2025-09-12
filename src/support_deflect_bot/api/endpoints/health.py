"""Health check endpoints for Support Deflect Bot API."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from ..models.responses import HealthResponse
from ..dependencies.engine import get_rag_engine, get_document_processor, get_query_service, get_embedding_service
from ...engine import UnifiedRAGEngine, UnifiedDocumentProcessor, UnifiedQueryService, UnifiedEmbeddingService
from ...utils.settings import APP_VERSION

router = APIRouter(prefix="/api/v1", tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Comprehensive health check for all system components."""
    try:
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Initialize status data
        providers_status = {}
        database_status = {}
        overall_status = "healthy"
        
        try:
            # Check RAG engine
            rag_engine = get_rag_engine()
            rag_status = rag_engine.get_system_status()
            
            # Check document processor 
            doc_processor = get_document_processor()
            doc_status = doc_processor.get_status()
            
            # Check query service
            query_service = get_query_service()
            query_status = query_service.get_status()
            
            # Check embedding service
            embedding_service = get_embedding_service()
            embedding_status = embedding_service.get_provider_status()
            
            # Combine provider status
            providers_status = {
                "rag_engine": {
                    "status": "healthy" if rag_status.get("overall_health") == "ok" else "unhealthy",
                    "details": rag_status
                },
                "query_service": {
                    "status": "healthy" if query_status.get("connected", False) else "unhealthy", 
                    "details": query_status
                },
                "embedding_service": {
                    "status": "healthy" if embedding_status else "unhealthy",
                    "available_providers": list(embedding_status.keys()) if embedding_status else []
                }
            }
            
            # Database status
            database_status = {
                "connected": doc_status.get("connected", False),
                "total_chunks": doc_status.get("total_chunks", 0),
                "collections": doc_status.get("collections", [])
            }
            
            # Determine overall status
            if not doc_status.get("connected", False):
                overall_status = "degraded"
            elif not any(p["status"] == "healthy" for p in providers_status.values()):
                overall_status = "unhealthy"
                
        except Exception as e:
            overall_status = "unhealthy"
            providers_status = {"error": str(e)}
            database_status = {"error": "Unable to check database status"}
        
        return HealthResponse(
            status=overall_status,
            timestamp=timestamp,
            version=APP_VERSION,
            providers=providers_status,
            database=database_status
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/ping")
async def ping() -> Dict[str, str]:
    """Simple ping endpoint for basic availability check."""
    return {
        "status": "ok", 
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "Support Deflect Bot API"
    }

@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """Readiness probe for container orchestration."""
    try:
        # Check if core services are ready
        rag_engine = get_rag_engine()
        doc_processor = get_document_processor() 
        
        # Basic readiness checks
        doc_status = doc_processor.get_status()
        
        ready = doc_status.get("connected", False)
        
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {
                "database": doc_status.get("connected", False),
                "engine": True  # Engine creation succeeded if we get here
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )

@router.get("/liveness")  
async def liveness_check() -> Dict[str, str]:
    """Liveness probe for container orchestration."""
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }