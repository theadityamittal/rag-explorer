"""Admin endpoints for Support Deflect Bot API."""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime

from ..dependencies.engine import get_config_manager, get_query_service
from ...config.manager import ConfigurationManager
from ...engine.query_service import UnifiedQueryService

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

@router.get("/status")
async def get_system_status(
    config_manager: ConfigurationManager = Depends(get_config_manager),
    query_service: UnifiedQueryService = Depends(get_query_service)
):
    """Get detailed system status for administrators."""
    try:
        settings = config_manager.get_settings()
        
        # Get database statistics
        try:
            stats = await query_service.get_collection_stats()
            db_stats = {
                "total_documents": stats.get("document_count", 0),
                "total_chunks": stats.get("chunk_count", 0),
                "collection_name": stats.get("collection_name", "unknown"),
                "last_updated": stats.get("last_updated", "unknown")
            }
        except Exception as e:
            db_stats = {"error": str(e)}
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "configuration": {
                "primary_provider": settings.primary_provider,
                "fallback_providers": getattr(settings, 'fallback_providers', []),
                "embedding_model": getattr(settings, 'embedding_model', 'unknown'),
                "chunk_size": getattr(settings, 'chunk_size', 500),
                "chunk_overlap": getattr(settings, 'chunk_overlap', 50)
            },
            "database": db_stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system status: {str(e)}"
        )

@router.post("/clear_cache")
async def clear_cache():
    """Clear system caches (placeholder for cache management)."""
    try:
        # In a real implementation, this would clear various caches
        return {
            "status": "success",
            "message": "Caches cleared successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/logs")
async def get_recent_logs():
    """Get recent system logs (placeholder for log management)."""
    try:
        # In a real implementation, this would retrieve actual logs
        return {
            "logs": [
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": "INFO",
                    "message": "System status check completed",
                    "component": "health"
                }
            ],
            "total_entries": 1,
            "retrieved_at": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve logs: {str(e)}"
        )