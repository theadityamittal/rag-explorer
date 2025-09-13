"""Health and monitoring endpoints for Support Deflect Bot API."""

from fastapi import APIRouter, Depends
from datetime import datetime
import asyncio

from ..models.responses import HealthResponse
from ..dependencies.engine import get_rag_engine, get_query_service, get_config_manager
from ...engine.rag_engine import UnifiedRAGEngine
from ...engine.query_service import UnifiedQueryService
from ...config.manager import ConfigurationManager
from ...utils.settings import APP_NAME, APP_VERSION

router = APIRouter(tags=["health"])

@router.get("/healthz", response_model=HealthResponse)
async def health_check(
    config_manager: ConfigurationManager = Depends(get_config_manager),
    rag_engine: UnifiedRAGEngine = Depends(get_rag_engine),
    query_service: UnifiedQueryService = Depends(get_query_service)
) -> HealthResponse:
    """Comprehensive health check endpoint."""
    try:
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Check provider status
        providers = {}
        try:
            # Test primary provider availability
            settings = config_manager.get_settings()
            primary_provider = settings.primary_provider
            
            # Simple ping test to the provider
            test_result = await rag_engine.answer_question(
                question="ping",
                max_chunks=1,
                use_context=False
            )
            
            providers[primary_provider] = {
                "status": "healthy",
                "response_time": test_result.get("response_time", 0.0),
                "last_check": timestamp
            }
            
        except Exception as e:
            providers["primary"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": timestamp
            }
        
        # Check database/vector store status
        database = {}
        try:
            # Test vector store connectivity
            test_search = await query_service.retrieve_documents(
                query="test",
                k=1
            )
            
            database = {
                "status": "healthy",
                "type": "chromadb",
                "documents_indexed": test_search.get("total_chunks", 0),
                "last_check": timestamp
            }
            
        except Exception as e:
            database = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": timestamp
            }
        
        # Determine overall status
        overall_status = "healthy"
        if any(p.get("status") == "unhealthy" for p in providers.values()):
            overall_status = "degraded"
        if database.get("status") == "unhealthy":
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=timestamp,
            version=APP_VERSION,
            providers=providers,
            database=database
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            version=APP_VERSION,
            providers={},
            database={"status": "unhealthy", "error": str(e)}
        )

@router.get("/health")
async def simple_health_check():
    """Simple health check for load balancers."""
    return {"status": "ok"}

@router.get("/ready")
async def readiness_check(
    config_manager: ConfigurationManager = Depends(get_config_manager)
):
    """Readiness check for Kubernetes."""
    try:
        # Basic configuration check
        settings = config_manager.get_settings()
        if not settings.primary_provider:
            return {"status": "not ready", "reason": "no provider configured"}, 503
        
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not ready", "reason": str(e)}, 503

@router.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint (placeholder for monitoring integration)."""
    return {
        "requests_total": 0,  # Would track actual metrics
        "requests_duration_seconds": 0.0,
        "active_connections": 0,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/llm_ping")
async def llm_ping(
    rag_engine: UnifiedRAGEngine = Depends(get_rag_engine)
):
    """Test LLM provider connectivity."""
    try:
        start_time = asyncio.get_event_loop().time()
        
        result = await rag_engine.answer_question(
            question="Say 'pong'",
            max_chunks=0,  # No context needed
            use_context=False
        )
        
        response_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "status": "ok",
            "response": result.get("answer", "").strip(),
            "provider": result.get("provider_used", "unknown"),
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, 500