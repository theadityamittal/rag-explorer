"""Main FastAPI application for Support Deflect Bot API."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from ..utils.settings import APP_NAME, APP_VERSION
from ..engine import UnifiedRAGEngine, UnifiedDocumentProcessor, UnifiedQueryService, UnifiedEmbeddingService
from .endpoints import routers
from .middleware.cors import add_cors_middleware
from .middleware.error_handling import add_error_handlers
from .middleware.logging import add_logging_middleware
from .dependencies.engine import set_rag_engine, set_document_processor, set_query_service, set_embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instances
_rag_engine: UnifiedRAGEngine = None
_document_processor: UnifiedDocumentProcessor = None
_query_service: UnifiedQueryService = None
_embedding_service: UnifiedEmbeddingService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _rag_engine, _document_processor, _query_service, _embedding_service
    
    # Startup
    logger.info("Initializing Support Deflect Bot API...")
    try:
        # Initialize all engine services
        _rag_engine = UnifiedRAGEngine()
        _document_processor = UnifiedDocumentProcessor()
        _query_service = UnifiedQueryService()
        _embedding_service = UnifiedEmbeddingService()
        
        # Set global instances for dependency injection
        set_rag_engine(_rag_engine)
        set_document_processor(_document_processor)
        set_query_service(_query_service)
        set_embedding_service(_embedding_service)
        
        logger.info("Engine initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Support Deflect Bot API...")
    # Add any cleanup logic here if needed

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Support Deflect Bot API - RAG-powered question answering and document indexing using unified engine services",
    docs_url="/docs",
    redoc_url="/redoc", 
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
add_cors_middleware(app)

# Add trusted host middleware  
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add custom middleware
add_error_handlers(app)
add_logging_middleware(app)

# Include all routers
for router in routers:
    app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {APP_NAME} API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "description": "RAG-powered question answering and document indexing API"
    }

@app.get("/version")
async def get_version():
    """Get API version information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "api_version": "v1",
        "timestamp": time.time()
    }

# Health check endpoint at root level (in addition to /api/v1/health)
@app.get("/health")
async def quick_health():
    """Quick health check endpoint."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "version": APP_VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")