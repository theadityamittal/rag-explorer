"""Main FastAPI application for Support Deflect Bot API."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from ..utils.settings import APP_NAME, APP_VERSION
from ..engine import UnifiedRAGEngine, UnifiedDocumentProcessor
from .endpoints import query, indexing, health, admin, batch
# from .middleware.error_handling import add_error_handlers
# from .middleware.logging import add_logging_middleware
from .dependencies.engine import get_rag_engine, get_document_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instances
_rag_engine: UnifiedRAGEngine = None
_document_processor: UnifiedDocumentProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _rag_engine, _document_processor
    
    # Startup
    logger.info("Initializing Support Deflect Bot API...")
    try:
        _rag_engine = UnifiedRAGEngine()
        _document_processor = UnifiedDocumentProcessor()
        logger.info("Engine initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Support Deflect Bot API...")
    if _rag_engine:
        # Add any cleanup logic here
        pass

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Support Deflect Bot API - RAG-powered question answering and document indexing",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add custom middleware
# add_error_handlers(app)
# add_logging_middleware(app)

# Include routers
app.include_router(query.router)
app.include_router(indexing.router)
app.include_router(health.router)
app.include_router(admin.router)
app.include_router(batch.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {APP_NAME} API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": APP_VERSION
    }

# Dependency injection functions for global instances
def get_global_rag_engine() -> UnifiedRAGEngine:
    """Get global RAG engine instance."""
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return _rag_engine

def get_global_document_processor() -> UnifiedDocumentProcessor:
    """Get global document processor instance."""
    if _document_processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    return _document_processor

# Override dependency functions
app.dependency_overrides[get_rag_engine] = get_global_rag_engine
app.dependency_overrides[get_document_processor] = get_global_document_processor

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)