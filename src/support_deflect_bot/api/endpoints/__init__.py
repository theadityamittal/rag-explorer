"""API endpoints for Support Deflect Bot."""

from .query import router as query_router
from .health import router as health_router
from .indexing import router as indexing_router
from .admin import router as admin_router
from .batch import router as batch_router

# Export all routers
routers = [
    query_router,
    health_router,
    indexing_router,
    admin_router,
    batch_router,
]

__all__ = [
    "query_router",
    "health_router", 
    "indexing_router",
    "admin_router",
    "batch_router",
    "routers",
]