"""Batch processing endpoints for Support Deflect Bot API."""

from fastapi import APIRouter

# Note: The main batch_ask endpoint is implemented in query.py
# This file is reserved for additional batch operations if needed

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

# Future batch operations can be added here
# For now, batch_ask is available at /api/v1/batch_ask via query.py