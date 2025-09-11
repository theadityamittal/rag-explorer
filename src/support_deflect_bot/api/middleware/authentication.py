"""Authentication middleware for Support Deflect Bot API."""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime

logger = logging.getLogger(__name__)

class AuthenticationMiddleware:
    """Simple authentication middleware."""
    
    def __init__(self, app: FastAPI, require_auth: bool = False):
        self.app = app
        self.require_auth = require_auth
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # For now, authentication is optional
        # In production, implement proper authentication logic
        
        await self.app(scope, receive, send)

def add_authentication_middleware(app: FastAPI, require_auth: bool = False) -> None:
    """Add authentication middleware to FastAPI application."""
    app.add_middleware(AuthenticationMiddleware, require_auth=require_auth)