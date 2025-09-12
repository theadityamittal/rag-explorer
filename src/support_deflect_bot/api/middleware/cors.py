"""CORS middleware configuration for Support Deflect Bot API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware to FastAPI application."""
    
    # Configure CORS settings
    origins = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # FastAPI dev server  
        "http://localhost:8080",  # Alternative dev port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language", 
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Request-ID",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
        ]
    )

def configure_development_cors(app: FastAPI) -> None:
    """Configure permissive CORS for development environment."""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=False,  # Disable credentials with wildcard origin
        allow_methods=["*"],
        allow_headers=["*"],
    )

def configure_production_cors(app: FastAPI, allowed_origins: list[str]) -> None:
    """Configure strict CORS for production environment."""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type", 
            "Authorization",
            "X-API-Key",
            "X-Request-ID",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
        ],
        max_age=86400,  # 24 hours
    )