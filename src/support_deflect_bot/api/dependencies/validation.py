"""Validation dependencies for FastAPI application."""

from typing import List, Optional
from fastapi import HTTPException, status, Query, Header
from ...utils.settings import ALLOW_HOSTS, TRUSTED_DOMAINS

def validate_domain_filter(domains: Optional[List[str]] = Query(None)) -> Optional[List[str]]:
    """Validate domain filtering parameters."""
    if not domains:
        return None
    
    # Basic validation for domain format
    for domain in domains:
        if not domain or not isinstance(domain, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid domain format: {domain}"
            )
        
        # Basic domain name validation
        if not domain.replace("-", "").replace(".", "").replace("_", "").isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid domain name: {domain}"
            )
    
    return domains

def validate_user_agent(user_agent: str = Header(None)) -> str:
    """Validate and provide default user agent."""
    if not user_agent:
        return "SupportDeflectBot/2.0 API"
    
    # Basic user agent validation
    if len(user_agent) > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User agent too long"
        )
    
    return user_agent

def validate_crawl_urls(urls: List[str]) -> List[str]:
    """Validate URLs for crawling against allowed hosts."""
    import re
    from urllib.parse import urlparse
    
    validated_urls = []
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    for url in urls:
        # Basic URL format validation
        if not url_pattern.match(url):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid URL format: {url}"
            )
        
        # Check against allowed hosts if configured
        if ALLOW_HOSTS:
            parsed = urlparse(url)
            hostname = parsed.hostname
            
            if hostname and hostname not in ALLOW_HOSTS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"URL not allowed: {hostname} not in allowed hosts"
                )
        
        validated_urls.append(url)
    
    return validated_urls

def validate_file_patterns(patterns: Optional[List[str]]) -> Optional[List[str]]:
    """Validate file patterns for indexing."""
    if not patterns:
        return None
    
    import re
    
    for pattern in patterns:
        try:
            # Test if pattern can be compiled as regex
            re.compile(pattern)
        except re.error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file pattern: {pattern}"
            )
    
    return patterns

def validate_pagination(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return")
) -> tuple[int, int]:
    """Validate pagination parameters."""
    return skip, limit