"""Indexing endpoints for Support Deflect Bot API."""

import time
import os
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..models.requests import IndexRequest, CrawlRequest
from ..models.responses import IndexResponse, CrawlResponse, ProcessingDetail
from ..dependencies.engine import get_document_processor
from ..dependencies.validation import validate_crawl_urls, validate_file_patterns
from ...engine import UnifiedDocumentProcessor

router = APIRouter(prefix="/api/v1", tags=["indexing"])

@router.post("/index", response_model=IndexResponse)
async def index_directory(
    request: IndexRequest,
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> IndexResponse:
    """Index local documents from a directory."""
    try:
        start_time = time.time()
        
        # Use default directory if none provided
        directory = request.directory or "./docs"
        
        # Validate directory exists
        if not os.path.exists(directory):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory not found: {directory}"
            )
        
        if not os.path.isdir(directory):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a directory: {directory}"
            )
        
        # Validate file patterns if provided
        patterns = validate_file_patterns(request.file_patterns)
        
        # Process directory
        result = processor.process_directory(
            directory_path=directory,
            recursive=request.recursive,
            force_reprocess=request.force,
            file_patterns=patterns
        )
        
        processing_time = time.time() - start_time
        
        # Parse result and create processing details
        details = []
        error_messages = []
        processed_count = 0
        failed_count = 0
        
        # Result structure varies, handle different formats
        if isinstance(result, dict):
            if "processed_files" in result:
                processed_count = len(result.get("processed_files", []))
                for file_path in result.get("processed_files", []):
                    details.append(ProcessingDetail(
                        status="success",
                        path=file_path,
                        url=None,
                        error=None
                    ))
            
            if "errors" in result:
                failed_count = len(result.get("errors", []))
                for error in result.get("errors", []):
                    error_messages.append(str(error))
                    details.append(ProcessingDetail(
                        status="failed",
                        path=error.get("path") if isinstance(error, dict) else None,
                        url=None,
                        error=str(error)
                    ))
        else:
            # Handle simple response formats
            processed_count = 1 if result else 0
        
        success = failed_count == 0 or processed_count > 0
        
        return IndexResponse(
            success=success,
            processed_count=processed_count,
            failed_count=failed_count,
            processing_time=processing_time,
            directory=directory,
            details=details,
            error_messages=error_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@router.post("/crawl", response_model=CrawlResponse) 
async def crawl_urls(
    request: CrawlRequest,
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> CrawlResponse:
    """Crawl and index web pages."""
    try:
        start_time = time.time()
        
        # Validate URLs
        validated_urls = validate_crawl_urls(request.urls)
        
        # Process URLs through document processor
        all_results = {}
        details = []
        error_messages = []
        processed_count = 0
        failed_count = 0
        
        for url in validated_urls:
            try:
                # Process single URL
                result = processor.process_web_content(
                    url=url,
                    max_depth=request.depth,
                    max_pages=request.max_pages,
                    same_domain_only=request.same_domain
                )
                
                if result:
                    processed_count += 1
                    details.append(ProcessingDetail(
                        status="success",
                        path=None,
                        url=url,
                        error=None
                    ))
                    all_results[url] = result
                else:
                    failed_count += 1
                    details.append(ProcessingDetail(
                        status="failed",
                        path=None,
                        url=url,
                        error="Processing returned no result"
                    ))
                    
            except Exception as url_error:
                failed_count += 1
                error_msg = str(url_error)
                error_messages.append(f"{url}: {error_msg}")
                details.append(ProcessingDetail(
                    status="failed",
                    path=None,
                    url=url,
                    error=error_msg
                ))
        
        processing_time = time.time() - start_time
        success = processed_count > 0
        
        return CrawlResponse(
            success=success,
            processed_count=processed_count,
            failed_count=failed_count,
            processing_time=processing_time,
            urls=validated_urls,
            crawl_details={
                "depth": request.depth,
                "max_pages": request.max_pages,
                "same_domain": request.same_domain
            },
            details=details,
            error_messages=error_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crawling failed: {str(e)}"
        )

@router.delete("/index")
async def clear_index(
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> dict:
    """Clear the document index."""
    try:
        # Clear the document store
        result = processor.clear_database()
        
        return {
            "success": True,
            "message": "Index cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear index: {str(e)}"
        )

@router.get("/index/stats")
async def get_index_stats(
    processor: UnifiedDocumentProcessor = Depends(get_document_processor)
) -> dict:
    """Get indexing statistics."""
    try:
        status = processor.get_status()
        
        return {
            "connected": status.get("connected", False),
            "total_chunks": status.get("total_chunks", 0),
            "collections": status.get("collections", []),
            "database_path": status.get("database_path", "unknown"),
            "last_updated": status.get("last_updated")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index stats: {str(e)}"
        )