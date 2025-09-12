"""Query endpoints for Support Deflect Bot API."""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..models.requests import AskRequest, SearchRequest, BatchAskRequest
from ..models.responses import AskResponse, SearchResponse, BatchAskResponse, Source, SearchResult
from ..dependencies.engine import get_rag_engine, get_query_service
from ...engine import UnifiedRAGEngine, UnifiedQueryService

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    engine: UnifiedRAGEngine = Depends(get_rag_engine)
) -> AskResponse:
    """Ask a question using the RAG pipeline."""
    try:
        start_time = time.time()
        
        # Call the synchronous engine method
        result = engine.answer_question(
            question=request.question,
            domains=request.domains,
            k=request.max_chunks,
            min_confidence=request.min_confidence
        )
        
        response_time = time.time() - start_time
        
        # Convert sources to response model format
        sources = []
        for citation in result.get("citations", []):
            sources.append(Source(
                id=citation.get("id"),
                content=citation.get("text", ""),
                metadata=citation.get("metadata", {}),
                distance=citation.get("distance", 0.0)
            ))
        
        return AskResponse(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            sources=sources,
            chunks_used=len(result.get("citations", [])),
            response_time=response_time,
            provider_used=result.get("provider_used", "unknown"),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question processing failed: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    query_service: UnifiedQueryService = Depends(get_query_service)
) -> SearchResponse:
    """Search documents using vector similarity."""
    try:
        start_time = time.time()
        
        # Call the synchronous query service
        result = query_service.search_similar_chunks(
            query=request.query,
            k=request.k,
            domain_filter=request.domains
        )
        
        response_time = time.time() - start_time
        
        # Convert chunks to search results
        search_results = []
        for chunk in result:
            search_results.append(SearchResult(
                id=chunk.get("id"),
                content=chunk.get("text", ""),
                metadata=chunk.get("metadata", {}),
                distance=chunk.get("distance", 0.0),
                score=max(0.0, 1.0 - chunk.get("distance", 1.0))
            ))
        
        return SearchResponse(
            results=search_results,
            total_count=len(search_results),
            query=request.query,
            response_time=response_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/batch_ask", response_model=BatchAskResponse)
async def batch_ask_questions(
    request: BatchAskRequest,
    engine: UnifiedRAGEngine = Depends(get_rag_engine)
) -> BatchAskResponse:
    """Process multiple questions in batch."""
    try:
        start_time = time.time()
        
        # Process questions individually
        results = []
        successful_count = 0
        
        for question in request.questions:
            try:
                question_start = time.time()
                result = engine.answer_question(
                    question=question,
                    domains=request.domains,
                    k=request.max_chunks,
                    min_confidence=request.min_confidence
                )
                question_time = time.time() - question_start
                
                # Convert sources to response model format
                sources = []
                for citation in result.get("citations", []):
                    sources.append(Source(
                        id=citation.get("id"),
                        content=citation.get("text", ""),
                        metadata=citation.get("metadata", {}),
                        distance=citation.get("distance", 0.0)
                    ))
                
                ask_response = AskResponse(
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0.0),
                    sources=sources,
                    chunks_used=len(result.get("citations", [])),
                    response_time=question_time,
                    provider_used=result.get("provider_used", "unknown"),
                    metadata=result.get("metadata", {})
                )
                
                results.append(ask_response)
                successful_count += 1
                
            except Exception as e:
                # Create error response for failed question
                error_response = AskResponse(
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    sources=[],
                    chunks_used=0,
                    response_time=0.0,
                    provider_used="error",
                    metadata={"error": str(e)}
                )
                results.append(error_response)
        
        total_time = time.time() - start_time
        
        return BatchAskResponse(
            results=results,
            total_questions=len(request.questions),
            successful_answers=successful_count,
            processing_time=total_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )