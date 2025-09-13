"""Query endpoints for Support Deflect Bot API."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ..models.requests import AskRequest, SearchRequest, BatchAskRequest
from ..models.responses import AskResponse, SearchResponse, BatchAskResponse, SearchResult, Source
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
        result = engine.answer_question(
            question=request.question,
            k=request.max_chunks,
            domains=request.domains,
            min_confidence=request.min_confidence,
        )

        # Convert sources to response model format
        sources = [
            Source(
                id=f"{c.get('path')}#{c.get('chunk_id')}",
                content=c.get("preview", ""),
                metadata={"path": c.get("path"), "chunk_id": c.get("chunk_id")},
                distance=0.0,
            ) for c in result.get("citations", [])
        ]

        return AskResponse(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            sources=sources,
            chunks_used=result.get("metadata", {}).get("chunks_found", 0),
            response_time=0.0,
            provider_used="auto",
            metadata=result.get("metadata", {}),
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
        processed = query_service.preprocess_query(request.query) 
        if not processed.get("valid", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query provided."
            )

        results = query_service.retrieve_documents(
            processed_query=processed,
            k=request.k,
            domains=request.domains
        )

        search_results = [
            SearchResult(
                id=f"{r.get('path')}#{r.get('chunk_id')}",
                content=r.get("preview", ""),
                metadata={"path": r.get("path"), "chunk_id": r.get("chunk_id")},
                distance=r.get("distance", 0.0),
                score=max(0.0, 1.0 - r.get("distance", 1.0))
            ) for r in results
        ]
        
        return SearchResponse(
            results=search_results,
            total_count=len(search_results),
            query=request.query,
            response_time=0.0
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
        import time
        start_time = time.time()
        
        # Process questions individually (could be optimized for parallel processing)
        results = []
        successful_count = 0
        
        for question in request.questions:
            try:
                result = engine.answer_question(
                    question=question,
                    k=request.max_chunks,
                    domains=request.domains,
                    min_confidence=request.min_confidence,
                )
                
                # Convert to AskResponse format
                sources = [
                    Source(
                        id=f"{c.get('path')}#{c.get('chunk_id')}",
                        content=c.get("preview", ""),
                        metadata={"path": c.get("path"), "chunk_id": c.get("chunk_id")},
                        distance=0.0,
                    ) for c in result.get("citations", [])
                ]
                
                results.append(AskResponse(
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0.0),
                    sources=sources,
                    chunks_used=result.get("metadata", {}).get("chunks_found", 0),
                    response_time=0.0,
                    provider_used="auto",
                    metadata=result.get("metadata", {}),
                ))
                # Increment successful count
                successful_count += 1
                
            except Exception as e:
                # Add error response for failed questions
                error_response = AskResponse(
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    sources=[],
                    chunks_used=0,
                    response_time=0.0,
                    provider_used="error",
                    metadata={"error": str(e), "question": question}
                )
                results.append(error_response)
        
        total_time = time.time() - start_time
        
        return BatchAskResponse(
            results=results,
            total_questions=len(request.questions),
            successful_answers=successful_count,
            total_processing_time=total_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )