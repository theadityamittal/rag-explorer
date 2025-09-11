import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.core.llm_local import llm_echo
from src.core.rag import answer_question
from src.core.retrieve import retrieve
from src.data.ingest import ingest_folder
from src.data.web_ingest import crawl_urls, index_urls
from support_deflect_bot_old.utils.batch import batch_ask
from support_deflect_bot_old.utils.metrics import Meter
from support_deflect_bot_old.utils.settings import (
    APP_NAME,
    APP_VERSION,
    CRAWL_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_SAME_DOMAIN,
    DEFAULT_SEEDS,
    DOCS_FOLDER,
)

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Optional CORS for a simple web UI later
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# meters
ASK_METER = Meter()
SEARCH_METER = Meter()


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=1000, description="Question to ask"
    )
    domains: Optional[List[str]] = Field(
        None, max_items=10, description="Optional domains to filter"
    )

    @field_validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")

    @field_validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class BatchAskRequest(BaseModel):
    questions: List[str] = Field(
        ..., min_items=1, max_items=10, description="List of questions"
    )

    @field_validator("questions")
    def validate_questions(cls, v):
        if not v:
            raise ValueError("Questions list cannot be empty")
        validated = []
        for q in v:
            if not q.strip():
                raise ValueError("Questions cannot be empty")
            if len(q) > 1000:
                raise ValueError("Question too long (max 1000 characters)")
            validated.append(q.strip())
        return validated


class CrawlRequest(BaseModel):
    urls: List[str] = Field(..., min_items=1, max_items=50, description="URLs to crawl")
    force: bool = Field(
        False, description="Force re-index even if content hasn't changed"
    )

    @field_validator("urls")
    def validate_urls(cls, v):
        validated = []
        for url in v:
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError("URLs must start with http:// or https://")
            validated.append(url)
        return validated


class CrawlDepthRequest(BaseModel):
    seeds: List[str] = Field(..., min_items=1, max_items=10, description="Seed URLs")
    depth: int = Field(1, ge=1, le=3, description="Crawl depth")
    max_pages: int = Field(30, ge=1, le=100, description="Maximum pages to crawl")
    same_domain: bool = Field(True, description="Restrict to same domain")
    force: bool = Field(
        False, description="Force re-index even if content hasn't changed"
    )

    @field_validator("seeds")
    def validate_seeds(cls, v):
        validated = []
        for url in v:
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError("Seed URLs must start with http:// or https://")
            validated.append(url)
        return validated


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/reindex")
def reindex():
    try:
        n = ingest_folder(DOCS_FOLDER)
        return {"chunks_indexed": n}
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Documentation folder not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Indexing failed")


@app.post("/search")
def search(req: SearchRequest):
    t0 = time.perf_counter()
    try:
        hits = retrieve(req.query, k=req.k)
        out = {
            "query": req.query,
            "results": [
                {
                    "text": h["text"][:400],
                    "path": h["meta"].get("path"),
                    "chunk_id": h["meta"].get("chunk_id"),
                    "distance": h.get("distance"),
                }
                for h in hits
            ],
        }
        return out
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        SEARCH_METER.observe(time.perf_counter() - t0)


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    try:
        result = answer_question(req.question, domains=req.domains)
        return result
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database or LLM connection failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        ASK_METER.observe(time.perf_counter() - t0)


@app.get("/metrics")
def metrics():
    return {
        "ask": ASK_METER.summary(),
        "search": SEARCH_METER.summary(),
        "version": app.version,
    }


@app.get("/llm_ping")
def llm_ping():
    try:
        text = llm_echo("Wake up!")
        return {"model": "ollama", "reply": text}
    except Exception:
        raise HTTPException(status_code=503, detail="LLM service unavailable")


@app.post("/batch_ask")
def batch_ask_endpoint(req: BatchAskRequest):
    try:
        return {"results": batch_ask(req.questions)}
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database or LLM connection failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Batch processing failed")


@app.post("/crawl")
def crawl(req: CrawlRequest):
    try:
        return {"result": index_urls(req.urls, force=req.force)}
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Crawl operation failed")


@app.post("/crawl_depth")
def crawl_depth(req: CrawlDepthRequest):
    try:
        result = crawl_urls(
            seeds=req.seeds,
            depth=req.depth,
            max_pages=req.max_pages,
            same_domain=req.same_domain,
            force=req.force,
        )
        return {"result": result}
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Depth crawl operation failed")


@app.post("/crawl_default")
def crawl_default():
    try:
        result = crawl_urls(
            seeds=DEFAULT_SEEDS,
            depth=CRAWL_DEPTH,
            max_pages=CRAWL_MAX_PAGES,
            same_domain=CRAWL_SAME_DOMAIN,
        )
        return {"result": result}
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Default crawl operation failed")
