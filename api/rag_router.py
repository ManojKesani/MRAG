"""
api/rag_router.py
=================
FastAPI router for agentic RAG queries.

Endpoints
---------
POST /rag/query          Submit a question (async background job)
GET  /rag/query/{job_id} Poll job status + partial results
POST /rag/query/sync     Synchronous query (waits for result, ≤60s)
GET  /rag/strategies     List available strategy names
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from rag.config import RAGConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

# In-memory job store (swap for Redis in production)
_rag_jobs: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    config: Optional[RAGConfig] = None


class QueryResponse(BaseModel):
    job_id: str
    status: str
    message: str


class QueryResult(BaseModel):
    job_id: str
    status: str                           # running | success | failed | max_iterations | no_store
    query: str
    final_answer: Optional[str] = None
    answer_sources: List[str] = []
    answer_confidence: Optional[float] = None
    iterations_used: Optional[int] = None
    notepad: Optional[List[Dict]] = None  # iteration log
    probe_diagnostics: Optional[str] = None   # store probe summary
    store_ready: Optional[bool] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────
async def _run_rag_job(job_id: str, query: str, config: Dict[str, Any]) -> None:
    _rag_jobs[job_id]["status"] = "running"
    try:
        from rag.graph import run_rag
        result = await run_rag(query=query, config=config)
        _rag_jobs[job_id].update(
            {
                "status": result.get("status", "success"),
                "final_answer": result.get("final_answer"),
                "answer_sources": result.get("answer_sources", []),
                "answer_confidence": result.get("answer_confidence"),
                "iterations_used": result.get("iteration"),
                "notepad": result.get("notepad"),
                "probe_diagnostics": result.get("probe_diagnostics"),
                "store_ready": result.get("store_ready"),
                "error": result.get("error"),
            }
        )
    except Exception as exc:
        logger.error(f"RAG job {job_id} failed: {exc}", exc_info=True)
        _rag_jobs[job_id].update({"status": "failed", "error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/strategies")
async def list_strategies():
    from rag.strategies import ALL_STRATEGIES
    return {"strategies": ALL_STRATEGIES}


@router.post("/query", response_model=QueryResponse, status_code=202)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a RAG query as a background job."""
    config = (request.config or RAGConfig()).model_dump()
    job_id = str(uuid.uuid4())
    _rag_jobs[job_id] = {
        "status": "queued",
        "query": request.query,
        "final_answer": None,
        "answer_sources": [],
        "answer_confidence": None,
        "iterations_used": None,
        "notepad": None,
        "error": None,
    }
    background_tasks.add_task(_run_rag_job, job_id, request.query, config)
    return QueryResponse(
        job_id=job_id,
        status="queued",
        message=f"RAG job queued for query: {request.query[:60]}…",
    )


@router.get("/query/{job_id}", response_model=QueryResult)
async def get_query_result(job_id: str):
    """Poll a RAG job."""
    job = _rag_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return QueryResult(job_id=job_id, query=job.get("query", ""), **{
        k: job.get(k) for k in (
            "status", "final_answer", "answer_sources",
            "answer_confidence", "iterations_used", "notepad",
            "probe_diagnostics", "store_ready", "error"
        )
    })


@router.post("/query/sync", response_model=QueryResult)
async def sync_query(request: QueryRequest):
    """
    Synchronous RAG query — waits up to 120s for the result.
    Suitable for direct API calls and testing.
    """
    config = (request.config or RAGConfig()).model_dump()
    job_id = str(uuid.uuid4())
    _rag_jobs[job_id] = {"status": "running", "query": request.query}

    try:
        from rag.graph import run_rag
        result = await asyncio.wait_for(
            run_rag(query=request.query, config=config),
            timeout=120,
        )
        return QueryResult(
            job_id=job_id,
            query=request.query,
            status=result.get("status", "success"),
            final_answer=result.get("final_answer"),
            answer_sources=result.get("answer_sources", []),
            answer_confidence=result.get("answer_confidence"),
            iterations_used=result.get("iteration"),
            notepad=result.get("notepad"),
            error=result.get("error"),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="RAG query timed out (120s)")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))