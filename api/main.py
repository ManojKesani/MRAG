"""
api/main.py
===========
FastAPI backend for M-RAG with permanent image storage.
Images are saved in ./uploads/ and served at http://localhost:8000/uploads/...
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import PipelineConfig
from core.settings import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = FastAPI(
    title="M-RAG Ingestion API",
    version="1.0.0",
    description="Multimodal RAG document ingestion pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== PERMANENT STORAGE ======================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Mount RAG router
from api.rag_router import router as rag_router
app.include_router(rag_router)

# ====================== IN-MEMORY JOB STORE ======================
_jobs: Dict[str, Dict[str, Any]] = {}

# ====================== RESPONSE MODELS ======================
class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | success | failed
    document_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CollectionsResponse(BaseModel):
    collections: list[str]


# ====================== BACKGROUND WORKER ======================
async def _run_pipeline(job_id: str, file_path: str, config: Dict[str, Any]) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        from ingestion.graph import run_ingestion
        final_state = await run_ingestion(
            file_path=file_path,
            config=config,
            run_name=f"job_{job_id}",
        )
        _jobs[job_id].update({
            "status": final_state.get("status", "success"),
            "document_id": final_state.get("document_id"),
            "metrics": final_state.get("metrics"),
            "error": final_state.get("error"),
        })
    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
        _jobs[job_id].update({"status": "failed", "error": str(exc)})
    # NO os.remove() — files stay forever in ./uploads/


# ====================== ROUTES ======================
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config/defaults", response_model=PipelineConfig)
async def get_default_config():
    """Return the default PipelineConfig (seeded from env vars)."""
    settings = get_settings()
    return PipelineConfig.from_settings(settings)


@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all Qdrant collections."""
    try:
        from qdrant_client import QdrantClient
        settings = get_settings()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=False,   # matches your local setup
        )
        names = [c.name for c in client.get_collections().collections]
        return CollectionsResponse(collections=names)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}")


@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config_json: Optional[str] = Form(None),
):
    """Upload a document and kick off ingestion."""
    # Validate file type
    allowed_exts = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_exts}",
        )

    # Parse config
    settings = get_settings()
    base_config = PipelineConfig.from_settings(settings)

    if config_json:
        try:
            overrides = json.loads(config_json)
            config = base_config.model_copy(update=overrides)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid config JSON: {exc}")
    else:
        config = base_config

    # Inject infra secrets
    config_dict = config.model_dump()
    config_dict["qdrant_host"] = settings.qdrant_host
    config_dict["qdrant_port"] = settings.qdrant_port
    config_dict["qdrant_api_key"] = settings.qdrant_api_key

    # === SAVE TO PERMANENT FOLDER (never deleted) ===
    file_id = str(uuid.uuid4())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    permanent_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")

    with open(permanent_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "document_id": None,
        "metrics": None,
        "error": None,
    }

    background_tasks.add_task(_run_pipeline, job_id, permanent_path, config_dict)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion queued for '{file.filename}'",
    )


@app.get("/ingest/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Poll the status of an ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatus(job_id=job_id, **job)


# ====================== DEV RUNNER ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)