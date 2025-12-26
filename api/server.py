import asyncio
import logging
import uuid
import os
import shutil
from typing import List, Optional, Any
from dataclasses import asdict

from fastapi import BackgroundTasks, FastAPI, HTTPException, File, UploadFile, Form, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.schemas import (
    GenerateResponse,
    JobResponse,
    JobStatus,
    JobSubmissionResponse,
    VideoResponse,
)
from api.security import verify_api_key
from src.pipeline.manager import FashionPipeline, PipelineResult
from src.utils.logger import setup_logging
from src.core.persistence import SQLiteJobStore
from src.core.models import TrainingConfig
from src.core.datasets import save_and_extract_dataset, validate_dataset

logger = setup_logging()

# Use SQLite persistence
db = SQLiteJobStore("jobs.db")

# Single engine instance
engine = FashionPipeline(output_root="api_runs")

app = FastAPI(
    title="Fashion Engine API",
    description="AI-powered fashion video generation engine",
    version="1.1.0",
)

# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mounts
# 1. API Runs (Outputs)
app.mount("/artifacts", StaticFiles(directory="api_runs"), name="runs")

# 2. Uploads (Inputs)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


class GenerateRequest(BaseModel):
    product_name: str
    category: str
    features: List[str]
    images: List[str]  # paths or URLs


async def _run_generation_task(job_id: str, request: GenerateRequest):
    """
    Background task wrapper to run the pipeline and update job status.
    """
    logger.info(f"▶️ Starting background job: {job_id}")
    db.update_job(job_id, {"status": JobStatus.PROCESSING})

    try:
        results: List[PipelineResult] = await engine.run_async(
            product_name=request.product_name,
            category=request.category,
            features=request.features,
            images=request.images,
        )

        clean_videos = []
        for res in results:
            rel_path = os.path.relpath(res.artifact.file_path, "api_runs").replace("\\", "/")
            clean_videos.append(
                VideoResponse(
                    video_id=res.artifact.video_id,
                    score=res.score.overall,
                    file_path=f"/artifacts/{rel_path}",
                    metrics={k: v for k, v in asdict(res.score).items() if isinstance(v, (int, float))},
                    notes=res.score.notes
                )
            )

        response = GenerateResponse(total_returned=len(clean_videos), videos=clean_videos)
        
        db.update_job(job_id, {"status": JobStatus.COMPLETED, "result": response.model_dump()})
        logger.info(f"✅ Job {job_id} completed.")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
        db.update_job(job_id, {"status": JobStatus.FAILED, "error": str(e)})


async def _run_training_task(job_id: str, dataset_path: str, generator_name: str, trigger_word: str):
    """
    Background task for Training/Fine-Tuning.
    """
    logger.info(f"🎓 Starting TRAINING job: {job_id} on {generator_name}")
    db.update_job(job_id, {"status": JobStatus.TRAINING})

    try:
        # Validate dataset again or gather stats
        count = validate_dataset(dataset_path)
        logger.info(f"Dataset has {count} images.")

        # Pick generator (Simple logic: use first one or match name)
        # For now, we grab the first one (Mock) because we didn't implement registry lookup yet
        # Phase 3 Refinement: Implement generator registry logic.
        target_gen = engine.generators[0] 
        
        # Create optimized config
        config = TrainingConfig(
            trigger_word=trigger_word,
            learning_rate=1e-4, # Optimized default
            network_rank=128    # Optimized default
        )
        
        training_job_id = await target_gen.train_async(
            dataset_path=dataset_path, 
            config=config
        )
        
        db.update_job(job_id, {
            "status": JobStatus.TRAINING_COMPLETED, 
            "result": {"adapter_id": training_job_id, "base_model": target_gen.profile.name}
        })
        logger.info(f"✅ Training Job {job_id} completed. Adapter: {training_job_id}")

    except Exception as e:
        logger.error(f"❌ Training Job {job_id} failed: {e}", exc_info=True)
        db.update_job(job_id, {"status": JobStatus.TRAINING_FAILED, "error": str(e)})


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_image(file: UploadFile = File(...)):
    """
    Handle image uploads. Returns the relative path. Protected by API Key.
    """
    try:
        file_id = str(uuid.uuid4())[:8]
        extension = file.filename.split(".")[-1]
        filename = f"{file_id}.{extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": filename, "path": f"uploads/{filename}", "url": f"/uploads/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs", response_model=JobSubmissionResponse, dependencies=[Depends(verify_api_key)])
async def submit_job(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Submit a video generation job. Returns immediately with a Job ID. Protected by API Key.
    """
    job_id = str(uuid.uuid4())
    initial_job = {
        "status": JobStatus.PENDING,
        "result": None,
        "error": None
    }
    db.create_job(job_id, initial_job)

    background_tasks.add_task(_run_generation_task, job_id, request)

    return JobSubmissionResponse(job_id=job_id, status=JobStatus.PENDING)


@app.post("/train", response_model=JobSubmissionResponse, dependencies=[Depends(verify_api_key)])
async def submit_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    trigger_word: str = Form(...),
    generator_name: str = Form("mock"),
):
    """
    Submit a Fine-Tuning Job. Accepts a ZIP dataset.
    """
    # 1. Handle Dataset
    try:
        dataset_path = await save_and_extract_dataset(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Create Job
    job_id = str(uuid.uuid4())
    initial_job = {
        "status": JobStatus.PENDING,
        "result": None,
        "error": None
    }
    db.create_job(job_id, initial_job)

    # 3. Background Task
    background_tasks.add_task(_run_training_task, job_id, dataset_path, generator_name, trigger_word)

    return JobSubmissionResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Poll the status of a specific job. Public endpoint (read-only).
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        result=job["result"],
        error=job["error"],
    )


@app.get("/api/health")
def health_check():
    # Count via DB
    active_count = db.list_active_jobs()
    return {"status": "Fashion Engine API is running", "jobs_active": active_count}

# 3. Web UI (Must be mounted last to avoid overriding API routes)
if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web")
