from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from api.schemas import GenerateResponse, VideoResponse
from src.pipeline.manager import FashionPipeline
import logging

logger = logging.getLogger("FashionEngine")

# Single engine instance used by the API
engine = FashionPipeline(output_root="api_runs")

app = FastAPI(
    title="Fashion Engine API",
    description="AI-powered fashion video generation engine",
    version="0.1.0",
)


class GenerateRequest(BaseModel):
    product_name: str
    category: str
    features: List[str]
    images: List[str]  # paths or URLs

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    results = engine.run(
        product_name=request.product_name,
        category=request.category,
        features=request.features,
        images=request.images
    )

    videos = [
        VideoResponse(
            video_id=artifact.video_id,
            score=score.overall,
            file_path=artifact.file_path
        )
        for artifact, score, _ in results
    ]

    return GenerateResponse(
        total_returned=len(videos),
        videos=videos
    )

@app.get("/")
def health_check():
    return {"status": "Fashion Engine API is running"}
