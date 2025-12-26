from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, ConfigDict


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TRAINING = "training"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"


class VideoResponse(BaseModel):
    video_id: str
    score: float
    file_path: str
    metrics: Dict[str, float]
    notes: List[str]


class GenerateResponse(BaseModel):
    total_returned: int
    videos: List[VideoResponse]


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None

