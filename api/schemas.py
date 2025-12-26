from pydantic import BaseModel
from typing import List

class VideoResponse(BaseModel):
    video_id: str
    score: float
    file_path: str

class GenerateResponse(BaseModel):
    total_returned: int
    videos: List[VideoResponse]
