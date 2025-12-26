import base64
import json
import logging
import json
import logging
import os
from typing import List, Optional

try:
    import cv2
    from openai import OpenAI
except ImportError:
    cv2 = None
    OpenAI = None
    # For mocking/testing purposes or if deps are missing
    pass

from src.core.models import QualityScore, VideoArtifact, Intent, Shot
from src.evaluation.scorer import QualityEvaluator, MockQualityEvaluator
from src.config.settings import settings

logger = logging.getLogger("FashionEngine")

class VisionQualityEvaluator(MockQualityEvaluator):
    """
    Evaluates video quality using Vision-Language Models (GPT-4o / VLLM).
    Extracts frames and asks the model to rate them.
    """
    def __init__(self):
        super().__init__()
        if OpenAI is None or cv2 is None:
            logger.warning("⚠️ OpenAI or OpenCV not installed. VisionEvaluator disabled.")
            self.client = None
            return

        api_key = settings.openai_api_key or "sk-dummy"
        base_url = settings.vllm_api_url or None # None defaults to OpenAI's public API
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o" if not base_url else "vllm-model" # Customizable model name via config if needed

    def _extract_frames(self, video_path: str, count: int = 3) -> List[str]:
        """
        Extracts 'count' frames spread evenly across the video.
        Returns list of base64 encoded strings.
        """
        if not os.path.exists(video_path):
             return []

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return []
            
        indices = [
            0,
            frame_count // 2,
            max(0, frame_count - 1)
        ]
        
        frames_b64 = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
                
        cap.release()
        return frames_b64

    def evaluate(
        self,
        artifact: VideoArtifact,
        intent: Intent,
        shot: Shot,
        full_prompt: str,
        reference_images: List[str],
    ) -> QualityScore:
        
        frames = self._extract_frames(artifact.file_path)
        if not frames:
            logger.warning(f"⚠️ Could not extract frames for {artifact.video_id}. Fallback to mock.")
            # Fallback to random score if extraction fails
            return super().evaluate(artifact, intent, shot, full_prompt, reference_images)

        # Build prompt
        system_prompt = """
        You are an expert cinematographic evaluator. 
        Analyze the provided video frames. 
        Rate the video on a scale of 1-10 for the following criteria:
        1. Realism: How photorealistic is the content?
        2. Brand Alignment: Does it match high-end fashion aesthetics?
        3. Product Visibility: Is the product clearly visible and undistorted?
        4. consistency: Is the visual quality consistent across frames?
        5. Motion: Does the implied motion look natural (not warped)?
        
        Return ONLY a JSON object:
        {
            "realism": float,
            "brand": float,
            "product": float,
            "consistency": float,
            "motion": float,
            "overall": float,
            "notes": "Short explanation"
        }
        """

        user_content = [
            {"type": "text", "text": f"Evaluate this video generated for: {full_prompt}"}
        ]
        
        for f in frames:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=300,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return QualityScore(
                realism=float(data.get("realism", 5)),
                brand_alignment=float(data.get("brand", 5)),
                product_visibility=float(data.get("product", 5)),
                visual_consistency=float(data.get("consistency", 5)),
                motion_quality=float(data.get("motion", 5)),
                overall=float(data.get("overall", 5)),
                notes=[data.get("notes", "Automated vision evaluation")]
            )

        except Exception as e:
            logger.error(f"Vision Evaluation failed: {e}")
            # Fallback
            return QualityScore(
                realism=0.0, brand_alignment=0.0, product_visibility=0.0, visual_consistency=0.0, motion_quality=0.0, overall=0.0, notes=[f"Eval Failed: {e}"]
            )
