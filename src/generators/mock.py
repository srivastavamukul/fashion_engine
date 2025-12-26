import base64
import uuid
import hashlib
import logging
import os
import time
import shutil
import random
import asyncio

from runwayml import RunwayML  # Assuming official SDK usage

from src.core.models import ModelProfile, VideoArtifact, TrainingConfig
from src.generators.base import VideoGenerator
from src.utils.logger import get_logger # Assuming get_loggernerator was a typo and should be get_logger

# Avoid importing heavy vision/CLIP utilities at module import time; import lazily if needed.


logger = get_logger("FashionEngine") # Changed to use get_logger from src.utils.logger


class AdvancedMockVideoGenerator(VideoGenerator):
    """Lightweight mock generator used for tests and local runs."""

    OUTPUT_DIR = "outputs_mock"

    profile = ModelProfile(
        name="advanced-mock",
        max_tokens=0,
        supports_seed=True,
        supports_negative_prompt=False,
        supports_training=True, # Added support for training
    )

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def generate(self, prompt, reference_paths, aspect_ratio="16:9", seed=None):
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        video_id = f"mock-{prompt_hash}-{int(time.time()*1000)}"
        # Valid 1-second H.264 MP4 (black frame) - Browser Compatible
        # This blob uses H.264 (avc1) instead of mp4v for better compatibility.
        MOCK_MP4_B64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAsptZGF0AAACrgYF//+//7fcP2TMuEAAAAAnZGlmZgH/gAAAAAAAAAAABgAAAAAsZGMgH/4AAAAAAAAGAAAAACxhY3AgH/4AAAAAAAAGAAAAABZkY3AgH/4AAAAAAAAGAAAAABZkY3AgH/4AAAAAAAAGAAAAABZkY3AgH/4AAAAAAAAGAAAAABhjbXAgH/4AAQAAAAAAABhjbXAgH/4AAQAAAAAAABhjbXAgH/4AAQAAAAAAABhjbXAgH/4AAQAAAAAAABhjbXAgH/4AAQAAAAAAABhjbXAgH/4AAQAAAAAAACBhY3AgH/4AAQAAAAAAAEG1lZGlhIGRhdGEgbmV0AAAAXuBtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAD6AABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAABWHRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAD6AAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAQAAAAEAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAA+gAAAAAAAEAAAAAAABibWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAD6AAAA+gAA1gAAAAAAHaWhZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATFtaW5mAAAAFHZtaGQAAAARAAAAAAAAAAAAAAApJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABM3N0YmwAAACxc3RzZAAAAAAAAAABAAAAhWF2YzEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAABAABIaAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAAFxhdmNDAWQAJf/hABlnZAAlrYy+F/LwYBAAZo6OMAQAAwAAAwB4kR7ADyQAAAMAAAMAd5EewA8kRUF0AAAAAElzdHQAAAAAAAAAAQAAAAEAAA1zdHNjAAAAAAAAAAEAAAABAAAAAQAAAAEAAAAcc3RzegAAAAAAAAAQAAAAAQAAAAEAAAAAAAAAFHN0Y28AAAAAAAAAAQAAAIAAAAAYc3RzcwAAAAAAAAABAAAAAQ=="
        
        filename = os.path.join(self.OUTPUT_DIR, f"{video_id}.mp4")
        
        with open(filename, "wb") as f:
            f.write(base64.b64decode(MOCK_MP4_B64))

        return VideoArtifact(
            file_path=filename,
            video_id=video_id,
            seed=seed if seed else 0,
            duration=1.0,
            model_used="advanced-mock",
            prompt=prompt,
            prompt_id=prompt_hash,
            metadata={"reference_count": len(reference_paths)},
        )

    async def train_async(self, dataset_path: str, config: TrainingConfig) -> str:
        """
        Simulates training with a realistic loss curve and early stopping.
        """
        job_id = f"train-mock-{int(time.time())}"
        logger.info(f"🎓 Starting Simulated Training (Config: lr={config.learning_rate}, rank={config.network_rank})")
        
        # Simulation Parameters
        steps = 20 # Reduced for speed in demo, normally config.max_train_steps
        current_loss = 2.0
        min_val_loss = float('inf')
        patience_counter = 0
        
        for step in range(steps):
             await asyncio.sleep(0.1) # Simulate computation
             
             # Simulate loss curve (mostly going down, occasional spike)
             current_loss *= 0.95 if random.random() > 0.1 else 1.1
             val_loss = current_loss * (1.1 + (random.random() * 0.1)) # Val loss slightly higher
             
             if step % 5 == 0:
                 logger.info(f"Step {step}/{steps} | Loss: {current_loss:.4f} | Val Loss: {val_loss:.4f}")
                 
                 # Early Stopping Check
                 if val_loss < min_val_loss:
                     min_val_loss = val_loss
                     patience_counter = 0
                 else:
                     patience_counter += 1
                     
                 if patience_counter >= config.early_stopping_patience:
                    logger.warning(f"🛑 Early Stopping triggered at step {step}. Val Loss did not improve.")
                    break

        return job_id


class RunwayVideoGenerator(VideoGenerator):
    profile = ModelProfile(
        name="runway-gen3-alpha",
        max_tokens=3000,
        supports_seed=True,
        supports_negative_prompt=False,  # Gen-3 doesn't support negatives yet
    )

    def __init__(self):
        # Initialize client (looks for RUNWAYML_API_SECRET env var)
        self.client = RunwayML()

    def _encode_image(self, image_path: str) -> str:
        """Converts local file to Data URI for API transmission."""
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        # Detect mime type roughly (production code should be more robust)
        ext = image_path.split(".")[-1].lower()
        mime = "png" if ext == "png" else "jpeg"
        return f"data:image/{mime};base64,{encoded_string}"

    def generate(self, prompt, reference_paths, aspect_ratio="16:9", seed=None):
        # 1. Prepare Inputs
        if not reference_paths:
            raise ValueError("Runway Gen-3 requires at least one reference image")

        # Gen-3 currently favors the 'first' image as the primary input
        base64_image = self._encode_image(reference_paths[0])

        logger.info(f"🎨 Submitting job to Runway (Prompt: {prompt[:30]}...)")

        # 2. Submit Task (Async)
        task = self.client.image_to_video.create(
            model="gen3a_turbo",  # or 'gen3a_alpha'
            prompt_image=base64_image,
            prompt_text=prompt,
            seed=seed,
            ratio=aspect_ratio.replace(":", "_"),  # runway uses '16_9' format often
        )

        task_id = task.id
        logger.info(f"⏳ Task {task_id} submitted. Polling for results...")

        # 3. Polling Loop (The Waiting Room)
        # We wait up to 5 minutes (300s)
        max_retries = 60
        for _ in range(max_retries):
            task_status = self.client.tasks.retrieve(task_id)
            status = task_status.status

            if status == "SUCCEEDED":
                video_url = task_status.output[0]  # API returns a URL, not a local file
                break
            elif status == "FAILED":
                raise RuntimeError(f"Runway Generation Failed: {task_status.failure}")
            elif status in ["THROTTLED"]:
                # Handle rate limits logic here if needed
                time.sleep(10)

            # Wait 5 seconds before asking again
            time.sleep(5)
        else:
            raise TimeoutError("Video generation timed out after 5 minutes")

        # 4. Download Video (Optional but recommended)
        # In production, you might just save the URL.
        # Here we pretend we downloaded it to a local path.
        # local_filename = download_file(video_url)

        # 5. Return Artifact
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]

        return VideoArtifact(
            file_path=video_url,  # Or local path if downloaded
            video_id=task_id,
            seed=seed if seed else 0,
            duration=5.0,  # Gen-3 is usually 5s or 10s fixed
            model_used="runway-gen3-alpha",
            prompt=prompt,
            prompt_id=prompt_hash,
            metadata={"remote_url": video_url},
        )
