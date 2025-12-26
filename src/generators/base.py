from typing import List, Optional
from src.core.models import VideoArtifact, ModelProfile

class VideoGenerator:
    """
    Abstract interface for all video generation backends.
    Any new AI model integration must inherit from this class.
    """
    
    # Every generator must declare its capabilities
    profile: ModelProfile

    def generate(
        self, 
        prompt: str, 
        reference_paths: List[str], 
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None
    ) -> VideoArtifact:
        """
        Generates a video based on the prompt and reference images.
        
        Args:
            prompt: The full text prompt.
            reference_paths: List of file paths to input images.
            aspect_ratio: Desired output ratio (e.g., "16:9", "9:16").
            seed: Optional integer for deterministic generation.

        Returns:
            VideoArtifact containing the file path and metadata.
        
        Raises:
            ConnectionError: For retryable network issues.
            ValueError: For non-retryable configuration issues.
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    async def generate_async(
        self, 
        prompt: str, 
        reference_paths: List[str], 
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None
    ) -> VideoArtifact:
        """
        Asynchronously generates a video.
        Default implementation wraps the synchronous generate method for compatibility.
        """
        import asyncio
        return await asyncio.to_thread(
            self.generate, 
            prompt=prompt, 
            reference_paths=reference_paths, 
            aspect_ratio=aspect_ratio, 
            seed=seed
        )