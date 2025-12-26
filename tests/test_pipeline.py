import asyncio
import pytest
from src.pipeline.manager import FashionPipeline
from src.generators.base import VideoGenerator
from src.core.models import VideoArtifact
from src.evaluation.scorer import MockQualityEvaluator


class DummyGenerator(VideoGenerator):
    profile = None

    def generate(self, prompt, reference_paths, aspect_ratio="16:9", seed=None):
        # Return a lightweight artifact without writing files
        return VideoArtifact(
            file_path="",
            video_id="testvid",
            seed=seed or 0,
            duration=4.0,
            model_used="dummy",
            prompt=prompt,
            prompt_id="pid",
            metadata={"simulated_latency": 0.1},
        )
        
    async def generate_async(self, prompt, reference_paths, aspect_ratio="16:9", seed=None):
        # Async implementation for testing
        await asyncio.sleep(0.01)
        return self.generate(prompt, reference_paths, aspect_ratio, seed)


def test_pipeline_run_sync_wrapper():
    """Test the synchronous wrapper which calls asyncio.run internally"""
    pipeline = FashionPipeline(generator=DummyGenerator(), evaluator=MockQualityEvaluator(), output_root="outputs_test")

    results = pipeline.run(product_name="Test Hoodie", category="hoodie", features=["feat"], images=["img.jpg"]) 

    assert isinstance(results, list)
    if results:
        artifact, score, shot = results[0]
        assert hasattr(artifact, "video_id")
        assert hasattr(score, "overall")

def test_pipeline_run_async_direct():
    """Test the async method directly using asyncio.run within the test"""
    pipeline = FashionPipeline(generator=DummyGenerator(), evaluator=MockQualityEvaluator(), output_root="outputs_test")
    
    async def run_test():
        return await pipeline.run_async(product_name="Test Hoodie", category="hoodie", features=["feat"], images=["img.jpg"]) 

    results = asyncio.run(run_test())

    assert isinstance(results, list)
    if results:
        artifact, score, shot = results[0]
        assert hasattr(artifact, "video_id")
        assert hasattr(score, "overall")
