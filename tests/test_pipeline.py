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


def test_pipeline_run_returns_list_and_scores():
    pipeline = FashionPipeline(generator=DummyGenerator(), evaluator=MockQualityEvaluator(), output_root="outputs_test")

    results = pipeline.run(product_name="Test Hoodie", category="hoodie", features=["feat"], images=["img.jpg"]) 

    assert isinstance(results, list)
    # If any results returned, ensure structure
    if results:
        artifact, score, shot = results[0]
        assert hasattr(artifact, "video_id")
        assert hasattr(score, "overall")
