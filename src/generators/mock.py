import os
import random
import hashlib
import logging
import asyncio
from typing import List, Optional
from src.core.models import VideoArtifact, ModelProfile, GenerationOutcome
from src.generators.base import VideoGenerator
from src.utils.decorators import smart_retry

logger = logging.getLogger("FashionEngine")


class AdvancedMockVideoGenerator(VideoGenerator):
    OUTPUT_DIR = "outputs"
    profile = ModelProfile("Gen-Mock-Elite-v3", 4096, True, True)

    def __init__(
        self,
        failure_rate: float = 0.15,
        hard_failure_ratio: float = 0.3,
        min_latency: float = 0.5,
        max_latency: float = 2.0,
        calls_per_minute: int = 600, # Increased for async demo
    ):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.failure_rate = failure_rate
        self.hard_failure_ratio = hard_failure_ratio
        self.min_latency = min_latency
        self.max_latency = max_latency

        # Simple Rate Limiter State
        self._call_timestamps = []
        self._rate_limit = calls_per_minute

        logger.info(
            f"🔌 [MOCK] Advanced Generator Active | Profile: {self.profile.name} | Error Rate: {int(failure_rate*100)}%"
        )

    def _check_rate_limit(self):
        # Note: This simple check isn't thread-safe for threads, but okay for asyncio if single-threaded event loop
        # For a more robust solution, we'd use a token bucket or similar
        import time
        now = time.time()
        # Remove calls older than 60 seconds
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]

        if len(self._call_timestamps) >= self._rate_limit:
            return True

        self._call_timestamps.append(now)
        return False

    def _decide_outcome(self, seed: Optional[int]) -> GenerationOutcome:
        # Deterministic seeds for testing
        if seed == 9999:
            return GenerationOutcome.HARD_FAILURE
        if seed == 8888:
            return GenerationOutcome.TRANSIENT_FAILURE
        if seed == 7777:
            return GenerationOutcome.RATE_LIMIT

        r = random.random()
        if r < self.failure_rate:
            if random.random() < self.hard_failure_ratio:
                return GenerationOutcome.HARD_FAILURE
            return GenerationOutcome.TRANSIENT_FAILURE

        return GenerationOutcome.SUCCESS

    @smart_retry(retries=3, delay=1.5)
    async def generate_async(self, prompt: str, reference_paths: List[str], aspect_ratio: str = "16:9", seed: Optional[int] = None) -> VideoArtifact:
        # 1. Rate limit
        if self._check_rate_limit():
            logger.warning("🔥 [MOCK] 429 Too Many Requests (Simulated)")
            raise ConnectionError("Rate limit exceeded (429)")

        # 2. Simulate latency (Async)
        latency = random.triangular(self.min_latency, self.max_latency, self.min_latency + 0.5)
        await asyncio.sleep(latency)

        # 3. Decide outcome
        outcome = self._decide_outcome(seed)

        if outcome == GenerationOutcome.HARD_FAILURE:
            logger.error("❌ [MOCK] 400 Bad Request: Model rejected prompt (Safety filter?)")
            raise ValueError("Safety filter triggered: Prompt violated content policy.")

        if outcome == GenerationOutcome.TRANSIENT_FAILURE:
            logger.warning("⚠️ [MOCK] 503 Service Unavailable: GPU overload")
            raise ConnectionError("Upstream service timeout (503)")

        if outcome == GenerationOutcome.RATE_LIMIT:
            logger.warning("🔥 [MOCK] 429 Too Many Requests (Forced by Seed)")
            raise ConnectionError("Rate limit exceeded (429)")

        # 4. Success path
        video_seed = seed or random.randint(100000, 999999)
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        video_id = hashlib.md5(f"{prompt_id}{video_seed}".encode()).hexdigest()[:8]

        filename = f"mock_gen_{video_id}.mp4"
        output_path = os.path.join(self.OUTPUT_DIR, filename)

        # We can still blocking-write for the mock or make it async if we want to be pure
        # For simplicity, small blocking write is fine here for a mock
        file_size_mb = random.uniform(1.5, 5.0)
        # Using a thread execution for file IO to avoid blocking event loop
        await asyncio.to_thread(self._write_dummy_file, output_path, file_size_mb)

        actual_duration = round(random.uniform(3.8, 4.2), 2)

        logger.info(f"✅ [MOCK] Generated {filename} ({file_size_mb:.2f}MB) in {latency:.2f}s")

        return VideoArtifact(
            file_path=output_path,
            video_id=video_id,
            seed=video_seed,
            duration=actual_duration,
            model_used=self.profile.name,
            prompt=prompt,
            prompt_id=prompt_id,
            metadata={
                "aspect_ratio": aspect_ratio,
                "simulated_latency": round(latency, 3),
                "server_node": f"gpu-cluster-{random.randint(1,5)}",
            },
        )
    
    def _write_dummy_file(self, path, size_mb):
        with open(path, "wb") as f:
            f.write(b"\0" * int(size_mb * 1024 * 1024))

    # Keep synchronous method for backward compatibility if needed, using run_until_complete
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Use generate_async instead")
