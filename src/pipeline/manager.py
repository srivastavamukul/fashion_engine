import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Tuple, Set

from src.config import settings
from src.core.models import (
    GenerationMode, 
    Intent, 
    PipelineResult, 
    Shot, 
    VideoArtifact
)
from src.engine.adapters import PromptAdapter, RunwayAdapter
from src.engine.director import CreativeDirector
from src.engine.mutator import ShotMutator
from src.engine.sanitizer import sanitize_and_repair_prompt
from src.evaluation.scorer import MockQualityEvaluator, QualityEvaluator
from src.evaluation.vision import VisionQualityEvaluator
from src.generators.base import VideoGenerator
from src.generators.mock import AdvancedMockVideoGenerator
from src.generators.integrations import StabilityVideoGenerator, LumaRayGenerator
from src.utils.logger import get_logger

logger = get_logger()

class FashionPipeline:
    """
    Orchestrates the full fashion video generation pipeline.
    """

    def __init__(
        self,
        generators: Optional[List[VideoGenerator]] = None,
        evaluator: Optional[QualityEvaluator] = None,
        director: Optional[CreativeDirector] = None,
        adapter: Optional[PromptAdapter] = None,
        mutator: Optional[ShotMutator] = None,
        output_root: Optional[str] = None,
    ):
        self.output_root = output_root or settings.output_root
        
        self.director = director or CreativeDirector(
            brand_path=settings.brand_profile_path,
            rules_path=settings.category_rules_path,
        )
        self.adapter = adapter or RunwayAdapter()
        self.mutator = mutator or ShotMutator()
        if settings.enable_vision_evaluator:
            # Use Vision Evaluator if enabled
            self.evaluator = evaluator or VisionQualityEvaluator()
        else:
            self.evaluator = evaluator or MockQualityEvaluator()

        # Initialize Generators
        self.generators = generators or []
        if not self.generators:
            # Load default chain based on settings
            if settings.enable_runway:
                # Assuming AdvancedMock is our "Runway" placeholder for now inside unit tests or simple usage
                # unless integrating actual RunwayGenerator if we had it.
                # For now using AdvancedMock as default if enabled, representing Runway
                self.generators.append(AdvancedMockVideoGenerator())
            
            if settings.enable_stability:
                self.generators.append(StabilityVideoGenerator())
            
            if settings.enable_luma:
                self.generators.append(LumaRayGenerator())
            
            # Fallback if nothing enabled
            if not self.generators:
                self.generators.append(AdvancedMockVideoGenerator())

        # Run Setup
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.output_root, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        for gen in self.generators:
             if hasattr(gen, "OUTPUT_DIR"):
                 gen.OUTPUT_DIR = self.run_dir

    def run(
        self,
        product_name: str,
        category: str,
        features: List[str],
        images: List[str],
    ) -> List[PipelineResult]:
        """
        Synchronous entry point for the pipeline.
        """
        return asyncio.run(
            self.run_async(product_name, category, features, images)
        )

    async def _generate_safely(self, generator: VideoGenerator, prompt: str, images: List[str], seed: int) -> VideoArtifact:
        """Safely run generator in async context (wrapping if needed)"""
        # If generator has native async, use it (checking if it overrides base which is async wrapper)
        return await generator.generate_async(prompt, images, seed=str(seed)) # seed as str/int safety depending on impl

    async def run_async(
        self, 
        product_name: str, 
        category: str, 
        features: List[str], 
        images: List[str]
    ) -> List[PipelineResult]:
        
        logger.info(f"🚀 Starting Pipeline Run for: {product_name} with {len(self.generators)} generators")

        try:
            intent = self.director.build_product_intent(
                product_name=product_name,
                category=category,
                key_features=features,
                image_count=len(images),
                images=images,
            )
        except Exception as e:
            logger.critical(f"❌ Failed to build intent: {e}")
            raise

        shots = self.director.generate_shots(
            intent,
            count=settings.target_videos, # Use target as count base
            mode=GenerationMode.STRICT,
        )

        if not shots:
            logger.warning("⚠️ No shots generated, using fallback.")
            shots = [
                Shot(
                    id=1,
                    pose="Centered static product shot",
                    environment="Neutral studio background",
                    camera_action="Static medium shot",
                    focus_points=["Overall product visibility"],
                )
            ]

        accepted: List[PipelineResult] = []
        rejected: List[PipelineResult] = []
        prompts_map = {}
        blocked_prompt_ids: Set[str] = set()

        semaphore = asyncio.Semaphore(settings.concurrency * len(self.generators)) # Scale semaphore by generators

        attempts = 0
        shot_index = 0

        # We want to fill accepted list
        # Strategy: Iterate shots, fan out to ALL generators, collect results
        
        while len(accepted) < settings.target_videos and attempts < settings.max_attempts:
            tasks = []
            
            # Pick a shot
            base_shot = shots[shot_index % len(shots)]
            shot_index += 1
            attempts += 1 # This counts as one "batch" attempt
            
            raw_prompt = self.adapter.format(intent, base_shot)
            prompt = sanitize_and_repair_prompt(raw_prompt, attempt=attempts)
            
            prompt_hash = hashlib.sha256(f"{prompt}-{base_shot.id}".encode("utf-8")).hexdigest()[:12]
            
            # Fan out to all generators
            for i, gen in enumerate(self.generators):
                seed = hash((base_shot.id, attempts, i)) % 1_000_000
                
                async def run_one(g=gen, p=prompt, s=base_shot, seed=seed, h=prompt_hash):
                    async with semaphore:
                        try:
                            # Start timer
                            t0 = time.time()
                            artifact = await g.generate_async(p, images, seed=seed)
                            latency = time.time() - t0
                            artifact.metadata["latency"] = latency
                            return (p, h, s, artifact)
                        except Exception as e:
                            logger.error(f"Gen {g.profile.name} failed: {e}")
                            return (p, h, s, e)

                tasks.append(run_one())

            # Execute Fan-Out
            if not tasks:
                break
                
            results = await asyncio.gather(*tasks)

            # Process Results
            for prompt, prompt_id, shot, result in results:
                if isinstance(result, Exception):
                    continue

                artifact = result
                prompts_map[artifact.prompt_id] = prompt

                score = self.evaluator.evaluate(
                    artifact=artifact,
                    intent=intent,
                    shot=shot,
                    full_prompt=prompt,
                    reference_images=images,
                )

                pipeline_result = PipelineResult(
                    artifact=artifact,
                    score=score,
                    shot=shot
                )

                if score.overall >= settings.quality_threshold:
                    accepted.append(pipeline_result)
                    logger.info(
                        f"✅ Accepted {artifact.video_id} [{artifact.model_used}] "
                        f"({score.overall:.2f}) "
                        f"{len(accepted)}/{settings.target_videos}"
                    )
                else:
                    rejected.append(pipeline_result)
                    logger.info(
                        f"❌ Rejected {artifact.video_id} [{artifact.model_used}] "
                        f"({score.overall:.2f})"
                    )

        # Fallback Strategy
        if not accepted:
            logger.warning("⚠️ Falling back to best rejected results.")
            rejected.sort(key=lambda r: r.score.overall, reverse=True)
            accepted = rejected[:settings.target_videos]

        accepted.sort(key=lambda r: r.score.overall, reverse=True)

        self._save_manifest(intent, accepted, prompts_map)

        logger.info(
            f"🏁 Pipeline complete | Accepted: {len(accepted)} | Active Generators: {len(self.generators)}"
        )

        return accepted

    def _save_manifest(self, intent: Intent, results: List[PipelineResult], prompts_map: dict):
        manifest = {
            "timestamp": time.time(),
            "intent": asdict(intent),
            "outputs": [
                {
                    "video_id": r.artifact.video_id,
                    "model": r.artifact.model_used,
                    "prompt_id": r.artifact.prompt_id,
                    "score": asdict(r.score),
                    "file_path": r.artifact.file_path,
                }
                for r in results
            ],
            "prompt_library": prompts_map,
        }

        try:
            with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"✅ Manifest saved to {self.run_dir}")
        except IOError as e:
            logger.error(f"Failed to save manifest: {e}")
