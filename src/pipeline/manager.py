import os
import time
import json
import logging
import asyncio
from dataclasses import asdict
from typing import Optional, List, Tuple

from src.core.models import GenerationMode, Shot, VideoArtifact, QualityScore
from src.engine.director import CreativeDirector
from src.engine.adapters import RunwayAdapter
from src.generators.base import VideoGenerator
from src.generators.mock import AdvancedMockVideoGenerator
from src.evaluation.scorer import QualityEvaluator, MockQualityEvaluator
from src.engine.sanitizer import sanitize_prompt

logger = logging.getLogger("FashionEngine")


class FashionPipeline:
    """
    Orchestrates the full fashion video generation pipeline.
    """

    def __init__(
        self,
        generator: Optional[VideoGenerator] = None,
        evaluator: Optional[QualityEvaluator] = None,
        output_root: str = "outputs",
    ):
        # Core engine
        self.director = CreativeDirector(
            brand_path="config/brand_profile.json",
            rules_path="config/category_rules.json",
        )
        self.adapter = RunwayAdapter()

        # Dependency injection
        self.generator = generator if generator else AdvancedMockVideoGenerator()
        self.evaluator = evaluator if evaluator else MockQualityEvaluator()

        # Run directory
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(output_root, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Redirect generator output
        if hasattr(self.generator, "OUTPUT_DIR"):
            self.generator.OUTPUT_DIR = self.run_dir

    def run(self, product_name, category, features, images) -> List[Tuple[VideoArtifact, QualityScore, Shot]]:
        """Synchronous wrapper for run_async"""
        return asyncio.run(self.run_async(product_name, category, features, images))

    async def run_async(self, product_name, category, features, images):
        logger.info(f"🚀 Starting Async Pipeline Run for: {product_name}")

        # 1️⃣ Build intent
        intent = self.director.build_product_intent(
            product_name=product_name,
            category=category,
            key_features=features,
            image_count=len(images),
        )
        logger.info(f"🔑 Intent Hash: {intent.get_hash()}")

        # 2️⃣ Generate shot plan
        shots = self.director.generate_shots(
            intent,
            count=3,
            mode=GenerationMode.STRICT,
        )

        # SAFETY: never allow empty shots
        if not shots:
            logger.warning("⚠️ No shots generated. Using fallback shot.")
            shots = [
                Shot(
                    id=1,
                    pose="Centered static product shot",
                    environment="Neutral studio background",
                    camera_action="Static medium shot",
                    focus_points=["Overall product visibility"],
                )
            ]

        TARGET_GOOD_VIDEOS = 10
        QUALITY_THRESHOLD = 7.0
        MAX_TOTAL_ATTEMPTS = 40
        CONCURRENCY = 5

        accepted_results = []
        rejected_results = []
        prompts_map = {}
        blocked_prompts = set()

        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def attempt_generation(shot: Shot, attempt_idx: int):
            if len(accepted_results) >= TARGET_GOOD_VIDEOS:
                return None

            raw_prompt = self.adapter.format(intent, shot)
            prompt = sanitize_prompt(raw_prompt)
            prompt_id = hash(prompt)

            if prompt_id in blocked_prompts:
                logger.info("⏭️ Skipping blocked prompt")
                return None

            async with semaphore:
                try:
                    artifact = await self.generator.generate_async(
                        prompt=prompt,
                        reference_paths=images,
                        seed=None,
                    )

                    prompts_map[artifact.prompt_id] = prompt

                    score = self.evaluator.evaluate(
                        artifact=artifact,
                        intent=intent,
                        shot=shot,
                        full_prompt=prompt,
                    )

                    result = {
                        "artifact": artifact,
                        "score": score,
                        "shot": shot,
                    }

                    if score.overall >= QUALITY_THRESHOLD:
                         # Thread-safe append in asyncio (it's single threaded event loop)
                        accepted_results.append(result)
                        logger.info(
                            f"✅ Accepted {artifact.video_id} | "
                            f"Score: {score.overall:.2f} "
                            f"({len(accepted_results)}/{TARGET_GOOD_VIDEOS})"
                        )
                        return result
                    else:
                        rejected_results.append(result)
                        logger.info(
                            f"❌ Rejected {artifact.video_id} | "
                            f"Score: {score.overall:.2f}"
                        )
                        return None

                except ValueError as e:
                    # HARD failure → blacklist prompt
                    logger.error(f"🚫 Blocking prompt due to safety failure: {e}")
                    blocked_prompts.add(prompt_id)
                except Exception as e:
                    logger.warning(f"⚠️ Generation attempt failed: {e}")
            return None

        # 3️⃣ Resilient generation loop (Parallel)
        tasks = []
        # We start by scheduling MAX_TOTAL_ATTEMPTS potentially, or we can do chunks.
        # Since we have "break" condition (TARGET_GOOD_VIDEOS), scheduling all might act wasteful if we hit target early.
        # But for simplicity in asyncio without complex queue management, we can schedule batches.
        
        # Let's use an approach where we schedule batches until target is met.
        attempts = 0
        shot_index = 0
        
        while len(accepted_results) < TARGET_GOOD_VIDEOS and attempts < MAX_TOTAL_ATTEMPTS:
            batch_size = min(CONCURRENCY, MAX_TOTAL_ATTEMPTS - attempts)
            current_batch_tasks = []
            
            for _ in range(batch_size):
                if attempts >= MAX_TOTAL_ATTEMPTS: 
                    break
                shot = shots[shot_index % len(shots)]
                current_batch_tasks.append(attempt_generation(shot, attempts))
                shot_index += 1
                attempts += 1
            
            if not current_batch_tasks:
                break
                
            await asyncio.gather(*current_batch_tasks)

        # 4️⃣ Graceful fallback
        if not accepted_results:
            logger.warning(
                "⚠️ No videos met quality threshold. "
                "Falling back to best available results."
            )
            rejected_results.sort(
                key=lambda r: r["score"].overall, reverse=True
            )
            accepted_results = rejected_results[:TARGET_GOOD_VIDEOS]

        # 5️⃣ Final ranking
        accepted_results.sort(
            key=lambda r: r["score"].overall, reverse=True
        )

        # 6️⃣ Save manifest
        self._save_manifest(intent, accepted_results, prompts_map)

        logger.info(
            f"🏁 Pipeline complete | "
            f"Returned {len(accepted_results)} videos | "
            f"Attempts: {attempts}"
        )

        return [
            (r["artifact"], r["score"], r["shot"])
            for r in accepted_results
        ]

    def _save_manifest(self, intent, results, prompts_map):
        manifest = {
            "timestamp": time.time(),
            "intent": asdict(intent),
            "outputs": [
                {
                    "video_id": r["artifact"].video_id,
                    "prompt_id": r["artifact"].prompt_id,
                    "score": asdict(r["score"]),
                    "file_path": r["artifact"].file_path,
                }
                for r in results
            ],
            "prompt_library": prompts_map,
        }

        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"✅ Manifest saved to {self.run_dir}")