from dataclasses import dataclass, field
from typing import Dict
import random
from src.core.models import QualityScore, VideoArtifact, Intent, Shot


@dataclass(frozen=True)
class ScoringConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "realism": 0.30,
        "brand_alignment": 0.30,
        "product_visibility": 0.20,
        "motion_quality": 0.20,
    })
    max_acceptable_latency: float = 1.5
    min_acceptable_duration: float = 3.8


class QualityEvaluator:
    def evaluate(self, artifact, intent, shot, full_prompt) -> QualityScore:
        raise NotImplementedError

    def select_top_videos(self, scored_videos, top_k=1):
        return sorted(scored_videos, key=lambda x: x[1].overall, reverse=True)[:top_k]


class MockQualityEvaluator(QualityEvaluator):
    def __init__(self, config: ScoringConfig = ScoringConfig()):
        self.config = config

    def _clamp(self, value, min_val=0.0, max_val=10.0):
        return max(min_val, min(value, max_val))

    def evaluate(self, artifact: VideoArtifact, intent: Intent, shot: Shot, full_prompt: str) -> QualityScore:
        # Realism: slightly penalize for latency
        latency = artifact.metadata.get("simulated_latency", 0.0)
        latency_penalty = max(0.0, (latency - self.config.max_acceptable_latency)) * 0.8

        realism = self._clamp(6.5 + random.uniform(-0.8, 0.8) - latency_penalty)

        # Brand alignment: crude heuristic based on presence of palette and vibe
        brand_alignment = 7.0
        if intent.brand_identity.vibe:
            brand_alignment += 0.5
        brand_alignment = self._clamp(brand_alignment + random.uniform(-1.0, 1.0))

        # Product visibility: reward shots that explicitly focus on overall visibility
        product_visibility = 8.5 if any("Overall product visibility" in f for f in shot.focus_points) else 6.0
        product_visibility = self._clamp(product_visibility + random.uniform(-0.6, 0.6))

        # Motion quality: based on closeness of duration to expected
        duration = getattr(artifact, "duration", self.config.min_acceptable_duration)
        motion_quality = 7.5 if duration >= self.config.min_acceptable_duration else 5.5
        motion_quality = self._clamp(motion_quality + random.uniform(-0.7, 0.7))

        # Guardrail detection: lower score severely if prompt contains banned words
        notes = []
        avoid_list = getattr(intent.guardrails, "avoid", [])
        prompt_text = getattr(artifact, "prompt", "") or full_prompt
        for banned in avoid_list:
            if banned.lower() in prompt_text.lower():
                notes.append("Guardrail violation detected")
                # Nuke overall
                overall = 0.0
                return QualityScore(realism=0.0, brand_alignment=0.0, product_visibility=0.0, motion_quality=0.0, overall=overall, notes=notes)

        # Compute weighted overall
        weights = self.config.weights
        overall = (
            realism * weights["realism"]
            + brand_alignment * weights["brand_alignment"]
            + product_visibility * weights["product_visibility"]
            + motion_quality * weights["motion_quality"]
        )

        overall = self._clamp(overall)

        return QualityScore(
            realism=realism,
            brand_alignment=brand_alignment,
            product_visibility=product_visibility,
            motion_quality=motion_quality,
            overall=overall,
            notes=notes,
        )