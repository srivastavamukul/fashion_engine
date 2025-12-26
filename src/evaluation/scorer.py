from dataclasses import dataclass, field
from typing import Dict, List

from src.core.models import Intent, QualityScore, Shot, VideoArtifact


@dataclass(frozen=True)
class ScoringConfig:
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "brand_alignment": 0.25,
            "product_visibility": 0.20,
            "visual_consistency": 0.25,
            "motion_quality": 0.20,
            # Total: 0.9 + base bias? Re-normalized weights should sum to ~1.0 ideally
            # Updated to: 0.25, 0.25, 0.20, 0.25, 0.05 (reserved)? 
            # Let's keep it simple: 0.25 * 4 = 1.0
        }
    )
    # Removing defaults from lambda to ensure clarity in diff, but relying on user knowing python
    # Cleaner defaults:
    # realism: 0.20, brand: 0.20, visibility: 0.20, motion: 0.20, consistency: 0.20
    max_acceptable_latency: float = 1.5
    min_acceptable_duration: float = 3.8


class QualityEvaluator:
    def evaluate(self, artifact, intent, shot, full_prompt, reference_images=None) -> QualityScore:
        raise NotImplementedError

    def select_top_videos(self, scored_videos, top_k=1):
        return sorted(scored_videos, key=lambda x: x[1].overall, reverse=True)[:top_k]


class MockQualityEvaluator(QualityEvaluator):
    def __init__(self, config: ScoringConfig = ScoringConfig()):
        self.config = config

    def _clamp(self, value, min_val=0.0, max_val=10.0):
        return max(min_val, min(value, max_val))

    def evaluate(
        self, artifact: VideoArtifact, intent: Intent, shot: Shot, full_prompt: str, reference_images: List[str] = None
    ) -> QualityScore:

        notes: List[str] = []

        # =========================
        # REALISM
        # =========================
        realism = 8.0
        latency = artifact.metadata.get("simulated_latency", 0.0)

        if latency > self.config.max_acceptable_latency:
            penalty = (latency - self.config.max_acceptable_latency) * 1.5
            realism -= penalty
            notes.append("Slight realism drop due to generation instability")
        else:
            notes.append("Stable and realistic motion")

        realism = self._clamp(realism)

        # =========================
        # BRAND ALIGNMENT
        # =========================
        brand_alignment = 7.5
        prompt_lower = full_prompt.lower()

        brand_name = intent.brand_identity.name.lower()
        if brand_name in prompt_lower:
            brand_alignment += 1.0
            notes.append("Brand name explicitly reinforced")

        vibe = intent.brand_identity.vibe.lower()
        if vibe in prompt_lower:
            brand_alignment += 0.5
            notes.append("Brand vibe reflected in scene description")

        palette_hits = sum(
            1 for c in intent.brand_identity.palette if c.lower() in prompt_lower
        )
        if palette_hits > 0:
            brand_alignment += 0.3
            notes.append("Color palette aligned with brand identity")

        brand_alignment = self._clamp(brand_alignment)

        # =========================
        # PRODUCT VISIBILITY
        # =========================
        product_visibility = 7.5

        if any("overall" in f.lower() for f in shot.focus_points):
            product_visibility += 0.8
            notes.append("Clear overall product visibility")

        if any("logo" in f.lower() for f in shot.focus_points):
            product_visibility += 0.7
            notes.append("Logo visibility emphasized")

        if len(shot.focus_points) >= 3:
            product_visibility += 0.5
            notes.append("Multiple product features highlighted")

        product_visibility = self._clamp(product_visibility)

        # =========================
        # VISUAL CONSISTENCY
        # =========================
        visual_consistency = 6.0  # Base score
        input_images = getattr(intent.meta, "reference_image_paths", []) or reference_images or []
        
        if input_images:
            # Mock logic: Check if files "exist" (strings not empty)
            if len(input_images) > 0:
                visual_consistency += 2.0
                notes.append(f"Visual consistency maintained with {len(input_images)} reference inputs")
            if "front" in str(input_images).lower():
                 visual_consistency += 1.0
                 notes.append("Strong frontal alignment with reference")
        else:
            notes.append("No reference images provided for consistency check")

        visual_consistency = self._clamp(visual_consistency)

        # =========================
        # MOTION QUALITY
        # =========================
        motion_quality = 7.5
        duration = getattr(artifact, "duration", self.config.min_acceptable_duration)

        if duration >= self.config.min_acceptable_duration:
            motion_quality += 0.8
            notes.append("Smooth cinematic pacing")
        else:
            motion_quality -= 1.5
            notes.append("Motion felt slightly rushed")

        if "slow" in shot.camera_action.lower():
            motion_quality += 0.5
            notes.append("Camera movement felt calm and controlled")

        motion_quality = self._clamp(motion_quality)

        # =========================
        # GUARDRAIL VIOLATION CHECK
        # =========================
        avoid_list = getattr(intent.guardrails, "avoid", [])
        
        # Naive check: Ignore the "AVOID:" line in the prompt to prevent false positives
        prompt_lines = [line for line in prompt_lower.split('\n') if not line.strip().startswith('avoid:')]
        clean_prompt = "\n".join(prompt_lines)

        for banned in avoid_list:
            if banned.lower() in clean_prompt:
                notes.append(f"Guardrail violation detected: {banned}")
                return QualityScore(
                    realism=0.0,
                    brand_alignment=0.0,
                    product_visibility=0.0,
                    visual_consistency=0.0,
                    motion_quality=0.0,
                    overall=0.0,
                    notes=notes,
                )

        # =========================
        # FINAL WEIGHTED SCORE
        # =========================
        weights = self.config.weights
        overall = (
            realism * weights.get("realism", 0.2)
            + brand_alignment * weights.get("brand_alignment", 0.2)
            + product_visibility * weights.get("product_visibility", 0.2)
            + motion_quality * weights.get("motion_quality", 0.2)
            + visual_consistency * weights.get("visual_consistency", 0.2)
        )

        overall = round(self._clamp(overall), 2)

        return QualityScore(
            realism=round(realism, 2),
            brand_alignment=round(brand_alignment, 2),
            product_visibility=round(product_visibility, 2),
            visual_consistency=round(visual_consistency, 2),
            motion_quality=round(motion_quality, 2),
            overall=overall,
            notes=notes,
        )
