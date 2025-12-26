from src.core.models import Intent, Shot
import logging

logger = logging.getLogger("FashionEngine")

class PromptAdapter:
    """
    Abstract base class for converting internal Intent/Shot models into
    model-specific text prompts.
    """
    def format(self, intent: Intent, shot: Shot) -> str:
        """
        Formats the intent and shot into a string prompt.
        
        Args:
            intent (Intent): The high-level product intent.
            shot (Shot): The specific shot configuration.
            
        Returns:
            str: The formatted prompt.
        """
        raise NotImplementedError

class RunwayAdapter(PromptAdapter):
    """
    Adapter optimized for RunwayML-style prompts (Gen-2/Gen-3).
    Focuses on clear structure, camera movement descriptions, and aesthetic cues.
    """
    def format(self, intent: Intent, shot: Shot) -> str:
        features_text = ", ".join(intent.meta.key_features) or "Standard details"
        palette_text = ", ".join(intent.brand_identity.palette)
        avoid_text = ", ".join(intent.guardrails.avoid)
        focus_text = ", ".join(shot.focus_points)

        # Logic check for conflicting tones
        if "dark" in intent.brand_identity.tone.lower() and "bright" in shot.environment.lower():
            logger.warning(f"⚠️ Logical Conflict: 'Dark' tone vs 'Bright' environment")

        return f"""
MODE: STRICT EXECUTION ONLY.
🛑 HARD CONSTRAINTS: {intent.guardrails.strict_rule}
AVOID: {avoid_text}
📸 SOURCE OF TRUTH: Preserve {features_text}.
🎬 SCENE: {intent.meta.product_name}, {shot.pose}, {shot.environment}, {shot.camera_action}
🎨 VIBE: {intent.brand_identity.vibe}, {palette_text}
💡 TECH: {intent.technical_specs.lighting_logic}, 50mm Lens
🎯 FOCUS: {focus_text}
""".strip()