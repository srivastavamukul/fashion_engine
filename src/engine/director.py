import json
import logging
import random

from src.core.models import (BrandIdentity, GenerationMode, Guardrails, Intent,
                             IntentMeta, SceneRules, Shot, TechnicalSpecs)

logger = logging.getLogger("FashionEngine")


class CreativeDirector:
    FALLBACK_RULES = {
        "poses": ["Static product shot"],
        "environments": ["Neutral studio background"],
        "camera": ["Static medium shot"],
        "focus_points": ["Overall product visibility"],
        "lighting_logic": "Balanced studio lighting",
    }

    def __init__(self, brand_path, rules_path, strict_mode=True):
        self.strict = strict_mode
        self.brand_dna = self._load_json(brand_path)
        self.category_rules = self._load_json(rules_path)
        self._validate_brand_dna()

    def _load_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"Strict Mode Violation: Cannot load {path} | {e}")
            return {}

    def _validate_brand_dna(self):
        required = [
            "brand_name",
            "core_aesthetic",
            "visual_specifications",
            "guardrails",
        ]
        for key in required:
            if key not in self.brand_dna and self.strict:
                raise ValueError(f"❌ Malformed Brand DNA: Missing '{key}'")

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def build_product_intent(
        self,
        product_name: str,
        category: str,
        key_features: list[str] | None = None,
        image_count: int = 1,
        images: list[str] | None = None,
    ) -> Intent:

        cat_key = category.lower()

        # 1️⃣ Resolve category rules
        if cat_key in self.category_rules:
            raw_rules = self.category_rules[cat_key]
        elif "default" in self.category_rules:
            logger.warning(f"⚠️ Category '{category}' not found. Using default.")
            raw_rules = self.category_rules["default"]
        else:
            raw_rules = self.FALLBACK_RULES

        # 2️⃣ Meta
        meta = IntentMeta(
            product_name=product_name,
            category=category,
            reference_image_count=len(images) if images else image_count,
            reference_image_paths=images or [],
            key_features=key_features or [],
        )

        # 3️⃣ Brand identity
        aesthetic = self.brand_dna["core_aesthetic"]
        brand = BrandIdentity(
            name=self.brand_dna["brand_name"],
            vibe=aesthetic["vibe"],
            tone=aesthetic["emotional_tone"],
            palette=aesthetic["color_palette"],
        )

        # 4️⃣ Technical specs
        visual = self.brand_dna["visual_specifications"]
        tech = TechnicalSpecs(
            lighting_global=visual["lighting"],
            camera_global=visual["camera_cinematography"],
            materials=visual["textures_and_materials"],
            lighting_logic=raw_rules.get("lighting_logic", "Balanced studio lighting"),
        )

        # 5️⃣ Scene rules
        scene = SceneRules(
            poses=raw_rules.get("poses", self.FALLBACK_RULES["poses"]),
            environments=raw_rules.get(
                "environments", self.FALLBACK_RULES["environments"]
            ),
            camera_actions=raw_rules.get("camera", self.FALLBACK_RULES["camera"]),
            focus_points=raw_rules.get(
                "focus_points", self.FALLBACK_RULES["focus_points"]
            ),
        )

        # 6️⃣ Guardrails
        guard = Guardrails(
            avoid=list(
                set(raw_rules.get("avoid", []) + self.brand_dna["guardrails"]["avoid"])
            ),
            strict_rule=self.brand_dna["guardrails"]["strict_rule"],
        )

        # 7️⃣ FULL INTENT (THIS WAS MISSING)
        return Intent(
            meta=meta,
            brand_identity=brand,
            technical_specs=tech,
            scene_rules=scene,
            guardrails=guard,
        )

    def generate_shots(
        self, intent: Intent, count=3, mode=GenerationMode.STRICT
    ) -> list[Shot]:
        shots: list[Shot] = []
        poses = intent.scene_rules.poses or self.FALLBACK_RULES["poses"]
        envs = intent.scene_rules.environments or self.FALLBACK_RULES["environments"]
        cams = intent.scene_rules.camera_actions or self.FALLBACK_RULES["camera"]
        focuses = intent.scene_rules.focus_points or self.FALLBACK_RULES["focus_points"]

        for i in range(max(1, count)):
            pose = random.choice(poses)
            env = random.choice(envs)
            cam = random.choice(cams)
            # pick up to 2 focus points
            focus = (
                random.sample(focuses, min(2, len(focuses)))
                if focuses
                else ["Overall product visibility"]
            )

            shots.append(
                Shot(
                    id=i + 1,
                    pose=pose,
                    environment=env,
                    camera_action=cam,
                    focus_points=focus,
                )
            )

        return shots
