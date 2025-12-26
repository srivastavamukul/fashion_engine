import json
import os
import logging
import hashlib
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from functools import wraps
import math

# ==========================================
# ⚙️ CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FashionEngine")

BRAND_PATH = os.getenv("BRAND_PROFILE", "brand_profile.json")
RULES_PATH = os.getenv("CATEGORY_RULES", "category_rules.json")

# ==========================================
# 🛠️ HELPER DECORATORS
# ==========================================

def smart_retry(retries=3, delay=1, backoff=2):
    """
    Elite Retry Logic:
    - Retries ConnectionError (Transient)
    - Fails fast on ValueError (Hard Logic Error)
    - Fails fast on RuntimeError (Configuration Error)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                
                # 🟡 CASE 1: RETRYABLE ERRORS (Network, Timeout, 503)
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(f"⚠️ [Retry {i+1}/{retries}] Transient error: {e}. Waiting {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff  # Exponential backoff
                
                # 🔴 CASE 2: FATAL ERRORS (Auth, Bad Request, 400)
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.critical(f"🛑 Non-retryable failure detected: {e}")
                    raise e  # Fail immediately
                
                # 🟠 CASE 3: UNEXPECTED ERRORS
                except Exception as e:
                    logger.error(f"⚠️ Unexpected error: {e}. Retrying just in case...")
                    time.sleep(current_delay)
            
            logger.error(f"❌ Operation failed after {retries} attempts.")
            raise ConnectionError("Max retries exceeded for transient failure")
        return wrapper
    return decorator

# ==========================================
# 🛡️ TYPED DATA MODELS
# ==========================================

class GenerationMode(Enum):
    STRICT = "strict"
    CREATIVE = "creative"

class GenerationOutcome(Enum):
    SUCCESS = "success"
    TRANSIENT_FAILURE = "transient_failure"
    HARD_FAILURE = "hard_failure"
    RATE_LIMIT = "rate_limit"

@dataclass(frozen=True)
class ModelProfile:
    """Defines the capabilities and limits of a video AI model."""
    name: str
    max_tokens: int
    supports_seed: bool
    supports_negative_prompt: bool

@dataclass(frozen=True)
class BrandIdentity:
    name: str
    vibe: str
    tone: str
    palette: List[str]

@dataclass(frozen=True)
class TechnicalSpecs:
    lighting_global: Dict[str, str]
    camera_global: Dict[str, str]
    materials: Dict[str, str]
    lighting_logic: str

@dataclass(frozen=True)
class SceneRules:
    poses: List[str]
    environments: List[str]
    camera_actions: List[str]
    focus_points: List[str]

@dataclass(frozen=True)
class Guardrails:
    avoid: List[str]
    strict_rule: str  # HARD constraint

@dataclass(frozen=True)
class IntentMeta:
    product_name: str
    category: str
    reference_image_count: int
    key_features: List[str]

@dataclass(frozen=True)
class Intent:
    """The Immutable Master Plan."""
    meta: IntentMeta
    brand_identity: BrandIdentity
    technical_specs: TechnicalSpecs
    scene_rules: SceneRules
    guardrails: Guardrails

    def get_hash(self) -> str:
        """Cryptographically strong hash of the full intent state."""
        json_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]

@dataclass(frozen=True)
class Shot:
    id: int
    pose: str
    environment: str
    camera_action: str
    focus_points: List[str]
    lighting_override: Optional[str] = None

@dataclass(frozen=True)
class VideoArtifact:
    """Standardized output from any video generator."""
    file_path: str
    video_id: str
    seed: int
    duration: float
    model_used: str
    prompt: str      # Full text for traceability
    prompt_id: str   # Hash for deduplication
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class QualityScore:
    realism: float
    brand_alignment: float
    product_visibility: float
    motion_quality: float
    overall: float
    notes: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class ScoringConfig:
    """Central control for all scoring weights and penalties."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "realism": 0.30,
        "brand_alignment": 0.30,
        "product_visibility": 0.20,
        "motion_quality": 0.20
    })
    
    # Thresholds
    max_acceptable_latency: float = 1.5
    min_acceptable_duration: float = 3.8
    optimal_duration: float = 4.0
    
    # Penalties
    latency_penalty_factor: float = 0.5  # Points lost per second of delay
    duration_penalty: float = 1.5        # Flat penalty for short videos
    complexity_penalty: float = 0.7      # Penalty for cluttered scenes
    guardrail_violation: float = 9.0     # Critical failure penalty (nukes score)

# ==========================================
# 🔌 PROMPT ADAPTERS
# ==========================================

class PromptAdapter:
    def format(self, intent: Intent, shot: Shot) -> str:
        raise NotImplementedError

class RunwayAdapter(PromptAdapter):
    """Optimized for Runway Gen-3: Imperative instructions, constraints first."""
    
    def format(self, intent: Intent, shot: Shot) -> str:
        features_text = ", ".join(intent.meta.key_features) or "Standard details"
        palette_text = ", ".join(intent.brand_identity.palette)
        avoid_text = ", ".join(intent.guardrails.avoid)
        focus_text = ", ".join(shot.focus_points)

        # Semantic Conflict Check
        if "dark" in intent.brand_identity.tone.lower() and "bright" in shot.environment.lower():
            logger.warning(f"⚠️ Logical Conflict in Shot {shot.id}: 'Dark' tone vs 'Bright' environment")

        template = f"""
MODE: STRICT EXECUTION ONLY.
Do not add creative elements.
Do not reinterpret instructions.

🛑 HARD CONSTRAINTS (MUST FOLLOW):
{intent.guardrails.strict_rule}
AVOID AT ALL COSTS: {avoid_text}

📸 SOURCE OF TRUTH:
The {intent.meta.reference_image_count} attached reference images are the ONLY source of truth.
Preserve exact product details: {features_text}.

🎬 SCENE SPECIFICATION:
Product: {intent.meta.product_name} ({intent.meta.category})
Action: {shot.pose}
Environment: {shot.environment}
Camera: {shot.camera_action}

🎨 AESTHETIC PROFILE:
Brand: {intent.brand_identity.name}
Vibe: {intent.brand_identity.vibe}
Tone: {intent.brand_identity.tone}
Palette: {palette_text}

💡 TECHNICAL EXECUTION:
Lighting: {intent.technical_specs.lighting_logic} ({intent.technical_specs.lighting_global.get('style', 'Natural')})
Lens: {intent.technical_specs.camera_global.get('focal_length', '50mm')}
Material Fidelity: {intent.technical_specs.materials.get('canvas', 'High fidelity')}

🎯 FOCUS:
{focus_text}
"""
        return template.strip()

# ==========================================
# 🎥 ADVANCED VIDEO GENERATOR (CHAOS ENGINE)
# ==========================================

class VideoGenerator:
    """Abstract interface for video generation engines."""
    
    profile: ModelProfile

    def generate(
        self, 
        prompt: str, 
        reference_paths: List[str], 
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None
    ) -> VideoArtifact:
        raise NotImplementedError

class AdvancedMockVideoGenerator(VideoGenerator):
    """
    Enterprise-Grade Mock Generator.
    Simulates:
    - Network Jitter (Log-Normal Latency)
    - Hard vs Transient Failures
    - Rate Limiting (Token Bucket)
    - Metadata Drift
    """

    OUTPUT_DIR = "outputs"

    profile = ModelProfile(
        name="Gen-Mock-Elite-v3",
        max_tokens=4096,
        supports_seed=True,
        supports_negative_prompt=True
    )

    def __init__(
        self,
        failure_rate: float = 0.15,      # 15% chance of any error
        hard_failure_ratio: float = 0.3, # 30% of errors are hard failures
        min_latency: float = 0.5,
        max_latency: float = 2.0,
        calls_per_minute: int = 60       # Rate limit simulation
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
            f"🔌 [MOCK] Advanced Generator Active | "
            f"Profile: {self.profile.name} | "
            f"Error Rate: {int(failure_rate*100)}%"
        )

    def _check_rate_limit(self):
        """Simulates 429 Too Many Requests."""
        now = time.time()
        # Remove calls older than 60 seconds
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        
        if len(self._call_timestamps) >= self._rate_limit:
            return True
        
        self._call_timestamps.append(now)
        return False

    def _decide_outcome(self, seed: Optional[int]) -> GenerationOutcome:
        """Determines the fate of the generation request."""
        
        # 🧙‍♂️ MAGIC SEEDS for Deterministic Testing
        if seed == 9999: return GenerationOutcome.HARD_FAILURE
        if seed == 8888: return GenerationOutcome.TRANSIENT_FAILURE
        if seed == 7777: return GenerationOutcome.RATE_LIMIT

        # 🎲 Random Chaos
        r = random.random()
        
        if r < self.failure_rate:
            # It's a failure. Is it Hard or Transient?
            if random.random() < self.hard_failure_ratio:
                return GenerationOutcome.HARD_FAILURE
            return GenerationOutcome.TRANSIENT_FAILURE
            
        return GenerationOutcome.SUCCESS

    @smart_retry(retries=3, delay=1.5)
    def generate(
        self,
        prompt: str,
        reference_paths: List[str],
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None
    ) -> VideoArtifact:

        # 1. Check Rate Limit
        if self._check_rate_limit():
            logger.warning("🔥 [MOCK] 429 Too Many Requests (Simulated)")
            raise ConnectionError("Rate limit exceeded (429)")

        # 2. Simulate Network Latency (Jitter)
        # Using triangular distribution for more realistic "tail latency"
        latency = random.triangular(self.min_latency, self.max_latency, self.min_latency + 0.5)
        time.sleep(latency)

        # 3. Determine Outcome
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

        # 4. Success Path: Generation
        video_seed = seed or random.randint(100000, 999999)
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        video_id = hashlib.md5(f"{prompt_id}{video_seed}".encode()).hexdigest()[:8]
        
        filename = f"mock_gen_{video_id}.mp4"
        output_path = os.path.join(self.OUTPUT_DIR, filename)

        # Simulate varied file sizes (Network transfer time simulation)
        file_size_mb = random.uniform(1.5, 5.0)
        with open(output_path, "wb") as f:
            f.write(b"\0" * int(file_size_mb * 1024 * 1024))

        # Metadata that might "drift" from request (Real models do this!)
        # e.g., you ask for 5.0s, model gives 4.8s
        actual_duration = round(random.uniform(3.8, 4.2), 2)
        
        logger.info(f"✅ [MOCK] Generated {filename} ({file_size_mb:.2f}MB) in {latency:.2f}s")

        return VideoArtifact(
            file_path=output_path,
            video_id=video_id,
            seed=video_seed,
            duration=actual_duration, 
            model_used=self.profile.name,
            prompt=prompt, # Store full prompt for traceability
            prompt_id=prompt_id,
            metadata={
                "aspect_ratio": aspect_ratio,
                "simulated_latency": round(latency, 3),
                "server_node": f"gpu-cluster-{random.randint(1,5)}"
            }
        )

# ==========================================
# 🧪 QUALITY EVALUATION
# ==========================================

class QualityEvaluator:
    def evaluate(self, artifact: VideoArtifact, intent: Intent, shot: Shot, full_prompt: str) -> QualityScore:
        raise NotImplementedError

    def select_top_videos(self, scored_videos: List[tuple], top_k: int = 1) -> List[tuple]:
        return sorted(scored_videos, key=lambda x: x[1].overall, reverse=True)[:top_k]


class MockQualityEvaluator(QualityEvaluator):
    """
    Elite Deterministic Evaluator.
    Features:
    - Configurable weights via Dependency Injection
    - Curve-based scoring (not just flat penalties)
    - Aspect Ratio & Resolution verification
    - Guardrail 'Nuke' logic (critical failures kill the score)
    """

    def __init__(self, config: ScoringConfig = ScoringConfig()):
        self.config = config

    def _clamp(self, value: float, min_val=0.0, max_val=10.0) -> float:
        """Helper to keep scores within logical bounds."""
        return max(min_val, min(value, max_val))

    def evaluate(
        self,
        artifact: VideoArtifact,
        intent: Intent,
        shot: Shot,
        full_prompt: str
    ) -> QualityScore:

        notes = []
        
        # ─────────────────────────────
        # 1️⃣ REALISM SCORE (Physics & Stability)
        # ─────────────────────────────
        realism = 9.5  # Start high for Mock

        # Advanced Latency Curve: The higher the latency, the harsher the penalty
        latency = artifact.metadata.get("simulated_latency", 0)
        if latency > self.config.max_acceptable_latency:
            # Calculate linear decay based on how much we exceeded the limit
            excess = latency - self.config.max_acceptable_latency
            penalty = excess * self.config.latency_penalty_factor
            realism -= penalty
            notes.append(f"High latency ({latency:.2f}s) suggests render struggle")

        # Duration Integrity Check
        if artifact.duration < self.config.min_acceptable_duration:
            realism -= self.config.duration_penalty
            notes.append(f"Duration failure: {artifact.duration}s < {self.config.min_acceptable_duration}s")

        realism = self._clamp(realism)

        # ─────────────────────────────
        # 2️⃣ BRAND ALIGNMENT SCORE (Identity & Safety)
        # ─────────────────────────────
        brand_alignment = 9.0

        brand_name = intent.brand_identity.name.lower()
        vibe = intent.brand_identity.vibe.lower()
        prompt_lower = full_prompt.lower()

        # Positive Reinforcement
        if brand_name in prompt_lower:
            brand_alignment += 0.5
            notes.append("✅ Brand name anchored in prompt")
        
        if vibe in prompt_lower:
            brand_alignment += 0.3
            notes.append("✅ Vibe keywords present")

        # CRITICAL GUARDRAIL CHECK ("The Nuke")
        # If a forbidden term appears, this score should collapse to near zero.
        for forbidden in intent.guardrails.avoid:
            if forbidden.lower() in prompt_lower:
                brand_alignment -= self.config.guardrail_violation
                notes.append(f"⛔ CRITICAL: Guardrail violation '{forbidden}' detected")
                break # Stop checking, damage is done

        brand_alignment = self._clamp(brand_alignment)

        # ─────────────────────────────
        # 3️⃣ PRODUCT VISIBILITY SCORE (Composition)
        # ─────────────────────────────
        product_visibility = 8.5

        # Complexity Analysis
        focus_count = len(shot.focus_points)
        if focus_count > 3:
            product_visibility -= self.config.complexity_penalty
            notes.append(f"Scene clutter risk: {focus_count} focus points")

        # Aspect Ratio Verification
        # If metadata aspect ratio doesn't match a standard vertical/horizontal logic
        meta_ar = artifact.metadata.get("aspect_ratio", "16:9")
        if meta_ar == "9:16" and "cinematic" in vibe:
             # Just a heuristic example: Vertical video for cinematic vibe might cut off product
             product_visibility -= 0.5
             notes.append("Aspect ratio mismatch for cinematic composition")

        # Logo Priority Check
        key_features_str = " ".join(intent.meta.key_features).lower()
        if "logo" in key_features_str:
            product_visibility += 0.5
            notes.append("✅ Logo priority active")

        product_visibility = self._clamp(product_visibility)

        # ─────────────────────────────
        # 4️⃣ MOTION QUALITY SCORE (Fluidity)
        # ─────────────────────────────
        motion_quality = 9.0

        # Motion Duration Ratio
        # A video that is exactly the requested length is likely smoother than one that was cut short
        duration_ratio = artifact.duration / self.config.optimal_duration
        if duration_ratio < 0.8:
            motion_quality -= 1.5
            notes.append("Choppy playback risk (duration too short)")
        
        # Latency correlation to jitter (Simulating GPU overload)
        if latency > 2.0:
            motion_quality -= 1.0
            notes.append("Possible frame jitter (high render load)")

        motion_quality = self._clamp(motion_quality)

        # ─────────────────────────────
        # 5️⃣ WEIGHTED AGGREGATION
        # ─────────────────────────────
        w = self.config.weights
        
        weighted_sum = (
            realism * w["realism"] +
            brand_alignment * w["brand_alignment"] +
            product_visibility * w["product_visibility"] +
            motion_quality * w["motion_quality"]
        )
        
        # Final sanity check: if any individual score is < 3.0, caps overall at 5.0
        # (A video cannot be "Great" if it fails purely on one axis)
        min_individual = min(realism, brand_alignment, product_visibility, motion_quality)
        if min_individual < 3.0:
            weighted_sum = min(weighted_sum, 5.0)
            notes.append("⚠️ Overall score capped due to critical failure in one metric")

        return QualityScore(
            realism=round(realism, 2),
            brand_alignment=round(brand_alignment, 2),
            product_visibility=round(product_visibility, 2),
            motion_quality=round(motion_quality, 2),
            overall=round(weighted_sum, 2),
            notes=notes
        )
    
# ==========================================
# 🧠 CREATIVE DIRECTOR
# ==========================================

class CreativeDirector:
    FALLBACK_RULES = {
        "poses": ["Static product shot"],
        "environments": ["Neutral studio background"],
        "camera": ["Static medium shot"],
        "focus_points": ["Overall product visibility"],
        "lighting_logic": "Balanced studio lighting"
    }

    def __init__(self, strict_mode=True):
        self.strict = strict_mode
        self.brand_dna = self._load_json(BRAND_PATH)
        self.category_rules = self._load_json(RULES_PATH)
        self._validate_brand_dna()
        
    def _load_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            if self.strict:
                logger.critical(f"❌ Critical Config Failure: {path} | Error: {e}")
                raise RuntimeError(f"Strict Mode Violation: Cannot load {path}")
            logger.warning(f"⚠️ Load failed for {path}. Proceeding with EMPTY/FALLBACK.")
            return {}

    def _validate_brand_dna(self):
        required = ["brand_name", "core_aesthetic", "visual_specifications", "guardrails"]
        if not self.brand_dna: return
        for key in required:
            if key not in self.brand_dna:
                if self.strict: 
                    raise ValueError(f"❌ Malformed Brand DNA: Missing '{key}'")

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4 

    def build_product_intent(self, product_name, category, key_features=None, image_count=1) -> Intent:
        cat_key = category.lower()
        
        if cat_key in self.category_rules:
            raw_rules = self.category_rules[cat_key]
        elif "default" in self.category_rules:
            logger.warning(f"⚠️ Category '{category}' not found. Using 'default'.")
            raw_rules = self.category_rules["default"]
        else:
            logger.error(f"❌ Category '{category}' AND 'default' missing. Using FALLBACK.")
            raw_rules = self.FALLBACK_RULES

        meta = IntentMeta(
            product_name=product_name,
            category=category,
            reference_image_count=image_count,
            key_features=key_features or []
        )

        brand_name = self.brand_dna.get("brand_name", "Unknown Brand")
        aes = self.brand_dna.get("core_aesthetic", {})
        
        brand = BrandIdentity(
            name=brand_name,
            vibe=aes.get("vibe", "Modern"),
            tone=aes.get("emotional_tone", "Neutral"),
            palette=aes.get("color_palette", ["Black", "White"])
        )

        vis = self.brand_dna.get("visual_specifications", {})
        tech = TechnicalSpecs(
            lighting_global=vis.get("lighting", {}),
            camera_global=vis.get("camera_cinematography", {}),
            materials=vis.get("textures_and_materials", {}),
            lighting_logic=raw_rules.get("lighting_logic", "Balanced studio lighting")
        )

        scene = SceneRules(
            poses=raw_rules.get("poses", self.FALLBACK_RULES["poses"]),
            environments=raw_rules.get("environments", self.FALLBACK_RULES["environments"]),
            camera_actions=raw_rules.get("camera", self.FALLBACK_RULES["camera"]),
            focus_points=raw_rules.get("focus_points", self.FALLBACK_RULES["focus_points"])
        )

        cat_avoid = raw_rules.get("avoid", [])
        brand_avoid = self.brand_dna.get("guardrails", {}).get("avoid", [])
        
        guard = Guardrails(
            avoid=list(set(cat_avoid + brand_avoid)),
            strict_rule=self.brand_dna.get("guardrails", {}).get("strict_rule", "Do not alter product.")
        )

        return Intent(meta, brand, tech, scene, guard)

    def generate_shots(self, intent: Intent, count=3, mode=GenerationMode.STRICT) -> List[Shot]:
        shots = []
        rules = intent.scene_rules
        used_combinations = set()
        
        attempts = 0
        max_attempts = count * 5

        while len(shots) < count and attempts < max_attempts:
            i = attempts
            
            if mode == GenerationMode.STRICT:
                pose = rules.poses[i % len(rules.poses)]
                env = rules.environments[i % len(rules.environments)]
                cam = rules.camera_actions[i % len(rules.camera_actions)]
            else:
                pose = random.choice(rules.poses)
                env = random.choice(rules.environments)
                cam = random.choice(rules.camera_actions)

            combo_key = (pose, env, cam)
            
            if combo_key in used_combinations:
                attempts += 1
                continue
            
            used_combinations.add(combo_key)
            
            shot = Shot(
                id=len(shots) + 1,
                pose=pose,
                environment=env,
                camera_action=cam,
                focus_points=rules.focus_points
            )
            shots.append(shot)
            attempts += 1
            
        if len(shots) < count:
            logger.warning(f"⚠️ Could only generate {len(shots)} unique variations.")

        return shots

# ==========================================
# 🎼 PIPELINE ORCHESTRATOR
# ==========================================

class FashionPipeline:
    def __init__(self, 
                 generator: Optional[VideoGenerator] = None, 
                 evaluator: Optional[QualityEvaluator] = None, 
                 output_root="outputs"):
        
        self.director = CreativeDirector(strict_mode=True)
        self.adapter = RunwayAdapter()
        
        # Dependency Injection with Advanced Mock as Default
        if generator:
            self.generator = generator
        else:
            self.generator = AdvancedMockVideoGenerator(
                failure_rate=0.10,       # 10% global error rate
                hard_failure_ratio=0.20, # 20% of errors are fatal
                calls_per_minute=60      # 60 calls/min rate limit
            )
            
        self.evaluator = evaluator if evaluator else MockQualityEvaluator()
        
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(output_root, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        if hasattr(self.generator, 'OUTPUT_DIR'):
            self.generator.OUTPUT_DIR = self.run_dir

    def _validate_inputs(self, images: List[str]):
        """Fail fast if input images do not exist."""
        for img_path in images:
            if not os.path.exists(img_path) and not img_path.startswith("http"): 
                logger.warning(f"⚠️ Input image not found: {img_path} (Simulation will proceed)")

    def run(self, product_name: str, category: str, features: List[str], images: List[str]):
        logger.info(f"🚀 Starting Pipeline Run for: {product_name}")
        
        self._validate_inputs(images)

        # 1. Build Intent
        intent = self.director.build_product_intent(
            product_name=product_name,
            category=category,
            key_features=features,
            image_count=len(images)
        )
        logger.info(f"🔑 Intent Hash: {intent.get_hash()}")

        # 2. Generate Shots
        shots = self.director.generate_shots(intent, count=3, mode=GenerationMode.STRICT)
        
        # 3. Execution Loop
        results = []
        prompts_map = {} 

        for shot in shots:
            prompt = self.adapter.format(intent, shot)
            
            # Token Check
            tokens = self.director.estimate_tokens(prompt)
            max_tokens = self.generator.profile.max_tokens
            if tokens > max_tokens:
                logger.warning(f"⚠️ Prompt {shot.id} exceeds token limit ({tokens}/{max_tokens})")

            # Generate Video
            artifact = self.generator.generate(
                prompt=prompt,
                reference_paths=images,
                seed=42
            )
            
            prompts_map[artifact.prompt_id] = prompt

            # Evaluate Video
            quality = self.evaluator.evaluate(artifact, intent, shot, prompt)
            
            results.append({
                "shot": shot,
                "artifact": artifact,
                "score": quality
            })

        # 4. Selection & Manifest
        best_videos = self.evaluator.select_top_videos(
            [(r["artifact"], r["score"], r["shot"]) for r in results], 
            top_k=1
        )

        self._save_manifest(intent, results, prompts_map)
        
        return best_videos

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
                } for r in results
            ],
            "prompt_library": prompts_map 
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"✅ Run Complete. Data saved to {self.run_dir}")

# ==========================================
# 🏁 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Setup Engine
    engine = FashionPipeline(output_root="my_campaign_runs")

    # 2. Inputs
    product_inputs = {
        "product_name": "Resilience Hoodie",
        "category": "hoodie",
        "features": [
            "Text: 'I AM STRONGER THAN THE FLAMES'", 
            "Backprint visible", 
            "Logo on chest"
        ],
        "images": ["uploads/front.jpg", "uploads/back.jpg", "uploads/detail.jpg"]
    }

    # 3. Run
    try:
        top_picks = engine.run(
            product_name=product_inputs["product_name"],
            category=product_inputs["category"],
            features=product_inputs["features"],
            images=product_inputs["images"]
        )

        print("\n🏆 WINNING VIDEOS:")
        for artifact, score, shot in top_picks:
            print(f"VIDEO ID: {artifact.video_id} | SCORE: {score.overall} | PATH: {artifact.file_path}")
            
    except Exception as e:
        logger.critical(f"🔥 Pipeline crashed: {e}")