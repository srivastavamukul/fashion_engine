from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

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
    strict_rule: str

@dataclass(frozen=True)
class IntentMeta:
    product_name: str
    category: str
    reference_image_count: int
    key_features: List[str]

@dataclass(frozen=True)
class Intent:
    meta: IntentMeta
    brand_identity: BrandIdentity
    technical_specs: TechnicalSpecs
    scene_rules: SceneRules
    guardrails: Guardrails

    def get_hash(self) -> str:
        import json
        import hashlib
        from dataclasses import asdict
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
    file_path: str
    video_id: str
    seed: int
    duration: float
    model_used: str
    prompt: str
    prompt_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class QualityScore:
    realism: float
    brand_alignment: float
    product_visibility: float
    motion_quality: float
    overall: float
    notes: List[str] = field(default_factory=list)