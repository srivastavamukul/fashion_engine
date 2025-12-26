import os
from typing import Optional
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # Pipeline Limits
    target_videos: int = Field(default=10, description="Number of accepted videos to generate")
    quality_threshold: float = Field(default=7.0, description="Minimum score for acceptance")
    max_attempts: int = Field(default=50, description="Maximum total attempts")
    concurrency: int = Field(default=5, description="Concurrent generation tasks")

    # Paths
    output_root: str = Field(default="campaign_runs", description="Root directory for outputs")
    brand_profile_path: str = Field(default="config/brand_profile.json", description="Path to brand profile")
    category_rules_path: str = Field(default="config/category_rules.json", description="Path to category rules")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # API
    api_host: str = Field(default="0.0.0.0", description="API Host")
    api_port: int = Field(default=8000, description="API Port")
    api_key: str = Field(default="dev-secret-key", description="API Key for Admin Access")

    # Generators
    enable_runway: bool = True
    enable_stability: bool = False
    enable_luma: bool = False
    
    # Evaluation
    enable_vision_evaluator: bool = False
    openai_api_key: Optional[str] = Field(default=None)
    vllm_api_url: Optional[str] = Field(default=None)

    runway_api_secret: Optional[str] = Field(default=None)
    stability_api_key: Optional[str] = Field(default=None)
    luma_api_key: Optional[str] = Field(default=None)

    @staticmethod
    def load() -> "Settings":
        """
        Load settings from environment variables or defaults.
        Simple manual env override implementation to avoid pydantic-settings dependency.
        """
        # Dictionary to hold overridden values
        overrides = {}
        
        # Mapping of env var to field name and type
        env_map = {
            "FASHION_TARGET_VIDEOS": ("target_videos", int),
            "FASHION_QUALITY_THRESHOLD": ("quality_threshold", float),
            "FASHION_MAX_ATTEMPTS": ("max_attempts", int),
            "FASHION_CONCURRENCY": ("concurrency", int),
            "FASHION_OUTPUT_ROOT": ("output_root", str),
            "FASHION_BRAND_PROFILE_PATH": ("brand_profile_path", str),
            "FASHION_CATEGORY_RULES_PATH": ("category_rules_path", str),
            "FASHION_LOG_LEVEL": ("log_level", str),
            "FASHION_API_HOST": ("api_host", str),
            "FASHION_API_PORT": ("api_port", int),
            "FASHION_API_KEY": ("api_key", str),
            "FASHION_ENABLE_RUNWAY": ("enable_runway", bool),
            "FASHION_ENABLE_STABILITY": ("enable_stability", bool),
            "FASHION_ENABLE_LUMA": ("enable_luma", bool),
            "FASHION_ENABLE_VISION_EVALUATOR": ("enable_vision_evaluator", bool),
            "OPENAI_API_KEY": ("openai_api_key", str),
            "VLLM_API_URL": ("vllm_api_url", str),
            "RUNWAYML_API_SECRET": ("runway_api_secret", str),
            "STABILITY_API_KEY": ("stability_api_key", str),
            "LUMA_API_KEY": ("luma_api_key", str),
        }

        for env_var, (field, type_) in env_map.items():
            val = os.getenv(env_var)
            if val is not None:
                try:
                    overrides[field] = type_(val)
                except ValueError:
                    pass # Keep default if parse fails
        
        return Settings(**overrides)

# Global settings instance
settings = Settings.load()
