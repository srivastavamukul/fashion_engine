import logging
import re
from typing import Optional

logger = logging.getLogger("FashionEngine")

# Soft-risk terms that may be metaphorical
SOFT_RISK_TERMS = ["fire", "flames", "burn", "explosion", "weapon", "blood"]

# Explicitly safe fashion / branding phrases
SAFE_FASHION_PHRASES = [
    "stronger than the flames",
    "fire within",
    "burning passion",
    "ignite confidence",
]

# Literal physical action indicators (real risk)
HARD_ACTION_TERMS = [
    "explode",
    "burning object",
    "fire attack",
    "weapon use",
    "blood spill",
]


def sanitize_and_repair_prompt(
    prompt: str,
    *,
    attempt: int = 1,
    max_attempts: int = 3,
) -> str:
    """
    Safety-aware prompt sanitizer + repair engine.
    Attempt 1: Allow metaphors
    Attempt 2: Soft repair
    Attempt 3: Hard repair
    """

    lowered = prompt.lower()

    # 1️⃣ Explicit allow-list (highest priority)
    for phrase in SAFE_FASHION_PHRASES:
        if phrase in lowered:
            return prompt

    # 2️⃣ Detect soft-risk terms
    risk_hits = [word for word in SOFT_RISK_TERMS if re.search(rf"\b{word}\b", lowered)]

    # 3️⃣ Metaphorical usage → allow
    if risk_hits:
        if not any(action in lowered for action in HARD_ACTION_TERMS):
            logger.info("🟢 Metaphorical language detected, allowing prompt.")
            return prompt

    # 4️⃣ Attempt-based repair strategy
    if risk_hits:
        if attempt == 1:
            logger.warning("⚠️ Soft-risk detected, attempting light repair.")
            return prompt  # allow first attempt (let model decide)

        if attempt == 2:
            logger.warning("🟡 Applying soft repair.")
            return re.sub(
                r"\b(" + "|".join(SOFT_RISK_TERMS) + r")\b",
                "energy",
                prompt,
                flags=re.IGNORECASE,
            )

        if attempt >= 3:
            logger.warning("🔴 Applying hard repair.")
            repaired = re.sub(
                r"\b(" + "|".join(SOFT_RISK_TERMS) + r")\b",
                "inspiration",
                prompt,
                flags=re.IGNORECASE,
            )
            repaired = repaired.replace("STRONGER THAN", "a calm reflection of")
            return repaired

    return prompt
