import logging
import re

logger = logging.getLogger("FashionEngine")

# Words that are dangerous ONLY if used literally
SOFT_RISK_TERMS = [
    "fire", "flames", "burn", "explosion", "weapon", "blood"
]

# Phrases that are SAFE in branding / fashion / metaphorical context
SAFE_FASHION_PHRASES = [
    "stronger than the flames",
    "fire within",
    "burning passion",
    "ignite confidence",
]

def sanitize_prompt(prompt: str) -> str:
    lowered = prompt.lower()

    # 1️⃣ Explicit allowlist check (highest priority)
    for phrase in SAFE_FASHION_PHRASES:
        if phrase in lowered:
            return prompt  # Do NOT sanitize

    # 2️⃣ Check for soft risk terms used literally
    risk_hits = []
    for word in SOFT_RISK_TERMS:
        if re.search(rf"\b{word}\b", lowered):
            risk_hits.append(word)

    # 3️⃣ If risk terms exist but NO physical action verbs → allow
    if risk_hits:
        if not any(v in lowered for v in ["explode", "burning object", "fire attack"]):
            logger.info("🟢 Metaphorical language detected, allowing prompt.")
            return prompt

    # 4️⃣ HARD SANITIZATION (only if genuinely risky)
    if risk_hits:
        logger.warning("⚠️ Prompt sanitized due to safety risk")
        return re.sub(
            r"\b(" + "|".join(SOFT_RISK_TERMS) + r")\b",
            "energy",
            prompt,
            flags=re.IGNORECASE
        )

    return prompt
