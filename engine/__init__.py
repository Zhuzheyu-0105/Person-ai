"""
Layer 2 Personality Engine · Package exports
"""
from .pipeline import run_pipeline
from .consistency_checker import CheckResult, Violation
from .emotion_engine_v3 import EmotionEngineV3, PADState, PERSONA_BASELINE, perceive_user_emotion
from . import config

__all__ = [
    "run_pipeline",
    "CheckResult", "Violation",
    "EmotionEngineV3", "PADState", "PERSONA_BASELINE", "perceive_user_emotion",
    "config",
]
