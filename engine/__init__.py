"""
Layer 2 Personality Engine · Package exports (v5)
"""
from .pipeline import run_pipeline, run_pipeline_v5, run_pipeline_with_fallback
from .consistency_checker import check, CheckResult, Violation
from .emotion_engine_v3 import (
    EmotionEngineV3, PADState, PERSONA_BASELINE,
    perceive_user_emotion, appraise_event, classify_emotion_natural,
)
from .emotion_classifier import EmotionClassifier, classify_emotion, get_classifier
from .relationship_memory import RelationshipMemory, get_relationship_memory
from . import config

__all__ = [
    # Pipeline
    "run_pipeline",
    "run_pipeline_v5",
    "run_pipeline_with_fallback",
    # Consistency
    "check",
    "CheckResult",
    "Violation",
    # Emotion engine
    "EmotionEngineV3",
    "PADState",
    "PERSONA_BASELINE",
    "perceive_user_emotion",
    "appraise_event",
    "classify_emotion_natural",
    # Emotion classifier (v5 lexicon)
    "EmotionClassifier",
    "classify_emotion",
    "get_classifier",
    # Relationship memory (v5)
    "RelationshipMemory",
    "get_relationship_memory",
    # Config
    "config",
]
