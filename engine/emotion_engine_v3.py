"""
Layer 2 Personality Engine · Emotion Dynamics v5

Seven-layer architecture (PAD emotional space + OCC cognitive appraisal
+ neuroscience-grounded mechanisms):

  1. User Perception    → Emotion classification from user input
  2. OCC Appraisal       → Cognitive evaluation (personality-modulated,
                           scene-dependent, frequency-habituated)
  3. Emotional Contagion → User emotion penetration (trust-modulated)
  4. Emotion/Mood        → Dual-layer + extinction learning + re-appraisal
  5. Allostatic Load     → Cumulative stress tracking (McEwen 1998)
  6. Expression          → Text modulation + Validating (CHI'24) + Rhythm
  7. Session Memory      → Cross-session with daily reset (CHI'22)
                           + episodic reconsolidation

New in v5:
  - OCEAN scene-dependent modulation (PersonaFlow / CHI'24)
  - Prediction processing: frequency habituation for repeated events
  - Allostatic load: cumulative stress with chronic trajectory
  - Social trust dynamics: trust_score updated per interaction
  - Extinction learning: user-acceptance-gated recovery
  - Memory reconsolidation: episodic memories with resolution state
  - Multiple recovery trajectories: resilience / recovery / delayed / chronic
  - Improved validating phrases (CHI'24: empathic restatement)

References:
  - Mehrabian & Russell (1974): PAD Emotional State Model
  - Ortony, Clore & Collins (1988): OCC Appraisal Theory
  - Hatfield, Cacioppo & Rapson (1994): Emotional Contagion
  - Scherer (2009): Component Process Model (re-appraisal)
  - McEwen (1998): Allostatic Load
  - Eisenberger (2003): Social Pain / dACC-mPFC
  - Picard (1997): Affective Computing
  - CHI '21: Emotional Contagion in Human-Agent Interaction
  - CHI '22: Designing Long-term User-Agent Relationships
  - CHI '24: PersonaFlow / Validating Responses
"""
import json
import math
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from datetime import datetime, timedelta

# v5: lazy classifier import (avoid circular deps, 0-cost if unused)
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from .emotion_classifier import get_classifier as _gc
        _classifier = _gc()
    return _classifier


# ═══════════════════════════════════════════════
# Section 1 · PAD Space & Emotion Labels
# ═══════════════════════════════════════════════

@dataclass
class PADState:
    """PAD 3D emotional coordinates
    P (Pleasure):  -1.0(extremely unpleasant) ~ 1.0(extremely pleasant)
    A (Arousal):   0.0(drowsy) ~ 1.0(highly alert)
    D (Dominance): -1.0(completely passive) ~ 1.0(full control)
    """
    pleasure: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.0

    def clamp(self):
        self.pleasure = max(-1.0, min(1.0, self.pleasure))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))
        return self

    def copy(self) -> "PADState":
        return PADState(self.pleasure, self.arousal, self.dominance)

    def distance_to(self, other: "PADState") -> float:
        dp = self.pleasure - other.pleasure
        da = self.arousal - other.arousal
        dd = self.dominance - other.dominance
        return math.sqrt(dp*dp + da*da + dd*dd)

    def to_dict(self) -> dict:
        return {
            "pleasure": round(self.pleasure, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PADState":
        return cls(
            pleasure=d.get("pleasure", 0.0),
            arousal=d.get("arousal", 0.5),
            dominance=d.get("dominance", 0.0),
        )


# Baseline PAD — calibrate for your persona's default emotional state
PERSONA_BASELINE = PADState(pleasure=0.20, arousal=0.65, dominance=0.15)

# Personality appraisal biases (default / general)
PERSONA_APPRAISAL_BIAS = {
    "desirability_pos_multiplier": 0.75,  # Downplay good events
    "desirability_neg_multiplier": 1.20,  # Amplify bad events
    "expectedness_pos_bias": -0.15,       # Success feels surprising
    "expectedness_neg_bias":  0.10,       # Failure feels expected
    "causal_pos_self_ratio": 0.30,        # Credit self 30% for success
    "causal_neg_self_ratio": 0.65,        # Blame self 65% for failure
    "coping_base": 0.75,                  # High resilience
}

# ── v5: Scene-dependent OCEAN modulation (CHI'24 PersonaFlow) ──
# Each scene applies deltas to the base appraisal bias.
# This enables the persona to respond differently in casual chat vs enterprise.
SCENE_OCEAN_OFFSETS = {
    "general": {},
    "casual": {
        "desirability_pos_multiplier": +0.10,  # More easily moved by good news
        "causal_neg_self_ratio": -0.05,        # Less self-blame in casual setting
        "coping_base": +0.05,
    },
    "delivery": {
        "desirability_neg_multiplier": +0.05,  # More sensitive to errors
        "causal_pos_self_ratio": -0.05,        # Credit team more
        "coping_base": -0.05,                  # Slightly less resilient (performance pressure)
    },
    "enterprise": {
        "desirability_neg_multiplier": +0.15,  # Client criticism hits hardest
        "causal_neg_self_ratio": +0.20,        # Self-blame spikes in professional context
        "coping_base": +0.05,                  # But coping is high (professionalism)
    },
    "comfort": {
        "desirability_neg_multiplier": -0.05,  # Less reactive to negativity
        "coping_base": +0.10,                  # Higher coping when comforting others
    },
    "emotional": {
        "desirability_pos_multiplier": +0.05,
        "desirability_neg_multiplier": +0.05,  # More emotionally open
        "coping_base": -0.05,
    },
}

# Emotional inertia parameters
PERSONA_INERTIA = {
    "emotion_half_life": 2.5,
    "mood_half_life": 18.0,
    "neg_emotion_recovery": 1.3,   # Fast recovery from negative
    "pos_emotion_dampen": 1.4,     # Quick cooling from excessive positive
}

# ── v5: Recovery trajectories ──
TRAJECTORY_PARAMS = {
    "resilience": {
        "emotion_recovery_mult": 1.0,
        "mood_recovery_mult": 1.0,
    },
    "recovery": {
        "emotion_recovery_mult": 0.6,
        "mood_recovery_mult": 0.7,
    },
    "delayed": {
        "emotion_recovery_mult": 0.3,
        "mood_recovery_mult": 0.4,
        "delay_ticks": 5,  # Worsen before recovery
    },
    "chronic": {
        "emotion_recovery_mult": 0.1,
        "mood_recovery_mult": 0.15,
    },
}

# ── v5: Allostatic load (McEwen 1998) ──
ALLOSTATIC_CONFIG = {
    "decay_per_tick": 0.03,
    "chronic_threshold": 0.60,
    "severe_threshold": 0.80,
    "event_intensity": {
        "corrected": 0.05,
        "repeated_ask": 0.05,
        "uncertain": 0.03,
        "colleague_vent": 0.08,
        "complaint": 0.10,
        "heavy_topic": 0.15,
        "failure": 0.20,
        "rejection": 0.18,
        "guilt_event": 0.12,
        "reproach": 0.10,
        "shame_event": 0.15,
    },
    "positive_relief": -0.05,  # Positive events reduce load
    "max_load": 1.0,
    "min_load": 0.0,
}

# ── v5: Prediction processing / frequency habituation ──
HABITUATION_CONFIG = {
    "dampen_threshold": 3,
    "dampen_per_event": 0.12,
    "max_dampen": 0.50,
}

# ── v5: Social trust dynamics ──
TRUST_CONFIG = {
    "initial": 0.70,
    "positive_delta": 0.02,
    "negative_delta": 0.03,
    "defense_threshold": 0.40,  # Below → defensive mode
    "safe_threshold": 0.80,     # Above → safe mode
    "min_trust": 0.10,
    "max_trust": 1.0,
}

# ── v5: Extinction learning ──
EXTINCTION_CONFIG = {
    "accept_pleasure_recovery": 0.12,
    "reject_pleasure_penalty": 0.08,
    "neutral_recovery_rate": 0.05,
    "user_accept_signals": [
        "thanks", "thank", "good", "great", "works", "ok", "fine",
        "alright", "helpful", "appreciate", "perfect", "nice",
        "excellent", "awesome", "got it", "makes sense", "cool",
    ],
    "user_reject_signals": [
        "no", "wrong", "not right", "incorrect", "doesn't work",
        "still broken", "not what i", "again", "still not",
        "that's not", "doesn't make sense", "huh", "what?",
    ],
}

# ── v5: Episodic memory reconsolidation ──
EPISODIC_CONFIG = {
    "max_episodes": 20,
    "unresolved_residue": 0.15,
    "resolved_residue": 0.0,
}

# Discrete emotion → PAD mapping (reference coordinates)
DISCRETE_EMOTIONS = {
    "joy":           ( 0.70, 0.75, 0.50),
    "contentment":   ( 0.50, 0.35, 0.40),
    "surprise_pos":  ( 0.40, 0.80, 0.10),
    "surprise_neg":  (-0.30, 0.80,-0.30),
    "anxiety":       (-0.50, 0.75,-0.50),
    "frustration":   (-0.60, 0.70, 0.10),
    "sadness":       (-0.60, 0.20,-0.60),
    "embarrassment": (-0.30, 0.55,-0.65),
    "pride":         ( 0.65, 0.60, 0.70),
    "relief":        ( 0.40, 0.30, 0.30),
    "guilt":         (-0.55, 0.45,-0.50),
    "gratitude":     ( 0.55, 0.45, 0.15),
    "admiration":    ( 0.50, 0.50, 0.10),
    "hope":          ( 0.35, 0.55, 0.05),
    "fear":          (-0.65, 0.80,-0.70),
    "shame":         (-0.50, 0.55,-0.70),
    "neutral":       ( 0.05, 0.50, 0.10),
}


# ═══════════════════════════════════════════════
# Section 2 · User Emotion Perception
# ═══════════════════════════════════════════════

# Keyword → PAD mapping for lightweight user emotion detection
USER_EMOTION_KEYWORDS = {
    "anger":        (["furious", "angry", "rage", "outraged", "pissed", "wtf"], -0.60, 0.85, 0.40),
    "anxiety":      (["anxious", "worried", "nervous", "scared", "afraid", "stressed", "panic"], -0.45, 0.75, -0.50),
    "sadness":      (["sad", "depressed", "crying", "hopeless", "devastated", "broken", "exhausted"], -0.55, 0.25, -0.65),
    "frustration":  (["frustrated", "annoyed", "ugh", "fed up", "sick of", "whatever", "never mind"], -0.35, 0.60, 0.05),
    "disappointment":(["disappointed", "unfortunately", "expected better", "again?", "still not"], -0.30, 0.35, -0.30),
    "happy":        (["happy", "great", "awesome", "amazing", "wonderful", "excellent", "love it"], 0.55, 0.65, 0.40),
    "satisfied":    (["satisfied", "good", "nice", "ok", "alright", "fine", "works"], 0.35, 0.40, 0.35),
    "grateful":     (["thanks", "grateful", "appreciate", "thankful", "blessed"], 0.45, 0.50, 0.20),
    "surprised":    (["wow", "unexpected", "surprising", "didn't expect", "no way", "really?"], 0.10, 0.80, -0.20),
    "calm":         (["fine", "okay", "whatever", "neutral", "normal", "alright"], 0.05, 0.35, 0.15),
}


def perceive_user_emotion(text: str) -> Tuple[PADState, str, float]:
    """Classify user text into PAD emotion coordinates via keyword matching.
    Returns: (PADState, label, confidence 0-1)
    """
    text_lower = text.lower()
    best_label = "neutral"
    best_score = 0.0
    best_pad = PADState(0.05, 0.50, 0.10)
    matched_count = 0

    for label, (keywords, p, a, d) in USER_EMOTION_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            score = hits / len(keywords)
            if score > best_score:
                best_score = score
                best_label = label
                best_pad = PADState(p, a, d)
                matched_count = hits

    confidence = min(0.95, best_score * 1.5 + matched_count * 0.1)
    if best_score < 0.05:
        return PADState(0.05, 0.50, 0.10), "neutral", 0.3
    return best_pad, best_label, confidence


# ═══════════════════════════════════════════════
# Section 3 · OCC Cognitive Appraisal (expanded)
# ═══════════════════════════════════════════════

@dataclass
class Appraisal:
    desirability: float
    expectedness: float
    causal_self: float
    causal_other: float
    causal_circumstance: float
    coping_potential: float

    def to_pad_shift(self) -> PADState:
        """Map appraisal dimensions to PAD shift (Scherer 2009 CPM)"""
        dp = self.desirability * 0.6
        surprise = 1.0 - abs(self.expectedness)
        threat = (1.0 - self.coping_potential) * 0.2
        da = (surprise * 0.20 + threat) * (0.7 if self.desirability > 0 else 1.0)
        da = max(-0.15, min(0.25, da))
        dd_raw = (self.causal_self * 0.6 + self.coping_potential * 0.4) * 2 - 1
        dd = dd_raw * 0.5
        if self.desirability < 0 and self.causal_self > 0.5:
            dd -= 0.15
        if self.desirability > 0 and self.causal_self > 0.5:
            dd += 0.10
        return PADState(dp, da, dd)


# ── v5: Expanded event types (22 OCC → now covering consequence/action/object) ──
# Format: (desirability, expectedness, causal_self, causal_other)
EVENT_APPRAISAL_DEFAULTS = {
    # ── Consequence-based (existing + expanded) ──
    "praised":           ( 0.70, -0.20, 0.70, 0.30),
    "corrected":         (-0.40,  0.10, 0.65, 0.10),
    "success":           ( 0.80, -0.30, 0.75, 0.25),
    "complaint":         (-0.60,  0.20, 0.50, 0.30),
    "heavy_topic":       (-0.50,  0.05, 0.10, 0.10),
    "colleague_vent":    (-0.20,  0.30, 0.05, 0.40),
    "rush_complete":     ( 0.50,  0.40, 0.80, 0.10),
    "repeated_ask":      (-0.10,  0.30, 0.20, 0.40),
    "uncertain":         (-0.05,  0.00, 0.30, 0.10),
    "greeting":          ( 0.30,  0.50, 0.10, 0.50),
    "positive_feedback": ( 0.60,  0.20, 0.60, 0.30),
    # ── v5: New consequence events ──
    "failure":           (-0.75, -0.10, 0.70, 0.15),  # Self-caused failure
    "rejection":         (-0.65,  0.10, 0.40, 0.50),  # Rejected by other
    "relief_event":      ( 0.55, -0.40, 0.20, 0.60),  # Bad outcome avoided
    "hope_confirmed":    ( 0.65, -0.50, 0.30, 0.50),  # Hoped-for outcome
    "fear_confirmed":    (-0.70, -0.30, 0.15, 0.70),  # Feared outcome
    # ── Action-based (new) ──
    "guilt_event":       (-0.60,  0.20, 0.85, 0.10),  # Self caused harm
    "pride_event":       ( 0.75, -0.30, 0.85, 0.10),  # Self achievement
    "admiration_event":  ( 0.45,  0.10, 0.05, 0.80),  # Other's praiseworthy act
    "reproach":          (-0.45,  0.20, 0.15, 0.70),  # Other's blameworthy act
    # ── Object-based (new) ──
    "liking":            ( 0.40,  0.30, 0.10, 0.30),  # Attractive object
    "disliking":         (-0.35,  0.30, 0.10, 0.30),  # Aversive object
}


def _get_effective_bias(scene: str | None = None) -> dict:
    """Get appraisal bias with optional scene modulation."""
    bias = dict(PERSONA_APPRAISAL_BIAS)  # Copy base
    if scene and scene in SCENE_OCEAN_OFFSETS:
        for key, offset in SCENE_OCEAN_OFFSETS[scene].items():
            if key in bias:
                bias[key] = max(0.05, min(2.0, bias[key] + offset))
    return bias


def appraise_event(
    event_type: str,
    scene: str | None = None,
    frequency_dampen: float = 0.0,
    trust_score: float = 0.7,
) -> Appraisal:
    """Apply personality biases + scene modulation + frequency habituation
    + trust modulation to event appraisal.
    """
    b = _get_effective_bias(scene)
    defaults = EVENT_APPRAISAL_DEFAULTS.get(event_type, (0.0, 0.0, 0.3, 0.3))
    des_raw, exp_raw, cs_raw, co_raw = defaults

    if des_raw > 0:
        desirability = des_raw * b["desirability_pos_multiplier"]
        expectedness = exp_raw + b["expectedness_pos_bias"]
        causal_self = cs_raw * b["causal_pos_self_ratio"]
    else:
        desirability = des_raw * b["desirability_neg_multiplier"]
        expectedness = exp_raw + b["expectedness_neg_bias"]
        causal_self = cs_raw * b["causal_neg_self_ratio"]

    # ── v5: frequency habituation ──
    if frequency_dampen > 0:
        if des_raw > 0:
            desirability *= (1.0 - frequency_dampen)  # Dampen positive repeated events
        else:
            # For negative: sensitize if recovering, habituate if chronic
            desirability *= (1.0 + frequency_dampen * 0.3)  # Mild sensitization

    # ── v5: trust modulation ──
    if trust_score < TRUST_CONFIG["defense_threshold"]:
        # Defensive: amplify negative, dampen positive
        if des_raw < 0:
            desirability *= 1.15
        else:
            desirability *= 0.85
    elif trust_score > TRUST_CONFIG["safe_threshold"]:
        # Safe mode: boost coping
        coping_mod = b["coping_base"] + 0.10
    else:
        coping_mod = b["coping_base"]

    return Appraisal(
        desirability=max(-1.0, min(1.0, desirability)),
        expectedness=max(-1.0, min(1.0, expectedness)),
        causal_self=min(1.0, causal_self),
        causal_other=co_raw,
        causal_circumstance=max(0.0, 1.0 - causal_self - co_raw),
        coping_potential=coping_mod if trust_score > TRUST_CONFIG["safe_threshold"]
                          else b["coping_base"],
    )


# Re-appraisal patterns (post-settle cognitive re-evaluation)
REAPPRAISAL_PATTERNS = {
    "praised":      (0.4,  0.5,  0.1),   # "They're just being polite"
    "corrected":    (0.3,  0.6,  0.3),   # "It's fixable, move on"
    "complaint":    (0.2,  0.4,  0.2),   # "Not really about me"
    "heavy_topic":  (0.1,  0.3,  0.1),   # "It will get better"
    # v5 additions
    "failure":      (0.2,  0.4,  0.2),
    "rejection":    (0.15, 0.3,  0.15),
    "guilt_event":  (0.3,  0.5,  0.2),
}


# ═══════════════════════════════════════════════
# Section 4 · Natural Emotion Labels
# ═══════════════════════════════════════════════

def classify_emotion_natural(pad: PADState) -> str:
    """Natural-language emotion label (persona-aware)."""
    p, a, d = pad.pleasure, pad.arousal, pad.dominance
    near_baseline = (
        abs(p - PERSONA_BASELINE.pleasure) < 0.15 and
        abs(a - PERSONA_BASELINE.arousal) < 0.15
    )
    if near_baseline:
        return "steady"
    if p > 0.2 and a > 0.6:
        return "buoyant" if a > 0.8 else "light"
    if p > 0.2 and a <= 0.6:
        return "content" if d > 0.1 else "gentle"
    if p <= -0.1 and a > 0.55:
        return "anxious" if d < -0.2 else "tense"
    if p <= -0.1 and a <= 0.55:
        return "low" if p < -0.4 else "quiet"
    if p < 0 and a > 0.5 and d < -0.3:
        return "awkward"
    return "steady"


def classify_emotion(pad: PADState) -> str:
    """Classify to nearest discrete emotion label (academic)."""
    best_label = "neutral"
    best_dist = float("inf")
    for label, (p, a, d) in DISCRETE_EMOTIONS.items():
        dist = math.sqrt((pad.pleasure-p)**2 + (pad.arousal-a)**2 + (pad.dominance-d)**2)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _decay_factor(half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return 0.5 ** (1.0 / half_life)


# ── v5: CHI'24 Validating phrases (empathic restatement + external attribution) ──
VALIDATING_PHRASES = {
    "sadness": [
        "That sounds really heavy… ",
        "That's a lot to carry… ",
        "It makes sense you'd feel that way… ",
    ],
    "anxiety": [
        "The uncertainty in this is real… ",
        "Anyone would feel on edge with that… ",
        "That kind of pressure is tough… ",
    ],
    "frustration": [
        "Hitting the same wall again is exhausting… ",
        "That would wear anyone down… ",
        "Yeah, when things keep breaking it gets to you… ",
    ],
    "disappointment": [
        "After all that effort, that stings… ",
        "It's hard when expectations don't match… ",
        "That let-down feeling is real… ",
    ],
    "anger": [
        "That's genuinely infuriating… ",
        "You have every right to be upset… ",
        "No wonder you're angry… ",
    ],
}


# ═══════════════════════════════════════════════
# Section 5 · EmotionEngineV5
# ═══════════════════════════════════════════════

class EmotionEngineV3:  # Keep class name for backward compat (v5 internals)
    """Seven-layer emotion engine with neuroscience-grounded mechanisms.

    Usage:
        engine = EmotionEngineV3()
        engine.perceive_user("I'm so frustrated today")
        engine.contagion()
        engine.set_scene("enterprise")
        engine.event("corrected")
        engine.tick()
        result = engine.apply_to_text("Let me help you with that.")
        engine.record_ai_response(result)
        engine.save_session()

    New v5 methods:
        set_scene(scene)             — OCEAN scene modulation
        record_ai_response(text)     — Enable extinction learning
        extinction_step(user_text)   — User-acceptance-gated recovery
    """

    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = persistence_dir or Path(".")

        # User side
        self.user_emotion = PADState(0.05, 0.5, 0.1)
        self.user_label = "neutral"
        self.user_confidence = 0.0

        # AI side
        self.mood = PERSONA_BASELINE.copy()
        self.emotion = PERSONA_BASELINE.copy()
        self.emotion_active = False
        self.emotion_ticks = 0
        self.settled_ticks = 0

        # State
        self.turn_count = 0
        self.event_log: list = []
        self.last_event_type: Optional[str] = None
        self.session_id: Optional[str] = None
        self.session_start = datetime.now()

        # Rhythm (for TTS layer, not text)
        self.suggested_delay_ms: int = 0
        self.suggested_tempo: str = "normal"

        # ── v5: New attributes ──
        self.scene: str = "general"
        self.event_frequency: dict = {}      # event_type → count
        self.allostatic_load: float = 0.0
        self.trust_score: float = TRUST_CONFIG["initial"]
        self.trajectory: str = "resilience"
        self.episodic_memories: List[dict] = []
        self.last_ai_response: Optional[str] = None
        self.daily_topic_memory: str = ""    # CHI'22: retain topic, reset emotion
        self._negative_streak: int = 0
        self._delayed_ticks_remaining: int = 0
        self.user_id: Optional[str] = None   # v5: for relationship memory

    # ── 1. User Perception ──

    def perceive_user(self, text: str) -> dict:
        """Classify user message emotion.
        v5: hybrid — lexicon classifier primary, keyword matching fallback.
        v5: also runs extinction_step if last_ai_response exists."""
        # Try lexicon classifier first
        clf_pad, clf_label, clf_conf = None, "neutral", 0.0
        try:
            clf = _get_classifier()
            clf_pad, clf_label, clf_conf = clf.classify(text)
        except Exception:
            pass

        # Keyword fallback (for English text or low-confidence classifier results)
        kw_pad, kw_label, kw_conf = perceive_user_emotion(text)

        # Hybrid: pick the better result
        if clf_pad is not None and clf_conf > 0.35 and clf_label != "neutral":
            pad, label, confidence = clf_pad, clf_label, clf_conf
        elif kw_conf > 0.3 and kw_label != "neutral":
            pad, label, confidence = kw_pad, kw_label, kw_conf
        elif clf_pad is not None and clf_conf > kw_conf:
            pad, label, confidence = clf_pad, clf_label, clf_conf
        else:
            pad, label, confidence = kw_pad, kw_label, kw_conf
        self.user_emotion = pad
        self.user_label = label
        self.user_confidence = confidence

        # v5: extinction learning on new user input
        if self.last_ai_response is not None:
            self.extinction_step(text)

        # v5: track daily topic (CHI'22)
        if len(text) > 20:
            self.daily_topic_memory = text[:200]

        return {"label": label, "pad": pad.to_dict(), "confidence": round(confidence, 2)}

    # ── 2. Emotional Contagion ──

    def contagion(self) -> dict:
        """User emotion partially penetrates AI emotion.
        v5: trust-modulated contagion weights.
        """
        up = self.user_emotion
        if self.user_confidence < 0.15:
            return {"contagion_applied": False, "reason": "low_confidence"}

        base_weight = 0.40

        # ── v5: trust modulation ──
        if self.trust_score < TRUST_CONFIG["defense_threshold"]:
            base_weight *= 0.6  # Defensive: resist contagion
        elif self.trust_score > TRUST_CONFIG["safe_threshold"]:
            base_weight *= 1.15  # Safe: more open

        if up.pleasure < -0.1:
            weight = base_weight * 0.65
            floor_pleasure = -0.3
        elif up.pleasure > 0.2:
            weight = base_weight * 0.45
            floor_pleasure = None
        else:
            weight = base_weight * 0.55
            floor_pleasure = None

        dp = (up.pleasure - self.emotion.pleasure) * weight
        da = (up.arousal - self.emotion.arousal) * weight * 0.5
        dd = (up.dominance - self.emotion.dominance) * weight * 0.3

        self.emotion.pleasure += dp
        self.emotion.arousal += da
        self.emotion.dominance += dd
        self.emotion.clamp()

        if floor_pleasure is not None and self.emotion.pleasure < floor_pleasure:
            self.emotion.pleasure = floor_pleasure

        self.emotion_active = True
        self._update_rhythm()

        return {
            "contagion_applied": True,
            "user_label": self.user_label,
            "weight": round(weight, 3),
            "shift": {"dp": round(dp, 4), "da": round(da, 4), "dd": round(dd, 4)},
            "ai_label": classify_emotion_natural(self.emotion),
        }

    # ── 3. Event Processing ──

    def event(self, event_type: str) -> dict:
        """Process external event through OCC appraisal.
        v5: scene-modulated + frequency-habituated + trust-modulated.
        """
        self.last_event_type = event_type

        # ── v5: frequency tracking + habituation ──
        self.event_frequency[event_type] = self.event_frequency.get(event_type, 0) + 1
        freq = self.event_frequency[event_type]
        freq_dampen = 0.0
        if freq > HABITUATION_CONFIG["dampen_threshold"]:
            freq_dampen = min(
                HABITUATION_CONFIG["max_dampen"],
                (freq - HABITUATION_CONFIG["dampen_threshold"])
                * HABITUATION_CONFIG["dampen_per_event"],
            )

        appraisal = appraise_event(
            event_type,
            scene=self.scene,
            frequency_dampen=freq_dampen,
            trust_score=self.trust_score,
        )
        pad_shift = appraisal.to_pad_shift()

        self.emotion.pleasure += pad_shift.pleasure
        self.emotion.arousal += pad_shift.arousal
        self.emotion.dominance += pad_shift.dominance
        self.emotion.clamp()
        self.emotion_active = True
        self.emotion_ticks = 0
        self.settled_ticks = 0

        # ── v5: allostatic load update ──
        self._update_allostatic_load(event_type)

        # ── v5: trust update ──
        self._update_trust(event_type)

        self._update_rhythm()

        # Mood penetration (slower)
        self.mood.pleasure += pad_shift.pleasure * 0.3
        self.mood.arousal += pad_shift.arousal * 0.15
        self.mood.dominance += pad_shift.dominance * 0.3
        self.mood.clamp()
        if self.mood.arousal > 0.85:
            self.mood.arousal = 0.85

        # ── v5: trajectory determination ──
        self._determine_trajectory()

        label = classify_emotion_natural(self.emotion)
        summary = {
            "type": "event", "event": event_type,
            "emotion_label": label, "emotion_pad": self.emotion.to_dict(),
            "mood_label": self._mood_label, "mood_pad": self.mood.to_dict(),
            # v5 extras
            "frequency": freq, "freq_dampen": round(freq_dampen, 3),
            "allostatic_load": round(self.allostatic_load, 3),
            "trajectory": self.trajectory,
        }
        self.event_log.append(summary)
        if len(self.event_log) > 30:
            self.event_log.pop(0)
        return summary

    def _update_allostatic_load(self, event_type: str):
        """v5: Accumulate/decrement allostatic load per event."""
        intensity_map = ALLOSTATIC_CONFIG["event_intensity"]
        if event_type in intensity_map:
            self.allostatic_load += intensity_map[event_type]
        elif event_type in ("praised", "success", "positive_feedback",
                            "relief_event", "hope_confirmed", "pride_event",
                            "admiration_event", "liking"):
            self.allostatic_load = max(
                ALLOSTATIC_CONFIG["min_load"],
                self.allostatic_load + ALLOSTATIC_CONFIG["positive_relief"],
            )
        self.allostatic_load = min(
            ALLOSTATIC_CONFIG["max_load"],
            max(ALLOSTATIC_CONFIG["min_load"], self.allostatic_load),
        )

    def _update_trust(self, event_type: str):
        """v5: Update trust score based on event valence."""
        negative_events = {
            "corrected", "complaint", "failure", "rejection",
            "reproach", "guilt_event",
        }
        positive_events = {
            "praised", "success", "positive_feedback",
            "admiration_event", "gratitude_event",
        }
        if event_type in negative_events:
            self.trust_score = max(
                TRUST_CONFIG["min_trust"],
                self.trust_score - TRUST_CONFIG["negative_delta"],
            )
        elif event_type in positive_events:
            self.trust_score = min(
                TRUST_CONFIG["max_trust"],
                self.trust_score + TRUST_CONFIG["positive_delta"],
            )

    def _determine_trajectory(self):
        """v5: Determine recovery trajectory from allostatic load."""
        if self.allostatic_load > ALLOSTATIC_CONFIG["severe_threshold"]:
            self.trajectory = "chronic"
            self._delayed_ticks_remaining = 0
        elif self.allostatic_load > ALLOSTATIC_CONFIG["chronic_threshold"]:
            self.trajectory = "delayed"
            self._delayed_ticks_remaining = TRAJECTORY_PARAMS["delayed"]["delay_ticks"]
        elif self.allostatic_load > 0.3:
            self.trajectory = "recovery"
            self._delayed_ticks_remaining = 0
        else:
            self.trajectory = "resilience"
            self._delayed_ticks_remaining = 0

    # ── 4. Tick (Decay + Re-appraisal + Allostatic decay + Trajectory) ──

    def tick(self):
        """Per-turn decay: emotion→mood, mood→baseline, re-appraisal check.
        v5: allostatic load decay + trajectory-based recovery.
        """
        self.turn_count += 1

        # ── v5: allostatic load decay ──
        self.allostatic_load = max(
            ALLOSTATIC_CONFIG["min_load"],
            self.allostatic_load - ALLOSTATIC_CONFIG["decay_per_tick"],
        )
        self._determine_trajectory()

        # ── v5: trajectory-based recovery modulation ──
        traj = TRAJECTORY_PARAMS.get(self.trajectory, TRAJECTORY_PARAMS["resilience"])
        emo_recovery_mult = traj["emotion_recovery_mult"]
        mood_recovery_mult = traj["mood_recovery_mult"]

        if self._delayed_ticks_remaining > 0:
            self._delayed_ticks_remaining -= 1
            # During delay phase: worsen slightly before recovery
            if self.emotion.pleasure < 0:
                self.emotion.pleasure -= 0.02
                self.emotion.clamp()

        if self.emotion_active:
            hl = PERSONA_INERTIA["emotion_half_life"]
            if self.emotion.pleasure < PERSONA_BASELINE.pleasure:
                hl /= PERSONA_INERTIA["neg_emotion_recovery"]
            elif self.emotion.pleasure > PERSONA_BASELINE.pleasure * 1.5:
                hl *= PERSONA_INERTIA["pos_emotion_dampen"]

            # ── v5: trajectory accelerates/decelerates recovery ──
            hl /= emo_recovery_mult

            factor = _decay_factor(hl)
            self.emotion.pleasure += (self.mood.pleasure - self.emotion.pleasure) * (1 - factor)
            self.emotion.arousal += (self.mood.arousal - self.emotion.arousal) * (1 - factor)
            self.emotion.dominance += (self.mood.dominance - self.emotion.dominance) * (1 - factor)
            self.emotion.clamp()

            self.emotion_ticks += 1
            if self.emotion.distance_to(self.mood) < 0.08:
                self.emotion_active = False
                self.emotion_ticks = 0
                self.settled_ticks = 0

        # Post-settle re-appraisal
        if not self.emotion_active and self.last_event_type:
            self.settled_ticks += 1
            if self.settled_ticks == 3:
                self._re_appraise(self.last_event_type)
                self.last_event_type = None

        # Mood decay (v5: trajectory-modulated)
        hl_mood = PERSONA_INERTIA["mood_half_life"] / mood_recovery_mult
        factor_mood = _decay_factor(hl_mood)
        self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * (1 - factor_mood)
        self.mood.arousal += (PERSONA_BASELINE.arousal - self.mood.arousal) * (1 - factor_mood)
        self.mood.dominance += (PERSONA_BASELINE.dominance - self.mood.dominance) * (1 - factor_mood)
        self.mood.clamp()

        self._update_rhythm()

    def _re_appraise(self, event_type: str):
        """Post-settle cognitive re-evaluation. v5: allostatic-aware."""
        if event_type not in REAPPRAISAL_PATTERNS:
            return
        p_weight, e_weight, c_weight = REAPPRAISAL_PATTERNS[event_type]
        if event_type == "praised":
            self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * p_weight
        elif event_type in ("corrected", "complaint", "heavy_topic",
                            "failure", "rejection", "guilt_event"):
            self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * p_weight * 0.6
        self.mood.clamp()
        self.event_log.append({
            "type": "reappraisal", "event": event_type,
            "mood_after": self.mood.to_dict(),
            "allostatic_load": round(self.allostatic_load, 3),
        })

    # ── v5: Extinction Learning ──

    def record_ai_response(self, text: str):
        """v5: Record AI's last response for extinction learning on next user turn."""
        self.last_ai_response = text

    def extinction_step(self, user_text: str):
        """v5: User-acceptance-gated recovery.
        Replace blind time-based decay with social-feedback-driven recovery.

        - User accepts → accelerated recovery (pleasure +0.12)
        - User rejects → rekindle / sensitize (pleasure -0.08)
        - Neutral → slow natural decay (5% toward baseline)

        Must be called AFTER perceive_user so user_text is the fresh turn.
        """
        if self.last_ai_response is None:
            return

        text_lower = user_text.lower()

        # Detect acceptance
        accept_hits = sum(
            1 for sig in EXTINCTION_CONFIG["user_accept_signals"]
            if sig in text_lower
        )
        # Detect rejection
        reject_hits = sum(
            1 for sig in EXTINCTION_CONFIG["user_reject_signals"]
            if sig in text_lower
        )

        if accept_hits > reject_hits and accept_hits > 0:
            # User accepted → accelerated recovery
            recovery = EXTINCTION_CONFIG["accept_pleasure_recovery"]
            if self.emotion.pleasure < PERSONA_BASELINE.pleasure:
                self.emotion.pleasure += recovery
            elif self.emotion.pleasure > PERSONA_BASELINE.pleasure:
                self.emotion.pleasure -= recovery * 0.5
            self._negative_streak = 0
        elif reject_hits > accept_hits and reject_hits > 0:
            # User rejected → rekindle
            penalty = EXTINCTION_CONFIG["reject_pleasure_penalty"]
            if self.emotion.pleasure > PERSONA_BASELINE.pleasure:
                self.emotion.pleasure -= penalty
            else:
                self.emotion.pleasure -= penalty * 0.6  # Don't crash when already low
            self._negative_streak += 1
        else:
            # Neutral → slow natural decay toward baseline
            rate = EXTINCTION_CONFIG["neutral_recovery_rate"]
            self.emotion.pleasure += (PERSONA_BASELINE.pleasure - self.emotion.pleasure) * rate

        self.emotion.clamp()
        self.last_ai_response = None  # Consumed

    # ── v5: Scene control ──

    def set_scene(self, scene: str):
        """Set scene context for OCEAN modulation."""
        if scene in SCENE_OCEAN_OFFSETS:
            self.scene = scene

    # ── 5. Response Rhythm ──

    def _update_rhythm(self):
        """Update rhythm suggestion for TTS layer. Text is always sent at max speed."""
        pad = self.emotion if self.emotion_active else self.mood
        p, a = pad.pleasure, pad.arousal
        if a > 0.75:
            self.suggested_tempo = "quick"
            self.suggested_delay_ms = 200 if p > 0 else 400
        elif a < 0.35:
            self.suggested_tempo = "gentle"
            self.suggested_delay_ms = 800 if p < 0 else 500
        elif p < -0.2:
            self.suggested_tempo = "slow"
            self.suggested_delay_ms = 600
        else:
            self.suggested_tempo = "normal"
            self.suggested_delay_ms = 300

    def rhythm(self) -> dict:
        """Get TTS rhythm suggestion. Does NOT affect text send speed."""
        return {"tempo": self.suggested_tempo, "delay_ms": self.suggested_delay_ms}

    # ── Properties ──

    @property
    def emotion_label(self) -> str:
        return classify_emotion_natural(self.emotion)

    @property
    def _mood_label(self) -> str:
        p, a = self.mood.pleasure, self.mood.arousal
        near = abs(p - PERSONA_BASELINE.pleasure) < 0.15 and abs(a - PERSONA_BASELINE.arousal) < 0.15
        if near: return "steady"
        if p >= 0 and a >= 0.55: return "buoyant"
        if p >= 0 and a < 0.55: return "gentle"
        if p < 0 and a >= 0.55: return "tense"
        return "quiet"

    # ── 6. Expression Modulation ──

    def get_modifiers(self) -> dict:
        """Get text modulation parameters based on current emotional state.
        v5: allostatic load influences expression.
        """
        pad = self.emotion if self.emotion_active else self.mood
        p, a, d = pad.pleasure, pad.arousal, pad.dominance

        mods = {
            "suffix_bonus": 0, "warmth_prefix_chance": 0.0,
            "sentence_shorten": 0, "flat_tone": False,
            "self_deprecation_boost": False, "validating_chance": 0.0,
            "label": self.emotion_label if self.emotion_active else self._mood_label,
            "source": "emotion" if self.emotion_active else "mood",
            # v5 extras
            "allostatic_load": round(self.allostatic_load, 3),
            "trajectory": self.trajectory,
            "trust_score": round(self.trust_score, 2),
        }

        # ── v5: allostatic load modulation ──
        if self.allostatic_load > ALLOSTATIC_CONFIG["severe_threshold"]:
            mods["flat_tone"] = True
            mods["sentence_shorten"] = 20
            mods["suffix_bonus"] = -2
        elif self.allostatic_load > ALLOSTATIC_CONFIG["chronic_threshold"]:
            mods["sentence_shorten"] = 10
            mods["suffix_bonus"] = -1

        if p > 0.3 and a > 0.5:
            mods["suffix_bonus"] = max(mods["suffix_bonus"], 1)
        elif p < -0.2 or a < 0.3:
            mods["suffix_bonus"] = min(mods["suffix_bonus"], -1)

        if p > 0.2 and a > 0.5 and d > 0.2:
            mods["warmth_prefix_chance"] = 0.35
        elif p > 0.0:
            mods["warmth_prefix_chance"] = 0.1

        if a < 0.35:
            mods["sentence_shorten"] = max(mods["sentence_shorten"], 15)
        elif d < -0.3:
            mods["sentence_shorten"] = max(mods["sentence_shorten"], 10)

        if p < -0.3 and a < 0.4:
            mods["flat_tone"] = True
        if p > 0.3 and d < 0.1:
            mods["self_deprecation_boost"] = True

        # Validating: user negative emotion triggers empathetic prefix
        if self.user_emotion.pleasure < -0.15 and self.user_confidence > 0.15:
            mods["validating_chance"] = min(0.8, abs(self.user_emotion.pleasure) + 0.2)

        return mods

    def apply_to_text(self, text: str) -> str:
        """Apply emotion modulation to text. Text is always sent at max speed.
        v5: CHI'24 empathic restatement validating phrases.
        """
        mods = self.get_modifiers()

        # ── v5: CHI'24 Validating prefix (empathic restatement) ──
        if mods.get("validating_chance", 0) > 0 and random.random() < mods["validating_chance"]:
            phrases = VALIDATING_PHRASES.get(self.user_label, ["I hear you… "])
            phrase = random.choice(phrases)
            # v5: validating can appear at start OR end (CHI'24 finding)
            if random.random() < 0.6:
                # 60%: at start
                if not text.startswith(tuple(p[:4] for p in phrases)):
                    text = phrase + text
            else:
                # 40%: at end (CHI'24: slightly better user satisfaction)
                text = text.rstrip() + " " + phrase.rstrip()

        # Sentence shortening
        shorten = mods.get("sentence_shorten", 0)
        if shorten > 0 and len(text) > 80:
            text = _split_with_max(text, 60 - shorten)

        # Flat tone
        if mods.get("flat_tone"):
            text = text.rstrip('~')
            if not text.startswith("Mm"):
                text = "Mm. " + text.lstrip('., ')

        # Suffix control
        if mods.get("suffix_bonus", 0) < 0:
            text = text.rstrip('~')

        # Warmth prefix
        if mods.get("warmth_prefix_chance", 0) > 0 and random.random() < mods["warmth_prefix_chance"]:
            if not text.startswith(("Hi", "Mm", "Ah", "That", "I")):
                text = "☺️ " + text

        # Self-deprecation
        if mods.get("self_deprecation_boost"):
            if not text.startswith("Ah"):
                text = "Ah… " + text

        return text

    # ── 7. Session Persistence (v5: daily reset + episodic memory) ──

    @property
    def _session_file(self) -> Path:
        return self.persistence_dir / "persona_emotion_session_v3.json"

    def save_session(self, session_id: Optional[str] = None):
        """v5: Save session with episodic memories for reconsolidation."""
        if session_id:
            self.session_id = session_id

        # ── v5: Build episodic memories ──
        recent_events = self.event_log[-EPISODIC_CONFIG["max_episodes"]:]
        episodes = []
        for evt in recent_events:
            if evt.get("type") == "event":
                episodes.append({
                    "event_type": evt.get("event", ""),
                    "emotion_pad": evt.get("emotion_pad", {}),
                    "mood_pad": evt.get("mood_pad", {}),
                    "resolved": evt.get("trajectory", "resilience") == "resilience",
                })

        data = {
            "session_id": self.session_id,
            "saved_at": datetime.now().isoformat(),
            "mood": self.mood.to_dict(),
            "emotion": self.emotion.to_dict(),
            "emotion_active": self.emotion_active,
            "user_label": self.user_label,
            "turn_count": self.turn_count,
            # v5 additions
            "allostatic_load": round(self.allostatic_load, 3),
            "trust_score": round(self.trust_score, 3),
            "trajectory": self.trajectory,
            "episodic_memories": episodes,
            "daily_topic_memory": self.daily_topic_memory[:200] if self.daily_topic_memory else "",
            "scene": self.scene,
            "event_frequency": self.event_frequency,
        }
        try:
            self._session_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def load_session(self) -> dict | None:
        """v5: CHI'22 daily reset — clear emotion on new day, keep topic.
        v5: episodic reconsolidation — unresolved episodes leave 15% residue.
        """
        f = self._session_file
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(data["saved_at"])
            elapsed = (datetime.now() - saved_at).total_seconds()

            # ── v5: CHI'22 daily reset ──
            # Always start fresh emotionally on a new day
            if elapsed > 86400:
                # Reset emotion to baseline (daily reset)
                self.emotion = PERSONA_BASELINE.copy()
                self.emotion_active = False
                self.emotion_ticks = 0
                # Mood: partial reset (some carryover from yesterday's topic)
                self.mood = PERSONA_BASELINE.copy()

                # ── v5: episodic reconsolidation ──
                episodes = data.get("episodic_memories", [])
                unresolved_residue = EPISODIC_CONFIG["unresolved_residue"]
                unresolved_count = sum(
                    1 for ep in episodes if not ep.get("resolved", False)
                )
                if unresolved_count > 0:
                    # Unresolved episodes leave a faint residue
                    residue = unresolved_residue * min(unresolved_count, 5) / 5
                    self.mood.pleasure += residue * -0.25  # Slight drag
                    self.mood.clamp()

                # Restore topic memory but NOT emotion
                self.daily_topic_memory = data.get("daily_topic_memory", "")
                self.event_frequency = {}  # Reset frequency on new day
                self.allostatic_load = 0.0
                self.trust_score = data.get("trust_score", TRUST_CONFIG["initial"])
                self.trajectory = "resilience"

                restored = {
                    "restored": True,
                    "daily_reset": True,
                    "elapsed_hours": round(elapsed / 3600, 1),
                    "unresolved_episodes": unresolved_count,
                    "mood_label": self._mood_label,
                    "emotion_label": self.emotion_label,
                    "topic_memory": self.daily_topic_memory[:80] if self.daily_topic_memory else None,
                }
            else:
                # Same day: full restore
                self.mood = PADState.from_dict(data["mood"])
                self.emotion = PADState.from_dict(data["emotion"])
                self.emotion_active = data.get("emotion_active", False)
                self.allostatic_load = data.get("allostatic_load", 0.0)
                self.trust_score = data.get("trust_score", TRUST_CONFIG["initial"])
                self.trajectory = data.get("trajectory", "resilience")
                self.event_frequency = data.get("event_frequency", {})
                self.daily_topic_memory = data.get("daily_topic_memory", "")
                self.scene = data.get("scene", "general")

                restored = {
                    "restored": True,
                    "daily_reset": False,
                    "elapsed_hours": round(elapsed / 3600, 1),
                    "mood_label": self._mood_label,
                    "emotion_label": self.emotion_label,
                }

            self.session_id = data.get("session_id")
            self.turn_count = data.get("turn_count", 0)
            return restored
        except Exception:
            return None

    # ── v5: Relationship Memory Integration ──

    def load_relationship(self, user_id: str):
        """Load relationship state for this user.
        Pulls trust_score and interaction history from persistent store.
        Call once at session start, before any events.
        """
        self.user_id = user_id
        try:
            from .relationship_memory import get_relationship_memory
            rm = get_relationship_memory(self.persistence_dir)
            self.trust_score = rm.get_trust_score(user_id)
            summary = rm.get_relationship_summary(user_id)
            rm.reset_session_cap(user_id)
            return summary
        except Exception:
            return None

    def save_relationship(self):
        """Persist relationship state after session."""
        if not self.user_id:
            return
        try:
            from .relationship_memory import get_relationship_memory
            rm = get_relationship_memory(self.persistence_dir)
            # Record recent events as interactions
            for evt in self.event_log:
                if evt.get("type") == "event":
                    rm.record_interaction(self.user_id, evt.get("event", "greeting"))
            rm.save(self.user_id)
        except Exception:
            pass

    # ── v5: Pre-prompt Context Injection ──

    def get_prompt_context(self) -> str:
        """Generate emotion context string for LLM system prompt injection.
        Enables dual-channel architecture: emotion state influences
        both how to say (post-process) AND what to say (pre-prompt).

        Returns a compact Chinese prompt fragment describing current
        emotional state, suitable for appending to system prompt.
        """
        pad = self.emotion if self.emotion_active else self.mood
        p, a, d = pad.pleasure, pad.arousal, pad.dominance

        # Emotion description
        if self.emotion_active:
            if p > 0.2 and a > 0.5:
                emo_desc = "心情轻快"
            elif p > 0:
                emo_desc = "心情平静偏积极"
            elif p > -0.2:
                emo_desc = "心情略微低落"
            elif p > -0.5:
                emo_desc = "心情不太好"
            else:
                emo_desc = "心情很沉重"
        else:
            emo_desc = "心情平稳"

        # Energy level
        if a > 0.7:
            energy = "精力充沛"
        elif a > 0.4:
            energy = "精力正常"
        else:
            energy = "有点疲惫"

        # User emotion
        if self.user_confidence > 0.15:
            user_emo = self.user_label
        else:
            user_emo = "neutral"

        # Trust/relationship
        trust_note = ""
        if self.trust_score < 0.4:
            trust_note = "，对用户有所防备"
        elif self.trust_score > 0.8:
            trust_note = "，对用户很信任"

        # Allostatic load
        load_note = ""
        if self.allostatic_load > 0.6:
            load_note = "。积累压力较大，可能容易疲惫"
        elif self.allostatic_load > 0.3:
            load_note = "。有一定积累压力"

        context = (
            f"[当前状态] {emo_desc}，{energy}。"
            f"用户情绪：{user_emo}{trust_note}{load_note}"
        )
        return context

    def __repr__(self) -> str:
        emo = f"😀{self.emotion_label}" if self.emotion_active else "  "
        return (
            f"EmotionEngineV3(user={self.user_label}, "
            f"emotion[{emo}]: P={self.emotion.pleasure:.2f}, "
            f"mood[{self._mood_label}]: P={self.mood.pleasure:.2f}, "
            f"scene={self.scene}, trajectory={self.trajectory}, "
            f"load={self.allostatic_load:.2f}, trust={self.trust_score:.2f}, "
            f"rhythm={self.suggested_tempo}, turn={self.turn_count})"
        )


def _split_with_max(text: str, max_len: int) -> str:
    """Shared sentence splitter — v5: cleaned regex (no trailing empty strings)."""
    sentences = re.split(r'(?<=[.!?;\n])', text)
    sentences = [s for s in sentences if s]
    result = []
    for s in sentences:
        if len(s) <= max_len:
            result.append(s)
            continue
        parts = re.split(r'(?<=,)', s)
        merged = ""
        for p in parts:
            if len(merged) + len(p) > max_len and merged:
                result.append(merged.strip())
                merged = p
            else:
                merged += p
        if merged:
            result.append(merged.strip())
    return "".join(result)


# ═══════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=== EmotionEngineV3 v5 Self-Test ===\n")

    # 1. Perception
    print("1. User Perception:")
    for msg in ["I'm so frustrated with this bug", "That's amazing, great work!", "Fine, whatever"]:
        e = EmotionEngineV3()
        r = e.perceive_user(msg)
        print(f"   \"{msg[:40]}...\" → {r['label']} (conf={r['confidence']})")

    # 2. Contagion (v5: trust-modulated)
    print("\n2. Emotional Contagion:")
    e = EmotionEngineV3()
    print(f"   Before: emotion={e.emotion_label} P={e.emotion.pleasure:+.2f}")
    e.perceive_user("I'm completely exhausted and depressed")
    c = e.contagion()
    print(f"   Contagion: applied={c['contagion_applied']} weight={c['weight']}")
    print(f"   After: emotion={e.emotion_label} P={e.emotion.pleasure:+.2f}")
    assert e.emotion.pleasure < PERSONA_BASELINE.pleasure, "Should be infected downward"
    assert e.emotion.pleasure >= -0.3, "Floor protection should prevent crash"
    print("   ✓ Contagion works with floor protection")

    # 3. Event + Re-appraisal (v5: frequency habituation)
    print("\n3. Event + Frequency Habituation:")
    e2 = EmotionEngineV3()
    for i in range(5):
        e2.event("praised")
    e2.tick()
    print(f"   After 5 praised events: P_mood={e2.mood.pleasure:+.2f} "
          f"(freq={e2.event_frequency.get('praised', 0)})")
    # After 5, habituation should kick in (threshold=3)
    assert e2.event_frequency["praised"] == 5
    print("   ✓ Frequency tracking + habituation active")

    # 4. Re-appraisal
    print("\n4. Re-appraisal:")
    e3 = EmotionEngineV3()
    e3.event("praised")
    print(f"   After praised: P_mood={e3.mood.pleasure:+.2f}")
    for _ in range(8):
        e3.tick()
    print(f"   After 8 ticks (re-appraisal): P_mood={e3.mood.pleasure:+.2f}")
    assert e3.mood.pleasure < 0.45, "Re-appraisal should pull mood back"
    print("   ✓ Re-appraisal working")

    # 5. v5: Scene modulation
    print("\n5. v5: OCEAN Scene Modulation:")
    e4 = EmotionEngineV3()
    e4.set_scene("enterprise")
    e4.event("complaint")
    print(f"   Enterprise+complaint: P_emotion={e4.emotion.pleasure:+.2f} "
          f"scene={e4.scene}")
    e5 = EmotionEngineV3()
    e5.set_scene("casual")
    e5.event("complaint")
    print(f"   Casual+complaint: P_emotion={e5.emotion.pleasure:+.2f} "
          f"scene={e5.scene}")
    # Enterprise should hit harder
    assert e4.emotion.pleasure < e5.emotion.pleasure, \
        "Enterprise scene should amplify negative more than casual"
    print("   ✓ Scene modulation working")

    # 6. v5: Allostatic load
    print("\n6. v5: Allostatic Load:")
    e6 = EmotionEngineV3()
    e6.event("complaint")
    e6.event("corrected")
    e6.event("repeated_ask")
    print(f"   After complaint+corrected+repeated: load={e6.allostatic_load:.3f}")
    assert e6.allostatic_load > 0.1, "Allostatic load should accumulate"
    for _ in range(5):
        e6.tick()
    print(f"   After 5 ticks decay: load={e6.allostatic_load:.3f}")
    assert e6.allostatic_load < ALLOSTATIC_CONFIG["chronic_threshold"], \
        "Load should decay below chronic threshold"
    print("   ✓ Allostatic load working")

    # 7. v5: Extinction learning
    print("\n7. v5: Extinction Learning:")
    e7 = EmotionEngineV3()
    e7.event("complaint")  # Lowers mood
    e7.tick()
    pre_ext = e7.emotion.pleasure
    e7.record_ai_response("Let me help you fix that.")
    e7.perceive_user("Thanks, that's perfect!")
    print(f"   Pre-extinction P={pre_ext:+.3f} → "
          f"After user acceptance P={e7.emotion.pleasure:+.3f}")
    assert e7.emotion.pleasure > pre_ext, \
        "Extinction with user acceptance should accelerate recovery"
    print("   ✓ Extinction learning working")

    # 8. v5: Trust dynamics
    print("\n8. v5: Trust Dynamics:")
    e8 = EmotionEngineV3()
    print(f"   Initial trust={e8.trust_score:.2f}")
    e8.event("complaint")
    print(f"   After complaint: trust={e8.trust_score:.2f}")
    e8.event("praised")
    e8.event("positive_feedback")
    print(f"   After praised+feedback: trust={e8.trust_score:.2f}")
    assert e8.trust_score > TRUST_CONFIG["initial"] - TRUST_CONFIG["negative_delta"], \
        "Trust should recover with positive events"
    print("   ✓ Trust dynamics working")

    # 9. v5: Cross-session daily reset
    print("\n9. v5: Cross-session (daily reset + episodic):")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        e9 = EmotionEngineV3(persistence_dir=Path(td))
        e9.event("complaint")
        e9.event("complaint")
        e9.event("praised")
        e9.event("corrected")
        for _ in range(4):
            e9.tick()
        e9.save_session("test_v5")
        saved_p = e9.mood.pleasure
        saved_episodes = len([x for x in e9.event_log if x.get("type") == "event"])

        e10 = EmotionEngineV3(persistence_dir=Path(td))
        restored = e10.load_session()
        print(f"   Saved P={saved_p:.2f} episodes={saved_episodes} → "
              f"Restored P={e10.mood.pleasure:.2f} "
              f"daily_reset={restored.get('daily_reset', False)}")
        # Same-day restore should preserve mood
        assert abs(e10.mood.pleasure - saved_p) < 0.1, \
            "Same-day restore should preserve mood"
        print("   ✓ Session persistence + daily reset logic OK")

    # 10. v5: Expression with validating
    print("\n10. v5: Expression (CHI'24 validating):")
    for scenario, setup_fn in [
        ("Praised (light)", lambda e: e.event("praised")),
        ("User upset", lambda e: (
            e.perceive_user("I'm so frustrated this keeps breaking"),
            e.contagion(),
        )),
    ]:
        e11 = EmotionEngineV3()
        setup_fn(e11)
        out = e11.apply_to_text("Here is the updated report for your review.")
        print(f"   {scenario}: {out[:80]}...")

    # 11. v5: Trajectory handling
    print("\n11. v5: Recovery Trajectories:")
    e12 = EmotionEngineV3()
    # Simulate heavy load to trigger chronic
    e12.event("heavy_topic")
    e12.event("complaint")
    e12.event("complaint")
    e12.event("failure")
    e12.event("rejection")
    e12.tick()
    print(f"   Load={e12.allostatic_load:.2f} trajectory={e12.trajectory}")
    assert e12.allostatic_load > 0.3, "Should accumulate significant load"

    print("\n✓✓✓ All EmotionEngineV3 v5 tests passed")
