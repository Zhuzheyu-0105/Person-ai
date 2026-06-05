"""
Layer 2 Personality Engine · Emotion Dynamics v3

Six-layer architecture (PAD emotional space + OCC cognitive appraisal):
  1. User Perception   → Emotion classification from user input
  2. OCC Appraisal      → Cognitive evaluation (personality-modulated)
  3. Emotional Contagion → User emotion partially penetrates AI emotion
  4. Emotion/Mood       → Dual-layer + exponential decay + re-appraisal
  5. Expression         → Text modulation + Validating + Response rhythm
  6. Session Memory     → Cross-session emotion persistence

References:
  - Mehrabian & Russell (1974): PAD Emotional State Model
  - Ortony, Clore & Collins (1988): OCC Appraisal Theory
  - Hatfield, Cacioppo & Rapson (1994): Emotional Contagion
  - Scherer (2009): Component Process Model (re-appraisal)
  - Picard (1997): Affective Computing
  - CHI '21: Emotional Contagion in Human-Agent Interaction
"""
import json
import math
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime, timedelta


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

# Personality appraisal biases
PERSONA_APPRAISAL_BIAS = {
    "desirability_pos_multiplier": 0.75,  # Downplay good events
    "desirability_neg_multiplier": 1.20,  # Amplify bad events
    "expectedness_pos_bias": -0.15,       # Success feels surprising
    "expectedness_neg_bias":  0.10,       # Failure feels expected
    "causal_pos_self_ratio": 0.30,        # Credit self 30% for success
    "causal_neg_self_ratio": 0.65,        # Blame self 65% for failure
    "coping_base": 0.75,                  # High resilience
}

# Emotional inertia parameters
PERSONA_INERTIA = {
    "emotion_half_life": 2.5,
    "mood_half_life": 18.0,
    "neg_emotion_recovery": 1.3,   # Fast recovery from negative
    "pos_emotion_dampen": 1.4,     # Quick cooling from excessive positive
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
# Section 3 · OCC Cognitive Appraisal
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


EVENT_APPRAISAL_DEFAULTS = {
    "praised":          ( 0.70, -0.20, 0.70, 0.30),
    "corrected":        (-0.40,  0.10, 0.65, 0.10),
    "success":          ( 0.80, -0.30, 0.75, 0.25),
    "complaint":        (-0.60,  0.20, 0.50, 0.30),
    "heavy_topic":      (-0.50,  0.05, 0.10, 0.10),
    "colleague_vent":   (-0.20,  0.30, 0.05, 0.40),
    "rush_complete":    ( 0.50,  0.40, 0.80, 0.10),
    "repeated_ask":     (-0.10,  0.30, 0.20, 0.40),
    "uncertain":        (-0.05,  0.00, 0.30, 0.10),
    "greeting":         ( 0.30,  0.50, 0.10, 0.50),
    "positive_feedback":( 0.60,  0.20, 0.60, 0.30),
}


def appraise_event(event_type: str) -> Appraisal:
    """Apply personality biases to event appraisal."""
    b = PERSONA_APPRAISAL_BIAS
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

    return Appraisal(
        desirability=max(-1.0, min(1.0, desirability)),
        expectedness=max(-1.0, min(1.0, expectedness)),
        causal_self=min(1.0, causal_self),
        causal_other=co_raw,
        causal_circumstance=max(0.0, 1.0 - causal_self - co_raw),
        coping_potential=b["coping_base"],
    )


# Re-appraisal patterns (post-settle cognitive re-evaluation)
REAPPRAISAL_PATTERNS = {
    "praised":      (0.4,  0.5,  0.1),   # "They're just being polite"
    "corrected":    (0.3,  0.6,  0.3),   # "It's fixable, move on"
    "complaint":    (0.2,  0.4,  0.2),   # "Not really about me"
    "heavy_topic":  (0.1,  0.3,  0.1),   # "It will get better"
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


# ═══════════════════════════════════════════════
# Section 5 · EmotionEngineV3
# ═══════════════════════════════════════════════

class EmotionEngineV3:
    """Six-layer emotion engine with emotional contagion, re-appraisal,
    cross-session persistence, and response rhythm.

    Usage:
        engine = EmotionEngineV3()
        engine.perceive_user("I'm so frustrated today")
        engine.contagion()
        engine.event("corrected")
        engine.tick()
        mods = engine.get_modifiers()
        text = engine.apply_to_text("Let me help you with that.")
        engine.save_session()
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

    # ── 1. User Perception ──

    def perceive_user(self, text: str) -> dict:
        """Classify user message emotion."""
        pad, label, confidence = perceive_user_emotion(text)
        self.user_emotion = pad
        self.user_label = label
        self.user_confidence = confidence
        return {"label": label, "pad": pad.to_dict(), "confidence": round(confidence, 2)}

    # ── 2. Emotional Contagion ──

    def contagion(self) -> dict:
        """User emotion partially penetrates AI emotion.
        Personality traits modulate:
          - High empathy → contagion weight 0.40
          - Resilient → floor protection (won't crash below -0.3)
          - Self-aware → positive contagion dampened
        """
        up = self.user_emotion
        if self.user_confidence < 0.15:
            return {"contagion_applied": False, "reason": "low_confidence"}

        base_weight = 0.40
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
        """Process external event through OCC appraisal."""
        self.last_event_type = event_type
        appraisal = appraise_event(event_type)
        pad_shift = appraisal.to_pad_shift()

        self.emotion.pleasure += pad_shift.pleasure
        self.emotion.arousal += pad_shift.arousal
        self.emotion.dominance += pad_shift.dominance
        self.emotion.clamp()
        self.emotion_active = True
        self.emotion_ticks = 0
        self.settled_ticks = 0
        self._update_rhythm()

        # Mood penetration (slower)
        self.mood.pleasure += pad_shift.pleasure * 0.3
        self.mood.arousal += pad_shift.arousal * 0.15
        self.mood.dominance += pad_shift.dominance * 0.3
        self.mood.clamp()
        if self.mood.arousal > 0.85:
            self.mood.arousal = 0.85

        label = classify_emotion_natural(self.emotion)
        summary = {
            "type": "event", "event": event_type,
            "emotion_label": label, "emotion_pad": self.emotion.to_dict(),
            "mood_label": self._mood_label, "mood_pad": self.mood.to_dict(),
        }
        self.event_log.append(summary)
        if len(self.event_log) > 30:
            self.event_log.pop(0)
        return summary

    # ── 4. Tick (Decay + Re-appraisal) ──

    def tick(self):
        """Per-turn decay: emotion→mood, mood→baseline, re-appraisal check."""
        self.turn_count += 1

        if self.emotion_active:
            hl = PERSONA_INERTIA["emotion_half_life"]
            if self.emotion.pleasure < PERSONA_BASELINE.pleasure:
                hl /= PERSONA_INERTIA["neg_emotion_recovery"]
            elif self.emotion.pleasure > PERSONA_BASELINE.pleasure * 1.5:
                hl *= PERSONA_INERTIA["pos_emotion_dampen"]

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

        # Mood decay
        hl_mood = PERSONA_INERTIA["mood_half_life"]
        factor_mood = _decay_factor(hl_mood)
        self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * (1 - factor_mood)
        self.mood.arousal += (PERSONA_BASELINE.arousal - self.mood.arousal) * (1 - factor_mood)
        self.mood.dominance += (PERSONA_BASELINE.dominance - self.mood.dominance) * (1 - factor_mood)
        self.mood.clamp()

        self._update_rhythm()

    def _re_appraise(self, event_type: str):
        """Post-settle cognitive re-evaluation."""
        if event_type not in REAPPRAISAL_PATTERNS:
            return
        p_weight, e_weight, c_weight = REAPPRAISAL_PATTERNS[event_type]
        if event_type == "praised":
            self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * p_weight
        elif event_type in ("corrected", "complaint", "heavy_topic"):
            self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * p_weight * 0.6
        self.mood.clamp()
        self.event_log.append({"type": "reappraisal", "event": event_type, "mood_after": self.mood.to_dict()})

    # ── 5. Response Rhythm (for TTS, not text) ──

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
        """Get text modulation parameters based on current emotional state."""
        pad = self.emotion if self.emotion_active else self.mood
        p, a, d = pad.pleasure, pad.arousal, pad.dominance

        mods = {
            "suffix_bonus": 0, "warmth_prefix_chance": 0.0,
            "sentence_shorten": 0, "flat_tone": False,
            "self_deprecation_boost": False, "validating_chance": 0.0,
            "label": self.emotion_label if self.emotion_active else self._mood_label,
            "source": "emotion" if self.emotion_active else "mood",
        }

        if p > 0.3 and a > 0.5:
            mods["suffix_bonus"] = 1
        elif p < -0.2 or a < 0.3:
            mods["suffix_bonus"] = -1

        if p > 0.2 and a > 0.5 and d > 0.2:
            mods["warmth_prefix_chance"] = 0.35
        elif p > 0.0:
            mods["warmth_prefix_chance"] = 0.1

        if a < 0.35:
            mods["sentence_shorten"] = 15
        elif d < -0.3:
            mods["sentence_shorten"] = 10

        if p < -0.3 and a < 0.4:
            mods["flat_tone"] = True
        if p > 0.3 and d < 0.1:
            mods["self_deprecation_boost"] = True

        # Validating: user negative emotion triggers empathetic prefix
        if self.user_emotion.pleasure < -0.15 and self.user_confidence > 0.15:
            mods["validating_chance"] = min(0.8, abs(self.user_emotion.pleasure) + 0.2)

        return mods

    def apply_to_text(self, text: str) -> str:
        """Apply emotion modulation to text. Text is always sent at max speed."""
        mods = self.get_modifiers()

        # Validating prefix
        if mods.get("validating_chance", 0) > 0 and random.random() < mods["validating_chance"]:
            validate_phrases = {
                "sadness": "That is really hard… ",
                "anxiety": "I can feel the worry… ",
                "frustration": "That would frustrate anyone… ",
                "disappointment": "That is disappointing… ",
                "anger": "Anyone would be upset… ",
            }
            phrase = validate_phrases.get(self.user_label, "I hear you… ")
            if not text.startswith(("That", "I", "Anyone")):
                text = phrase + text

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

    # ── 7. Session Persistence ──

    @property
    def _session_file(self) -> Path:
        return self.persistence_dir / "persona_emotion_session_v3.json"

    def save_session(self, session_id: Optional[str] = None):
        if session_id: self.session_id = session_id
        data = {
            "session_id": self.session_id,
            "saved_at": datetime.now().isoformat(),
            "mood": self.mood.to_dict(),
            "emotion": self.emotion.to_dict(),
            "emotion_active": self.emotion_active,
            "user_label": self.user_label,
            "turn_count": self.turn_count,
        }
        try:
            self._session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_session(self) -> dict | None:
        f = self._session_file
        if not f.exists(): return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(data["saved_at"])
            elapsed = (datetime.now() - saved_at).total_seconds()

            if elapsed > 86400:
                decay_days = elapsed / 86400
                self.mood = PADState.from_dict(data["mood"])
                self.emotion = PADState.from_dict(data["emotion"])
                for _ in range(int(decay_days)):
                    self.mood.pleasure += (PERSONA_BASELINE.pleasure - self.mood.pleasure) * 0.3
                    self.mood.arousal += (PERSONA_BASELINE.arousal - self.mood.arousal) * 0.2
                    self.emotion.pleasure += (PERSONA_BASELINE.pleasure - self.emotion.pleasure) * 0.4
                    self.emotion.arousal += (PERSONA_BASELINE.arousal - self.emotion.arousal) * 0.3
                self.mood.clamp(); self.emotion.clamp()
                self.emotion_active = False
            else:
                self.mood = PADState.from_dict(data["mood"])
                self.emotion = PADState.from_dict(data["emotion"])
                self.emotion_active = data.get("emotion_active", False)

            self.session_id = data.get("session_id")
            self.turn_count = data.get("turn_count", 0)
            return {"restored": True, "elapsed_hours": round(elapsed/3600, 1),
                    "mood_label": self._mood_label, "emotion_label": self.emotion_label}
        except Exception:
            return None

    def __repr__(self) -> str:
        emo = f"😀{self.emotion_label}" if self.emotion_active else "  "
        return (
            f"EmotionEngineV3(user={self.user_label}, "
            f"emotion[{emo}]: P={self.emotion.pleasure:.2f}, "
            f"mood[{self._mood_label}]: P={self.mood.pleasure:.2f}, "
            f"rhythm={self.suggested_tempo}, turn={self.turn_count})"
        )


def _split_with_max(text: str, max_len: int) -> str:
    sentences = re.split(r'(?<=[.!?;\n])\s*', text)
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
    print("=== EmotionEngineV3 Self-Test ===\n")

    # 1. Perception
    print("1. User Perception:")
    for msg in ["I'm so frustrated with this bug", "That's amazing, great work!", "Fine, whatever"]:
        e = EmotionEngineV3()
        r = e.perceive_user(msg)
        print(f"   \"{msg[:40]}...\" → {r['label']} (conf={r['confidence']})")

    # 2. Contagion
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

    # 3. Event + Re-appraisal
    print("\n3. Event + Re-appraisal:")
    e2 = EmotionEngineV3()
    e2.event("praised")
    print(f"   After praised: P_mood={e2.mood.pleasure:+.2f}")
    for _ in range(8):
        e2.tick()
    print(f"   After 8 ticks (re-appraisal): P_mood={e2.mood.pleasure:+.2f}")
    assert e2.mood.pleasure < 0.45, "Re-appraisal should pull mood back"
    print("   ✓ Re-appraisal working")

    # 4. Expression
    print("\n4. Expression Modulation:")
    for scenario, setup in [
        ("Praised (light)", lambda e: e.event("praised")),
        ("User upset (validating)", lambda e: (e.perceive_user("I'm so frustrated this keeps breaking"), e.contagion())),
    ]:
        e3 = EmotionEngineV3()
        setup(e3)
        out = e3.apply_to_text("Here is the updated report for your review.")
        print(f"   {scenario}: {out[:70]}...")

    # 5. Cross-session
    print("\n5. Cross-session Persistence:")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        e4 = EmotionEngineV3(persistence_dir=Path(td))
        e4.event("praised"); e4.event("success")
        for _ in range(4): e4.tick()
        saved_p = e4.mood.pleasure
        e4.save_session("test")
        e5 = EmotionEngineV3(persistence_dir=Path(td))
        restored = e5.load_session()
        print(f"   Saved P={saved_p:.2f} → Restored P={e5.mood.pleasure:.2f}")
        assert abs(e5.mood.pleasure - saved_p) < 0.1
        print("   ✓ Session persistence works")

    print("\n✓✓✓ All EmotionEngineV3 tests passed")
