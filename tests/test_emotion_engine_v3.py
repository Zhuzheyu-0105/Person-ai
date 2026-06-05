"""
Tests for engine.emotion_engine_v3 — six-layer emotion engine.
"""
import pytest
import tempfile
from pathlib import Path
from engine.emotion_engine_v3 import (
    EmotionEngineV3, PADState, PERSONA_BASELINE,
    perceive_user_emotion, appraise_event, classify_emotion_natural,
    _decay_factor,
)


class TestPADState:
    def test_default_initialization(self):
        pad = PADState()
        assert pad.pleasure == 0.0
        assert pad.arousal == 0.5
        assert pad.dominance == 0.0

    def test_clamp_bounds(self):
        pad = PADState(pleasure=2.0, arousal=2.0, dominance=2.0)
        pad.clamp()
        assert pad.pleasure == 1.0
        assert pad.arousal == 1.0
        assert pad.dominance == 1.0

    def test_clamp_lower_bounds(self):
        pad = PADState(pleasure=-2.0, arousal=-1.0, dominance=-2.0)
        pad.clamp()
        assert pad.pleasure == -1.0
        assert pad.arousal == 0.0
        assert pad.dominance == -1.0

    def test_copy_independent(self):
        pad = PADState(0.5, 0.6, 0.7)
        copy = pad.copy()
        copy.pleasure = 0.0
        assert pad.pleasure == 0.5  # original unchanged

    def test_distance_to(self):
        a = PADState(0.0, 0.0, 0.0)
        b = PADState(3.0, 4.0, 0.0)
        assert abs(a.distance_to(b) - 5.0) < 0.001

    def test_to_dict_roundtrip(self):
        pad = PADState(0.1234, 0.5678, -0.4321)
        d = pad.to_dict()
        restored = PADState.from_dict(d)
        assert abs(restored.pleasure - pad.pleasure) < 0.001
        assert abs(restored.arousal - pad.arousal) < 0.001
        assert abs(restored.dominance - pad.dominance) < 0.001


class TestDecayFactor:
    def test_half_life_2(self):
        f = _decay_factor(2.0)
        assert 0.7 < f < 0.75  # 0.5^(1/2) ≈ 0.707

    def test_half_life_zero(self):
        assert _decay_factor(0.0) == 0.0

    def test_half_life_negative(self):
        assert _decay_factor(-1.0) == 0.0


class TestPerceiveUserEmotion:
    def test_frustration_detected(self):
        pad, label, conf = perceive_user_emotion("I'm so frustrated with this bug")
        assert label == "frustration"
        assert conf > 0.1

    def test_happiness_detected(self):
        pad, label, conf = perceive_user_emotion("This is great, amazing work!")
        assert label in ("happy", "satisfied")

    def test_sadness_detected(self):
        pad, label, conf = perceive_user_emotion("I'm feeling really sad and depressed today")
        assert label == "sadness"
        assert conf > 0.1

    def test_neutral_fallback(self):
        pad, label, conf = perceive_user_emotion("qwerty asdfgh zxcvbn")
        assert label == "neutral"
        assert conf <= 0.3

    def test_empty_text(self):
        pad, label, conf = perceive_user_emotion("")
        assert label == "neutral"


class TestAppraiseEvent:
    def test_praised_event(self):
        a = appraise_event("praised")
        assert a.desirability > 0
        assert a.coping_potential > 0

    def test_corrected_event(self):
        a = appraise_event("corrected")
        assert a.desirability < 0
        # causal_self = 0.65 * 0.65 ≈ 0.42 (neg bias ratio)
        assert a.causal_self > 0.4  # personality bias: takes responsibility

    def test_complaint_event(self):
        a = appraise_event("complaint")
        assert a.desirability < 0

    def test_unknown_event_defaults(self):
        a = appraise_event("nonexistent_event_type")
        assert isinstance(a.desirability, float)


class TestEmotionEngineV3:

    def test_init_defaults(self):
        e = EmotionEngineV3()
        assert e.emotion_label in ("steady", "gentle", "content")
        assert e.user_label == "neutral"
        assert e.turn_count == 0

    def test_perceive_user(self):
        e = EmotionEngineV3()
        r = e.perceive_user("I'm so frustrated with this")
        assert r["label"] == "frustration"
        assert "pad" in r

    def test_contagion_positive(self):
        e = EmotionEngineV3()
        e.perceive_user("This is amazing, great work!")
        result = e.contagion()
        assert result["contagion_applied"]
        assert e.emotion.pleasure > PERSONA_BASELINE.pleasure

    def test_contagion_negative_with_floor(self):
        e = EmotionEngineV3()
        e.perceive_user("I'm completely exhausted and depressed")
        result = e.contagion()
        assert result["contagion_applied"]
        assert e.emotion.pleasure >= -0.3  # floor protection

    def test_contagion_low_confidence_skipped(self):
        e = EmotionEngineV3()
        e.perceive_user("xyz random gibberish")
        # force low confidence
        e.user_confidence = 0.0
        result = e.contagion()
        assert not result["contagion_applied"]

    def test_event_triggers_emotion(self):
        e = EmotionEngineV3()
        r = e.event("praised")
        assert r["type"] == "event"
        assert len(e.event_log) >= 1

    def test_event_log_capped_at_30(self):
        e = EmotionEngineV3()
        for i in range(35):
            e.event("greeting")
        assert len(e.event_log) <= 30

    def test_tick_decays_emotion(self):
        e = EmotionEngineV3()
        e.event("praised")
        p_before = e.emotion.pleasure
        for _ in range(10):
            e.tick()
        p_after = e.emotion.pleasure
        # After enough ticks, emotion should drift back toward mood/baseline
        assert abs(p_after - p_before) > 0.01  # some change happened

    def test_reappraisal_after_settling(self):
        e = EmotionEngineV3()
        e.event("praised")
        # Tick enough to settle + trigger reappraisal at tick 3-post-settle
        for _ in range(15):
            e.tick()
        # Should have at least one reappraisal log entry
        reappraisals = [log for log in e.event_log if log.get("type") == "reappraisal"]
        assert len(reappraisals) >= 1

    def test_get_modifiers(self):
        e = EmotionEngineV3()
        mods = e.get_modifiers()
        assert "suffix_bonus" in mods
        assert "label" in mods
        assert "source" in mods

    def test_apply_to_text_basic(self):
        e = EmotionEngineV3()
        text = "Here is the report."
        result = e.apply_to_text(text)
        assert len(result) > 0

    def test_apply_to_text_with_validating(self):
        e = EmotionEngineV3()
        e.perceive_user("I'm so sad and depressed today")
        e.contagion()
        text = "Let me help with that."
        result = e.apply_to_text(text)
        # May or may not get validating prefix (random), but should still produce text
        assert len(result) >= len(text)

    def test_session_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            e1 = EmotionEngineV3(persistence_dir=Path(td))
            e1.event("praised")
            e1.event("success")
            for _ in range(4):
                e1.tick()
            saved_p = e1.mood.pleasure
            e1.save_session("test-session")

            e2 = EmotionEngineV3(persistence_dir=Path(td))
            restored = e2.load_session()
            assert restored is not None
            assert restored["restored"]
            assert abs(e2.mood.pleasure - saved_p) < 0.1

    def test_load_session_nonexistent(self):
        e = EmotionEngineV3(persistence_dir=Path("/nonexistent/path/xyz"))
        assert e.load_session() is None

    def test_repr(self):
        e = EmotionEngineV3()
        r = repr(e)
        assert "EmotionEngineV3" in r
        assert "user=" in r

    def test_rhythm_property(self):
        e = EmotionEngineV3()
        r = e.rhythm()
        assert "tempo" in r
        assert "delay_ms" in r
        assert r["tempo"] in ("normal", "gentle", "quick", "slow")


class TestEmotionEngineIntegration:
    """Full conversation flow tests."""

    def test_full_conversation_cycle(self):
        e = EmotionEngineV3()
        # Turn 1: user greets
        e.perceive_user("Hi there!")
        e.contagion()
        e.event("greeting")
        e.tick()
        assert e.turn_count == 1

        # Turn 2: user is frustrated
        e.perceive_user("This is so frustrating, nothing works")
        e.contagion()
        e.event("complaint")
        e.tick()
        assert e.emotion.pleasure < PERSONA_BASELINE.pleasure

        # Turn 3+: emotion should recover
        for _ in range(10):
            e.tick()
        # Should approach baseline
        assert abs(e.emotion.pleasure - PERSONA_BASELINE.pleasure) < 0.4

    def test_persona_baseline_is_stable(self):
        """PERSONA_BASELINE should be within valid PAD range."""
        assert -1.0 <= PERSONA_BASELINE.pleasure <= 1.0
        assert 0.0 <= PERSONA_BASELINE.arousal <= 1.0
        assert -1.0 <= PERSONA_BASELINE.dominance <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
