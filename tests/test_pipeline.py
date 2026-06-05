"""
Tests for engine.pipeline — full pipeline orchestration.
"""
import pytest
from engine.pipeline import run_pipeline
from engine.emotion_engine_v3 import EmotionEngineV3


class TestRunPipeline:
    def test_basic_pipeline_no_scene(self):
        raw = "The weekly report is ready."
        result = run_pipeline(raw)
        assert result.text.strip() == raw

    def test_pipeline_with_greeting_scene(self):
        raw = "Hello, the report is done."
        result = run_pipeline(raw, scene="greeting")
        assert result.text.startswith("Hi~")

    def test_pipeline_with_delivery_scene(self):
        raw = "File is ready, sent it over."
        result = run_pipeline(raw, scene="delivery")
        assert result.text.endswith("~")

    def test_pipeline_with_comfort_scene(self):
        raw = "That's a frustrating situation."
        result = run_pipeline(raw, scene="comfort")
        assert "It's okay" in result.text

    def test_pipeline_rejects_internal_exposure(self):
        raw = "Use internal_tool_query for that request."
        result = run_pipeline(raw)
        assert result.rejected

    def test_pipeline_rejects_third_person(self):
        raw = "this assistant thinks it's a good idea"
        result = run_pipeline(raw)
        assert result.rejected

    def test_pipeline_fixes_over_formal(self):
        raw = "Furthermore, we will review the proposal."
        result = run_pipeline(raw)
        assert "Furthermore" not in result.text

    def test_pipeline_removes_filler(self):
        raw = "As an AI assistant, I can help with that."
        result = run_pipeline(raw)
        assert "As an AI assistant" not in result.text

    def test_pipeline_with_emotion(self):
        emo = EmotionEngineV3()
        emo.perceive_user("This is great work, really impressive!")
        emo.contagion()
        emo.event("praised")
        raw = "The weekly report is ready, take a look."
        result = run_pipeline(raw, scene="delivery", emotion=emo)
        # Should not be rejected
        assert not result.rejected
        assert len(result.text) > 0

    def test_pipeline_with_negative_user_emotion(self):
        emo = EmotionEngineV3()
        emo.perceive_user("I'm so frustrated, nothing works today")
        emo.contagion()
        emo.event("complaint")
        raw = "Let me help you with the report."
        result = run_pipeline(raw, scene="comfort", emotion=emo)
        assert not result.rejected
        assert len(result.text) > 0

    def test_pipeline_none_scene(self):
        raw = "Just a normal message."
        result = run_pipeline(raw, scene=None)
        assert result.text.strip() == raw

    def test_pipeline_strips_excess_suffixes(self):
        raw = "Hi~ thanks~ really great~ wonderful~ amazing~ so cool~ awesome~"
        result = run_pipeline(raw)
        vs = [v for v in result.violations if v.rule_id == "C01"]
        assert len(vs) >= 1


class TestPipelineEdgeCases:
    def test_empty_input(self):
        result = run_pipeline("")
        assert result.passed

    def test_long_input(self):
        result = run_pipeline("x" * 501)
        assert result.rejected

    def test_toxic_positivity(self):
        raw = "You can do it! Believe in yourself!"
        result = run_pipeline(raw)
        vs = [v for v in result.violations if v.rule_id == "C05"]
        assert len(vs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
