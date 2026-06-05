"""
Layer 2 Personality Engine · Main Pipeline (v4)

LLM raw output → Tone Injection → Style Fix → Emotion Modulation → Consistency Check → Output

Usage:
    from engine import run_pipeline, EmotionEngineV3

    emo = EmotionEngineV3()
    emo.perceive_user("user message text")
    emo.contagion()
    emo.event("greeting")

    result = run_pipeline("raw reply text", emotion=emo)
    emo.tick()
    emo.save_session()
"""
from .tone_injector import inject
from .style_fixer import fix
from .consistency_checker import check, CheckResult
from .emotion_engine_v3 import EmotionEngineV3


def run_pipeline(
    raw_text: str,
    scene: str | None = None,
    emotion: EmotionEngineV3 | None = None,
) -> CheckResult:
    """Complete personality engine pipeline (v4).

    Args:
        raw_text: LLM raw output
        scene:     Optional scene tag (greeting/delivery/comfort/...)
        emotion:   Optional EmotionEngineV3 instance

    Returns:
        CheckResult with final text and validation result.
    """
    text = raw_text
    text = inject(text, scene=scene)
    text = fix(text)
    if emotion is not None:
        text = emotion.apply_to_text(text)
    return check(text)


if __name__ == "__main__":
    from .emotion_engine_v3 import EmotionEngineV3

    raw = "The weekly update covers three areas: policy changes, regional implementation, and local meetings. Take a look at the adjustments."

    # Test: praised + contagion
    print("=== v4 Pipeline with Emotion ===")
    emo = EmotionEngineV3()
    emo.perceive_user("This is great work, really impressive!")
    emo.contagion()
    emo.event("praised")
    r = run_pipeline(raw, scene="delivery", emotion=emo)
    print(f"User: {emo.user_label}  Emotion: {emo.emotion_label}")
    print(f"Output: {r.text}")
    emo.tick()

    # Test: negative user + validating
    print("\n=== Negative User + Validating ===")
    emo2 = EmotionEngineV3()
    emo2.perceive_user("I'm so frustrated, nothing works today")
    emo2.contagion()
    emo2.event("complaint")
    r2 = run_pipeline(raw, scene="comfort", emotion=emo2)
    print(f"User: {emo2.user_label}  Emotion: {emo2.emotion_label}")
    print(f"Output: {r2.text}")
    emo2.tick()

    print("\n✓ Pipeline v4 OK")
