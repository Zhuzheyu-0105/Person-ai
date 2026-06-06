"""
Layer 2 Personality Engine · Main Pipeline (v5)

LLM raw output → Tone Injection → Style Fix → Emotion Modulation → Consistency Check → Output

v5 changes:
  - Dual-channel: emotion state affects both pre-prompt AND post-process
  - reject_fallback: auto retry on consistency rejection (max 3 attempts)
  - Integrated extinction learning + record_ai_response in full lifecycle
  - Scene parameter propagated through entire chain
  - run_pipeline_v5() for complete lifecycle (perceive → contagion → event → pipeline → record → tick)

Usage:
    from engine import run_pipeline, run_pipeline_v5, EmotionEngineV3

    # Minimal (backward compatible)
    emo = EmotionEngineV3()
    emo.perceive_user("user message text")
    emo.contagion()
    emo.event("greeting")
    result = run_pipeline("raw reply text", emotion=emo)
    emo.tick()
    emo.save_session()

    # v5 full lifecycle
    result = run_pipeline_v5(
        user_text="I'm frustrated",
        raw_reply="Let me help you with that.",
        emo=emo,
        event_type="complaint",
        scene="enterprise",
    )
"""
from typing import Optional, Callable
from .tone_injector import inject
from .style_fixer import fix
from .consistency_checker import check, CheckResult
from .emotion_engine_v3 import EmotionEngineV3


def run_pipeline(
    raw_text: str,
    scene: str | None = None,
    emotion: EmotionEngineV3 | None = None,
) -> CheckResult:
    """Complete personality engine pipeline (v5).

    Args:
        raw_text: LLM raw output
        scene:     Optional scene tag (greeting/delivery/comfort/enterprise/...)
        emotion:   Optional EmotionEngineV3 instance

    Returns:
        CheckResult with final text and validation result.
    """
    text = raw_text
    text = inject(text, scene=scene)
    text = fix(text)
    if emotion is not None:
        text = emotion.apply_to_text(text)
    return check(text, scene=scene)


def run_pipeline_with_fallback(
    raw_text: str,
    scene: str | None = None,
    emotion: EmotionEngineV3 | None = None,
    regenerate_fn: Optional[Callable[[str], str]] = None,
    max_retries: int = 3,
) -> CheckResult:
    """v5: Run pipeline with automatic retry on rejection.

    When consistency check rejects, calls regenerate_fn(rejection_reason)
    to get a new LLM response, then retries the pipeline.

    Args:
        raw_text:      LLM raw output
        scene:         Scene tag
        emotion:       EmotionEngineV3 instance
        regenerate_fn: Function(reason) → new_text for LLM regeneration
        max_retries:   Maximum regeneration attempts (default: 3)

    Returns:
        CheckResult — passes if any attempt succeeds, otherwise last rejection.
    """
    result = run_pipeline(raw_text, scene=scene, emotion=emotion)

    retries = 0
    while result.rejected and retries < max_retries and regenerate_fn is not None:
        retries += 1
        # Build rejection reason for LLM
        reasons = "; ".join(v.detail for v in result.violations)
        rejection_context = (
            f"Your previous response was rejected for these reasons: {reasons}. "
            f"Please regenerate, keeping the same content but fixing the issues. "
            f"Scene: {scene or 'general'}."
        )
        new_text = regenerate_fn(rejection_context)
        if not new_text:
            break
        result = run_pipeline(new_text, scene=scene, emotion=emotion)

    return result


def run_pipeline_v5(
    user_text: str,
    raw_reply: str,
    emo: EmotionEngineV3,
    event_type: str = "greeting",
    scene: str | None = None,
    regenerate_fn: Optional[Callable[[str], str]] = None,
) -> CheckResult:
    """v5: Complete emotion+pipeline lifecycle in one call.

    Full cycle:
      1. perceive_user(user_text) → lexicon classifier + extinction_step
      2. contagion() → trust-modulated emotional infection
      3. set_scene(scene) → OCEAN scene modulation
      4. event(event_type) → OCC appraisal with frequency habituation
      5. run_pipeline(raw_reply) → tone injection + style fix + expression
      6. record_ai_response(result.text) → enable extinction on next turn
      7. tick() → allostatic decay + trajectory recovery

    The prompt context (for pre-prompt injection) is accessible via
    emo.get_prompt_context() after calling this function.

    Args:
        user_text:      User's message text
        raw_reply:      LLM raw reply text
        emo:            EmotionEngineV3 instance (state is mutated)
        event_type:     Event type for OCC appraisal
        scene:          Scene tag for OCEAN modulation
        regenerate_fn:  Optional LLM regeneration callback for reject fallback

    Returns:
        CheckResult with final validated text.
    """
    # Step 1-2: Perceive + contagion
    emo.perceive_user(user_text)
    emo.contagion()

    # Step 3: Scene modulation
    if scene:
        emo.set_scene(scene)

    # Step 4: Event appraisal
    emo.event(event_type)

    # Step 5-6: Pipeline + record
    result = run_pipeline_with_fallback(
        raw_text=raw_reply,
        scene=scene or emo.scene,
        emotion=emo,
        regenerate_fn=regenerate_fn,
    )
    emo.record_ai_response(result.text)

    # Step 7: Tick
    emo.tick()

    return result


if __name__ == "__main__":
    from .emotion_engine_v3 import EmotionEngineV3

    raw = "The weekly update covers three areas: policy changes, regional implementation, and local meetings. Take a look at the adjustments."

    # Test: praised + contagion (backward compat)
    print("=== v5 Pipeline (backward compat) ===\n")
    emo = EmotionEngineV3()
    emo.perceive_user("This is great work, really impressive!")
    emo.contagion()
    emo.event("praised")
    r = run_pipeline(raw, scene="delivery", emotion=emo)
    print(f"User: {emo.user_label}  Emotion: {emo.emotion_label}")
    print(f"Scene: {emo.scene}  Trust: {emo.trust_score:.2f}")
    print(f"Output: {r.text[:100]}...")
    emo.tick()

    # Test: v5 full lifecycle
    print("\n=== v5 Full Lifecycle ===\n")
    emo2 = EmotionEngineV3()
    result = run_pipeline_v5(
        user_text="I'm so frustrated, nothing works today",
        raw_reply=raw,
        emo=emo2,
        event_type="complaint",
        scene="enterprise",
    )
    print(f"User: {emo2.user_label}  Emotion: {emo2.emotion_label}")
    print(f"Load: {emo2.allostatic_load:.2f}  Trajectory: {emo2.trajectory}")
    print(f"Scene: {emo2.scene}  Trust: {emo2.trust_score:.2f}")
    print(f"Output: {result.text[:100]}...")

    # Test: neg user + validating (backward compat)
    print("\n=== Negative User + Validating (v5 CHI'24 phrases) ===\n")
    emo3 = EmotionEngineV3()
    emo3.perceive_user("I've been feeling really down lately")
    emo3.contagion()
    emo3.event("heavy_topic")
    r3 = run_pipeline(raw, scene="comfort", emotion=emo3)
    print(f"User: {emo3.user_label}  Emotion: {emo3.emotion_label}")
    print(f"Output: {r3.text[:100]}...")
    emo3.tick()

    # Test: reject fallback
    print("\n=== Reject Fallback ===\n")
    emo4 = EmotionEngineV3()
    toxic_raw = "You can do it! Believe in yourself! Just keep going!"
    # Without fallback: C05 fixes it
    r4 = run_pipeline(toxic_raw, emotion=emo4)
    print(f"Without fallback: {'PASS' if r4.passed else ('FIXED' if r4.fixed else 'REJECT')}")
    print(f"  Output: {r4.text[:80]}...")
    # With fallback: would regenerate if rejected
    r5 = run_pipeline_with_fallback(
        "this assistant thinks you should try harder", 
        regenerate_fn=lambda reason: "I think you might want to give it another shot.",
    )
    print(f"With fallback (C02 reject → regenerate): {'PASS' if r5.passed else 'REJECT'}")
    print(f"  Output: {r5.text[:80]}...")

    print("\n✓ Pipeline v5 OK")
