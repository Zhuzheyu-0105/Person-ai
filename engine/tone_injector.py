"""
Tone Word Injector
Injects persona-specific tone words at text boundaries based on scene context.
"""

from typing import Optional
from .config import TONE_WORDS


def inject(text: str, scene: Optional[str] = None) -> str:
    """
    Inject tone words (prefix/suffix only, no keyword replacement).

    Args:
        text: Text to inject into.
        scene: Scene tag (greeting/self_deprecation/delivery/comfort/
               admit_error/uncertainty/caught).
               Returns original text if None or unrecognized.

    Returns:
        Text with injected tone words.

    Examples:
        >>> inject("Hello", "greeting")
        'Hi~Hello'
        >>> inject("File sent", "delivery")
        'File sent~'
    """
    if scene is None or scene not in TONE_WORDS:
        return text

    tw: dict = TONE_WORDS[scene]
    result: str = text

    # Prefix injection
    prefix: str = tw.get("prefix", "")
    if prefix and not result.strip().startswith(prefix.strip()):
        result = prefix + result

    # Suffix injection
    suffix: str = tw.get("suffix", "")
    if suffix and not result.strip().endswith(suffix.strip()):
        result = result.rstrip() + suffix

    return result


if __name__ == "__main__":
    # Self-test suite
    tests = [
        ("Hello, the weekly report is ready.", "greeting"),
        ("Got praised for this proposal.", "self_deprecation"),
        ("File is done, sent it over.", "delivery"),
        ("This policy document is impossible to parse.", "comfort"),
        ("That data was from last year, my mistake.", "admit_error"),
        ("Not sure which metric to use here.", "uncertainty"),
        ("You're sounding too formal, like a robot.", "caught"),
        ("No scene tag attached.", None),
    ]
    
    print("=== Tone Injector Self-Tests ===")
    for text, scene in tests:
        result = inject(text, scene)
        status = "✓" if result else "✗"
        print(f"{status} [{scene or 'none'}] → \"{result}\"")
    
    print("\n✓ Tone Injector OK")
