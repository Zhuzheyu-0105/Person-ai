"""
Sentence Style Fixer
Long sentence splitting · Formal→Casual · Filler removal
"""
import re
from .config import SENTENCE_RULES

def split_long_sentences(text: str, max_len: int = None) -> str:
    """Split sentences exceeding max_len at punctuation boundaries."""
    if max_len is None:
        max_len = SENTENCE_RULES["max_sentence_len"]

    sentences = re.split(r'(?<=[.!?;\n])', text)
    sentences = [s for s in sentences if s]  # filter empty strings from zero-width match at end
    result = []
    for s in sentences:
        if len(s) <= max_len:
            result.append(s)
            continue
        # Try splitting at commas
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


def de_formalize(text: str) -> str:
    """Replace formal language with casual equivalents."""
    mapping = SENTENCE_RULES["formal_to_casual"]
    result = text
    for formal, casual in mapping.items():
        result = result.replace(formal, casual)
    return result


def remove_filler(text: str) -> str:
    """Remove filler phrases and clean up whitespace."""
    for pattern in SENTENCE_RULES["remove_patterns"]:
        text = text.replace(pattern, "")
    # Clean up excessive whitespace and leading punctuation
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^[,.\s]*', '', text)
    return text.strip()


def fix(text: str) -> str:
    """Pipeline: split → de-formalize → remove filler."""
    text = split_long_sentences(text)
    text = de_formalize(text)
    text = remove_filler(text)
    return text


if __name__ == "__main__":
    test = "Based on my analysis, from a policy perspective, we believe the most urgent task is to complete the data review and submit results to the leadership team for confirmation within this week."
    print(f"Original ({len(test)} chars): {test}")
    result = fix(test)
    print(f"Fixed ({len(result)} chars): {result}")
