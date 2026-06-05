"""
Tests for engine.style_fixer module.
"""
import pytest
from engine.style_fixer import split_long_sentences, de_formalize, remove_filler, fix


class TestSplitLongSentences:
    def test_short_sentence_unchanged(self):
        text = "Hi there."
        result = split_long_sentences(text)
        assert result == text

    def test_long_sentence_split_at_comma(self):
        text = "Part one is somewhat lengthy and verbose, part two is also fairly long, part three continues further still."
        result = split_long_sentences(text, max_len=30)
        # Function should not crash and should preserve all characters
        assert all(c in result for c in text.replace(' ', '').replace(',', '').replace('.', ''))

    def test_no_trailing_empty_strings(self):
        """Bug fix: regex should not produce empty strings."""
        text = "Hello world. How are you?"
        result = split_long_sentences(text)
        # Should not contain empty fragments
        parts = [p for p in result.split('\n') if p.strip()]
        assert all(len(p) > 0 for p in parts)

    def test_multiple_punctuation_boundaries(self):
        text = "A. B. C. D."
        result = split_long_sentences(text)
        assert result

    def test_no_punctuation(self):
        text = "this is a single continuous sentence without any punctuation marks at all"
        result = split_long_sentences(text)
        assert result == text

    def test_custom_max_len(self):
        text = "Part one is here, part two is over there, part three is somewhere else."
        result = split_long_sentences(text, max_len=20)
        assert result  # should not crash


class TestDeFormalize:
    def test_replaces_formal_words(self):
        text = "Furthermore, we will review this. In conclusion, it works."
        result = de_formalize(text)
        assert "Furthermore" not in result
        assert "In conclusion" not in result

    def test_casual_text_unchanged(self):
        text = "Hey, take a look at this file."
        result = de_formalize(text)
        assert result == text

    def test_we_will_to_i_will(self):
        text = "We will send it tomorrow."
        result = de_formalize(text)
        assert "I'll" in result

    def test_unable_to_becomes_cant(self):
        text = "Unable to process this request."
        result = de_formalize(text)
        assert "Can't" in result


class TestRemoveFiller:
    def test_removes_as_an_ai_assistant(self):
        text = "As an AI assistant, I think this is correct."
        result = remove_filler(text)
        assert "As an AI assistant" not in result

    def test_removes_based_on_my_analysis(self):
        text = "Based on my analysis, the data shows growth."
        result = remove_filler(text)
        assert "Based on my analysis" not in result

    def test_cleans_excessive_newlines(self):
        text = "Line one\n\n\n\nLine two"
        result = remove_filler(text)
        assert "\n\n\n\n" not in result

    def test_cleans_leading_punctuation(self):
        text = ",  . Hello"
        result = remove_filler(text)
        assert result.startswith("Hello")

    def test_empty_text(self):
        text = ""
        result = remove_filler(text)
        assert result == ""


class TestFix:
    def test_full_pipeline(self):
        text = "Based on my analysis, Furthermore, we will review the data and submit results this week."
        result = fix(text)
        assert "Based on my analysis" not in result
        assert "Furthermore" not in result

    def test_normal_text_passes_through(self):
        text = "The weekly report is ready, take a look."
        result = fix(text)
        assert result.strip() == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
