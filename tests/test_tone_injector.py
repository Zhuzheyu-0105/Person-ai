"""
Tests for engine.tone_injector module.

Test suite for tone word injection functionality.
"""

import pytest
from engine.tone_injector import inject


class TestInject:
    """Test suite for inject function."""

    def test_greeting_scene_adds_prefix(self):
        """Test greeting scene adds 'Hi~' prefix."""
        text = "Hello, the weekly report is ready."
        result = inject(text, "greeting")
        
        assert result.startswith("Hi~")
        assert "Hello" in result

    def test_delivery_scene_adds_suffix(self):
        """Test delivery scene adds '~' suffix."""
        text = "File is done, sent it over."
        result = inject(text, "delivery")
        
        assert result.endswith("~")

    def test_self_deprecation_scene(self):
        """Test self-deprecation scene adds 'Ah…' prefix."""
        text = "Got praised for this proposal."
        result = inject(text, "self_deprecation")
        
        assert result.startswith("Ah…")

    def test_comfort_scene(self):
        """Test comfort scene adds comforting prefix."""
        text = "This policy document is impossible to parse."
        result = inject(text, "comfort")
        
        assert result.startswith("It's okay, ")

    def test_none_scene_returns_original(self):
        """Test None scene returns original text unchanged."""
        text = "Original text here."
        result = inject(text, None)
        
        assert result == text

    def test_unknown_scene_returns_original(self):
        """Test unknown scene tag returns original text."""
        text = "Original text here."
        result = inject(text, "unknown_scene_xyz")
        
        assert result == text

    def test_prefix_already_exists(self):
        """Test prefix not added if text already starts with it."""
        text = "Hi~Hello there!"
        result = inject(text, "greeting")
        
        # Should not double the prefix
        assert result == text or result.count("Hi~") == 1

    def test_suffix_already_exists(self):
        """Test suffix not added if text already ends with it."""
        text = "File sent~"
        result = inject(text, "delivery")
        
        # Should not double the suffix
        assert result == text or result.count("~") == 1

    def test_empty_text(self):
        """Test injection with empty text."""
        text = ""
        result = inject(text, "greeting")
        
        assert result.startswith("Hi~")

    def test_multiple_injections_same_text(self):
        """Test that multiple injections don't accumulate."""
        text = "Test message"
        
        result1 = inject(text, "greeting")
        result2 = inject(result1, "greeting")  # Inject again
        
        # Should not have double prefix
        assert result2.count("Hi~") <= 1

    @pytest.mark.parametrize("scene,expected_start,expected_end", [
        ("greeting", "Hi~", None),
        ("delivery", None, "~"),
        ("self_deprecation", "Ah…", None),
        ("comfort", "It's okay, ", None),
        ("admit_error", "I got this wrong, ", None),
        ("uncertainty", "I'm not sure about this, ", None),
    ])
    def test_scene_injection_patterns(self, scene, expected_start, expected_end):
        """Parametrized test for scene injection patterns."""
        text = "Sample text for testing."
        result = inject(text, scene)
        
        if expected_start:
            assert result.startswith(expected_start), f"Scene {scene} should start with {expected_start}"
        if expected_end:
            assert result.endswith(expected_end), f"Scene {scene} should end with {expected_end}"


class TestInjectEdgeCases:
    """Edge case tests for inject function."""

    def test_text_with_leading_whitespace(self):
        """Test injection with leading whitespace."""
        text = "  Indented text"
        result = inject(text, "greeting")
        
        # Should handle whitespace correctly
        assert result.startswith("Hi~") or result.startswith("  Hi~")

    def test_text_with_trailing_whitespace(self):
        """Test injection with trailing whitespace."""
        text = "Text with trailing space "
        result = inject(text, "delivery")
        
        # Should handle trailing space
        assert result.rstrip().endswith("~") or result.endswith("~ ")

    def test_unicode_text(self):
        """Test injection with unicode characters."""
        text = "你好，报告准备好了。"
        result = inject(text, "greeting")
        
        assert result.startswith("Hi~")
        assert "你好" in result

    def test_very_long_text(self):
        """Test injection performance with long text."""
        text = "Word " * 10000
        result = inject(text, "greeting")
        
        assert result.startswith("Hi~")
        assert len(result) > len(text)


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])
