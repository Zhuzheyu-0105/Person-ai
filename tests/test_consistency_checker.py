"""
Tests for engine.consistency_checker module — C01-C08 rules.
"""
import pytest
from engine.consistency_checker import check, CheckResult, Violation


class TestC01SuffixFrequency:
    def test_normal_text_passes(self):
        r = check("Weekly report is ready.")
        assert r.passed

    def test_excess_suffixes_trimmed(self):
        r = check("Hi~ thanks~ really great~ wonderful~ amazing~ so cool~ awesome~")
        vs = [v for v in r.violations if v.rule_id == "C01"]
        assert len(vs) >= 1
        assert vs[0].action == "fixed"
        # Should have fewer ~ after fix
        assert r.text.count("~") <= 5

    def test_fix_when_not_rejected(self):
        r = check("Hi~ thanks~ really great~ wonderful~ amazing~")
        assert not r.rejected
        assert r.fixed


class TestC02FirstPerson:
    def test_this_assistant_rejected(self):
        r = check("this assistant thinks the proposal is good")
        assert r.rejected
        vs = [v for v in r.violations if v.rule_id == "C02"]
        assert len(vs) == 1

    def test_the_bot_rejected(self):
        r = check("the bot can help with that")
        assert r.rejected

    def test_first_person_passes(self):
        r = check("I think the proposal is good")
        assert not r.rejected


class TestC04ReplyLength:
    def test_within_limit_passes(self):
        r = check("Short reply.")
        assert not r.rejected

    def test_over_limit_rejected(self):
        r = check("L" * 501)
        assert r.rejected
        vs = [v for v in r.violations if v.rule_id == "C04"]
        assert len(vs) == 1


class TestC05ToxicPositivity:
    def test_you_can_do_it_replaced(self):
        r = check("You can do it!")
        vs = [v for v in r.violations if v.rule_id == "C05"]
        assert len(vs) == 1
        assert vs[0].action == "fixed"
        assert "take your time" in r.text

    def test_believe_in_yourself_replaced(self):
        r = check("Believe in yourself!")
        assert "take your time" in r.text

    def test_normal_encouragement_passes(self):
        r = check("I'm here if you need help.")
        assert r.passed


class TestC06InternalExposure:
    def test_internal_tool_rejected(self):
        r = check("Use internal_tool_query for that")
        assert r.rejected
        vs = [v for v in r.violations if v.rule_id == "C06"]
        assert len(vs) == 1

    def test_file_path_rejected(self):
        r = check("The config is at /sensitive/path/config.yaml")
        assert r.rejected

    def test_normal_text_passes(self):
        r = check("Let me look that up for you.")
        assert r.passed


class TestC07BareTable:
    def test_bare_table_rejected(self):
        r = check("Results:\n| A | B |\n| 1 | 2 |")
        assert r.rejected
        vs = [v for v in r.violations if v.rule_id == "C07"]
        assert len(vs) == 1

    def test_table_with_explanation_passes(self):
        r = check("Results:\n| A | B |\n| 1 | 2 |\n\nThis shows the quarterly breakdown.")
        assert not r.rejected

    def test_no_table_passes(self):
        r = check("Just some text, no tables here.")
        assert r.passed


class TestC08OverFormal:
    def test_furthermore_replaced(self):
        r = check("Furthermore, we should consider the data.")
        vs = [v for v in r.violations if v.rule_id == "C08"]
        assert len(vs) >= 1
        assert "Furthermore" not in r.text

    def test_in_conclusion_replaced(self):
        r = check("In conclusion, this is fine.")
        assert "In conclusion" not in r.text

    def test_casual_text_passes(self):
        r = check("Hey, take a look at this.")
        assert r.passed


class TestCheckResultProperties:
    def test_passed_clean(self):
        r = CheckResult(text="Hello")
        assert r.passed
        assert not r.fixed
        assert not r.rejected

    def test_fixed_but_not_rejected(self):
        r = CheckResult(
            text="Hello~",
            violations=[Violation("C01", "Suffix", "fixed", "fixed")],
            rejected=False,
        )
        assert r.fixed
        assert not r.passed
        assert not r.rejected

    def test_rejected(self):
        r = CheckResult(
            text="",
            violations=[Violation("C06", "Internal", "leaked", "rejected")],
            rejected=True,
        )
        assert r.rejected
        assert not r.passed
        assert not r.fixed


class TestCheckEdgeCases:
    def test_empty_text(self):
        r = check("")
        assert r.passed  # empty text has no violations

    def test_only_suffixes(self):
        r = check("~~~~~")
        vs = [v for v in r.violations if v.rule_id == "C01"]
        assert len(vs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
