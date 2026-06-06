"""
Consistency Checker (v5)
C01-C08 rule-by-rule validation → fix or reject

v5 changes:
  - C05: fixed full-text replacement bug → only replace matched phrase
  - C01/C04/C08: scene-dependent rule activation (CHI'20: 80-90% consistency target)
  - Scene parameter accepted by check() to modulate rule severity
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from .config import CONSISTENCY_RULES, SENTENCE_RULES


@dataclass
class Violation:
    rule_id: str
    rule_name: str
    detail: str
    action: str  # "fixed" | "rejected" | "warning"


@dataclass
class CheckResult:
    text: str
    violations: list = field(default_factory=list)
    rejected: bool = False

    @property
    def passed(self) -> bool:
        return not self.rejected and len(self.violations) == 0

    @property
    def fixed(self) -> bool:
        return not self.rejected and len(self.violations) > 0


def _count_suffix(text: str) -> int:
    """Count emotive suffix characters."""
    return len(re.findall(r'[~]', text))


def _get_effective_rule(rule: dict, key: str, scene: Optional[str], default=None):
    """Get rule parameter, with optional scene override."""
    if scene and "scene_override" in rule:
        override = rule["scene_override"].get(scene, {})
        if key in override:
            return override[key]
    return rule.get(key, default)


def check(text: str, scene: Optional[str] = None) -> CheckResult:
    """Execute all 8 consistency checks. v5: scene-aware.

    Args:
        text:  Text to validate.
        scene: Optional scene tag (casual/delivery/enterprise/comfort/emotional).
               Modulates C01/C04/C08 severity per CHI'20 findings.
    """
    result = CheckResult(text=text)

    # ── C01: Suffix frequency (v5: scene-dependent) ──
    suffix_count = _count_suffix(text)
    char_count = len(text)
    max_per_200 = _get_effective_rule(CONSISTENCY_RULES[0], "max_per_200", scene, 3)
    max_allowed = max(1, char_count // 200) * max_per_200
    if suffix_count > max_allowed:
        excess = suffix_count - max_allowed
        for _ in range(excess):
            text = re.sub(r'[~](?!.*[~])', '', text, count=1)
        result.text = text
        result.violations.append(Violation("C01", "Suffix Frequency",
            f"{suffix_count} suffixes > max {max_allowed}, trimmed to {max_allowed}", "fixed"))

    # ── C02: First person ──
    for forbidden in CONSISTENCY_RULES[1]["forbidden"]:
        if forbidden.lower() in text.lower():
            result.rejected = True
            result.violations.append(Violation("C02", "First Person",
                f"Forbidden term '{forbidden}' → REJECTED", "rejected"))
            return result

    # ── C03: Self-reference (warning only) ──
    # Skipped in pipeline — caller decides based on context.

    # ── C04: Reply length (v5: scene-dependent) ──
    max_chars = _get_effective_rule(CONSISTENCY_RULES[3], "max_chars", scene, 500)
    fail_action = _get_effective_rule(CONSISTENCY_RULES[3], "fail_action", scene, "reject")
    if len(text) > max_chars:
        if fail_action == "warn":
            result.violations.append(Violation("C04", "Reply Length",
                f"{len(text)} chars > max {max_chars} → warning (scene={scene})", "warning"))
        else:
            result.rejected = True
            result.violations.append(Violation("C04", "Reply Length",
                f"{len(text)} chars > max {max_chars} → REJECTED", "rejected"))
            return result

    # ── C05: Toxic positivity (v5: fixed — replace ONLY matched phrase, not whole text) ──
    for forbidden in CONSISTENCY_RULES[4]["forbidden"]:
        if forbidden.lower() in text.lower():
            # v5: Replace only the matched phrase with comfort phrase
            replacement = CONSISTENCY_RULES[4]["replacement"]
            text = text.replace(forbidden, replacement)
            result.text = text
            result.violations.append(Violation("C05", "Toxic Positivity",
                f"'{forbidden}' → replaced in-place with '{replacement}'", "fixed"))
            break

    # ── C06: Internal exposure ──
    for forbidden in CONSISTENCY_RULES[5]["forbidden"]:
        if forbidden.lower() in text.lower():
            result.rejected = True
            result.violations.append(Violation("C06", "Internal Exposure",
                f"Detected '{forbidden}' → ABSOLUTE REJECT", "rejected"))
            return result

    # ── C07: Bare table ──
    lines = text.strip().split('\n')
    has_table = any(line.strip().startswith('|') for line in lines)
    if has_table:
        last_table_line = max(i for i, line in enumerate(lines) if line.strip().startswith('|'))
        after_table = '\n'.join(lines[last_table_line + 1:]).strip()
        if not after_table or len(after_table) < 10:
            result.violations.append(Violation("C07", "Bare Table",
                "Table has no explanation text → REJECTED", "rejected"))
            result.rejected = True
            return result

    # ── C08: Over-formal (v5: scene-dependent) ──
    c08_fail_action = _get_effective_rule(CONSISTENCY_RULES[7], "fail_action", scene, "replace_per_rule")
    for forbidden in CONSISTENCY_RULES[7]["forbidden"]:
        if forbidden.lower() in text.lower():
            if c08_fail_action == "warn":
                result.violations.append(Violation("C08", "Over-Formal",
                    f"'{forbidden}' allowed in scene={scene}", "warning"))
            else:
                replacement = SENTENCE_RULES["formal_to_casual"].get(forbidden, "")
                text = text.replace(forbidden, replacement)
                result.text = text
                result.violations.append(Violation("C08", "Over-Formal",
                    f"'{forbidden}' → '{replacement}'", "fixed"))

    return result


if __name__ == "__main__":
    tests = [
        # Original tests
        ("Hi~ thanks~ really great~ wonderful~ amazing~", None, "C01 suffix overflow"),
        ("this assistant thinks the proposal is good", None, "C02 first person"),
        ("L" * 501, None, "C04 length"),
        ("You can do it! Just keep trying.", None, "C05 toxic positivity (fixed)"),
        ("Use internal_tool_query then admin_command to process", None, "C06 internal exposure"),
        ("Furthermore, In conclusion, please confirm.", None, "C08 over-formal"),
        ("Hi~ the weekly report is ready. Main updates are in the policy section.", None, "Normal"),
        # v5 scene tests
        ("Furthermore, the report shows excellent progress in Q2.", "enterprise", "C08 allowed in enterprise"),
        ("L" * 600, "enterprise", "C04 relaxed in enterprise (800 char limit)"),
    ]
    for text, scene, label in tests:
        r = check(text, scene=scene)
        status = "✓PASS" if r.passed else ("⚠FIXED" if r.fixed else "✗REJECT")
        scene_str = f" [scene={scene}]" if scene else ""
        print(f"{status} [{label}]{scene_str}")
        for v in r.violations:
            print(f"  → {v.rule_id} {v.rule_name}: {v.detail}")
        if not r.rejected:
            preview = r.text[:80]
            if len(r.text) > 80:
                preview += "..."
            print(f"  Output: {preview}")
        print()
