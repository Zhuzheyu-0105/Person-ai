"""
Consistency Checker
C01-C08 rule-by-rule validation → fix or reject
"""
import re
from dataclasses import dataclass, field
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


def check(text: str) -> CheckResult:
    """Execute all 8 consistency checks."""
    result = CheckResult(text=text)

    # C01: Suffix frequency
    suffix_count = _count_suffix(text)
    char_count = len(text)
    max_allowed = max(1, char_count // 200) * CONSISTENCY_RULES[0]["max_per_200"]
    if suffix_count > max_allowed:
        excess = suffix_count - max_allowed
        for _ in range(excess):
            text = re.sub(r'[~](?!.*[~])', '', text, count=1)
        result.text = text
        result.violations.append(Violation("C01", "Suffix Frequency",
            f"{suffix_count} suffixes > max {max_allowed}, trimmed to {max_allowed}", "fixed"))

    # C02: First person
    for forbidden in CONSISTENCY_RULES[1]["forbidden"]:
        if forbidden.lower() in text.lower():
            result.rejected = True
            result.violations.append(Violation("C02", "First Person",
                f"Forbidden term '{forbidden}' → REJECTED", "rejected"))
            return result

    # C03: Self-reference (warning only, not rejection)
    # Skipped in pipeline — caller decides based on context.

    # C04: Reply length
    if len(text) > CONSISTENCY_RULES[3]["max_chars"]:
        result.rejected = True
        result.violations.append(Violation("C04", "Reply Length",
            f"{len(text)} chars > max 500 → REJECTED", "rejected"))
        return result

    # C05: Toxic positivity
    for forbidden in CONSISTENCY_RULES[4]["forbidden"]:
        if forbidden.lower() in text.lower():
            text = CONSISTENCY_RULES[4]["replacement"]
            result.text = text
            result.violations.append(Violation("C05", "Toxic Positivity",
                f"'{forbidden}' → replaced with comfort phrase", "fixed"))
            break

    # C06: Internal exposure
    for forbidden in CONSISTENCY_RULES[5]["forbidden"]:
        if forbidden.lower() in text.lower():
            result.rejected = True
            result.violations.append(Violation("C06", "Internal Exposure",
                f"Detected '{forbidden}' → ABSOLUTE REJECT", "rejected"))
            return result

    # C07: Bare table
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

    # C08: Over-formal
    for forbidden in CONSISTENCY_RULES[7]["forbidden"]:
        if forbidden.lower() in text.lower():
            text = text.replace(forbidden, SENTENCE_RULES["formal_to_casual"].get(forbidden, ""))
            result.text = text
            result.violations.append(Violation("C08", "Over-Formal",
                f"'{forbidden}' → replaced", "fixed"))

    return result


if __name__ == "__main__":
    tests = [
        ("Hi~ thanks~ really great~ wonderful~ amazing~", "C01 suffix overflow"),
        ("this assistant thinks the proposal is good", "C02 first person"),
        ("L" * 501, "C04 length"),
        ("You can do it! Believe in yourself!", "C05 toxic positivity"),
        ("Use internal_tool_query then admin_command to process", "C06 internal exposure"),
        ("Furthermore, In conclusion, please confirm.", "C08 over-formal"),
        ("Hi~ the weekly report is ready. Main updates are in the policy section.", "Normal"),
    ]
    for text, label in tests:
        r = check(text)
        status = "✓PASS" if r.passed else ("⚠FIXED" if r.fixed else "✗REJECT")
        print(f"{status} [{label}]")
        for v in r.violations:
            print(f"  → {v.rule_id} {v.rule_name}: {v.detail}")
        if not r.rejected:
            print(f"  Output: {r.text[:80]}...")
        print()
