# Structured Persona Engine

**A four-layer framework for building verifiable, consistent AI digital personas.**

Unlike prompt-only systems (Character.AI, GPTs) that rely on a paragraph of text,
or fine-tuned models (Replika) where personality is a black box, this engine
provides **triangulated personality definition** with **programmatic guardrails**
and **quantified relationship memory**.

---

## Why This Exists

Most AI personas have no guardrails. They break character. They leak internal
details. They can't explain *why* they behave a certain way.

This engine solves that:

- **Triangulation** — OCEAN (Big Five) + MBTI + Custom Traits describe the same
  personality from three angles, cross-validating each other
- **Evidence chain** — every OCEAN score traces back to specific utterance data
- **C01-C08 guardrails** — 8 programmatic rules that catch and fix persona breaks
  before the user sees them
- **Quantified relationships** — intimacy decays mathematically over time; persona
  tone adapts to trust level

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         Layer 1: Personality Definition      │
│     OCEAN + MBTI + 5 Custom Traits           │
│     + Language Feature Corpus                │
│     → Defined in docs/layer1_schema.json     │
├─────────────────────────────────────────────┤
│         Layer 2: Personality Engine          │
│     Tone Injection → Style Fix → C01-C08     │
│     → Implemented in engine/                 │
├─────────────────────────────────────────────┤
│         Layer 3: Multi-Modal Output           │
│     Text (Production) · TTS · Avatar         │
│     Channel-specific adaptation              │
├─────────────────────────────────────────────┤
│         Layer 4: Memory System               │
│     Episodic · Semantic · Relationship       │
│     Intimacy decay: -0.02/week, floor 0.05   │
└─────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/your-org/structured-persona-engine.git
cd structured-persona-engine

# Run all self-tests
python -m engine.pipeline
python -m engine.tone_injector
python -m engine.style_fixer
python -m engine.consistency_checker

# Basic usage
from engine import run_pipeline

result = run_pipeline("Your AI's raw reply text", scene="delivery")
if result.passed:
    send_to_user(result.text)
else:
    for v in result.violations:
        print(f"{v.rule_id}: {v.detail}")
```

---

## The 8 Guardrails (C01-C08)

| Rule | What It Checks | Action |
|------|---------------|--------|
| **C01** | Too many emotive suffixes (~) | Strip excess |
| **C02** | Third-person self-reference ("this assistant") | REJECT |
| **C03** | Formal self-name in casual chat | Strip |
| **C04** | Reply over 500 characters | REJECT |
| **C05** | Toxic positivity clichés | Replace with comfort phrase |
| **C06** | Internal tool names, paths, infrastructure | **ABSOLUTE REJECT** |
| **C07** | Table with no explanation text | REJECT |
| **C08** | Stilted formal language | Replace with casual equivalents |

---

## How It Compares

| Capability | Character.AI / GPTs | Replika (Fine-tune) | elizaOS (Rules) | **This Engine** |
|---|---|---|---|---|
| OCEAN Big Five quantified | ❌ | ❌ | ❌ | ✅ Scores + percentiles |
| Evidence chain (quote → score) | ❌ | ❌ | ❌ | ✅ Fully traceable |
| Programmatic guardrails | ❌ | ❌ | ❌ | ✅ C01-C08 |
| Relationship intimacy decay | ❌ | Implicit | ❌ | ✅ Formula + cron |
| Knowledge/Schema separation | ❌ | N/A | ❌ | ✅ Three independent layers |
| Swap persona (reusability) | Low | N/A | Medium | **High** — swap KB only |

### Key Differentiators

1. **Triangulation** — OCEAN + MBTI + Custom Traits cross-validate. A single
   behavior (e.g., reaction to praise) is explainable through all three frameworks.
2. **C01-C08 are globally unique** — no other open-source persona project has
   explicit, program-level identity guardrails.
3. **Evidence chain is auditable** — every personality score can be questioned
   and traced back to source material.

### Limitations

- No emotional dynamics (no "current mood" state machine)
- Purely reactive (no proactive messaging)
- Memory capacity bounded (not a vector DB)
- Personality is rule-injected, not neurally emergent

---

## Project Structure

```
.
├── README.md
├── docs/
│   ├── architecture.md          # Full architecture documentation
│   └── layer1_schema.json       # Example Layer 1 JSON Schema
├── engine/
│   ├── __init__.py              # Package exports
│   ├── config.py                # Persona configuration (TONE_WORDS, C01-C08 rules)
│   ├── pipeline.py              # Main pipeline (inject → fix → check)
│   ├── tone_injector.py         # Scene-based tone word injection
│   ├── style_fixer.py           # Sentence splitting + de-formalization
│   └── consistency_checker.py   # C01-C08 rule enforcement
└── .gitignore
```

---

## Building Your Own Persona

1. **Define the knowledge base** — collect real utterances, background, evaluations
2. **Derive OCEAN scores** — map each quote to a dimension with evidence weight (★★★/★★/★)
3. **Infer MBTI** — from social preference, information processing, decision, lifestyle patterns
4. **Extract custom traits** — domain-specific behaviors mapped to OCEAN sub-facets
5. **Build language corpus** — tone words, sentence style, expression patterns
6. **Fill the JSON Schema** — use `docs/layer1_schema.json` as a template
7. **Configure the engine** — update `engine/config.py` with your persona's rules

Full methodology: see `docs/architecture.md`.

---

## License

MIT
