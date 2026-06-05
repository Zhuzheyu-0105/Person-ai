# Structured Persona Engine — Architecture

A four-layer framework for building verifiable, consistent digital personas.
Unlike prompt-only or fine-tune-only approaches, this architecture provides
**triangulated personality definition**, **programmatic guardrails**, and
**quantified relationship memory**.

---

## The Four Layers

```
┌─────────────────────────────────────────────┐
│              Layer 1: Personality Definition │
│  OCEAN (Big Five) + MBTI + Custom Traits     │
│  + Language Feature Corpus                   │
├─────────────────────────────────────────────┤
│              Layer 2: Personality Engine      │
│  Tone Injection → Style Fix → Emotion → C01-C08│
│  Programmatic Guardrails                     │
├─────────────────────────────────────────────┤
│              Layer 3: Multi-Modal Output      │
│  Text (Production) · TTS · Avatar (Future)   │
│  Channel-Specific Adaptation                 │
├─────────────────────────────────────────────┤
│              Layer 4: Memory System           │
│  Episodic · Semantic · Relationship (Intimacy)│
└─────────────────────────────────────────────┘
```

---

## Layer 1: Triangulated Personality

### OCEAN Big Five (Quantified)

Instead of a vague character description, personality is measured on all five
dimensions with scores, percentiles, labels, and **traceable evidence**.

Each dimension score is anchored to specific behaviors and utterances:

| Dimension | Score | Percentile | Label |
|-----------|-------|------------|-------|
| Openness | 70 | 70th | Moderately High |
| Conscientiousness | 75 | 75th | Moderately High |
| Extraversion | 50 | 50th | Average |
| Agreeableness | 65 | 65th | Moderately High |
| Neuroticism | 45 | 40th | Below Average |

### MBTI Type

Defined by four facets with evidence for each:

| Facet | Value | Rationale |
|-------|-------|-----------|
| I/E | I | Explain social preference pattern |
| N/S | N | Explain information processing style |
| F/T | F | Explain decision-making bias |
| J/P | J | Explain lifestyle orientation |

### Custom Traits

Derived from the persona's real-world speech patterns and behavior,
mapped to OCEAN sub-facets. Each trait has:
- **Intensity** (0–1): how strongly it manifests
- **OCEAN mapping**: which Big Five sub-facet it corresponds to
- **Trigger**: when this trait activates
- **Behavior**: how the persona responds
- **Anti-pattern**: what this persona would never do

### Evidence Chain

Every OCEAN score is traceable to specific utterances. Each quote is
tagged with the personality dimension it supports and an impact weight
(★/★★/★★★). This makes the personality definition **auditable** — you
can question any score and trace it back to source material.

---

## Layer 2: Programmatic Guardrails (C01-C08)

The engine enforces 8 consistency rules programmatically — not via prompt
constraints alone. **This is the globally unique differentiator** — no other
open-source persona project has explicit, program-level identity guardrails.

### Pipeline

```
LLM Raw Output → Tone Injection → Style Fix → Emotion Modulation → Consistency Check → Final Output
```

### The 8 Rules

| ID | Rule | Mechanism | Fail Action |
|----|------|-----------|-------------|
| C01 | Suffix Frequency | Max 3 emotive suffixes per 200 chars | Strip excess |
| C02 | First Person | Must use "I", never third-person self-reference | REJECT |
| C03 | Self-Reference | Formal name only in official contexts | Strip |
| C04 | Reply Length | Single reply ≤ 500 chars | REJECT |
| C05 | Toxic Positivity | No forced encouragement clichés | Replace with comfort |
| C06 | Internal Exposure | Never expose tool names, paths, infrastructure | ABSOLUTE REJECT |
| C07 | Bare Table | Tables must have explanation text | REJECT |
| C08 | Over-Formal | No stilted formal language in casual contexts | Replace |

### Engine Modules

| Module | Function | Input → Output |
|--------|----------|----------------|
| `tone_injector.py` | Scene-based tone word injection | Text + Scene → Injected text |
| `style_fixer.py` | Sentence splitting + de-formalization + filler removal | Raw text → Clean text |
| `consistency_checker.py` | C01-C08 rule validation | Text → CheckResult(violations) |
| `pipeline.py` | Full pipeline orchestration | Raw text + Scene → CheckResult |
| `emotion_engine_v3.py` | Six-layer emotion dynamics (PAD + OCC) | User input + Events → Modulated text |
| `config.py` | All configurable parameters (tone words, rules, boundaries) | — |

---

## Layer 3: Multi-Modal Output

| Modality | Status | Notes |
|----------|--------|-------|
| Text | Production | Primary channel, personality injected |
| TTS (Standard) | Verified | Edge TTS compatible |
| TTS (Voice Clone) | Optional | Requires GPU/cloud |
| Avatar | Future | Live2D/VRM |

### Channel Adaptation

Different channels enforce different rules — e.g., C06 (internal exposure) is
strictly enforced on public channels but may be relaxed for trusted development
channels.

---

## Layer 4: Memory System

### Three Memory Types

| Type | Backend | Purpose |
|------|---------|---------|
| Episodic | Session DB + FTS5 search | What was discussed, when, with whom |
| Semantic | File-based knowledge base | Domain expertise, policy database |
| Relationship | Intimacy algorithm + daily cron | Quantitative relationship tracking |

### Relationship Intimacy Algorithm

| Trigger | Δ Intimacy | Cap |
|---------|-----------|-----|
| Active correction | +0.03 | — |
| Positive feedback | +0.01 | — |
| Complaint/dissatisfaction | -0.05 | — |
| No contact (weekly decay) | -0.02 | Floor 0.05 |
| Per-session cap | — | +0.10 max |

### Intimacy Tiers

| Tier | Range | Tone | Internal Exposure |
|------|-------|------|-------------------|
| Stranger | 0 ~ 0.3 | Standard persona | Forbidden |
| Acquaintance | 0.3 ~ 0.6 | Slightly relaxed | Forbidden |
| Close | 0.6 ~ 0.9 | Natural conversation | Allowed |
| High Trust | 0.9 ~ 1.0 | Most authentic | Allowed |

---

## Building Your Own Persona

1. **Define the knowledge base** — collect real utterances, background, evaluations
2. **Derive OCEAN scores** — map each quote to a dimension with evidence weight (★★★/★★/★)
3. **Infer MBTI** — from social preference, information processing, decision, lifestyle patterns
4. **Extract custom traits** — domain-specific behaviors mapped to OCEAN sub-facets
5. **Build language corpus** — tone words, sentence style, expression patterns
6. **Fill the JSON Schema** — use `docs/layer1_schema.json` as a template
7. **Configure the engine** — update `engine/config.py` with your persona's rules

---

## Competitive Positioning

| Capability | Prompt-only (Character.AI) | Fine-tune (Replika) | Rule Engine (elizaOS) | **Structured Persona** |
|---|---|---|---|---|
| OCEAN Quantified | ❌ | ❌ | ❌ | ✅ 5D + percentile |
| Evidence Traceability | ❌ | ❌ | ❌ | ✅ Quote → Score |
| Programmatic Guardrails | ❌ | ❌ | ❌ | ✅ C01-C08 |
| Relationship Memory (Quantified) | ❌ | Implicit | ❌ | ✅ Decay formula |
| Knowledge/Schema Separation | ❌ | N/A | ❌ | ✅ 3-layer assets |
| Reusability (swap persona) | Low | N/A | Medium | **High** — swap knowledge base only |

### Unique Advantages

1. **Triangulation**: Three frameworks (OCEAN + MBTI + Custom Traits) describe the
   same personality from different angles, cross-validating each other.

2. **Evidence chain**: Every OCEAN score can be traced to specific utterance data.
   No other system provides this level of auditability.

3. **C01-C08 guardrails**: Globally unique programmatic identity enforcement.
   Prompt-only systems frequently break character; fine-tuned systems have no
   explicit rules at all.

### Known Limitations

- Purely reactive (no proactive messaging)
- Memory capacity bounded (vs. vector-DB systems like mem0)
- Personality is "worn" (rule-injected), not "grown" (neural emergence)
- User emotion perception is keyword-based (lighter than dedicated ML models)

### Emotion Engine (v3)

The v3 release adds a six-layer emotion architecture based on academic gold standards:

1. **User Perception** — Keyword+confidence-based emotion classification → PAD coordinates
2. **OCC Appraisal** — Every event passes through cognitive evaluation with personality biases
3. **Emotional Contagion** — User emotion partially penetrates AI emotion (CHI'21)
4. **Emotion/Mood Dual-Layer** — Fast emotion spikes + slow mood background, exponential decay
5. **Expression Modulation** — PAD state drives text style (self-deprecation, validating, warmth)
6. **Session Persistence** — Emotion state saved/restored across sessions with time-based decay

Re-appraisal (Scherer CPM) triggers post-settle: praised events are re-evaluated downward
("they're just being polite"), corrections trigger rapid recovery ("it's fixable, move on").

---

## Project Structure

```
structured-persona-engine/
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture.md          ← This file
│   └── layer1_schema.json       ← JSON Schema template
├── engine/
│   ├── __init__.py
│   ├── config.py                ← Persona configuration
│   ├── pipeline.py              ← Main pipeline
│   ├── tone_injector.py         ← Tone word injection
│   ├── style_fixer.py           ← Sentence style correction
│   ├── consistency_checker.py   ← C01-C08 validation
│   └── emotion_engine_v3.py     ← Six-layer emotion engine
├── tests/
│   ├── test_tone_injector.py
│   ├── test_style_fixer.py
│   ├── test_consistency_checker.py
│   ├── test_pipeline.py
│   └── test_emotion_engine_v3.py
└── .gitignore
```

## Getting Started

```bash
# Run all tests
python -m pytest tests/ -v

# Run self-tests
python -m engine.pipeline
python -m engine.tone_injector
python -m engine.style_fixer
python -m engine.consistency_checker

# Use in your project
from engine import run_pipeline

result = run_pipeline("your AI's raw reply text", scene="delivery")
if result.passed:
    send_to_user(result.text)
else:
    log_violations(result.violations)
```
