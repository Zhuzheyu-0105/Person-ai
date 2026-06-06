# Structured Persona Engine

**A four-layer framework for building verifiable, consistent AI digital personas.**

Unlike prompt-only systems (Character.AI, GPTs) that rely on a paragraph of text,
or fine-tuned models (Replika) where personality is a black box, this engine
provides **triangulated personality definition** with **programmatic guardrails**,
a **seven-layer neuroscience-grounded emotion engine**, and **quantified relationship memory**.

---

## Why This Exists

Most AI personas have no guardrails. They break character. They leak internal
details. They can't explain *why* they behave a certain way. Their emotions
feel shallow because there's no actual state machine behind them.

This engine solves that:

- **Triangulation** — OCEAN (Big Five) + MBTI + Custom Traits describe the same
  personality from three angles, cross-validating each other
- **Evidence chain** — every OCEAN score traces back to specific utterance data
- **C01-C08 guardrails** — 8 programmatic rules that catch and fix persona breaks
  before the user sees them
- **Emotion engine v5** — PAD emotional space + OCC cognitive appraisal +
  allostatic load + trust dynamics + extinction learning + episodic memory
  reconsolidation. Backed by 11 academic references (Hatfield 1994 to CHI '24).
- **Quantified relationships** — intimacy decays mathematically over time; trust
  score updates per interaction; persona tone adapts to relationship depth

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│           Layer 1: Personality Definition         │
│       OCEAN + MBTI + 5 Custom Traits              │
│       + Language Feature Corpus                   │
│       → Defined in docs/layer1_schema.json        │
├──────────────────────────────────────────────────┤
│           Layer 2: Personality Engine (v5)         │
│       Emotion Engine (PAD+OCC+Neuro)              │
│       Tone Injection → Style Fix → C01-C08        │
│       → Implemented in engine/                    │
├──────────────────────────────────────────────────┤
│           Layer 3: Multi-Modal Output              │
│       Text (Production) · TTS · Avatar            │
│       ⚠ DISCONTINUED — see note below             │
├──────────────────────────────────────────────────┤
│           Layer 4: Memory System                   │
│       Episodic · Semantic · Relationship          │
│       Intimacy decay: -0.01/day, floor 0.05       │
└──────────────────────────────────────────────────┘
```

### ⚠ Layer 3 (Voice / Video / Avatar) — Discontinued

Effective June 2026, Layer 3 multi-modal output development is frozen.

This project has zero funding. Running a TTS voice clone costs GPU compute.
Building a Live2D/VRM avatar pipeline costs time and resources that don't exist.
The author is a graduate student — there is no budget for cloud GPU, voice API
subscriptions, or avatar tooling.

**What remains:** The text output pipeline (Layer 2) is production-ready.
If you want to plug in your own TTS or avatar, the `TONE_WORDS` and `rhythm()`
interfaces are there — but no further development will happen on this layer.

Pull requests for Layer 3 integrations are welcome. But this project will not
fund or build them.

---

## Quick Start

```bash
# Clone
git clone https://github.com/Zhuzheyu-0105/persona.git
cd structured-persona-engine

# Run tests
python -m pytest tests/ -v

# Run self-tests
python -m engine.emotion_engine_v3
python -m engine.emotion_classifier
python -m engine.pipeline
python -m engine.consistency_checker
python -m engine.relationship_memory

# Basic usage (backward compatible)
from engine import run_pipeline
result = run_pipeline("Your AI's raw reply text", scene="delivery")

# v5 full lifecycle
from engine import run_pipeline_v5, EmotionEngineV3

emo = EmotionEngineV3()
emo.load_relationship("user_123")        # Load persistent trust state
result = run_pipeline_v5(
    user_text="I'm frustrated with this bug",
    raw_reply="Let me help you fix that.",
    emo=emo,
    event_type="complaint",
    scene="enterprise",
)
# Now inject into LLM system prompt:
prompt_context = emo.get_prompt_context()
# → "[当前状态] 心情不太好，精力充沛。用户情绪：frustration"
```

---

## v5 Emotion Engine

The v5 release adds a seven-layer emotion architecture grounded in psychology
and neuroscience:

| Layer | Mechanism | Academic Basis |
|-------|-----------|---------------|
| 1. Perception | Chinese lexicon classifier (0.03ms) + keyword hybrid | CHI '23 |
| 2. OCC Appraisal | 22 event types, scene-modulated OCEAN, frequency habituation | Ortony et al. 1988 |
| 3. Contagion | Trust-modulated emotional infection with floor protection | Hatfield et al. 1994 |
| 4. Emotion/Mood | Dual-layer + extinction learning + re-appraisal | Scherer 2009 |
| 5. Allostatic Load | Cumulative stress with chronic trajectory switching | McEwen 1998 |
| 6. Expression | CHI'24 validating phrases + rhythm + self-deprecation | CHI '24 |
| 7. Memory | Daily reset (CHI'22) + episodic reconsolidation | CHI '22 |

New methods: `set_scene()`, `get_prompt_context()`, `load_relationship()`,
`save_relationship()`, `extinction_step()`, `record_ai_response()`.

---

## The 8 Guardrails (C01-C08)

| Rule | What It Checks | Action |
|------|---------------|--------|
| **C01** | Too many emotive suffixes (~) | Strip excess (scene-dependent) |
| **C02** | Third-person self-reference ("this assistant") | REJECT |
| **C03** | Formal self-name in casual chat | Strip |
| **C04** | Reply over 500/800 characters | REJECT (enterprise: warn) |
| **C05** | Toxic positivity clichés | Replace matched phrase in-place |
| **C06** | Internal tool names, paths, infrastructure | **ABSOLUTE REJECT** |
| **C07** | Table with no explanation text | REJECT |
| **C08** | Stilted formal language | Replace (enterprise: warn) |

---

## How It Compares

| Capability | Character.AI / GPTs | Replika | elizaOS | **This Engine** |
|---|---|---|---|---|
| OCEAN Big Five quantified | ❌ | ❌ | ❌ | ✅ Scores + percentiles |
| Evidence chain (quote → score) | ❌ | ❌ | ❌ | ✅ Fully traceable |
| Programmatic guardrails | ❌ | ❌ | ❌ | ✅ C01-C08 |
| Emotion state machine | ❌ (prompt-level) | Implicit | ❌ | ✅ PAD+OCC+Neuro |
| Scene-dependent personality | ❌ | ❌ | ❌ | ✅ 5 OCEAN scenes |
| Relationship intimacy decay | ❌ | Implicit | ❌ | ✅ Formula + persistence |
| Pre-prompt emotion context | ❌ | ❌ | ❌ | ✅ `get_prompt_context()` |
| Swap persona (reusability) | Low | N/A | Medium | **High** — swap KB only |

### Key Differentiators

1. **Triangulation** — OCEAN + MBTI + Custom Traits cross-validate
2. **C01-C08 are globally unique** — no other open-source project has
   program-level identity guardrails
3. **Emotion engine is academic-prototype grade** — 11 cited papers, from
   Hatfield (1994) to CHI (2024), implemented in production Python
4. **Evidence chain** — every personality score can be traced back to source material

### Known Limitations

- User emotion perception is lexicon-based (F1 ~0.7) — lighter than BERT
- Engine is post-processing; does not control LLM generation token-by-token
- Memory capacity bounded (not a vector DB)
- Personality is rule-injected, not neurally emergent
- Layer 3 (voice/video) development discontinued due to budget constraints

---

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── pytest.ini
├── docs/
│   ├── architecture.md              # Full architecture documentation
│   └── layer1_schema.json           # Example Layer 1 JSON Schema
├── engine/
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Persona configuration + scene rules
│   ├── pipeline.py                  # v5 pipeline (full lifecycle + fallback)
│   ├── tone_injector.py             # Scene-based tone word injection
│   ├── style_fixer.py               # Sentence splitting + de-formalization
│   ├── consistency_checker.py       # C01-C08 rule enforcement (scene-aware)
│   ├── emotion_engine_v3.py         # v5 emotion engine (~1400 lines)
│   ├── emotion_classifier.py        # Chinese lexicon emotion classifier
│   └── relationship_memory.py       # Persistent per-user relationship state
└── tests/
    ├── test_tone_injector.py
    ├── test_style_fixer.py
    ├── test_consistency_checker.py
    ├── test_pipeline.py
    └── test_emotion_engine_v3.py
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


