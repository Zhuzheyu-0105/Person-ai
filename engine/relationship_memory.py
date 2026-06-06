"""
Relationship Memory Module (v1)

Persistent per-user relationship tracking with intimacy algorithm.
Connects to EmotionEngineV3 trust_score.

Architecture:
  - Per-user JSON files stored in persistence directory
  - Tracks: intimacy, trust, interaction count, last_seen, criticism ratio
  - Intimacy algorithm from Social Penetration Theory (Altman & Taylor 1973)
  - Daily decay with floor protection

Usage:
    from engine.relationship_memory import RelationshipMemory
    rm = RelationshipMemory(persistence_dir)
    rm.record_interaction(user_id, event_type)
    rm.get_intimacy_tier(user_id)  # → "stranger"/"acquaintance"/"close"/"high_trust"
    rm.get_effective_trust(user_id)  # → float for engine trust_score
"""

import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict


# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

# Intimacy: 0.0 (stranger) → 1.0 (deeply bonded)
INTIMACY_TRIGGERS = {
    "praised":           +0.015,
    "positive_feedback": +0.010,
    "success":           +0.012,
    "gratitude_event":   +0.020,
    "admiration_event":  +0.018,
    "greeting":          +0.002,   # Mere contact, slight bump
    "corrected":         -0.010,
    "complaint":         -0.030,
    "repeated_ask":      -0.008,
    "colleague_vent":    +0.005,   # Venting = trust, paradoxically
    "failure":           -0.020,
    "rejection":         -0.040,
    "guilt_event":       -0.015,
    "reproach":          -0.025,
    "default":           0.0,
}

# Daily decay (no contact = slight drift apart)
INTIMACY_DAILY_DECAY = 0.01
INTIMACY_FLOOR = 0.05
INTIMACY_CEILING = 1.0
PER_SESSION_CAP = 0.08

# Trust score modulation from relationship
# Higher intimacy → trust boost, but trust also moves independently
TRUST_INTIMACY_BOOST = 0.15  # Max trust boost at intimacy=1.0

# Intimacy tiers (affect tone, internal exposure rules)
INTIMACY_TIERS = {
    "stranger":      (0.00, 0.30),
    "acquaintance":  (0.30, 0.55),
    "close":         (0.55, 0.80),
    "high_trust":    (0.80, 1.00),
}


# ═══════════════════════════════════════════════
# Relationship Memory
# ═══════════════════════════════════════════════

class RelationshipMemory:
    """Persistent per-user relationship state.

    Tracks intimacy, cumulative trust, interaction history,
    and criticism ratio. Survives session restarts.
    """

    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = persistence_dir or Path(".")
        self._store: Dict[str, dict] = {}  # user_id → relationship data
        self._dirty: set = set()

    @property
    def _base_dir(self) -> Path:
        d = self.persistence_dir / "relationships"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _user_file(self, user_id: str) -> Path:
        safe_id = user_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._base_dir / f"{safe_id}.json"

    def _load_user(self, user_id: str) -> dict:
        """Load relationship data for a user, with daily decay applied."""
        if user_id in self._store:
            return self._store[user_id]

        f = self._user_file(user_id)
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                data = self._default_user_data()
        else:
            data = self._default_user_data()

        # Apply decay for days without contact
        if "last_seen" in data:
            try:
                last_seen = datetime.fromisoformat(data["last_seen"])
                days_elapsed = (datetime.now() - last_seen).days
                if days_elapsed > 0:
                    data["intimacy"] = max(
                        INTIMACY_FLOOR,
                        data.get("intimacy", 0.1) - INTIMACY_DAILY_DECAY * days_elapsed,
                    )
            except Exception:
                pass

        self._store[user_id] = data
        return data

    @staticmethod
    def _default_user_data() -> dict:
        return {
            "intimacy": 0.10,
            "trust_score": 0.70,
            "interaction_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "criticism_count": 0,
            "last_seen": None,
            "first_seen": datetime.now().isoformat(),
            "session_intimacy_delta": 0.0,  # Reset per session
        }

    def record_interaction(self, user_id: str, event_type: str):
        """Record an interaction event, updating intimacy and trust."""
        data = self._load_user(user_id)
        data["last_seen"] = datetime.now().isoformat()
        data["interaction_count"] += 1

        # Intimacy delta
        delta = INTIMACY_TRIGGERS.get(event_type, INTIMACY_TRIGGERS["default"])
        session_delta = data.get("session_intimacy_delta", 0.0)
        if session_delta + abs(delta) <= PER_SESSION_CAP:
            data["intimacy"] = min(
                INTIMACY_CEILING,
                max(INTIMACY_FLOOR, data["intimacy"] + delta),
            )
            data["session_intimacy_delta"] = session_delta + abs(delta)

        # Trust tracking
        negative_events = {
            "corrected", "complaint", "failure", "rejection", "reproach", "guilt_event",
        }
        positive_events = {
            "praised", "positive_feedback", "success", "admiration_event", "gratitude_event",
        }
        if event_type in negative_events:
            data["negative_count"] += 1
            data["criticism_count"] += 1
            data["trust_score"] = max(0.10, data["trust_score"] - 0.02)
        elif event_type in positive_events:
            data["positive_count"] += 1
            data["trust_score"] = min(1.0, data["trust_score"] + 0.015)

        data["criticism_ratio"] = (
            data["criticism_count"] / max(data["interaction_count"], 1)
        )

        self._dirty.add(user_id)

    def reset_session_cap(self, user_id: str):
        """Reset per-session intimacy cap (call at start of new session)."""
        data = self._load_user(user_id)
        data["session_intimacy_delta"] = 0.0
        self._dirty.add(user_id)

    def get_intimacy(self, user_id: str) -> float:
        return self._load_user(user_id).get("intimacy", 0.1)

    def get_intimacy_tier(self, user_id: str) -> str:
        intimacy = self.get_intimacy(user_id)
        for tier, (lo, hi) in INTIMACY_TIERS.items():
            if lo <= intimacy < hi:
                return tier
        return "high_trust"

    def get_trust_score(self, user_id: str) -> float:
        """Get effective trust score for this user.
        Combines stored trust with intimacy boost.
        """
        data = self._load_user(user_id)
        base_trust = data.get("trust_score", 0.70)
        intimacy = data.get("intimacy", 0.10)
        boost = TRUST_INTIMACY_BOOST * intimacy
        return min(1.0, base_trust + boost)

    def get_interaction_count(self, user_id: str) -> int:
        return self._load_user(user_id).get("interaction_count", 0)

    def get_criticism_ratio(self, user_id: str) -> float:
        """Fraction of interactions that were critical/corrective."""
        data = self._load_user(user_id)
        return data.get("criticism_ratio", 0.0)

    def get_relationship_summary(self, user_id: str) -> dict:
        data = self._load_user(user_id)
        return {
            "user_id": user_id,
            "intimacy": round(data.get("intimacy", 0.1), 3),
            "tier": self.get_intimacy_tier(user_id),
            "trust_score": round(self.get_trust_score(user_id), 3),
            "interactions": data.get("interaction_count", 0),
            "criticism_ratio": round(data.get("criticism_ratio", 0.0), 3),
            "last_seen": data.get("last_seen"),
            "first_seen": data.get("first_seen"),
        }

    def save(self, user_id: Optional[str] = None):
        """Persist relationship data to disk."""
        if user_id:
            self._save_one(user_id)
        else:
            for uid in list(self._dirty):
                self._save_one(uid)

    def _save_one(self, user_id: str):
        if user_id not in self._store:
            return
        try:
            self._user_file(user_id).write_text(
                json.dumps(self._store[user_id], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._dirty.discard(user_id)
        except Exception:
            pass

    def load_all(self) -> list[str]:
        """Load all known user IDs from disk."""
        if not self._base_dir.exists():
            return []
        return [
            f.stem for f in self._base_dir.glob("*.json") if f.is_file()
        ]

    def forget_user(self, user_id: str):
        """Remove a user's relationship data."""
        self._store.pop(user_id, None)
        self._dirty.discard(user_id)
        f = self._user_file(user_id)
        if f.exists():
            f.unlink(missing_ok=True)


# ═══════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════

_relationship_memory: RelationshipMemory | None = None


def get_relationship_memory(persistence_dir: Optional[Path] = None) -> RelationshipMemory:
    global _relationship_memory
    if _relationship_memory is None:
        _relationship_memory = RelationshipMemory(persistence_dir)
    return _relationship_memory


# ═══════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    print("=== Relationship Memory Self-Test ===\n")

    with tempfile.TemporaryDirectory() as td:
        rm = RelationshipMemory(persistence_dir=Path(td))

        # Test: build relationship through interactions
        user_id = "test_user_001"
        events = [
            ("greeting", "Initial contact"),
            ("praised", "User praises work"),
            ("positive_feedback", "User gives good feedback"),
            ("corrected", "User corrects mistake"),
            ("praised", "User praises again"),
            ("complaint", "User complains"),
            ("corrected", "User corrects again"),
        ]

        for event_type, desc in events:
            rm.record_interaction(user_id, event_type)
            summary = rm.get_relationship_summary(user_id)
            print(f"  {desc:30s} → intimacy={summary['intimacy']:.3f} "
                  f"trust={summary['trust_score']:.3f} "
                  f"tier={summary['tier']}")

        # Persist
        rm.save(user_id)

        # Load fresh
        rm2 = RelationshipMemory(persistence_dir=Path(td))
        loaded = rm2.get_relationship_summary(user_id)
        print(f"\n  After reload: intimacy={loaded['intimacy']:.3f} "
              f"trust={loaded['trust_score']:.3f} "
              f"crit_ratio={loaded['criticism_ratio']:.3f}")

        # Assertions
        assert loaded["interactions"] == 7
        assert loaded["intimacy"] > 0.05  # Above floor
        assert loaded["trust_score"] < 0.80  # Criticisms dragged it down
        assert loaded["criticism_ratio"] > 0.2  # Some criticism

    print("\n✓ Relationship Memory OK")
