"""
driveauth/fraud_state.py
-----------------------
§6.2 — Adaptive fraud state machine. Replaces the old ``_RateLimiter`` (which
was only a flat 3-strikes-then-lock window).

Verification rigor is a STATE, not a constant. The ladder tightens after
suspicious signals and relaxes after sustained clean behaviour:

    Normal ── soft flag ──▶ Elevated ── more flags / confirmed fraud ──▶ Heightened
      ▲                                                                      │
      └──────────── clean streak / decay ──────────────────────────────────┘
                                                                             │
                                              repeated failure / "not mine" ─▶ Locked

This is a deterministic FSM on purpose (§6.2 reasoning): it must react instantly
to a single event with no retraining, be exhaustively testable, and be immune to
adversarial poisoning that an online-learning model would be exposed to. It
updates per-user thresholds/rigor only — never model weights.

State is persisted per driver so it survives a worker restart.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("driveauth.fraud")

_DECAY_HOURS = float(os.getenv("NOVA_FRAUD_LADDER_DECAY_HOURS", "24.0"))
_CLEAN_STREAK_TO_RELAX = int(os.getenv("NOVA_FRAUD_CLEAN_STREAK", "5"))


class FraudState(str, enum.Enum):
    NORMAL     = "normal"
    ELEVATED   = "elevated"
    HEIGHTENED = "heightened"
    LOCKED     = "locked"


# Per-state rigor overrides consumed by the PolicyEngine.
#   min_modalities   : how many biometric modalities must be confident
#   force_step_up    : STEP_UP is mandatory even if Trust/Risk would ACCEPT
#   block            : refuse financial transactions outright
#   trust_margin     : add this to the Trust bar (harder to pass)
_RIGOR = {
    FraudState.NORMAL:     dict(min_modalities=1, force_step_up=False, block=False, trust_margin=0.00),
    FraudState.ELEVATED:   dict(min_modalities=2, force_step_up=False, block=False, trust_margin=0.05),
    FraudState.HEIGHTENED: dict(min_modalities=3, force_step_up=True,  block=False, trust_margin=0.10),
    FraudState.LOCKED:     dict(min_modalities=3, force_step_up=True,  block=True,  trust_margin=0.15),
}


class FraudStateMachine:

    def __init__(self, state_path: Path, driver_id: str):
        self._path = state_path
        self._driver = driver_id
        self._lock = threading.Lock()
        self._state = FraudState.NORMAL
        self._flags: list[float] = []       # timestamps of soft flags
        self._clean_streak = 0
        self._confirmed_fraud = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text()).get(self._driver, {})
            self._state = FraudState(data.get("state", "normal"))
            self._flags = list(data.get("flags", []))
            self._clean_streak = int(data.get("clean_streak", 0))
            self._confirmed_fraud = int(data.get("confirmed_fraud", 0))
        except Exception as exc:
            logger.warning(f"FraudState: load failed ({exc})")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            alldata = {}
            if self._path.exists():
                try:
                    alldata = json.loads(self._path.read_text())
                except Exception:
                    alldata = {}
            alldata[self._driver] = {
                "state": self._state.value,
                "flags": self._flags,
                "clean_streak": self._clean_streak,
                "confirmed_fraud": self._confirmed_fraud,
                "updated": time.time(),
            }
            self._path.write_text(json.dumps(alldata))
        except Exception as exc:
            logger.warning(f"FraudState: save failed ({exc})")

    # ── decay ────────────────────────────────────────────────────────────────

    def _decay(self) -> None:
        now = time.time()
        window = _DECAY_HOURS * 3600.0
        self._flags = [t for t in self._flags if now - t < window]

    # ── transitions ──────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        self._decay()
        n = len(self._flags)
        if self._state == FraudState.LOCKED:
            return  # only an explicit reset leaves LOCKED
        if self._confirmed_fraud >= 1 or n >= 2:
            self._state = FraudState.HEIGHTENED
        elif n == 1:
            self._state = FraudState.ELEVATED
        else:
            self._state = FraudState.NORMAL

    def record_soft_flag(self, reason: str = "") -> FraudState:
        """A step-up event, biometric mismatch, or context anomaly."""
        with self._lock:
            self._flags.append(time.time())
            self._clean_streak = 0
            self._recompute()
            logger.info(f"FraudState[{self._driver}]: soft flag ({reason}) → {self._state.value}")
            self._save()
            return self._state

    def record_confirmed_fraud(self) -> FraudState:
        """User marked a transaction 'not mine', or a confirmed dispute."""
        with self._lock:
            self._confirmed_fraud += 1
            self._clean_streak = 0
            if self._confirmed_fraud >= 2:
                self._state = FraudState.LOCKED
            else:
                self._recompute()
            logger.warning(f"FraudState[{self._driver}]: confirmed fraud → {self._state.value}")
            self._save()
            return self._state

    def record_clean(self) -> FraudState:
        """A clean, accepted transaction. Relaxes the ladder over time."""
        with self._lock:
            self._clean_streak += 1
            if (self._state in (FraudState.ELEVATED, FraudState.HEIGHTENED)
                    and self._clean_streak >= _CLEAN_STREAK_TO_RELAX):
                self._flags = self._flags[1:] if self._flags else []
                self._confirmed_fraud = max(0, self._confirmed_fraud - 1)
                self._clean_streak = 0
                self._recompute()
                logger.info(f"FraudState[{self._driver}]: clean streak → relaxed to {self._state.value}")
                self._save()
            return self._state

    def reset(self) -> None:
        """Explicit manual reset (e.g. after phone-side re-auth clears a lock)."""
        with self._lock:
            self._state = FraudState.NORMAL
            self._flags = []
            self._clean_streak = 0
            self._confirmed_fraud = 0
            self._save()
            logger.info(f"FraudState[{self._driver}]: reset to normal")

    # ── read ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> FraudState:
        with self._lock:
            self._decay()
            return self._state

    def rigor(self) -> dict:
        return dict(_RIGOR[self.state])
