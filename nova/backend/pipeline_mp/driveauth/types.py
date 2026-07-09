"""
driveauth/types.py
------------------
Shared dataclasses and the Decision enum used across the driveauth package.

Kept dependency-free (numpy only) so every submodule can import from here
without a cycle. ``ModalityResult`` is re-declared here (rather than imported
from the old ``biometric_gate``) so the driveauth package is self-contained;
it is field-compatible with the old one, so verifiers that already return the
old ``ModalityResult`` still work unchanged.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Decision(str, enum.Enum):
    """The three outward-facing outcomes (proposal §4.2)."""
    ACCEPT           = "ACCEPT"
    STEP_UP_REQUIRED = "STEP_UP_REQUIRED"
    REJECT           = "REJECT"

    def legacy(self) -> str:
        """
        Map to the old gate's pass/step_up/deny vocabulary so existing
        call-site branches that compared against those strings still work.
        """
        return {
            Decision.ACCEPT:           "pass",
            Decision.STEP_UP_REQUIRED: "step_up",
            Decision.REJECT:           "deny",
        }[self]


@dataclass
class ModalityResult:
    """Result of a single biometric matcher (voice / face / finger)."""
    score:      float | None       # None = unavailable / skipped
    confident:  bool               # signal clean enough to trust?
    latency_ms: float = 0.0
    # Quality metric from the pre-matching QualityGate (§8a.5); 1.0 = perfect.
    quality:    float = 1.0
    # OOD flag from OODDetector (§8a.6): True = input looks out-of-distribution.
    ood:        bool  = False


@dataclass
class QualityFlags:
    """Per-modality pre-matching quality outcomes (§8a.5)."""
    voice_ok:   bool  = True
    face_ok:    bool  = True
    finger_ok:  bool  = True
    voice_q:    float = 1.0
    face_q:     float = 1.0
    finger_q:   float = 1.0
    # A sensor that was present but is now unresponsive (§8a.7). Distinct from
    # "modality simply not fitted on this trim".
    hardware_fault: bool = False
    notes:      list[str] = field(default_factory=list)


@dataclass
class RiskContext:
    """
    Inputs to the RiskModel (§7). This is where GPS / CAN / behaviour /
    transaction context live — deliberately OUTSIDE the Trust computation.
    """
    # Vehicle context (read-only, from CAN — §8a.3)
    gps_lat:            float | None = None
    gps_lon:            float | None = None
    gps_accuracy_m:     float = 50.0
    speed_kmh:          float = 0.0
    ignition_on:        bool  = True
    is_tunnel:          bool  = False
    # Temporal
    time_hour:          float = 12.0        # 0–23 local
    # Transaction
    amount:             float = 0.0
    currency:           str   = "INR"
    beneficiary:        str   = ""
    action:             str   = ""          # transfer | pay_toll | pay_fuel | ...
    beneficiary_known:  bool  = False
    # Behavioural passive signal (kept as a RISK input, never a TRUST input)
    behavioral_score:   float | None = None
    # Rolling per-user history summary (from the local store)
    amount_mean:        float = 0.0
    amount_std:         float = 0.0
    dist_from_home_km:  float = 0.0
    in_trusted_zone:    bool  = True


@dataclass
class DriveAuthResult:
    """
    The four simultaneous outputs of a single authentication call (§4.2),
    plus the routing / audit metadata the PolicyEngine and AuditLog need.
    """
    trust_score:      float
    risk_score:       float
    confidence_score: float
    decision:         Decision
    # Which tier the transaction fell into (micro / standard / high_value / guest)
    tier:             str = "standard"
    # Human-readable reason codes (§4.2 "explanations")
    explanations:     list[str] = field(default_factory=list)
    # Step-up routing (§4.3a)
    step_up_method:   str | None = None      # "otp_mobile" | "biometric_recapture_pin" | None
    step_up_fallback: str | None = None
    # The rule that made the final call, for auditability
    policy_rule:      str = ""
    # Fraud-ladder state at decision time (§6.2)
    fraud_state:      str = "normal"
    # Per-modality snapshot for the audit log (§8.3)
    modality_scores:  dict[str, Any] = field(default_factory=dict)
    # The exact thresholds active at decision time (§8.3)
    active_thresholds: dict[str, float] = field(default_factory=dict)
    ood_flags:        dict[str, bool]  = field(default_factory=dict)
    is_payment:       bool = False

    @property
    def score(self) -> float:
        """
        Back-compat shim: the old FusionResult exposed a single ``.score``.
        Call sites that logged ``result.score`` keep working — they now read
        the Trust Score, which is the closest analogue.
        """
        return self.trust_score

    @property
    def legacy_decision(self) -> str:
        return self.decision.legacy()


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))
