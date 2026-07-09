"""
pipeline_mp/driveauth/
======================
DriveAuth Edge — the Trust/Risk-separated replacement for the monolithic
BiometricGate fusion in ``biometric_gate.py``.

Design (mirrors the DriveAuth Edge proposal):

  Sensor capture  →  QualityGate (§8a.5)  ──gate──▶ matchers
                                                     │
       voice / face / finger ModalityResults ────────┤
                                                     ▼
                                            TrustFusion (§4.3)   ← biometrics ONLY
                                                     │            (no behaviour / GPS)
                                            RiskModel  (§7)      ← GPS / CAN / history
                                                     │            (Risk, kept separate)
                                            OODDetector (§8a.6) ─┐
                                            QualityFlags ────────┼▶ Confidence (§4.3 step 6)
                                                     ▼           ┘
                                            PolicyEngine (§8a.4 / §8a.10)
                                              Trust + Risk + Confidence + tier
                                                     ▼
                                            Decision: ACCEPT / STEP_UP_REQUIRED / REJECT
                                                     │
                                        FraudStateMachine (§6.2) adjusts rigor
                                                     │
                                 STEP_UP_REQUIRED → OTP (§4.3a) ─▶ fallback if no signal

The key fix over the old ``FusionScorer``: behaviour / location / context data
NEVER enters the Trust number. It drives a separate **Risk Score** instead
(proposal §4.3). ``BehavioralMonitor`` is kept as an input to the RiskModel,
but its score is no longer blended into Trust.

The public entry point is :class:`DriveAuthGate` (``gate.py``), which keeps the
exact ``.load()`` / ``.intercept()`` / ``.require_auth(tier=...)`` signatures of
the old ``BiometricGate`` so the two call sites (stt_worker, llm_worker) don't
need restructuring — only the class name in the import changes.
"""

from __future__ import annotations

from .types import (
    Decision,
    ModalityResult,
    QualityFlags,
    RiskContext,
    DriveAuthResult,
)
from .gate import DriveAuthGate

__all__ = [
    "DriveAuthGate",
    "DriveAuthResult",
    "Decision",
    "ModalityResult",
    "QualityFlags",
    "RiskContext",
]
