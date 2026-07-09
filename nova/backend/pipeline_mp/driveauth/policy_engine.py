"""
driveauth/policy_engine.py
-------------------------
§8a.4 + §8a.10 — Dedicated Policy Engine, separated from the ML scoring layer.

The ML layer (trust_fusion, risk_model, confidence) produces numbers. This
engine applies human-auditable, versioned rules ON TOP of those numbers to reach
the final ACCEPT / STEP_UP_REQUIRED / REJECT decision. Compliance/fintech
stakeholders can review and change policy here without touching any model, and
the models can be validated independently of business-rule changes.

Deterministic on purpose (see the "why not ML for the decision head" reasoning):
explainability for disputes, guaranteed monotonicity (more risk never buys less
scrutiny), a hard floor immune to upstream score manipulation, and an independent
update cadence from the ML pipeline.

Tiers (§8a.10), by transaction amount/category:
    micro       — single strong modality ok if Trust high & Risk low; no step-up
    standard    — 2-of-3 (or full available set); OTP step-up on ambiguous scores
    high_value  — all available modalities + mandatory OTP step-up regardless of Trust
    guest       — PIN/card-present, hard cap; OTP not applicable
"""

from __future__ import annotations

import logging
import os

from .types import Decision, DriveAuthResult, RiskContext

logger = logging.getLogger("driveauth.policy")

POLICY_VERSION = os.getenv("NOVA_POLICY_VERSION", "driveauth-1.0")

# Trust bars per tier (before fraud-ladder margin is added).
_TRUST_ACCEPT = {
    "micro":      float(os.getenv("NOVA_TRUST_ACCEPT_MICRO",  "0.75")),
    "standard":   float(os.getenv("NOVA_TRUST_ACCEPT_STD",    "0.82")),
    "high_value": float(os.getenv("NOVA_TRUST_ACCEPT_HIGH",   "0.88")),
    "guest":      1.01,   # unreachable — guests never clear on biometrics
}
_TRUST_REJECT = float(os.getenv("NOVA_TRUST_REJECT", "0.55"))

# Risk bands
_RISK_LOW  = float(os.getenv("NOVA_RISK_APPROVE", "0.35"))
_RISK_HIGH = float(os.getenv("NOVA_RISK_REJECT",  "0.80"))

# Confidence floor — below this we never blind-ACCEPT (§4.3 step 6)
_CONF_FLOOR = float(os.getenv("NOVA_CONF_FLOOR", "0.55"))

# Amount thresholds for tiering (currency-naive default; per-currency in prod)
_MICRO_MAX = float(os.getenv("NOVA_TIER_MICRO_MAX", "200.0"))
_HIGH_MIN  = float(os.getenv("NOVA_TIER_HIGH_MIN",  "50000.0"))
_GUEST_MAX = float(os.getenv("NOVA_TIER_GUEST_MAX", "1000.0"))


def classify_tier(ctx: RiskContext, is_guest: bool = False) -> str:
    if is_guest:
        return "guest"
    amt = ctx.amount
    if amt <= _MICRO_MAX and ctx.beneficiary_known:
        return "micro"
    if amt >= _HIGH_MIN or not ctx.beneficiary_known:
        return "high_value"
    return "standard"


class PolicyEngine:
    """
    Pure function of (trust, risk, confidence, tier, fraud-rigor, modality count).
    No model, no I/O — every branch is testable and every decision carries the
    rule name that produced it.
    """

    def decide(
        self,
        *,
        trust: float,
        risk: float,
        confidence: float,
        tier: str,
        n_confident_modalities: int,
        fraud_rigor: dict,
        explanations: list[str],
    ) -> tuple[Decision, str, dict[str, float], str | None]:
        """
        Returns (decision, policy_rule, active_thresholds, step_up_method).
        ``step_up_method`` is "otp_mobile" when a step-up is required, else None.
        """
        trust_bar = _TRUST_ACCEPT.get(tier, _TRUST_ACCEPT["standard"])
        trust_bar += float(fraud_rigor.get("trust_margin", 0.0))
        min_mods  = int(fraud_rigor.get("min_modalities", 1))
        force_su  = bool(fraud_rigor.get("force_step_up", False))
        blocked   = bool(fraud_rigor.get("block", False))

        active = {
            "trust_accept": round(trust_bar, 3),
            "trust_reject": _TRUST_REJECT,
            "risk_low": _RISK_LOW,
            "risk_high": _RISK_HIGH,
            "conf_floor": _CONF_FLOOR,
            "min_modalities": float(min_mods),
        }

        # ── Hard blocks first (defense-in-depth, immune to score manipulation) ─
        if blocked:
            return Decision.REJECT, f"{POLICY_VERSION}:fraud_locked", active, None

        if tier == "guest":
            # Guests never authenticate on biometrics — routed to PIN/card flow.
            explanations.append("guest_mode_requires_pin")
            return Decision.STEP_UP_REQUIRED, f"{POLICY_VERSION}:guest_pin_required", active, "pin_card_present"

        if risk >= _RISK_HIGH:
            explanations.append("risk_above_hard_ceiling")
            return Decision.REJECT, f"{POLICY_VERSION}:risk_ceiling", active, None

        if trust < _TRUST_REJECT:
            explanations.append("trust_below_floor")
            return Decision.REJECT, f"{POLICY_VERSION}:trust_floor", active, None

        if n_confident_modalities < min_mods:
            explanations.append(f"need_{min_mods}_modalities_have_{n_confident_modalities}")
            return Decision.STEP_UP_REQUIRED, f"{POLICY_VERSION}:insufficient_modalities", active, "otp_mobile"

        # ── High-value: OTP is mandatory regardless of how good Trust looks ────
        if tier == "high_value" or force_su:
            rule = "high_value_mandatory_stepup" if tier == "high_value" else "fraud_ladder_stepup"
            explanations.append(rule)
            return Decision.STEP_UP_REQUIRED, f"{POLICY_VERSION}:{rule}", active, "otp_mobile"

        # ── Confidence gate: never blind-ACCEPT on dirty inputs (§4.3 step 6) ──
        if confidence < _CONF_FLOOR:
            explanations.append("low_confidence_inputs")
            return Decision.STEP_UP_REQUIRED, f"{POLICY_VERSION}:low_confidence", active, "otp_mobile"

        # ── Normal path ───────────────────────────────────────────────────────
        if trust >= trust_bar and risk <= _RISK_LOW:
            return Decision.ACCEPT, f"{POLICY_VERSION}:accept_{tier}", active, None

        # Ambiguous middle band → step up
        explanations.append("ambiguous_trust_or_risk")
        return Decision.STEP_UP_REQUIRED, f"{POLICY_VERSION}:ambiguous", active, "otp_mobile"
