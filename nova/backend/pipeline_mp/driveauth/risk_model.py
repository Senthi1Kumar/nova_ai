"""
driveauth/risk_model.py
----------------------
§7 — Adaptive Risk model. THIS is where GPS / geofence / ignition+speed (CAN) /
time-of-day / amount / beneficiary-novelty / behaviour live.

The whole point of the driveauth rewrite: this data produces a **Risk Score**
and is kept fully separate from the **Trust Score** (proposal §4.3). The old
``FusionScorer`` blended a ``BehavioralMonitor`` score straight into Trust —
that conflation is the main bug this fixes. ``behavioral_score`` is still an
input here, but it is a RISK input, never a TRUST input.

Risk is "how risky is this transaction, independent of who is initiating it".
Higher = riskier. It is intentionally interpretable: a gradient-boosted tree or
small MLP when a trained model is present, otherwise a transparent additive
rule set. Runs on CPU (cheap, and keeps GPU headroom for STT/LLM/TTS — matches
Nova's per-worker GPU allocation design).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np

from .types import RiskContext, clip01

logger = logging.getLogger("driveauth.risk")

# Risk decision bands (consumed by the PolicyEngine, not applied here)
RISK_APPROVE = float(os.getenv("NOVA_RISK_APPROVE", "0.35"))   # risk ≤ this = low
RISK_REJECT  = float(os.getenv("NOVA_RISK_REJECT",  "0.80"))   # risk ≥ this = hard-risky


class RiskModel:
    """
    Produces a Risk Score in [0, 1] from a RiskContext.

    If a trained model (``risk_gbt.onnx`` — a GBT/MLP over the feature vector)
    is present in the store, it is used and run on CPU. Otherwise a transparent
    additive fallback is used so the system is fully functional out of the box
    and every risk contribution is auditable.
    """

    _FEATURE_ORDER = (
        "amount_z", "amount_norm", "beneficiary_novel", "dist_from_home",
        "out_of_zone", "night", "moving_fast", "ignition_off_anomaly",
        "tunnel", "behavior_anomaly",
    )

    def __init__(self, session=None):
        self._session = session
        self._input_name = session.get_inputs()[0].name if session is not None else None

    @classmethod
    def load(cls, store_dir: str) -> "RiskModel":
        path = Path(store_dir) / "risk_gbt.onnx"
        session = None
        if path.exists():
            try:
                import onnxruntime as ort  # type: ignore
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                # Force CPU: risk model is cheap; keep the GPU for STT/LLM/TTS.
                session = ort.InferenceSession(
                    str(path), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("RiskModel: trained model loaded (CPU)")
            except Exception as exc:
                logger.warning(f"RiskModel: model load failed ({exc}) — using additive fallback")
        else:
            logger.info("RiskModel: no trained model — using transparent additive fallback")
        return cls(session)

    # ── feature engineering ──────────────────────────────────────────────────

    def _features(self, ctx: RiskContext) -> dict[str, float]:
        amount_z = 0.0
        if ctx.amount_std > 1e-6:
            amount_z = (ctx.amount - ctx.amount_mean) / ctx.amount_std
        amount_z = float(np.clip(amount_z, -3.0, 6.0))

        night = 1.0 if (ctx.time_hour < 5.0 or ctx.time_hour >= 23.0) else 0.0
        # A transaction while moving fast is atypical for a deliberate payment.
        moving_fast = clip01((ctx.speed_kmh - 20.0) / 80.0)
        # Ignition off but a transaction firing is a mild anomaly.
        ign_anom = 0.0 if ctx.ignition_on else 1.0
        behavior_anom = 0.0
        if ctx.behavioral_score is not None:
            # behavioral_score ~1.0 = matches enrolled driving style; invert to anomaly.
            behavior_anom = clip01(1.0 - ctx.behavioral_score)

        return {
            "amount_z":             amount_z,
            "amount_norm":          clip01(ctx.amount / 100_000.0),
            "beneficiary_novel":    0.0 if ctx.beneficiary_known else 1.0,
            "dist_from_home":       clip01(ctx.dist_from_home_km / 50.0),
            "out_of_zone":          0.0 if ctx.in_trusted_zone else 1.0,
            "night":                night,
            "moving_fast":          moving_fast,
            "ignition_off_anomaly": ign_anom,
            "tunnel":               1.0 if ctx.is_tunnel else 0.0,
            "behavior_anomaly":     behavior_anom,
        }

    def _vector(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([[feats[k] for k in self._FEATURE_ORDER]], dtype=np.float32)

    # ── scoring ──────────────────────────────────────────────────────────────

    def score(self, ctx: RiskContext) -> tuple[float, list[str]]:
        """Returns (risk_score_0_1, contributing_reason_codes)."""
        feats = self._features(ctx)

        if self._session is not None:
            try:
                out = self._session.run(None, {self._input_name: self._vector(feats)})[0]
                risk = clip01(float(np.ravel(out)[0]))
                reasons = self._reasons(feats)
                return risk, reasons
            except Exception as exc:
                logger.warning(f"RiskModel: inference failed ({exc}) — additive fallback")

        # Transparent additive fallback. Weights chosen to be interpretable;
        # each term is a named risk driver you can point at during an audit.
        risk = (
            0.22 * clip01(feats["amount_z"] / 4.0)
            + 0.14 * feats["amount_norm"]
            + 0.16 * feats["beneficiary_novel"]
            + 0.12 * feats["dist_from_home"]
            + 0.12 * feats["out_of_zone"]
            + 0.06 * feats["night"]
            + 0.06 * feats["moving_fast"]
            + 0.04 * feats["ignition_off_anomaly"]
            + 0.02 * feats["tunnel"]
            + 0.06 * feats["behavior_anomaly"]
        )
        return clip01(risk), self._reasons(feats)

    @staticmethod
    def _reasons(feats: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if feats["amount_z"] > 2.0:
            reasons.append("amount_far_above_usual")
        if feats["amount_norm"] > 0.5:
            reasons.append("large_absolute_amount")
        if feats["beneficiary_novel"] > 0.5:
            reasons.append("first_time_beneficiary")
        if feats["out_of_zone"] > 0.5:
            reasons.append("unfamiliar_location")
        if feats["dist_from_home"] > 0.6:
            reasons.append("far_from_home")
        if feats["night"] > 0.5:
            reasons.append("unusual_hour")
        if feats["moving_fast"] > 0.3:
            reasons.append("transaction_while_moving")
        if feats["behavior_anomaly"] > 0.4:
            reasons.append("driving_style_anomaly")
        return reasons
