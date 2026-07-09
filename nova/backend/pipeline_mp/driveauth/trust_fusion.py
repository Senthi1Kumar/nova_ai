"""
driveauth/trust_fusion.py
-------------------------
§4.3 — Trust Score fusion. Replacement for the old ``FusionScorer``.

Fuses ONLY the biometric modalities — voice, face, fingerprint. No behaviour,
no GPS, no context. Those moved to ``risk_model.py``. This is the central fix:
the Trust Score answers exactly one question — "how confident are we this is
the enrolled person, right now" — and nothing about *where* the car is or *how*
it's being driven is allowed to inflate or deflate that number.

Dynamic weighting: we keep the existing ``DynamicOrchestrator`` (Tier-1 ONNX
PolicyMLP on CPU, Tier-2 SmolLM2 on GPU) because it's already correctly built
for this hardware. We simply drop the ``behavior`` slot from its weight dict
and re-normalise across voice/face/finger. If the orchestrator isn't present,
we fall back to static biometric-only weights.

Live quality flags down-weight a degraded modality automatically (a blurry
face frame contributes less), on top of whatever the orchestrator returns.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .types import ModalityResult, clip01

logger = logging.getLogger("driveauth.trust")

# Static biometric-only weights (behaviour removed). Used when the orchestrator
# is unavailable. Re-normalised over whatever modalities are actually present.
_STATIC_W = {
    "voice":  float(os.getenv("NOVA_TRUST_W_VOICE",  "0.30")),
    "face":   float(os.getenv("NOVA_TRUST_W_FACE",   "0.40")),
    "finger": float(os.getenv("NOVA_TRUST_W_FINGER", "0.30")),
}


class TrustFusion:
    """
    Biometric-only weighted fusion → Trust Score in [0, 1].

    ``orchestrator`` is an optional DynamicOrchestrator. When present, its
    weights are used but the ``behavior`` component is stripped and the
    remaining three are re-normalised.
    """

    def __init__(self, orchestrator=None):
        self._orch = orchestrator

    def _weights(self, ctx=None) -> dict[str, float]:
        if self._orch is not None and ctx is not None:
            try:
                w, _thresh_delta = self._orch.get_weights(ctx)
                # Strip behaviour — it is NOT a Trust input any more.
                w = {k: float(v) for k, v in w.items() if k in ("voice", "face", "finger")}
                s = sum(w.values())
                if s > 1e-6:
                    return {k: v / s for k, v in w.items()}
            except Exception as exc:
                logger.warning(f"TrustFusion: orchestrator weights failed ({exc}) — static")
        return dict(_STATIC_W)

    def fuse(
        self,
        voice:  ModalityResult,
        face:   ModalityResult,
        finger: ModalityResult,
        orch_ctx=None,
    ) -> tuple[float, dict[str, float]]:
        """
        Returns (trust_score, effective_weights).

        A modality is included only if it produced a confident score. Each
        included modality's weight is scaled by its live quality (§8a.5) before
        re-normalisation, so a low-quality-but-confident capture counts for less.
        """
        base_w = self._weights(orch_ctx)
        candidates: dict[str, tuple[float, float]] = {}   # name → (score, weight)

        def _add(name: str, res: ModalityResult) -> None:
            if res.score is not None and res.confident:
                w = base_w.get(name, 0.0) * max(res.quality, 0.05)
                candidates[name] = (res.score, w)

        _add("voice",  voice)
        _add("face",   face)
        _add("finger", finger)

        if not candidates:
            return 0.0, {}

        total_w = sum(w for _, w in candidates.values())
        if total_w <= 1e-9:
            return 0.0, {}

        eff: dict[str, float] = {}
        fused = 0.0
        for name, (score, w) in candidates.items():
            ew = w / total_w
            eff[name] = ew
            fused += ew * score
        return clip01(fused), eff
