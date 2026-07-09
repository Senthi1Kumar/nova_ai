"""
driveauth/confidence.py
----------------------
§4.3 step 6 — Confidence Score.

Distinct from Trust. Trust = "is this the enrolled person". Confidence =
"how much should we trust our own scores this time" — a measure of the system's
internal agreement and input cleanliness, NOT of the user.

Three ingredients:
  1. Quality flags (§8a.5) — were the captures clean?
  2. OOD flags (§8a.6)     — do the inputs look like anything we enrolled on?
  3. Modality variance     — do the biometrics agree with each other, or is one
                             saying "yes" and another "no"?

A low Confidence Score with a high Trust Score should bias the decision toward
STEP_UP rather than blind ACCEPT (the PolicyEngine consumes this). That decouples
"the model is sure" from "the model's inputs were clean" — which matters a lot in
a noisy, vibrating, variable-lighting cabin.
"""

from __future__ import annotations

import logging

import numpy as np

from .types import ModalityResult, QualityFlags, clip01

logger = logging.getLogger("driveauth.confidence")


class ConfidenceScorer:

    @staticmethod
    def score(
        voice:  ModalityResult,
        face:   ModalityResult,
        finger: ModalityResult,
        quality: QualityFlags,
        ood_flags: dict[str, bool],
    ) -> tuple[float, list[str]]:
        """Returns (confidence_0_1, reason_codes)."""
        reasons: list[str] = []

        present = [r for r in (voice, face, finger)
                   if r.score is not None and r.confident]

        # ── 1. agreement / variance across present modalities ────────────────
        if len(present) >= 2:
            scores = np.array([r.score for r in present], dtype=np.float32)
            spread = float(scores.max() - scores.min())
            agreement = clip01(1.0 - spread)      # wide spread → low agreement
            if spread > 0.35:
                reasons.append("modalities_disagree")
        elif len(present) == 1:
            agreement = 0.6                         # single modality: capped confidence
            reasons.append("single_modality_only")
        else:
            return 0.0, ["no_confident_modality"]

        # ── 2. capture quality ───────────────────────────────────────────────
        q_vals = []
        if voice.score is not None:  q_vals.append(quality.voice_q)
        if face.score is not None:   q_vals.append(quality.face_q)
        if finger.score is not None: q_vals.append(quality.finger_q)
        quality_score = float(np.mean(q_vals)) if q_vals else 0.5
        if quality_score < 0.5:
            reasons.append("low_capture_quality")

        # ── 3. OOD penalty ───────────────────────────────────────────────────
        ood_hits = sum(1 for v in ood_flags.values() if v)
        ood_penalty = clip01(ood_hits / max(len(present), 1))
        if ood_hits:
            reasons.append("out_of_distribution_input")

        # ── 4. hardware fault penalty (§8a.7) ────────────────────────────────
        fault_penalty = 0.25 if quality.hardware_fault else 0.0
        if quality.hardware_fault:
            reasons.append("sensor_hardware_fault")

        confidence = (
            0.45 * agreement
            + 0.35 * quality_score
            + 0.20 * (1.0 - ood_penalty)
        ) - fault_penalty

        confidence = clip01(confidence)
        logger.debug(f"Confidence: {confidence:.3f} agree={agreement:.2f} "
                     f"q={quality_score:.2f} ood={ood_hits} reasons={reasons}")
        return confidence, reasons
