"""
driveauth/ood_detector.py
-------------------------
§8a.6 — Out-of-distribution detection per modality.

Catches inputs that don't resemble anything the matcher was enrolled on
(a costume mask, heavily-processed audio, an object on the fingerprint sensor).
A high raw match score paired with a high OOD flag must NOT be trusted — it
feeds the Confidence Score (never the Trust Score), which can then route an
otherwise-passing transaction to STEP_UP / REJECT.

Approach: unsupervised distance-to-enrollment-distribution. We don't need
labelled examples of every attack — we only need "does this live embedding sit
where the enrolled embeddings sit". Implemented as a Mahalanobis-style distance
against the per-driver enrollment statistics (mean + diagonal covariance),
falling back to cosine-distance-to-centroid when only a single enrolled vector
exists. Runs on CPU; it's a handful of vector ops.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("driveauth.ood")

# Distance beyond this (in std devs, or cosine distance) → flag as OOD.
_OOD_Z_THRESH      = float(os.getenv("NOVA_OOD_Z_THRESH", "3.0"))
_OOD_COSINE_THRESH = float(os.getenv("NOVA_OOD_COSINE_THRESH", "0.55"))


class _ModalityOOD:
    """OOD stats for one modality (mean, diagonal std, centroid)."""

    def __init__(self, mean: np.ndarray | None, std: np.ndarray | None):
        self._mean = mean
        self._std  = std

    @classmethod
    def from_store(cls, stats_path: Path) -> "_ModalityOOD":
        mean = std = None
        if stats_path.exists():
            try:
                data = np.load(stats_path)
                mean = data["mean"].astype(np.float32)
                std  = data["std"].astype(np.float32)
                std  = np.where(std < 1e-6, 1e-6, std)
                logger.info(f"OOD: loaded enrollment stats from {stats_path.name}")
            except Exception as exc:
                logger.warning(f"OOD: stats load failed ({exc})")
        return cls(mean, std)

    def is_ood(self, embedding: np.ndarray | None) -> tuple[bool, float]:
        """Returns (is_ood, distance_metric)."""
        if embedding is None:
            return False, 0.0    # nothing captured → not our call to flag
        if self._mean is None:
            return False, 0.0    # no enrollment stats → cannot judge, don't false-flag

        emb = embedding.astype(np.float32).ravel()
        if emb.shape != self._mean.shape:
            # Shape mismatch is itself suspicious, but more likely a wiring bug —
            # log and don't hard-flag, so a dimension change doesn't lock users out.
            logger.warning(f"OOD: shape mismatch {emb.shape} vs {self._mean.shape}")
            return False, 0.0

        if self._std is not None:
            z = np.abs(emb - self._mean) / self._std
            dist = float(np.sqrt(np.mean(z ** 2)))   # RMS z-score
            return dist > _OOD_Z_THRESH, dist

        # Cosine fallback
        a = emb / (np.linalg.norm(emb) + 1e-8)
        b = self._mean / (np.linalg.norm(self._mean) + 1e-8)
        cos_dist = float(1.0 - np.dot(a, b))
        return cos_dist > _OOD_COSINE_THRESH, cos_dist


class OODDetector:
    """Holds per-modality OOD stats and evaluates live embeddings."""

    def __init__(self, voice: _ModalityOOD, face: _ModalityOOD, finger: _ModalityOOD):
        self.voice  = voice
        self.face   = face
        self.finger = finger

    @classmethod
    def load(cls, store_dir: str, driver_id: str = "driver1") -> "OODDetector":
        store = Path(store_dir) / "ood_stats"
        return cls(
            voice  = _ModalityOOD.from_store(store / f"voice_{driver_id}.npz"),
            face   = _ModalityOOD.from_store(store / f"face_{driver_id}.npz"),
            finger = _ModalityOOD.from_store(store / f"finger_{driver_id}.npz"),
        )

    def evaluate(
        self,
        *,
        voice_emb:  np.ndarray | None = None,
        face_emb:   np.ndarray | None = None,
        finger_emb: np.ndarray | None = None,
    ) -> dict[str, bool]:
        v_ood, v_d = self.voice.is_ood(voice_emb)
        f_ood, f_d = self.face.is_ood(face_emb)
        p_ood, p_d = self.finger.is_ood(finger_emb)
        flags = {"voice": v_ood, "face": f_ood, "finger": p_ood}
        if any(flags.values()):
            logger.info(f"OOD: flags={flags} dist(v/f/p)={v_d:.2f}/{f_d:.2f}/{p_d:.2f}")
        return flags
