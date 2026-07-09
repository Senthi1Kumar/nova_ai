"""
driveauth/quality_gate.py
-------------------------
§8a.5 — Sensor quality assessment BEFORE matching.

Each capture is quality-scored first; a capture failing its modality's minimum
threshold is rejected (re-capture prompted) rather than matched anyway and then
down-weighted after the fact. Cheaper and more reliable than post-hoc weighting.

All checks are deterministic signal-processing heuristics (SNR / clip / blur /
occlusion / contact-area) — no learned model, because "is this signal clean
enough" is an objective physical measurement with a well-defined pass bar, and
we want it to fail predictably and transparently.

This module does NOT do matching; it only gates whether matching should run.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .types import QualityFlags

logger = logging.getLogger("driveauth.quality")

# ── Thresholds (env-overridable) ─────────────────────────────────────────────
_VOICE_MIN_SNR_DB   = float(os.getenv("NOVA_Q_VOICE_MIN_SNR", "6.0"))
_VOICE_CLIP_FRAC    = float(os.getenv("NOVA_Q_VOICE_CLIP_FRAC", "0.02"))   # >2% clipped → fail
_VOICE_MIN_SECONDS  = float(os.getenv("NOVA_Q_VOICE_MIN_SEC", "1.0"))
_FACE_MIN_SHARPNESS = float(os.getenv("NOVA_Q_FACE_MIN_SHARP", "40.0"))    # variance-of-Laplacian
_FACE_MIN_BRIGHT    = float(os.getenv("NOVA_Q_FACE_MIN_BRIGHT", "25.0"))
_FACE_MAX_BRIGHT    = float(os.getenv("NOVA_Q_FACE_MAX_BRIGHT", "235.0"))
_FINGER_MIN_CONTACT = float(os.getenv("NOVA_Q_FINGER_MIN_CONTACT", "0.35"))  # ridge coverage frac


def _snr_db(audio: np.ndarray) -> float:
    """Crude SNR: ratio of the loud (speech) frames to the quiet (noise) floor."""
    if audio.size == 0:
        return 0.0
    frame = 400  # 25 ms @ 16k
    n = (audio.size // frame) * frame
    if n < frame:
        return 0.0
    energies = (audio[:n].reshape(-1, frame) ** 2).mean(axis=1)
    energies = np.sqrt(energies + 1e-12)
    noise = np.percentile(energies, 10)
    speech = np.percentile(energies, 90)
    if noise <= 1e-9:
        return 40.0
    return float(20.0 * np.log10(max(speech / noise, 1e-6)))


def score_voice(audio_f32: np.ndarray, sample_rate: int = 16_000) -> tuple[bool, float, list[str]]:
    """Returns (ok, quality_0_1, notes)."""
    notes: list[str] = []
    if audio_f32 is None or audio_f32.size == 0:
        return False, 0.0, ["voice_no_audio"]

    duration = audio_f32.size / sample_rate
    if duration < _VOICE_MIN_SECONDS:
        notes.append("voice_too_short")

    clip_frac = float(np.mean(np.abs(audio_f32) > 0.995))
    if clip_frac > _VOICE_CLIP_FRAC:
        notes.append("voice_clipping")

    snr = _snr_db(audio_f32)
    if snr < _VOICE_MIN_SNR_DB:
        notes.append("voice_low_snr")

    # Quality blends SNR headroom and clip-freedom into 0..1
    q_snr  = float(np.clip((snr - _VOICE_MIN_SNR_DB) / 24.0 + 0.5, 0.0, 1.0))
    q_clip = float(np.clip(1.0 - clip_frac / max(_VOICE_CLIP_FRAC, 1e-6), 0.0, 1.0))
    quality = 0.6 * q_snr + 0.4 * q_clip
    ok = (duration >= _VOICE_MIN_SECONDS) and (clip_frac <= _VOICE_CLIP_FRAC) and (snr >= _VOICE_MIN_SNR_DB)
    return ok, quality, notes


def score_face(frame_gray: np.ndarray | None) -> tuple[bool, float, list[str]]:
    """
    Returns (ok, quality_0_1, notes). ``frame_gray`` is a HxW uint8/float array
    (the IR/greyscale frame). If None (camera unavailable), fails closed.
    """
    notes: list[str] = []
    if frame_gray is None or getattr(frame_gray, "size", 0) == 0:
        return False, 0.0, ["face_no_frame"]

    f = frame_gray.astype(np.float32)
    # Sharpness via variance of a 3x3 Laplacian (blur/occlusion proxy)
    lap = (
        -4.0 * f
        + np.roll(f, 1, 0) + np.roll(f, -1, 0)
        + np.roll(f, 1, 1) + np.roll(f, -1, 1)
    )
    sharpness = float(lap.var())
    brightness = float(f.mean())

    if sharpness < _FACE_MIN_SHARPNESS:
        notes.append("face_blurry_or_occluded")
    if brightness < _FACE_MIN_BRIGHT:
        notes.append("face_underexposed")
    if brightness > _FACE_MAX_BRIGHT:
        notes.append("face_overexposed")

    q_sharp  = float(np.clip(sharpness / (_FACE_MIN_SHARPNESS * 4.0), 0.0, 1.0))
    q_bright = float(np.clip(1.0 - abs(brightness - 130.0) / 130.0, 0.0, 1.0))
    quality = 0.65 * q_sharp + 0.35 * q_bright
    ok = (sharpness >= _FACE_MIN_SHARPNESS) and (_FACE_MIN_BRIGHT <= brightness <= _FACE_MAX_BRIGHT)
    return ok, quality, notes


def score_finger(contact_fraction: float | None, ridge_clarity: float | None = None
                 ) -> tuple[bool, float, list[str]]:
    """
    Returns (ok, quality_0_1, notes).
    ``contact_fraction`` and ``ridge_clarity`` come from the sensor SDK's own
    quality metric (0..1). If the sensor gave us nothing, fail closed.
    """
    notes: list[str] = []
    if contact_fraction is None:
        return False, 0.0, ["finger_no_metric"]

    if contact_fraction < _FINGER_MIN_CONTACT:
        notes.append("finger_low_contact")

    clarity = 1.0 if ridge_clarity is None else float(np.clip(ridge_clarity, 0.0, 1.0))
    quality = float(np.clip(0.5 * contact_fraction / max(_FINGER_MIN_CONTACT, 1e-6) * 0.5
                            + 0.5 * clarity, 0.0, 1.0))
    ok = contact_fraction >= _FINGER_MIN_CONTACT
    return ok, quality, notes


class QualityGate:
    """
    Runs the per-modality quality checks and returns a QualityFlags summary.
    Inputs are optional — only the modalities actually captured this turn need
    to be passed. A modality not passed is treated as "not attempted" (ok=True,
    quality=1.0) so it doesn't spuriously fail the gate; whether it was required
    is the PolicyEngine's job, not the QualityGate's.
    """

    def evaluate(
        self,
        *,
        voice_audio:      np.ndarray | None = None,
        face_frame_gray:  np.ndarray | None = None,
        finger_contact:   float | None = None,
        finger_clarity:   float | None = None,
        hardware_fault:   bool = False,
    ) -> QualityFlags:
        flags = QualityFlags(hardware_fault=hardware_fault)

        if voice_audio is not None:
            ok, q, notes = score_voice(voice_audio)
            flags.voice_ok, flags.voice_q = ok, q
            flags.notes.extend(notes)

        if face_frame_gray is not None:
            ok, q, notes = score_face(face_frame_gray)
            flags.face_ok, flags.face_q = ok, q
            flags.notes.extend(notes)

        if finger_contact is not None:
            ok, q, notes = score_finger(finger_contact, finger_clarity)
            flags.finger_ok, flags.finger_q = ok, q
            flags.notes.extend(notes)

        if flags.notes:
            logger.debug(f"QualityGate: flags={flags.notes}")
        return flags
