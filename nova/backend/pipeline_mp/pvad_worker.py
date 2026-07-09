"""
pvad_worker.py — Personalized VAD (Speaker-Conditioned Audio Gate)

Thin wrapper that delegates to the two-stage pVAD implementation in
nova/backend/pvad/ (FireRedGate + SpeakerGate).

Stage 1: FireRedGate (DFSMN, 0.6M params, 10ms frames)
  - Generic voice/silence/music/noise discrimination
  - Falls back to RMS energy gate if FireRedVAD unavailable

Stage 2: SpeakerGate (ECAPA-TDNN cosine similarity, 250ms stride)
  - Driver identity verification using voiced-only audio ring buffer
  - Only runs when Stage 1 detects voice (voice fraction gate)

Gate logic:
  - Silence / music / noise  -> Stage 2 skipped -> gate OPEN (fail-open)
  - Driver's voice detected  -> Stage 2 score >= threshold -> gate OPEN
  - Stranger's voice detected -> Stage 2 score < threshold -> gate CLOSED

Env vars:
  NOVA_PVAD_ENABLED          1 (default) | 0
  NOVA_PVAD_DRIVER_ID        driver1 (default)
  NOVA_PVAD_THRESHOLD        0.35 (default)
  NOVA_PVAD_HYSTERESIS       2 (default)
  NOVA_PVAD_VOICE_FRAC       0.30 (default)
  NOVA_PVAD_STAGE1           firered (default) | rms
  NOVA_PVAD_ENERGY           0.035 (default, RMS fallback threshold)
"""

import os
import sys

# Ensure nova/backend/ is on sys.path so pvad package is importable
_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from pvad.worker import run_pvad_worker  # noqa: E402, F401

__all__ = ["run_pvad_worker"]
