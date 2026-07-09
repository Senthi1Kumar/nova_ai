"""
worker.py — Two-Stage Personalized VAD Process Entry Point

Stage 1: FireRedGate (DFSMN, 0.6M params, 10ms frames)
  → Generic voice/silence/music/noise discrimination

Stage 2: SpeakerGate (ECAPA-TDNN cosine similarity, 250ms stride)
  → Driver identity verification (only runs when Stage 1 detects voice)

Gate logic:
  - Silence / music / noise  → Stage 2 skipped → gate OPEN (fail-open)
  - Driver's voice detected  → Stage 2 score ≥ threshold → gate OPEN
  - Stranger's voice detected → Stage 2 score < threshold → gate CLOSED

Env vars:
  NOVA_PVAD_ENABLED          1 (default) | 0
  NOVA_PVAD_DRIVER_ID        driver1 (default)
  NOVA_PVAD_THRESHOLD        0.35 (default)
  NOVA_PVAD_HYSTERESIS       2 (default, consecutive same-direction strides before emitting)
  NOVA_PVAD_VOICE_FRAC       0.30 (default, min Stage-1 voiced fraction to run ECAPA)
  NOVA_PVAD_STAGE1           firered (default) | rms
  NOVA_FIREREDASR2S_DIR      path to FireRedASR2S repo (auto-detected if omitted)
  NOVA_FIREREDVAD_MODEL_DIR  path to FireRedVAD model weights (auto-detected if omitted)
"""

import logging
import os
import sys
from queue import Empty

import numpy as np

logger = logging.getLogger("pVADWorker")

_STEP_SAMPLES = 160   # 10ms at 16kHz — must match firered_gate._STRIDE_SAMPLES


def run_pvad_worker(pvad_in_queue, ws_out_queue, stop_event):
    import setproctitle
    setproctitle.setproctitle("nova-pvad-worker")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(name)s) %(levelname)s: %(message)s",
    )

    # ── Add this module's parent to path so sibling imports work ──────────────
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # ── Config ────────────────────────────────────────────────────────────────
    enabled        = os.getenv("NOVA_PVAD_ENABLED", "1") == "1"
    driver_id      = os.getenv("NOVA_PVAD_DRIVER_ID", "driver1")
    threshold      = float(os.getenv("NOVA_PVAD_THRESHOLD", "0.35"))
    hysteresis     = int(os.getenv("NOVA_PVAD_HYSTERESIS", "2"))
    voice_frac_min = float(os.getenv("NOVA_PVAD_VOICE_FRAC", "0.30"))
    stage1_backend = os.getenv("NOVA_PVAD_STAGE1", "firered").lower()  # "firered" | "rms"
    rms_fallback   = float(os.getenv("NOVA_PVAD_ENERGY", "0.035"))

    if not enabled:
        logger.info("pVAD disabled — draining queue and exiting.")
        while not stop_event.is_set():
            try:
                pvad_in_queue.get(timeout=0.1)
            except Empty:
                pass
        return

    # ── Stage 1: FireRedGate ──────────────────────────────────────────────────
    firered_gate = None
    if stage1_backend == "firered":
        try:
            from pvad.firered_gate import FireRedGate  # type: ignore
            firered_gate = FireRedGate.load()
            logger.info("pVAD Stage-1: FireRedGate ready")
        except Exception as e:
            logger.warning(f"pVAD Stage-1: FireRedGate failed ({e}) — falling back to RMS gate")

    # ── Stage 2: SpeakerGate ──────────────────────────────────────────────────
    from pvad.speaker_gate import SpeakerGate  # type: ignore

    l3_dir = os.path.join(backend_dir, "nova-l7", "L-3")
    speaker_gate = SpeakerGate.load(
        l3_dir=l3_dir,
        driver_id=driver_id,
        threshold=threshold,
        hysteresis=hysteresis,
        voice_frac_min=voice_frac_min,
    )
    logger.info(
        f"pVAD: ready — driver='{driver_id}' threshold={threshold} "
        f"stage1={'FireRedGate' if firered_gate else 'RMS'} "
        f"stage2={'ECAPA' if speaker_gate._ecapa else 'fail-open'} "
        f"hysteresis={hysteresis} voice_frac_min={voice_frac_min}"
    )

    # ── RMS fallback helper ───────────────────────────────────────────────────
    # When FireRedGate isn't available, generate voice_flags using simple RMS
    # per 10ms step — same behaviour as the old pvad_worker.py energy gate.
    _rms_step_buf = np.zeros(_STEP_SAMPLES, dtype=np.float32)
    _rms_step_pos = 0

    def _rms_voice_flags(audio_f32: np.ndarray) -> list[bool]:
        nonlocal _rms_step_buf, _rms_step_pos
        flags = []
        pos = 0
        while pos < len(audio_f32):
            space = _STEP_SAMPLES - _rms_step_pos
            take  = min(space, len(audio_f32) - pos)
            _rms_step_buf[_rms_step_pos:_rms_step_pos + take] = audio_f32[pos:pos + take]
            _rms_step_pos += take
            pos           += take
            if _rms_step_pos == _STEP_SAMPLES:
                rms = float(np.sqrt(np.mean(_rms_step_buf ** 2)))
                flags.append(rms >= rms_fallback)
                _rms_step_pos = 0
        return flags

    # ── Main loop ─────────────────────────────────────────────────────────────
    prev_gate = True   # track previous gate state to only emit on changes

    while not stop_event.is_set():
        try:
            chunk = pvad_in_queue.get(timeout=0.05)
        except Empty:
            continue

        if not isinstance(chunk, (bytes, bytearray)):
            continue

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if len(audio_int16) == 0:
            continue

        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        # ── Stage 1: get per-10ms voice flags ────────────────────────────────
        if firered_gate is not None:
            frames      = firered_gate.feed(audio_int16)
            voice_flags = [f.is_voice for f in frames]
        else:
            voice_flags = _rms_voice_flags(audio_f32)

        # ── Stage 2: accumulate + ECAPA when stride completes ────────────────
        result = speaker_gate.feed(audio_f32, voice_flags)

        if result is None:
            continue  # stride not yet complete

        # Only emit pvad_gate on state changes (avoids flooding ws_out_queue)
        if result.gate_vote != prev_gate:
            prev_gate = result.gate_vote
            score = round(result.score, 3) if result.score is not None else 1.0
            ws_out_queue.put({
                "type":  "pvad_gate",
                "pass":  result.gate_vote,
                "score": score,
            })
            logger.info(
                f"pVAD gate {'OPEN' if result.gate_vote else 'CLOSED'} "
                f"(score={score}, voice_frac={result.voice_frac:.2f})"
            )

        # Continuous barge-in signal: driver voice confirmed this stride.
        # Emit every stride (~250ms) so the gateway can interrupt TTS promptly.
        if result.score is not None and result.gate_vote:
            ws_out_queue.put({
                "type":       "pvad_voice_detected",
                "score":      round(result.score, 3),
                "voice_frac": round(result.voice_frac, 3),
            })

    logger.info("pVAD worker stopped.")
