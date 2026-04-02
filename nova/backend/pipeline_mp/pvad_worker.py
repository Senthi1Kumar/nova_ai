"""
pvad_worker.py — Personalized VAD (Speaker-Conditioned Audio Gate)

Runs as a separate process alongside KWS/STT/LLM/TTS workers.
Receives copies of raw mic audio, runs ECAPA-TDNN speaker embedding on
a rolling 1s window, and emits pvad_gate events to the gateway.

The gateway gates FSM transitions (IDLE→LISTENING) on these events —
audio still flows to KWS/STT without delay (no latency added to pipeline).

Architecture: parallel advisory gate, not inline sequential filter.
"""

import multiprocessing as mp
import logging
import os
import sys
import numpy as np
from pathlib import Path
from queue import Empty

logger = logging.getLogger("pVADWorker")

# 1s window, 0.5s stride at 16kHz
_WINDOW_SAMPLES = 16_000
_STRIDE_SAMPLES = 8_000
_DEFAULT_THRESHOLD = 0.58


def run_pvad_worker(
    pvad_in_queue: "mp.Queue",
    ws_out_queue: "mp.Queue",
    stop_event: "mp.Event",
) -> None:
    import setproctitle
    setproctitle.setproctitle("nova-pvad-worker")

    logging.basicConfig(level=logging.INFO)

    # Resolve paths — pvad_worker lives in pipeline_mp/, L-3 is ../nova-l7/L-3/
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    l3_dir = os.path.join(backend_dir, "nova-l7", "L-3")
    sys.path.insert(0, l3_dir)

    threshold = float(os.getenv("NOVA_PVAD_THRESHOLD", str(_DEFAULT_THRESHOLD)))
    driver_id = os.getenv("NOVA_PVAD_DRIVER_ID", "driver1")
    enabled = os.getenv("NOVA_PVAD_ENABLED", "1") == "1"

    if not enabled:
        logger.info("pVAD disabled (NOVA_PVAD_ENABLED=0) — draining queue only.")
        while not stop_event.is_set():
            try:
                pvad_in_queue.get(timeout=0.1)
            except Empty:
                pass
        return

    # ── Load driver voiceprint ──────────────────────────────────────────────────
    driver_embedding: np.ndarray | None = None
    try:
        from crypto_utils import load_array
        voiceprint_path = Path(l3_dir) / "data" / "voiceprints" / f"{driver_id}.enc"
        if voiceprint_path.exists():
            driver_embedding = load_array(voiceprint_path)
            logger.info(f"pVAD: loaded voiceprint for '{driver_id}' (dim={driver_embedding.shape})")
        else:
            logger.warning(
                f"pVAD: no voiceprint found for '{driver_id}' at {voiceprint_path}. "
                "Running in fail-open mode (pvad_pass=True always)."
            )
    except Exception as e:
        logger.error(f"pVAD: failed to load voiceprint: {e}. Fail-open mode.")
        driver_embedding = None

    # ── Load ECAPA-TDNN ─────────────────────────────────────────────────────────
    ecapa_model = None
    if driver_embedding is not None:
        try:
            # Re-use same monkeypatch as verify.py for hf_hub compat
            import huggingface_hub as _hf
            _orig = _hf.hf_hub_download
            def _patched(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token") or None
                try:
                    return _orig(*args, **kwargs)
                except Exception as e:
                    filename = args[1] if len(args) > 1 else kwargs.get("filename", "")
                    is_404 = ("404" in str(e) or "Not Found" in str(e) or
                              "EntryNotFound" in type(e).__name__ or
                              "RemoteEntryNotFound" in type(e).__name__)
                    if "custom.py" in str(filename) and is_404:
                        raise ValueError("File not found on HF hub") from e
                    raise
            _hf.hf_hub_download = _patched

            from speechbrain.inference.speaker import EncoderClassifier
            ecapa_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(l3_dir, "pretrained_models", "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
            )
            logger.info("pVAD: ECAPA-TDNN loaded on CPU.")
        except Exception as e:
            logger.error(f"pVAD: ECAPA-TDNN load failed: {e}. Fail-open mode.")
            ecapa_model = None

    # ── Rolling ring buffer ─────────────────────────────────────────────────────
    ring_buf = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
    samples_accumulated = 0  # total samples seen since last stride reset
    stride_buf = np.zeros(_STRIDE_SAMPLES, dtype=np.float32)
    stride_pos = 0

    def _infer_score(window_f32: np.ndarray) -> float:
        """Run ECAPA on window, return cosine similarity vs driver_embedding."""
        import torch
        waveform = torch.from_numpy(window_f32).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = ecapa_model.encode_batch(waveform).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return 0.0
        emb = emb / norm
        return float(np.dot(emb, driver_embedding))

    last_gate = True  # previous emitted decision (start open)

    # Hysteresis: require N consecutive same-direction windows before emitting
    # a state change. Prevents single-window noise bursts from toggling the gate.
    _HYSTERESIS = int(os.getenv("NOVA_PVAD_HYSTERESIS", "2"))
    _consecutive_same = 0
    _pending_gate = True   # what the last window voted

    logger.info(
        f"pVAD: running — threshold={threshold}, driver='{driver_id}', "
        f"hysteresis={_HYSTERESIS}, energy={os.getenv('NOVA_PVAD_ENERGY', '0.035')}"
    )

    # ── Main loop ───────────────────────────────────────────────────────────────
    while not stop_event.is_set():
        try:
            chunk = pvad_in_queue.get(timeout=0.05)
        except Empty:
            continue

        if not isinstance(chunk, (bytes, bytearray)):
            continue

        # Convert int16 bytes → float32 [-1, 1]
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if len(audio_int16) == 0:
            continue
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        # Accumulate into stride buffer
        pos = 0
        while pos < len(audio_f32):
            space = _STRIDE_SAMPLES - stride_pos
            take = min(space, len(audio_f32) - pos)
            stride_buf[stride_pos:stride_pos + take] = audio_f32[pos:pos + take]
            stride_pos += take
            pos += take

            if stride_pos >= _STRIDE_SAMPLES:
                # Stride complete — shift ring buffer and append new stride
                ring_buf = np.roll(ring_buf, -_STRIDE_SAMPLES)
                ring_buf[-_STRIDE_SAMPLES:] = stride_buf
                samples_accumulated += _STRIDE_SAMPLES
                stride_pos = 0

                # Only infer once we have a full 1s window
                if samples_accumulated < _WINDOW_SAMPLES:
                    continue

                # Fail-open if model/embedding not available
                if ecapa_model is None or driver_embedding is None:
                    if not last_gate:
                        ws_out_queue.put({"type": "pvad_gate", "pass": True, "score": 1.0})
                        last_gate = True
                    continue

                try:
                    # Energy gate: raised to 0.035 (~-29 dBFS) to sit above typical
                    # room noise (fan, TV) which sits at 0.020–0.030 RMS.
                    # Only run ECAPA inference at actual speech-level energy.
                    rms = float(np.sqrt(np.mean(ring_buf ** 2)))
                    speech_energy_threshold = float(os.getenv("NOVA_PVAD_ENERGY", "0.035"))
                    score = None
                    if rms < speech_energy_threshold:
                        # Silence / background noise — gate should be open
                        this_vote = True
                    else:
                        score = _infer_score(ring_buf.copy())
                        this_vote = score >= threshold
                        logger.debug(f"pVAD score={score:.3f} rms={rms:.4f} pass={this_vote}")

                    # Hysteresis: only emit when N consecutive windows agree
                    if this_vote == _pending_gate:
                        _consecutive_same += 1
                    else:
                        _pending_gate = this_vote
                        _consecutive_same = 1

                    if _consecutive_same >= _HYSTERESIS and this_vote != last_gate:
                        last_gate = this_vote
                        if this_vote:
                            ws_out_queue.put({"type": "pvad_gate", "pass": True, "score": 1.0})
                            logger.info("pVAD gate OPEN (silence or driver voice)")
                        else:
                            ws_out_queue.put({"type": "pvad_gate", "pass": False, "score": round(score, 3) if score is not None else 0.0})
                            logger.info(f"pVAD gate CLOSED (score={score:.3f} rms={rms:.4f})" if score is not None else f"pVAD gate CLOSED (rms={rms:.4f})")
                except Exception as e:
                    logger.error(f"pVAD inference error: {e}")

    logger.info("pVAD worker stopped.")
