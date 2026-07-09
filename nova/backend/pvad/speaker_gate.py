"""
speaker_gate.py — Stage 2: Driver Identity Verification

ECAPA-TDNN cosine similarity gate. Verifies the current speaker matches
the enrolled driver voiceprint. Only runs when Stage 1 (FireRedGate)
confirms genuine voice is present (voice fraction gate).

Design:
  - Rolling 500ms window, 250ms stride
  - Each 10ms step is tagged with an is_voice flag from Stage 1
  - When stride is complete, if ≥30% of frames were voiced → run ECAPA
  - Otherwise (silence / music / noise dominated) → skip ECAPA, fail-open
  - Preprocessing matches enroll/verify: pre-emphasis + RMS normalisation

Run standalone for latency benchmarking:
    cd nova/backend
    python pvad/speaker_gate.py audio.wav --driver_id driver1 [--l3_dir ...]
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Audio geometry
_SAMPLE_RATE    = 16_000
_WINDOW_SAMPLES = 16_000  # 1.0s voiced-only ring fed to ECAPA
_STRIDE_SAMPLES = 4_000   # 0.25s stride — one inference per stride
_STEP_SAMPLES   = 160     # 10ms — must match FireRedGate._STRIDE_SAMPLES
_VOICED_MIN     = 4_000   # 0.25s minimum voiced audio before running ECAPA

# Preprocessing — must match enroll.py / verify.py exactly
_PREEMPH_COEFF = 0.97
_RMS_TARGET    = 0.08
_RMS_FLOOR     = 1e-6

# Hysteresis defaults
_DEFAULT_THRESHOLD    = 0.35
_DEFAULT_HYSTERESIS   = 2
_VOICE_FRAC_MIN       = 0.30   # ≥30% voiced frames in stride required to run ECAPA


def _preprocess(audio: np.ndarray) -> np.ndarray:
    """Pre-emphasis + RMS normalisation — identical to stt_moonshine_worker."""
    out = np.empty_like(audio)
    out[0]  = audio[0]
    out[1:] = audio[1:] - _PREEMPH_COEFF * audio[:-1]
    rms = float(np.sqrt(np.mean(out ** 2)))
    if rms > _RMS_FLOOR:
        out = out * (_RMS_TARGET / rms)
    return out


@dataclass
class SpeakerFrame:
    """Result emitted once per 250ms ECAPA stride."""
    stride_idx:   int
    score:        float | None   # cosine similarity; None = skipped (silence/not-warmed)
    voice_frac:   float          # fraction of 10ms frames that were voiced
    gate_vote:    bool           # True = driver / silence (fail-open), False = stranger
    latency_ms:   float          # ECAPA inference wall-clock time (0 if skipped)


class SpeakerGate:
    """
    ECAPA-TDNN driver verification gate.

    Usage
    -----
    gate = SpeakerGate.load(l3_dir, driver_id="driver1")
    for chunk_f32, voice_flags in zip(audio_chunks, gate_flags):
        result = gate.feed(chunk_f32, voice_flags)
        if result is not None:
            if not result.gate_vote:
                block_barge_in()

    voice_flags: list[bool], one entry per 10ms step (from FireRedGate.feed()).
    The gate accumulates them in sync with its own 10ms step buffer.
    """

    @classmethod
    def load(
        cls,
        l3_dir:     str  | None = None,
        driver_id:  str         = "driver1",
        threshold:  float       = _DEFAULT_THRESHOLD,
        hysteresis: int         = _DEFAULT_HYSTERESIS,
        voice_frac_min: float   = _VOICE_FRAC_MIN,
        device:     str  | None = None,
    ) -> "SpeakerGate":
        if l3_dir is None:
            backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            l3_dir = os.path.join(backend_dir, "nova-l7", "L-3")
        sys.path.insert(0, l3_dir)

        # Load driver voiceprint
        driver_embedding: np.ndarray | None = None
        try:
            from crypto_utils import load_array  # type: ignore
            vp_path = Path(l3_dir) / "data" / "voiceprints" / f"{driver_id}.enc"
            if vp_path.exists():
                emb  = load_array(vp_path)
                norm = np.linalg.norm(emb)
                driver_embedding = emb / norm if norm > 1e-8 else emb
                logger.info(f"SpeakerGate: voiceprint loaded for '{driver_id}' dim={emb.shape}")
            else:
                logger.warning(f"SpeakerGate: no voiceprint at {vp_path} — fail-open mode")
        except Exception as e:
            logger.error(f"SpeakerGate: voiceprint load error: {e} — fail-open mode")

        # Load ECAPA-TDNN
        ecapa_model = None
        if driver_embedding is not None:
            try:
                # torchaudio compat (SpeechBrain expects list_audio_backends)
                import torchaudio as _ta
                if not hasattr(_ta, "list_audio_backends"):
                    _ta.list_audio_backends = lambda: ["ffmpeg"]

                # HF-hub compat monkeypatch (same as verify.py)
                import huggingface_hub as _hf
                _orig = _hf.hf_hub_download
                def _patched(*a, **kw):
                    if "use_auth_token" in kw:
                        kw["token"] = kw.pop("use_auth_token") or None
                    try:
                        return _orig(*a, **kw)
                    except Exception as ex:
                        fname  = a[1] if len(a) > 1 else kw.get("filename", "")
                        is_404 = ("404" in str(ex) or "Not Found" in str(ex) or
                                  "EntryNotFound" in type(ex).__name__ or
                                  "RemoteEntryNotFound" in type(ex).__name__)
                        if "custom.py" in str(fname) and is_404:
                            raise ValueError("File not found on HF hub") from ex
                        raise
                _hf.hf_hub_download = _patched

                import torch
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
                ecapa_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join(l3_dir, "pretrained_models_pvad", "spkrec-ecapa-voxceleb"),
                    run_opts={"device": device},
                )
                logger.info(f"SpeakerGate: ECAPA-TDNN loaded on {device.upper()}")
            except Exception as e:
                logger.error(f"SpeakerGate: ECAPA load error: {e} — fail-open mode")
                device = "cpu"

        return cls(
            ecapa_model=ecapa_model,
            driver_embedding=driver_embedding,
            threshold=threshold,
            hysteresis=hysteresis,
            voice_frac_min=voice_frac_min,
            device=device or "cpu",
        )

    # ------------------------------------------------------------------
    def __init__(
        self,
        ecapa_model,
        driver_embedding: np.ndarray | None,
        threshold:    float,
        hysteresis:   int,
        voice_frac_min: float,
        device:       str,
    ):
        self._ecapa        = ecapa_model
        self._driver_emb   = driver_embedding
        self.threshold     = threshold
        self.hysteresis    = hysteresis
        self.voice_frac_min = voice_frac_min
        self._device       = device

        # Voiced-only ring buffer for ECAPA (only voiced frames accumulated here)
        self._voiced_ring   = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
        self._voiced_filled = 0   # samples accumulated so far (capped at _WINDOW_SAMPLES)
        # Time-based stride accumulator (for gate timing, not ECAPA input)
        self._stride   = np.zeros(_STRIDE_SAMPLES, dtype=np.float32)
        self._stride_pos = 0

        # 10ms step accumulator (mirrors FireRedGate buffering)
        self._step_buf = np.zeros(_STEP_SAMPLES, dtype=np.float32)
        self._step_pos = 0

        # Voice fraction tracking per stride
        self._voice_steps = 0
        self._total_steps = 0

        # Warmup (need 2 strides = 500ms for full ring)
        self._warmed_up  = False
        self._stride_idx = 0

        # Hysteresis state
        self._last_gate       = True    # start open
        self._pending_gate    = True
        self._consecutive_same = 0

    def reset(self):
        self._voiced_ring[:]  = 0
        self._voiced_filled   = 0
        self._stride[:]       = 0
        self._stride_pos      = 0
        self._step_buf[:]     = 0
        self._step_pos        = 0
        self._voice_steps     = 0
        self._total_steps     = 0
        self._warmed_up       = False
        self._stride_idx      = 0
        self._last_gate       = True
        self._pending_gate    = True
        self._consecutive_same = 0

    def feed(self, audio_f32: np.ndarray, voice_flags: list[bool]) -> SpeakerFrame | None:
        """
        Process a variable-length float32 chunk and its per-10ms voice flags.

        voice_flags has one entry per completed 10ms step (from FireRedGate.feed()).
        Returns a SpeakerFrame each time the 250ms ECAPA stride completes, else None.
        """
        pos      = 0
        flag_idx = 0
        result   = None

        while pos < len(audio_f32):
            space = _STEP_SAMPLES - self._step_pos
            take  = min(space, len(audio_f32) - pos)
            self._step_buf[self._step_pos:self._step_pos + take] = audio_f32[pos:pos + take]
            self._step_pos += take
            pos            += take

            if self._step_pos < _STEP_SAMPLES:
                break  # incomplete 10ms step

            # 10ms step complete
            is_voice = voice_flags[flag_idx] if flag_idx < len(voice_flags) else True
            flag_idx += 1
            self._step_pos = 0

            # Accumulate into time-based stride (for gate output timing)
            ec_space = _STRIDE_SAMPLES - self._stride_pos
            take2    = min(ec_space, _STEP_SAMPLES)  # always _STEP_SAMPLES since _STRIDE is multiple
            self._stride[self._stride_pos:self._stride_pos + take2] = self._step_buf[:take2]
            self._stride_pos  += take2
            self._voice_steps += int(is_voice)
            self._total_steps += 1

            # Voiced-only ring: only accumulate frames where speech is detected
            if is_voice:
                self._voiced_ring = np.roll(self._voiced_ring, -_STEP_SAMPLES)
                self._voiced_ring[-_STEP_SAMPLES:] = self._step_buf
                self._voiced_filled = min(self._voiced_filled + _STEP_SAMPLES, _WINDOW_SAMPLES)

            if self._stride_pos < _STRIDE_SAMPLES:
                continue

            # ECAPA stride complete (250ms accumulated)
            result = self._on_stride()
            self._stride_pos  = 0
            self._voice_steps = 0
            self._total_steps = 0

        return result

    # ------------------------------------------------------------------
    def _on_stride(self) -> SpeakerFrame:
        voice_frac  = self._voice_steps / max(1, self._total_steps)
        stride_idx  = self._stride_idx
        self._stride_idx += 1

        # Warmup: need 2 strides before emitting scored results
        if not self._warmed_up:
            self._warmed_up = (self._stride_idx >= 2)
            return SpeakerFrame(stride_idx, None, voice_frac, True, 0.0)

        # Fail-open if no model or voiceprint
        if self._ecapa is None or self._driver_emb is None:
            return SpeakerFrame(stride_idx, None, voice_frac, True, 0.0)

        # Skip ECAPA if too little voiced audio (silence / noise dominated stride,
        # or not enough voiced audio accumulated yet for a stable embedding)
        if voice_frac < self.voice_frac_min or self._voiced_filled < _VOICED_MIN:
            logger.debug(f"SpeakerGate stride {stride_idx}: skip ECAPA "
                         f"(voice_frac={voice_frac:.2f}, voiced_filled={self._voiced_filled})")
            gate_vote = True  # non-voice → fail-open
            score = None
            latency_ms = 0.0
        else:
            # Use voiced-only ring: ECAPA sees pure speech, matching enroll.py
            score, latency_ms = self._infer(self._voiced_ring.copy())
            gate_vote = score >= self.threshold
            logger.debug(f"SpeakerGate stride {stride_idx}: "
                         f"score={score:.3f} voice_frac={voice_frac:.2f} pass={gate_vote}")

        # Hysteresis
        if gate_vote == self._pending_gate:
            self._consecutive_same += 1
        else:
            self._pending_gate     = gate_vote
            self._consecutive_same = 1

        emit_change = (self._consecutive_same >= self.hysteresis
                       and gate_vote != self._last_gate)
        if emit_change:
            self._last_gate = gate_vote
            logger.info(
                f"SpeakerGate gate {'OPEN' if gate_vote else 'CLOSED'} "
                f"(score={score:.3f} voice_frac={voice_frac:.2f})"
                if score is not None else
                f"SpeakerGate gate {'OPEN' if gate_vote else 'CLOSED'} (silence)"
            )

        return SpeakerFrame(stride_idx, score, voice_frac, self._last_gate, latency_ms)

    def _infer(self, window_f32: np.ndarray) -> tuple[float, float]:
        import torch
        t0        = time.perf_counter()
        processed = _preprocess(window_f32)
        waveform  = torch.from_numpy(processed).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._ecapa.encode_batch(waveform).squeeze().cpu().numpy()
        norm  = np.linalg.norm(emb)
        assert self._driver_emb is not None  # guaranteed by caller
        score = float(np.dot(emb / norm, self._driver_emb)) if norm > 1e-8 else 0.0
        return score, (time.perf_counter() - t0) * 1000.0

    @property
    def last_gate(self) -> bool:
        return self._last_gate


# ── Standalone benchmark / smoke-test ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, statistics, soundfile as sf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pvad.firered_gate import FireRedGate  # type: ignore

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="SpeakerGate standalone benchmark")
    ap.add_argument("wav",         help="WAV file to process (16kHz mono)")
    ap.add_argument("--driver_id", default="driver1")
    ap.add_argument("--l3_dir",    default=None)
    ap.add_argument("--threshold", type=float, default=0.58)
    ap.add_argument("--chunk_ms",  type=int,   default=20)
    ap.add_argument("--no_firered", action="store_true",
                    help="Disable Stage-1; use RMS fallback for voice_flags")
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="int16")
    assert sr == 16000
    audio_f32    = audio.astype(np.float32) / 32768.0
    chunk_samples = int(sr * args.chunk_ms / 1000)

    # Stage 1
    firered_gate = None
    if not args.no_firered:
        try:
            firered_gate = FireRedGate.load()
        except Exception as e:
            print(f"[warn] FireRedGate load failed: {e} — using RMS fallback")

    # Stage 2
    speaker_gate = SpeakerGate.load(l3_dir=args.l3_dir, driver_id=args.driver_id,
                                     threshold=args.threshold)

    latencies, results = [], []
    rms_threshold = 0.035

    for start in range(0, len(audio), chunk_samples):
        chunk_i16 = audio[start:start + chunk_samples]
        chunk_f32 = audio_f32[start:start + chunk_samples]

        if firered_gate:
            frames = firered_gate.feed(chunk_i16)
            voice_flags = [f.is_voice for f in frames]
        else:
            # Simple RMS gate per 160-sample step for fallback
            voice_flags = []
            for i in range(0, len(chunk_i16), 160):
                step = chunk_f32[i:i+160]
                if len(step) == 160:
                    rms = float(np.sqrt(np.mean(step**2)))
                    voice_flags.append(rms >= rms_threshold)

        frame = speaker_gate.feed(chunk_f32, voice_flags)
        if frame is not None and frame.score is not None:
            results.append(frame)
            latencies.append(frame.latency_ms)
            print(f"stride {frame.stride_idx:4d}  "
                  f"score={frame.score:.3f}  "
                  f"voice_frac={frame.voice_frac:.2f}  "
                  f"gate={'OPEN' if frame.gate_vote else 'CLOSED'}  "
                  f"latency={frame.latency_ms:.1f}ms")

    print(f"\n{'='*60}")
    if latencies:
        print(f"ECAPA strides    : {len(latencies)}")
        print(f"Latency mean     : {statistics.mean(latencies):.1f}ms")
        print(f"Latency p50      : {statistics.median(latencies):.1f}ms")
        print(f"Latency p95      : {sorted(latencies)[int(0.95*len(latencies))]:.1f}ms")
        print(f"Latency max      : {max(latencies):.1f}ms")
        scores = [r.score for r in results]
        print(f"Score mean/min/max: {np.mean(scores):.3f} / {min(scores):.3f} / {max(scores):.3f}")
    else:
        print("No ECAPA strides completed (audio too short or all silence?)")
    print(f"{'='*60}\n")
