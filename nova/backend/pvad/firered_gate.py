"""
firered_gate.py — Stage 1: Generic Voice Activity Gate

Wraps FireRedStreamVad (DFSMN, 0.6M params) to classify each 10ms audio
frame as voice vs silence / music / noise.

Design:
  - Feed arbitrary-length int16 chunks via .feed()
  - Internally buffers to 160-sample (10ms) steps
  - Maintains a 400-sample (25ms) rolling window fed to detect_frame()
  - Returns one bool per completed 10ms step

Run standalone for benchmarking / smoke-test:
    cd nova/backend
    python pvad/firered_gate.py path/to/audio.wav [--model_dir ...]
"""

import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

# FireRedVAD frame geometry (fixed by the model)
_WINDOW_SAMPLES = 400   # 25ms at 16kHz — input to detect_frame()
_STRIDE_SAMPLES = 160   # 10ms at 16kHz — step between calls


def _add_fireredasr2s_to_path(fireredasr2s_dir: str | None = None) -> str:
    """
    Resolve FireRedASR2S repo root, add the inner Python package dir to sys.path.
    Returns the repo root (not the package subdir).

    Repo layout:
        FireRedASR2S/               ← fireredasr2s_dir (repo root)
        └── fireredasr2s/           ← Python package root (added to sys.path)
            └── fireredvad/         ← importable as `import fireredvad`
    """
    if fireredasr2s_dir is None:
        # Default: sibling of nova_ai root → <workspace>/FireRedASR2S
        backend_dir = os.path.abspath(os.path.join(_HERE, ".."))
        repo_root   = os.path.abspath(os.path.join(backend_dir, "..", ".."))
        fireredasr2s_dir = os.getenv(
            "NOVA_FIREREDASR2S_DIR",
            os.path.join(repo_root, "FireRedASR2S"),
        )
    # The importable packages live one level deeper inside fireredasr2s/
    pkg_dir = os.path.join(fireredasr2s_dir, "fireredasr2s")
    if os.path.isdir(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    return fireredasr2s_dir


@dataclass
class GateFrame:
    """Result for a single 10ms step."""
    step_idx:   int     # 0-based, increments every 10ms
    is_voice:   bool
    raw_prob:   float   # sigmoid output from DFSMN
    latency_ms: float   # wall-clock inference time for this step


class FireRedGate:
    """
    Streaming FireRedVAD wrapper. Feed arbitrary int16 chunks; get per-10ms voice flags.

    Usage
    -----
    gate = FireRedGate.load(model_dir)
    for raw_bytes in mic_stream:
        chunk = np.frombuffer(raw_bytes, dtype=np.int16)
        for frame in gate.feed(chunk):
            if frame.is_voice:
                ...  # genuine voice activity

    Thread / process safety: NOT thread-safe. Use one instance per process.
    """

    @classmethod
    def load(
        cls,
        model_dir:        str | None = None,
        speech_threshold: float = 0.4,   # permissive — let Stage 2 do speaker check
        smooth_window:    int   = 3,      # 3 × 10ms = 30ms smoothing
        min_speech_frame: int   = 4,      # 40ms onset — sensitive for barge-in
        min_silence_frame: int  = 15,     # 150ms hangover before closing
        use_gpu:          bool  = False,
        fireredasr2s_dir: str | None = None,
    ) -> "FireRedGate":
        root = _add_fireredasr2s_to_path(fireredasr2s_dir)

        if model_dir is None:
            model_dir = os.getenv("NOVA_FIREREDVAD_MODEL_DIR", None)
        if model_dir is None:
            # Look next to this file first (nova/backend/pvad/pretrained_models_pvad/...)
            _local = os.path.join(_HERE, "pretrained_models_pvad", "FireRedVAD", "Stream-VAD")
            _remote = os.path.join(root, "pretrained_models_pvad", "FireRedVAD", "Stream-VAD")
            model_dir = _local if os.path.isdir(_local) else _remote

        from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig  # type: ignore

        cfg = FireRedStreamVadConfig(
            use_gpu=use_gpu,
            speech_threshold=speech_threshold,
            smooth_window_size=smooth_window,
            min_speech_frame=min_speech_frame,
            min_silence_frame=min_silence_frame,
        )
        vad = FireRedStreamVad.from_pretrained(model_dir, cfg)
        logger.info(f"FireRedGate: loaded from {model_dir}")
        return cls(vad)

    # ------------------------------------------------------------------
    def __init__(self, stream_vad):
        self._vad = stream_vad

        # Rolling 25ms window fed to detect_frame()
        self._window     = np.zeros(_WINDOW_SAMPLES, dtype=np.int16)
        # Accumulator for next 10ms step
        self._stride_buf = np.zeros(_STRIDE_SAMPLES, dtype=np.int16)
        self._stride_pos = 0
        # Warmup: need at least one full window before results are meaningful
        self._samples_seen = 0
        self._step_idx     = 0

    def reset(self):
        """Clear all buffers and FSMN cache. Call between sessions."""
        self._window[:] = 0
        self._stride_buf[:] = 0
        self._stride_pos = 0
        self._samples_seen = 0
        self._step_idx = 0
        self._vad.reset()

    def feed(self, audio_int16: np.ndarray) -> list[GateFrame]:
        """
        Process a variable-length int16 chunk (16kHz, mono).
        Returns one GateFrame per completed 10ms step.
        Incomplete steps are buffered until the next call.
        """
        frames: list[GateFrame] = []
        pos = 0

        while pos < len(audio_int16):
            space = _STRIDE_SAMPLES - self._stride_pos
            take  = min(space, len(audio_int16) - pos)
            self._stride_buf[self._stride_pos:self._stride_pos + take] = audio_int16[pos:pos + take]
            self._stride_pos += take
            pos              += take

            if self._stride_pos < _STRIDE_SAMPLES:
                break   # incomplete step — wait for more audio

            # Full 10ms step: roll window and call detect_frame
            self._window      = np.roll(self._window, -_STRIDE_SAMPLES)
            self._window[-_STRIDE_SAMPLES:] = self._stride_buf
            self._stride_pos  = 0
            self._samples_seen += _STRIDE_SAMPLES

            if self._samples_seen < _WINDOW_SAMPLES:
                # Warmup: rolling window not yet full, return fail-open
                frames.append(GateFrame(self._step_idx, True, 1.0, 0.0))
                self._step_idx += 1
                continue

            t0 = time.perf_counter()
            result = self._vad.detect_frame(self._window.copy())
            latency_ms = (time.perf_counter() - t0) * 1000.0

            frames.append(GateFrame(
                step_idx   = self._step_idx,
                is_voice   = result.is_speech,
                raw_prob   = result.smoothed_prob,
                latency_ms = latency_ms,
            ))
            self._step_idx += 1

        return frames

    @property
    def step_duration_ms(self) -> float:
        return _STRIDE_SAMPLES / 16.0   # = 10ms


# ── Standalone benchmark / smoke-test ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import statistics

    import soundfile as sf

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="FireRedGate standalone benchmark")
    ap.add_argument("wav",          help="WAV file to process (16kHz mono int16)")
    ap.add_argument("--model_dir",  default=None)
    ap.add_argument("--threshold",  type=float, default=0.4)
    ap.add_argument("--chunk_ms",   type=int,   default=20,
                    help="Simulated mic chunk size in ms (default 20ms = 320 samples)")
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="int16")
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    chunk_samples = int(sr * args.chunk_ms / 1000)

    gate = FireRedGate.load(model_dir=args.model_dir, speech_threshold=args.threshold)

    latencies, voice_count, total_frames = [], 0, 0
    speech_segments = []
    in_speech, seg_start = False, 0

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        for frame in gate.feed(chunk):
            total_frames += 1
            if frame.latency_ms > 0:
                latencies.append(frame.latency_ms)
            if frame.is_voice:
                voice_count += 1
            # Track speech segments
            t_ms = frame.step_idx * 10
            if frame.is_voice and not in_speech:
                in_speech  = True
                seg_start  = t_ms
            elif not frame.is_voice and in_speech:
                in_speech  = False
                speech_segments.append((seg_start / 1000, t_ms / 1000))

    if in_speech:
        speech_segments.append((seg_start / 1000, total_frames * 10 / 1000))

    dur_s = len(audio) / sr
    print(f"\n{'='*60}")
    print(f"File       : {args.wav}  ({dur_s:.2f}s)")
    print(f"Frames     : {total_frames}  ({total_frames * 10}ms total)")
    print(f"Voice      : {voice_count} frames ({100*voice_count/max(1,total_frames):.1f}%)")
    print(f"Segments   : {speech_segments}")
    if latencies:
        print("\nInference latency (per 10ms step):")
        print(f"  mean  : {statistics.mean(latencies):.2f}ms")
        print(f"  p50   : {statistics.median(latencies):.2f}ms")
        print(f"  p95   : {sorted(latencies)[int(0.95*len(latencies))]:.2f}ms")
        print(f"  max   : {max(latencies):.2f}ms")
    print(f"{'='*60}\n")
