"""WebRTC AEC3 + Silero VAD barge-in detector.

Detects when the user starts speaking during Nova's TTS playback — the classic
full-duplex voice-agent "interrupt me" UX. Pipeline:

    mic_16k ─────────┐
                     ├─▶ livekit APM (AEC3) ─▶ cleaned_16k ─▶ Silero VAD
    tts_ref_16k ─────┘                                           │
                                                                 ▼
                                              consecutive voice frames ≥ N?
                                                          → True

Fail-open: load/inference errors return (False, 0.0), so a broken detector
simply disables barge-in without affecting the rest of the pipeline.

Env vars:
    NOVA_AEC_ENABLED             1 to enable (default 0)
    NOVA_AEC_VAD_THRESHOLD       Silero prob above which a frame counts (default 0.8)
    NOVA_AEC_CONSECUTIVE_FRAMES  Frames required to trigger barge-in (default 3)
    NOVA_AEC_REF_BUFFER_MS       TTS reference ring buffer size (default 500)
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from livekit.rtc.apm import AudioProcessingModule

logger = logging.getLogger("AECBargeIn")

_SR = 16000
_FRAME_SAMPLES = 160           # livekit APM hard requirement: 10 ms @ 16 kHz
_VAD_WINDOW = 512              # Silero v5 hard requirement: 512 samples @ 16 kHz


def is_enabled() -> bool:
    return os.getenv("NOVA_AEC_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _fenv(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _ienv(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class AECBargeInDetector:
    """Process-local singleton — each worker / main process gets its own APM."""

    _instance: "Optional[AECBargeInDetector]" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "AECBargeInDetector":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._apm: "Optional[AudioProcessingModule]" = None
        self._vad = None
        self._torch = None
        self._AudioFrame = None
        self._load_failed = False
        self._load_lock = threading.Lock()

        # Ring buffer of TTS reference samples at 16 kHz (int16). Mic chunks pop
        # a length-matched prefix; zero-pad if reference runs dry (no TTS audio).
        self._ref_buf: deque = deque()
        max_ref_ms = _ienv("NOVA_AEC_REF_BUFFER_MS", 500)
        self._max_ref_samples = max(_FRAME_SAMPLES, int(max_ref_ms * _SR / 1000))

        # Cleaned mic accumulator — Silero VAD consumes fixed 512-sample windows.
        self._vad_buf = np.zeros(0, dtype=np.float32)

        # Barge-in hysteresis: count consecutive voice frames before triggering.
        self._voice_frame_count = 0
        self._threshold = _fenv("NOVA_AEC_VAD_THRESHOLD", 0.8)
        self._consecutive = _ienv("NOVA_AEC_CONSECUTIVE_FRAMES", 3)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        if self._apm is not None and self._vad is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._apm is not None and self._vad is not None:
                return True
            if self._load_failed:
                return False
            try:
                from livekit.rtc import AudioFrame  # type: ignore
                from livekit.rtc.apm import AudioProcessingModule  # type: ignore
                from silero_vad import load_silero_vad  # type: ignore
                import torch  # type: ignore

                self._AudioFrame = AudioFrame
                self._apm = AudioProcessingModule(
                    echo_cancellation=True,
                    noise_suppression=True,
                )
                self._vad = load_silero_vad(onnx=True)
                self._torch = torch
                logger.info(
                    f"AEC barge-in loaded (threshold={self._threshold:.2f}, "
                    f"consecutive={self._consecutive}, ref_buffer={self._max_ref_samples} samples)"
                )
                return True
            except Exception as e:
                logger.warning(f"AEC barge-in load failed — disabling: {e}")
                self._load_failed = True
                return False

    # ------------------------------------------------------------------
    # Reference (TTS output) ring buffer
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_24k_to_16k(audio_i16: np.ndarray) -> np.ndarray:
        """Downsample 24 kHz int16 → 16 kHz int16 via 2:3 linear interpolation.

        Quality is not critical: this is ONLY used as the AEC reference signal,
        and AEC3's adaptive delay estimator is tolerant of minor artifacts.
        """
        if audio_i16.size == 0:
            return audio_i16
        n_in = audio_i16.size
        n_out = int(round(n_in * 16000 / 24000))
        if n_out <= 0:
            return np.zeros(0, dtype=np.int16)
        x_in = np.linspace(0, n_in - 1, n_in, dtype=np.float32)
        x_out = np.linspace(0, n_in - 1, n_out, dtype=np.float32)
        resampled = np.interp(x_out, x_in, audio_i16.astype(np.float32))
        return resampled.astype(np.int16)

    def push_reference(self, audio_24k_i16: np.ndarray) -> None:
        """Feed a chunk of TTS audio (24 kHz int16) into the reference buffer."""
        if not is_enabled():
            return
        if not self._load():
            return
        try:
            audio_16k = self._resample_24k_to_16k(audio_24k_i16)
            self._ref_buf.extend(audio_16k.tolist())
            # Trim buffer to max size (drop oldest — reference should mirror
            # what the speaker is playing NOW, not minutes ago).
            overflow = len(self._ref_buf) - self._max_ref_samples
            for _ in range(max(0, overflow)):
                self._ref_buf.popleft()
        except Exception as e:
            logger.debug(f"push_reference failed: {e}")

    def reset(self) -> None:
        """Clear reference buffer + hysteresis counter between conversations."""
        self._ref_buf.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._voice_frame_count = 0

    # ------------------------------------------------------------------
    # Mic processing (returns True on barge-in detection)
    # ------------------------------------------------------------------

    def process_mic(self, mic_16k_i16: np.ndarray) -> tuple[bool, float]:
        """Process a mic chunk. Returns (barge_in_detected, last_vad_prob).

        Barge-in is triggered when `NOVA_AEC_CONSECUTIVE_FRAMES` consecutive
        Silero-VAD windows on the cleaned mic exceed `NOVA_AEC_VAD_THRESHOLD`.
        """
        if not is_enabled():
            return False, 0.0
        if not self._load():
            return False, 0.0
        if mic_16k_i16.size == 0:
            return False, 0.0
        try:
            assert self._apm is not None and self._AudioFrame is not None
            # Ensure int16 contiguous
            mic = np.ascontiguousarray(mic_16k_i16, dtype=np.int16)

            # ── 1. Process mic in 10 ms frames through AEC, paired with ref ─
            cleaned_f32_chunks: list[np.ndarray] = []
            for i in range(0, len(mic), _FRAME_SAMPLES):
                mic_frame_samples = mic[i:i + _FRAME_SAMPLES]
                if len(mic_frame_samples) < _FRAME_SAMPLES:
                    mic_frame_samples = np.pad(
                        mic_frame_samples, (0, _FRAME_SAMPLES - len(mic_frame_samples))
                    )

                # Pop matching reference length; pad with zeros if buffer dry
                ref_samples = np.zeros(_FRAME_SAMPLES, dtype=np.int16)
                for j in range(_FRAME_SAMPLES):
                    if self._ref_buf:
                        ref_samples[j] = self._ref_buf.popleft()
                    else:
                        break  # rest stays zero

                ref_frame = self._AudioFrame(
                    ref_samples.tobytes(),
                    sample_rate=_SR, num_channels=1, samples_per_channel=_FRAME_SAMPLES,
                )
                mic_frame = self._AudioFrame(
                    mic_frame_samples.tobytes(),
                    sample_rate=_SR, num_channels=1, samples_per_channel=_FRAME_SAMPLES,
                )
                self._apm.process_reverse_stream(ref_frame)
                self._apm.process_stream(mic_frame)
                cleaned_i16 = np.frombuffer(bytes(mic_frame.data), dtype=np.int16)
                cleaned_f32_chunks.append(cleaned_i16.astype(np.float32) / 32768.0)

            if not cleaned_f32_chunks:
                return False, 0.0
            cleaned = np.concatenate(cleaned_f32_chunks)

            # ── 2. Accumulate cleaned audio, run Silero VAD on 512-sample windows
            self._vad_buf = np.concatenate([self._vad_buf, cleaned])
            last_prob = 0.0
            barge_in = False
            while self._vad_buf.size >= _VAD_WINDOW:
                window = self._vad_buf[:_VAD_WINDOW]
                self._vad_buf = self._vad_buf[_VAD_WINDOW:]
                last_prob = self._vad_prob(window)
                if last_prob > self._threshold:
                    self._voice_frame_count += 1
                else:
                    self._voice_frame_count = 0
                if self._voice_frame_count >= self._consecutive:
                    self._voice_frame_count = 0  # reset so re-fires require fresh streak
                    barge_in = True
                    # Keep draining so vad_buf doesn't grow unbounded, but
                    # don't report multiple barge-ins in one call.
            return barge_in, last_prob
        except Exception as e:
            logger.debug(f"process_mic failed: {e}")
            return False, 0.0

    def _vad_prob(self, window_f32: np.ndarray) -> float:
        try:
            assert self._vad is not None and self._torch is not None
            t = self._torch.from_numpy(window_f32)
            p = self._vad(t, _SR)
            return float(p.item()) if hasattr(p, "item") else float(p)
        except Exception:
            return 0.0
