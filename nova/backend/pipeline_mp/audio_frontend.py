"""Unified mic front-end for Nova: WebRTC APM (AEC3 + NS + AGC + HPF) + Silero VAD.

Replaces the narrower `aec_barge_in.py` (kept for one release as a thin shim).

Pipeline (always-on, single instance per process):

    mic_16k ─────────┐
                     ├─▶ livekit APM ─▶ cleaned_16k ─▶ Silero VAD ─▶ speech events
    tts_ref_16k ─────┘                                                │
                                                                      ▼
                                              barge-in / start_of_speech / end_of_speech

Two-streak hysteresis state machine on top of Silero:
    - in_speech=False → after N consecutive voice frames emit "start" + flip in_speech=True
    - in_speech=True  → after M consecutive silence frames emit "end"  + flip in_speech=False

`process_mic()` now returns (barge_in, vad_prob, speech_event, cleaned_i16):
    barge_in:       True only on the rising edge of "start" (same legacy semantics)
    vad_prob:       last Silero probability seen in this call
    speech_event:   None | "start" | "end"   (one event max per call)
    cleaned_i16:    APM-processed mic bytes (caller should forward to STT)

`is_enabled()` controls whether the heavy APM/VAD path runs at all. When disabled,
process_mic returns a passthrough of the raw mic bytes so callers can hand the
return value to STT unconditionally.

Env vars:
    NOVA_FRONTEND_ENABLED               1 to enable (default: inherits NOVA_AEC_ENABLED)
    NOVA_FRONTEND_VAD_THRESHOLD         Silero prob above which a frame counts as voice (default 0.6)
    NOVA_FRONTEND_START_FRAMES          Voice frames required to emit "start" (default 3)
    NOVA_FRONTEND_END_FRAMES            Silence frames required to emit "end" (default 12 → ~384 ms)
    NOVA_FRONTEND_REF_BUFFER_MS         TTS reference ring buffer size (default 500)

Legacy (still honored, mapped onto the above for one release):
    NOVA_AEC_ENABLED                  → NOVA_FRONTEND_ENABLED
    NOVA_AEC_VAD_THRESHOLD            → NOVA_FRONTEND_VAD_THRESHOLD
    NOVA_AEC_CONSECUTIVE_FRAMES       → NOVA_FRONTEND_START_FRAMES
    NOVA_AEC_REF_BUFFER_MS            → NOVA_FRONTEND_REF_BUFFER_MS
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

logger = logging.getLogger("AudioFrontend")

_SR = 16000
_FRAME_SAMPLES = 160          # livekit APM: 10 ms @ 16 kHz
_VAD_WINDOW = 512             # Silero v5: fixed 512 samples @ 16 kHz (~32 ms)


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def is_enabled() -> bool:
    # Front-end enabled if either the new flag OR the legacy flag says so.
    return _truthy(os.getenv("NOVA_FRONTEND_ENABLED")) or _truthy(os.getenv("NOVA_AEC_ENABLED"))


def _fenv(*names: str, default: float) -> float:
    for n in names:
        v = os.getenv(n)
        if v is not None:
            try:
                return float(v)
            except ValueError:
                pass
    return default


def _ienv(*names: str, default: int) -> int:
    for n in names:
        v = os.getenv(n)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return default


class AudioFrontend:
    """Process-local singleton owning APM + VAD + speech-event state machine."""

    _instance: "Optional[AudioFrontend]" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "AudioFrontend":
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

        # TTS reference ring buffer (16 kHz int16). Mic chunks pop matched lengths.
        self._ref_buf: deque = deque()
        max_ref_ms = _ienv("NOVA_FRONTEND_REF_BUFFER_MS", "NOVA_AEC_REF_BUFFER_MS", default=500)
        self._max_ref_samples = max(_FRAME_SAMPLES, int(max_ref_ms * _SR / 1000))

        # Cleaned-mic accumulator for fixed-size VAD windows.
        self._vad_buf = np.zeros(0, dtype=np.float32)

        # State machine
        self._threshold = _fenv("NOVA_FRONTEND_VAD_THRESHOLD", "NOVA_AEC_VAD_THRESHOLD", default=0.5)
        self._start_frames = _ienv(
            "NOVA_FRONTEND_START_FRAMES", "NOVA_AEC_CONSECUTIVE_FRAMES", default=3
        )
        self._end_frames = _ienv("NOVA_FRONTEND_END_FRAMES", default=24)  # ~24 * 32 ms ≈ 770 ms
        # When TTS pushed reference samples within this many seconds, we use the
        # AEC-cleaned signal for VAD (echo-resistant barge-in). Otherwise VAD
        # runs on the raw mic so quiet listening isn't attenuated by AEC.
        self._ref_recent_window_s = _fenv("NOVA_FRONTEND_REF_RECENT_S", default=1.0)
        self._last_ref_push_at = 0.0
        self._voice_streak = 0
        self._silence_streak = 0
        self._in_speech = False

    # -- loading ----------------------------------------------------------------
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
                # AGC OFF: we'd rather VAD see raw dynamic range during quiet
                # listening than have AGC compress speech onset down toward
                # background noise. AEC + NS + HPF are still on.
                self._apm = AudioProcessingModule(
                    echo_cancellation=True,
                    noise_suppression=True,
                    auto_gain_control=False,
                    high_pass_filter=True,
                )
                self._vad = load_silero_vad(onnx=True)
                self._torch = torch
                logger.info(
                    f"AudioFrontend loaded "
                    f"(threshold={self._threshold:.2f}, "
                    f"start={self._start_frames}f, end={self._end_frames}f, "
                    f"ref_buffer={self._max_ref_samples} samples)"
                )
                return True
            except Exception as e:
                logger.warning(f"AudioFrontend load failed — disabling: {e}")
                self._load_failed = True
                return False

    # -- TTS reference ----------------------------------------------------------
    @staticmethod
    def _resample_24k_to_16k(audio_i16: np.ndarray) -> np.ndarray:
        if audio_i16.size == 0:
            return audio_i16
        n_in = audio_i16.size
        n_out = int(round(n_in * 16000 / 24000))
        if n_out <= 0:
            return np.zeros(0, dtype=np.int16)
        x_in = np.linspace(0, n_in - 1, n_in, dtype=np.float32)
        x_out = np.linspace(0, n_in - 1, n_out, dtype=np.float32)
        return np.interp(x_out, x_in, audio_i16.astype(np.float32)).astype(np.int16)

    def push_reference(self, audio_24k_i16: np.ndarray) -> None:
        if not is_enabled() or not self._load():
            return
        try:
            audio_16k = self._resample_24k_to_16k(audio_24k_i16)
            self._ref_buf.extend(audio_16k.tolist())
            # Stamp wall-clock so process_mic knows TTS was playing recently.
            import time as _time
            self._last_ref_push_at = _time.time()
            overflow = len(self._ref_buf) - self._max_ref_samples
            for _ in range(max(0, overflow)):
                self._ref_buf.popleft()
        except Exception as e:
            logger.debug(f"push_reference failed: {e}")

    def reset(self) -> None:
        """Clear all per-conversation state (between turns / sessions)."""
        self._ref_buf.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._voice_streak = 0
        self._silence_streak = 0
        self._in_speech = False

    def clear_speech_state(self) -> None:
        """Reset only the speech-event hysteresis (NOT the APM / ref buffer).

        Use when entering a phase where we want a fresh rising edge — e.g.
        on TTS start, so the first real barge-in after AEC converges fires
        a clean `start` event regardless of any echo-induced state from the
        guard window.
        """
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._voice_streak = 0
        self._silence_streak = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    # -- mic processing ---------------------------------------------------------
    def process_mic(
        self, mic_16k_i16: np.ndarray
    ) -> tuple[bool, float, Optional[str], np.ndarray]:
        """Run a mic chunk through APM + VAD + speech-event hysteresis.

        Returns (barge_in, last_vad_prob, speech_event, cleaned_i16):
            speech_event in {None, "start", "end"} — at most one per call
            barge_in == True iff this call produced "start"
            cleaned_i16: APM-processed mic in int16, same length as input;
                         returned even when disabled (passthrough = original)
        """
        if mic_16k_i16.size == 0:
            return False, 0.0, None, mic_16k_i16
        if not is_enabled() or not self._load():
            return False, 0.0, None, mic_16k_i16
        try:
            assert self._apm is not None and self._AudioFrame is not None
            mic = np.ascontiguousarray(mic_16k_i16, dtype=np.int16)

            cleaned_i16_chunks: list[np.ndarray] = []
            for i in range(0, len(mic), _FRAME_SAMPLES):
                mic_frame_samples = mic[i:i + _FRAME_SAMPLES]
                pad = _FRAME_SAMPLES - len(mic_frame_samples)
                if pad > 0:
                    mic_frame_samples = np.pad(mic_frame_samples, (0, pad))

                ref_samples = np.zeros(_FRAME_SAMPLES, dtype=np.int16)
                for j in range(_FRAME_SAMPLES):
                    if not self._ref_buf:
                        break
                    ref_samples[j] = self._ref_buf.popleft()

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
                cleaned_i16_chunks.append(np.frombuffer(bytes(mic_frame.data), dtype=np.int16))

            if not cleaned_i16_chunks:
                return False, 0.0, None, mic_16k_i16
            # Trim trailing pad so cleaned length matches input length.
            cleaned_full = np.concatenate(cleaned_i16_chunks)
            cleaned_full = cleaned_full[: len(mic_16k_i16)]
            cleaned_f32 = cleaned_full.astype(np.float32) / 32768.0

            # VAD source: cleaned audio only when TTS played recently
            # (echo-resistant barge-in); raw audio otherwise so quiet listening
            # isn't attenuated by AEC adapting to a silent reference.
            import time as _time
            tts_recent = (_time.time() - self._last_ref_push_at) < self._ref_recent_window_s
            if tts_recent:
                vad_input_f32 = cleaned_f32
            else:
                raw_f32 = mic.astype(np.float32) / 32768.0
                vad_input_f32 = raw_f32[: cleaned_f32.size]
            self._vad_buf = np.concatenate([self._vad_buf, vad_input_f32])
            last_prob = 0.0
            event: Optional[str] = None
            barge_in = False
            while self._vad_buf.size >= _VAD_WINDOW:
                window = self._vad_buf[:_VAD_WINDOW]
                self._vad_buf = self._vad_buf[_VAD_WINDOW:]
                last_prob = self._vad_prob(window)
                if last_prob > self._threshold:
                    self._voice_streak += 1
                    self._silence_streak = 0
                    if (not self._in_speech) and self._voice_streak >= self._start_frames:
                        self._in_speech = True
                        self._voice_streak = 0
                        if event is None:
                            event = "start"
                            barge_in = True
                else:
                    self._silence_streak += 1
                    self._voice_streak = 0
                    if self._in_speech and self._silence_streak >= self._end_frames:
                        self._in_speech = False
                        self._silence_streak = 0
                        if event is None:
                            event = "end"
                        # don't break — keep draining buffer
            return barge_in, last_prob, event, cleaned_full
        except Exception as e:
            logger.debug(f"process_mic failed: {e}")
            return False, 0.0, None, mic_16k_i16

    def _vad_prob(self, window_f32: np.ndarray) -> float:
        try:
            assert self._vad is not None and self._torch is not None
            t = self._torch.from_numpy(window_f32)
            p = self._vad(t, _SR)
            return float(p.item()) if hasattr(p, "item") else float(p)
        except Exception:
            return 0.0


