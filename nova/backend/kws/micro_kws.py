"""
MicroKWS — Lightweight wake-word engine using micro-wake-word TFLite models.

Drop-in alternative to StreamingKWSv2.  Uses a pre-trained, streaming-quantised
TFLite model (~50-200 KB) for inference.  No enrollment needed — the model is
trained offline via the micro-wake-word training pipeline.

Runtime dependencies (CPU-only):
    pymicro-features   — C spectrogram preprocessor
    ai-edge-litert     — TFLite interpreter
"""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("MicroKWS")

# Default model location (relative to this file)
_DEFAULT_MODEL = Path(__file__).parent / "models" / "micro_nova.tflite"

# MicroFrontend produces one 40-feature frame per 160 samples (10 ms @ 16 kHz).
_FRAME_BYTES = 160 * 2  # 320 bytes per frame (int16)
_NUM_FEATURES = 40


class MicroKWS:
    """Streaming wake-word detector backed by a micro-wake-word TFLite model.

    Interface mirrors ``StreamingKWSv2`` so it can be used in ``kws_worker.py``
    without any changes to the audio processing loop.

    Parameters
    ----------
    model_path : str or None
        Path to a streaming-quantised ``.tflite`` model file.
        Defaults to ``kws/models/micro_nova.tflite``.
    threshold : float
        Per-window probability above which a frame counts as positive.
    consecutive_triggers : int
        How many consecutive positive windows are needed to fire.
    cooldown_sec : float
        Minimum gap (seconds) between two detections.
    debug : bool
        Log near-miss scores for tuning.
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        consecutive_triggers: int = 2,
        cooldown_sec: float = 2.0,
        debug: bool = True,
    ):
        self._model_path = model_path or str(_DEFAULT_MODEL)
        self.threshold = threshold
        self.consecutive_triggers = consecutive_triggers
        self.cooldown_sec = cooldown_sec
        self.debug = debug

        # Loaded by load_model()
        self._model = None          # microwakeword.inference.Model
        self._stride: int = 1       # spectrogram frames per invoke
        self._input_slices: int = 1 # total spectrogram frames the model expects

        # Streaming state
        self._muted = False
        self._last_trigger_time: float = 0.0
        self._consecutive_count: int = 0

        # Audio / spectrogram buffers
        self._audio_remainder = b""
        self._micro_frontend = None  # pymicro_features.MicroFrontend
        self._spec_buffer: np.ndarray | None = None  # (input_slices, 40)
        self._spec_count: int = 0          # frames written into buffer so far
        self._frames_since_inference: int = 0

    # ------------------------------------------------------------------
    # Public interface (matches StreamingKWSv2)
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | None = None) -> bool:
        """Load (or reload) a TFLite model.  Returns True on success."""
        path = model_path or self._model_path
        if not Path(path).is_file():
            logger.warning("MicroKWS model not found: %s", path)
            return False

        try:
            # Import lazily so missing deps surface only when engine is selected
            from microwakeword.inference import Model as MWWModel
            from pymicro_features import MicroFrontend

            self._model = MWWModel(path)
            self._stride = self._model.stride
            self._input_slices = self._model.input_feature_slices

            # Spectrogram sliding window
            self._spec_buffer = np.zeros(
                (self._input_slices, _NUM_FEATURES), dtype=np.float32
            )
            self._spec_count = 0
            self._frames_since_inference = 0
            self._audio_remainder = b""
            self._consecutive_count = 0

            # Persistent frontend — keeps PCAN / AGC state across calls
            self._micro_frontend = MicroFrontend()

            logger.info(
                "MicroKWS loaded  model=%s  input_slices=%d  stride=%d  quantized=%s",
                Path(path).name,
                self._input_slices,
                self._stride,
                self._model.is_quantized_model,
            )
            return True

        except Exception:
            logger.exception("Failed to load MicroKWS model")
            self._model = None
            return False

    def process_chunk(self, audio_16k: np.ndarray) -> tuple[bool, float]:
        """Feed a chunk of 16 kHz float32 audio.  Returns (triggered, latency_ms)."""
        if self._model is None or self._muted:
            return False, 0.0

        now = time.perf_counter()
        if now - self._last_trigger_time < self.cooldown_sec:
            return False, 0.0

        t0 = now

        # ── 1. float32 → int16 bytes ─────────────────────────────────────
        pcm_int16 = np.clip(audio_16k * 32768, -32768, 32767).astype(np.int16)
        pcm_bytes = self._audio_remainder + pcm_int16.tobytes()
        self._audio_remainder = b""

        # ── 2. Feed through MicroFrontend in 320-byte chunks ─────────────
        idx = 0
        triggered = False
        latency = 0.0

        while idx + _FRAME_BYTES <= len(pcm_bytes):
            result = self._micro_frontend.process_samples(
                pcm_bytes[idx : idx + _FRAME_BYTES]
            )
            consumed = result.samples_read * 2  # samples_read is in samples, convert to bytes
            if consumed <= 0:
                idx += _FRAME_BYTES  # avoid infinite loop
            else:
                idx += consumed

            if result.features is None or len(result.features) != _NUM_FEATURES:
                continue

            # New spectrogram frame (40 floats)
            frame = np.array(result.features, dtype=np.float32)

            # Shift sliding window left, insert new frame at the end
            if self._spec_count < self._input_slices:
                self._spec_buffer[self._spec_count] = frame
                self._spec_count += 1
            else:
                self._spec_buffer[:-1] = self._spec_buffer[1:]
                self._spec_buffer[-1] = frame

            self._frames_since_inference += 1

            # ── 3. Run inference when we have a stride-worth of new frames ─
            if (
                self._frames_since_inference >= self._stride
                and self._spec_count >= self._input_slices
            ):
                prob = self._run_inference()
                self._frames_since_inference = 0

                if prob >= self.threshold:
                    self._consecutive_count += 1
                else:
                    self._consecutive_count = 0

                if self.debug and prob >= self.threshold * 0.6:
                    logger.debug(
                        "MicroKWS  prob=%.3f  consec=%d/%d",
                        prob,
                        self._consecutive_count,
                        self.consecutive_triggers,
                    )

                if self._consecutive_count >= self.consecutive_triggers:
                    latency = (time.perf_counter() - t0) * 1000
                    self._consecutive_count = 0
                    self._last_trigger_time = time.perf_counter()
                    triggered = True
                    break  # one trigger per call is enough

        # Store leftover bytes for next call
        if idx < len(pcm_bytes):
            self._audio_remainder = pcm_bytes[idx:]

        return triggered, latency

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False
        self._consecutive_count = 0
        self._last_trigger_time = time.perf_counter()

    def reset(self) -> None:
        """Clear all streaming state (useful after long silence or mute cycle)."""
        from pymicro_features import MicroFrontend

        if self._spec_buffer is not None:
            self._spec_buffer[:] = 0
        self._spec_count = 0
        self._frames_since_inference = 0
        self._consecutive_count = 0
        self._audio_remainder = b""
        self._micro_frontend = MicroFrontend()

        # Re-zero TFLite interpreter state (ring buffers in Conv layers)
        if self._model is not None:
            for s in range(len(self._model.input_details)):
                dtype = (
                    np.int8 if self._model.is_quantized_model else np.float32
                )
                self._model.model.set_tensor(
                    self._model.input_details[s]["index"],
                    np.zeros(self._model.input_details[s]["shape"], dtype=dtype),
                )

    def enroll(self, *args, **kwargs) -> None:  # noqa: ARG002
        """No-op — MicroKWS models are trained offline."""
        logger.info("MicroKWS does not support runtime enrollment. Train offline instead.")

    def load_version(self, version: str) -> bool:
        """Load a versioned model from kws/models/versions/<version>/micro_nova.tflite."""
        versioned = Path(__file__).parent / "models" / "versions" / version / "micro_nova.tflite"
        if versioned.is_file():
            return self.load_model(str(versioned))
        logger.warning("MicroKWS version not found: %s", versioned)
        return False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_inference(self) -> float:
        """Feed the current spectrogram buffer through the TFLite model.
        Returns wake-word probability in [0, 1]."""
        model = self._model
        chunk = self._spec_buffer.copy()

        if model.is_quantized_model:
            chunk = model.quantize_input_data(chunk, model.input_details[0])

        model.model.set_tensor(
            model.input_details[0]["index"],
            chunk.reshape(model.input_details[0]["shape"]),
        )
        model.model.invoke()

        output = model.model.get_tensor(model.output_details[0]["index"])[0][0]

        if model.is_quantized_model:
            output = model.dequantize_output_data(output, model.output_details[0])

        return float(output)
