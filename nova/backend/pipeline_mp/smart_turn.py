"""Smart-Turn v3 semantic end-of-turn classifier.

Wraps pipecat-ai/smart-turn-v3 ONNX model. Takes float32 mono 16 kHz audio,
returns probability (0-1) that the speaker has finished their thought.

Fail-open contract: `predict()` returns 1.0 on any load / inference failure
or unreliable input (empty or <250 ms audio), so the caller's typical
"if prob >= threshold: finalize" gate naturally falls through to the
pre-existing silence-based finalize path. This means a broken smart-turn
can never stall the pipeline — worst case it just reverts to old behavior.

Usage:
    from smart_turn import SmartTurnPredictor
    predictor = SmartTurnPredictor()   # lazy-loads on first predict()
    prob = predictor.predict(audio_float32)
    if prob >= 0.5:
        # user is done — finalize utterance

Env vars:
    NOVA_SMART_TURN              1 to enable (default 0 — safety)
    NOVA_SMART_TURN_THRESHOLD    finalize when prob >= this (default 0.5)
    NOVA_SMART_TURN_MODEL_PATH   override local model path
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from transformers import WhisperFeatureExtractor

logger = logging.getLogger("SmartTurn")

_SR = 16000
_MAX_SAMPLES = 8 * _SR   # smart-turn v3 expects ≤8 s of audio
_MIN_SAMPLES = _SR // 4  # <250 ms = unreliable; fall through to silence gate
_FAIL_OPEN = 1.0         # returned on any error so callers finalize normally
_MODEL_URL = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "smart_turn"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "smart_turn_v3.2_cpu.onnx"

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def is_enabled() -> bool:
    return os.getenv("NOVA_SMART_TURN", "0").strip().lower() in _TRUTHY


def threshold() -> float:
    try:
        return float(os.getenv("NOVA_SMART_TURN_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


class SmartTurnPredictor:
    """Thread-safe lazy ONNX wrapper. Returns 0.0 on load/inference failure."""

    _instance: "Optional[SmartTurnPredictor]" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "SmartTurnPredictor":
        # Process-local singleton — each worker process gets its own session.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._session: "Optional[InferenceSession]" = None
        self._extractor: "Optional[WhisperFeatureExtractor]" = None
        self._load_failed = False
        self._load_lock = threading.Lock()

    def _ensure_model_file(self) -> str:
        path = Path(os.getenv("NOVA_SMART_TURN_MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
        if path.is_file():
            return str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading Smart-Turn v3 model → {path}")
        urllib.request.urlretrieve(_MODEL_URL, str(path))
        return str(path)

    def _load(self) -> bool:
        if self._session is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._session is not None:
                return True
            if self._load_failed:
                return False
            try:
                import onnxruntime as ort  # type: ignore
                from transformers import WhisperFeatureExtractor  # type: ignore

                model_path = self._ensure_model_file()
                self._session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                self._extractor = WhisperFeatureExtractor.from_pretrained(
                    "openai/whisper-tiny"
                )
                logger.info("Smart-Turn v3 loaded (CPU).")
                return True
            except Exception as e:
                logger.warning(f"Smart-Turn load failed — disabling: {e}")
                self._load_failed = True
                self._session = None
                self._extractor = None
                return False

    def predict(self, audio_f32: np.ndarray) -> float:
        """Return end-of-turn probability in [0, 1].

        Fail-open: returns 1.0 on any load/inference failure, or when input is
        missing / too short to be reliable (<250 ms). A 1.0 return means the
        caller's `prob >= threshold` gate passes and the utterance finalizes
        normally — smart-turn being unavailable can never stall the pipeline.
        """
        if audio_f32 is None or audio_f32.size < _MIN_SAMPLES:
            return _FAIL_OPEN
        if not self._load():
            return _FAIL_OPEN
        assert self._extractor is not None and self._session is not None
        try:
            audio = np.ascontiguousarray(audio_f32, dtype=np.float32)
            # Clip to last 8 s (model's max window)
            if audio.size > _MAX_SAMPLES:
                audio = audio[-_MAX_SAMPLES:]
            features = self._extractor(
                audio,
                sampling_rate=_SR,
                max_length=_MAX_SAMPLES,
                padding="max_length",
                return_attention_mask=False,
                return_tensors="np",
            )
            # Subscript access is stable across transformers versions —
            # .input_features attribute was removed in some 4.4x releases.
            input_features = np.asarray(features["input_features"], dtype=np.float32)
            out = self._session.run(None, {"input_features": input_features})
            prob = float(out[0].flatten()[0])
            # Clamp to [0, 1] in case the head drifts outside sigmoid range
            if prob < 0.0:
                return 0.0
            if prob > 1.0:
                return 1.0
            return prob
        except Exception as e:
            logger.warning(f"Smart-Turn inference failed — fail-open: {e}")
            return _FAIL_OPEN
