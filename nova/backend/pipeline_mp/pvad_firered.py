"""
FireRedChat pVAD wrapper for Nova.

Architecture (FireRedChat paper, Sec 2.2.1):
    audio_16k (160 samples) → mel spec (80 bins) → causal conv
    ECAPA-TDNN spk emb (192-dim) → concat along channel
    → GRU (2-layer, 256 hidden) → linear → sigmoid → speaking prob

Streaming at 10 ms granularity — stateful across calls via mel_buffer + gru_buffer.
Model: FireRedTeam/FireRedChat-pvad (pvad.onnx) on HuggingFace.

Usage:
    pvad = FireRedPVAD(model_dir="/path/to/pvad", voiceprint_16k_wav=ref)
    prob = pvad.is_target_speaking(audio_frame_float32_160)  # → 0.0-1.0
    pvad.reset()  # between turns
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("FireRedPVAD")

_MODEL_DIR_DEFAULT = os.path.join(
    os.path.expanduser("~"), "models", "FireRedChat-pvad"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_ecapa_embedding(wav_path_or_array, model_dir: str) -> np.ndarray:
    """Extract ECAPA-TDNN (192-dim) embedding from reference audio.

    Uses the exact SpeechBrain model bundled with the pVAD checkpoint
    to ensure embedding-space compatibility.
    """
    import torch
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
    except ImportError:
        raise ImportError(
            "speechbrain required for ECAPA-TDNN. "
            "Install: pip install speechbrain"
        )

    savedir = os.path.join(model_dir, "spkrec-ecapa-voxceleb")
    model = EncoderClassifier.from_hparams(
        source=str(savedir),
        savedir=savedir,
        run_opts={"device": "cpu"},
    )

    if isinstance(wav_path_or_array, (str, Path)):
        import torchaudio  # type: ignore
        waveform, sr = torchaudio.load(str(wav_path_or_array))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    elif isinstance(wav_path_or_array, np.ndarray):
        waveform = torch.from_numpy(wav_path_or_array).unsqueeze(0).float()
    else:
        raise TypeError(f"Expected str, Path, or np.ndarray, got {type(wav_path_or_array)}")

    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    emb = embedding.squeeze().cpu().numpy().astype(np.float32)
    emb = emb / np.linalg.norm(emb)  # L2 normalise (cosine-sim ready)
    return emb


def _load_enrolled_voiceprint(enc_path: str) -> np.ndarray:
    """Load a pre-saved encrypted voiceprint from Nova's L-3 storage."""
    enc_path = Path(enc_path)
    if not enc_path.exists():
        raise FileNotFoundError(f"Voiceprint not found: {enc_path}")
    sys_path = str(Path(__file__).resolve().parents[1] / "nova-l7" / "L-3")
    import sys as _sys
    _sys.path.insert(0, sys_path)
    try:
        from crypto_utils import load_array
        fp = load_array(enc_path)  # Path, not str — crypto_utils.load_and_decrypt calls .exists()
    finally:
        _sys.path.remove(sys_path)
    fp = fp.astype(np.float32)
    fp = fp / np.linalg.norm(fp)
    return fp


# ── main class ───────────────────────────────────────────────────────────────

class FireRedPVAD:
    """Streaming personalised VAD.

    10 ms per call — feed exactly 160 float32 samples (16 kHz mono).
    Maintains internal mel + GRU state; caller must call reset() between turns.
    """

    def __init__(
        self,
        model_dir: str | Path = _MODEL_DIR_DEFAULT,
        voiceprint_16k_wav: str | Path | None = None,
        voiceprint_enc: str | Path | None = None,
        voiceprint_embedding: np.ndarray | None = None,
    ) -> None:
        """
        Provide ONE of:
          - voiceprint_16k_wav: path to a 16kHz mono WAV of the target speaker
          - voiceprint_enc: path to Nova's encrypted .enc voiceprint
          - voiceprint_embedding: pre-computed 192-dim float32 array
        """
        model_dir = Path(model_dir)
        onnx_path = model_dir / "pvad.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"pvad.onnx not found at {onnx_path}. "
                f"Download: huggingface-cli download FireRedTeam/FireRedChat-pvad "
                f"--local-dir {model_dir}"
            )

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            raise ImportError("onnxruntime required. Install: pip install onnxruntime")

        self._model_dir = str(model_dir)

        # ── resolve voiceprint ──────────────────────────────────────────
        if voiceprint_embedding is not None:
            self._spk_emb = voiceprint_embedding.astype(np.float32)
        elif voiceprint_enc is not None:
            self._spk_emb = _load_enrolled_voiceprint(str(voiceprint_enc))
        elif voiceprint_16k_wav is not None:
            self._spk_emb = _extract_ecapa_embedding(str(voiceprint_16k_wav), self._model_dir)
        else:
            raise ValueError(
                "Provide one of: voiceprint_16k_wav, voiceprint_enc, or voiceprint_embedding"
            )

        # Validate shape
        if self._spk_emb.shape != (192,):
            raise ValueError(f"Speaker embedding must be (192,), got {self._spk_emb.shape}")

        # ── ONNX session ────────────────────────────────────────────────
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU is fine — 160 samples per call is trivial
        self._sess = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )

        # ── streaming state (per-session) ───────────────────────────────
        self._lock = threading.Lock()
        self.reset()

        logger.info(
            f"FireRedPVAD loaded (model={onnx_path.name}, "
            f"spk_emb norm={np.linalg.norm(self._spk_emb):.4f})"
        )

    # ── public API ───────────────────────────────────────────────────────────

    def is_target_speaking(self, audio_frame_16k_f32: np.ndarray) -> float:
        """Feed one 10 ms frame (160 float32 samples @ 16 kHz).

        Returns probability 0-1 that the TARGET speaker is speaking.
        Caller must convert int16 → float32 (/ 32768) before passing.
        """
        if audio_frame_16k_f32.size != 160:
            raise ValueError(
                f"Expected exactly 160 samples (10 ms @ 16 kHz), "
                f"got {audio_frame_16k_f32.size}"
            )

        # Ensure float32 contiguous
        audio = np.ascontiguousarray(audio_frame_16k_f32, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)  # (1, 160)

        spk = self._spk_emb.reshape(1, 192)  # (1, 192)

        with self._lock:
            outputs = self._sess.run(
                None,
                {
                    "input_audio": audio,
                    "spkemb": spk,
                    "mel_buffer": self._mel_buf,
                    "gru_buffer": self._gru_buf,
                },
            )
            # Linear_out [1,1], Sigmoid_out [1,1], mel_buffer_out, gru_buffer_out
            prob = float(outputs[1].item())  # sigmoid_out
            self._mel_buf = outputs[2]
            self._gru_buf = outputs[3]

        return prob

    def reset(self) -> None:
        """Clear streaming state. Call at the start of every turn."""
        with self._lock:
            # mel_buffer: (1, 80, 15) sliding window
            self._mel_buf = np.zeros((1, 80, 15), dtype=np.float32)
            # gru_buffer: (2, 1, 256) — 2-layer GRU hidden state
            self._gru_buf = np.zeros((2, 1, 256), dtype=np.float32)

    @property
    def spk_embedding(self) -> np.ndarray:
        """Return the 192-dim speaker embedding (read-only)."""
        return self._spk_emb.copy()
