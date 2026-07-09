# pipeline/dynamic_orchestrator.py
"""
Two-tier dynamic weight orchestrator for BiometricGate.

Tier 1 — PolicyMLP: ONNX, CPU, <1ms, always-on.
Tier 2 — SmolLM2-135M-Instruct: GPU, ~50ms, fires only on low-confidence
          or novel context (MLP uncertainty > 0.35).

Training flow:
  1. Run SmolLM2 as teacher on your auth event logs → generates (context, weights) pairs
  2. Train PolicyMLP on those pairs (supervised) → export to ONNX
  3. PolicyMLP replaces SmolLM2 for 95% of calls at runtime

Integration:
  In BiometricGate.load(), add:
      self.orchestrator = DynamicOrchestrator.load(store_dir)
  In FusionScorer.fuse(), replace static weights with:
      weights, thresh_delta = gate.orchestrator.get_weights(context)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("DynamicOrchestrator")

# ── Base weights (fallback if orchestrator unavailable) ───────────────────────
# NOTE (DriveAuth): 'behavior' is intentionally REMOVED from the Trust fusion
# weights. Behaviour/context is a Risk signal (driveauth/risk_model.py), never a
# Trust signal (proposal §4.3). Weights are biometric-only and renormalised.
_BASE_W = {"voice": 0.30, "face": 0.40, "finger": 0.30}

# ── Uncertainty threshold — above this, escalate to SmolLM2 ──────────────────
_UNCERTAINTY_THRESH = float(os.getenv("NOVA_ORCH_UNCERTAINTY", "0.35"))


# ══════════════════════════════════════════════════════════════════════════════
#  Context vector builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorContext:
    # Signal quality
    ambient_noise_rms:      float = 0.02
    snr_db:                 float = 20.0
    vehicle_speed_kmh:      float = 0.0
    time_hour:              float = 12.0      # 0–23
    # Modality confidence (0–1, from hardware quality checks before scoring)
    voice_signal_conf:      float = 1.0       # RMS energy, spectral clarity
    face_camera_quality:    float = 1.0       # blur score, IR saturation
    finger_sensor_quality:  float = 1.0       # ridge clarity from sensor API
    behavioral_data_secs:   float = 0.0       # seconds of sensor history available
    # Auth history (rolling window)
    auth_streak_successes:  int   = 0
    auth_streak_failures:   int   = 0
    last_deny_secs_ago:     float = 9999.0
    # Transaction context
    transaction_tier:       int   = 0         # 0=none, 1=normal, 2=high-value
    # Environment flags
    is_highway:             bool  = False
    is_parked:              bool  = False
    is_tunnel:              bool  = False
    # Modality raw scores (from previous fast-pass, if available)
    voice_raw_score:        float = 0.5
    face_raw_score:         float = 0.5
    finger_raw_score:       float = 0.5
    behavioral_raw_score:   float = 0.5

    def to_vector(self) -> np.ndarray:
        """20-dimensional float32 feature vector."""
        hour_rad = self.time_hour * 2 * math.pi / 24.0
        return np.array([
            self.ambient_noise_rms,
            self.snr_db / 40.0,              # normalise to ~0–1
            self.vehicle_speed_kmh / 200.0,
            math.sin(hour_rad),              # cyclic time encoding
            math.cos(hour_rad),
            self.voice_signal_conf,
            self.face_camera_quality,
            self.finger_sensor_quality,
            min(self.behavioral_data_secs / 30.0, 1.0),
            min(self.auth_streak_successes / 5.0, 1.0),
            min(self.auth_streak_failures / 3.0, 1.0),
            float(self.transaction_tier) / 2.0,
            float(self.is_highway),
            float(self.is_parked),
            float(self.is_tunnel),
            self.voice_raw_score,
            self.face_raw_score,
            self.finger_raw_score,
            self.behavioral_raw_score,
            min(self.last_deny_secs_ago / 300.0, 1.0),
        ], dtype=np.float32)

    def to_prompt(self) -> str:
        """Natural language description for SmolLM2 fallback."""
        env_parts = []
        if self.is_highway:
            env_parts.append(f"highway driving at {self.vehicle_speed_kmh:.0f} km/h")
        elif self.is_parked:
            env_parts.append("vehicle parked")
        else:
            env_parts.append(f"city driving at {self.vehicle_speed_kmh:.0f} km/h")
        if self.is_tunnel:
            env_parts.append("in tunnel")
        env_parts.append(f"ambient noise RMS={self.ambient_noise_rms:.3f}, SNR={self.snr_db:.1f}dB")
        env_parts.append(f"time={self.time_hour:.0f}h")

        quality_parts = []
        if self.voice_signal_conf < 0.5:
            quality_parts.append(f"voice signal weak (conf={self.voice_signal_conf:.2f})")
        if self.face_camera_quality < 0.5:
            quality_parts.append(f"IR camera degraded (quality={self.face_camera_quality:.2f})")
        if self.finger_sensor_quality < 0.5:
            quality_parts.append(f"fingerprint sensor weak (quality={self.finger_sensor_quality:.2f})")
        if self.behavioral_data_secs < 5:
            quality_parts.append(f"limited behavioral data ({self.behavioral_data_secs:.0f}s)")

        return (
            f"Vehicle biometric context: {', '.join(env_parts)}. "
            f"{'Issues: ' + ', '.join(quality_parts) + '. ' if quality_parts else ''}"
            f"Auth history: {self.auth_streak_successes} recent successes, "
            f"{self.auth_streak_failures} failures. "
            f"Transaction tier: {self.transaction_tier}. "
            "Adjust biometric fusion weights (voice, face, fingerprint, behavior) "
            "to maximise verification reliability given these conditions. "
            "Respond ONLY with JSON: "
            '{"voice": 0.XX, "face": 0.XX, "finger": 0.XX, "behavior": 0.XX, '
            '"thresh_delta": 0.XX, "reason": "..."}'
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 1 — PolicyMLP (ONNX, CPU)
# ══════════════════════════════════════════════════════════════════════════════

class PolicyMLP:
    """
    Tiny 3-layer MLP exported to ONNX.
    Input: 20-dim context vector.
    Output: [w_voice, w_face, w_finger, w_behavior, uncertainty, thresh_delta]
    
    Architecture (PyTorch, export once):
        Linear(20, 64) → LayerNorm → ReLU → Dropout(0.1)
        Linear(64, 32) → LayerNorm → ReLU → Dropout(0.1)
        Linear(32, 6)
          → [:4] softmax  (weights, forced sum-to-1)
          → [4]  sigmoid  (uncertainty estimate, 0–1)
          → [5]  tanh×0.15 (threshold delta, ±0.15)
    
    ~2MB ONNX file. No GPU needed. Inference <0.5ms on Jetson CPU.
    See train_orchestrator.py for training + export.
    """

    def __init__(self, session):
        self._session   = session
        self._input_name = session.get_inputs()[0].name

    @classmethod
    def load(cls, model_path: str) -> "PolicyMLP | None":
        try:
            import onnxruntime as ort
            # Force CPU — save GPU for LLM/STT
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"PolicyMLP loaded from {model_path}")
            return cls(session)
        except Exception as exc:
            logger.warning(f"PolicyMLP: load failed ({exc}) — will use static weights")
            return None

    def infer(self, ctx: OrchestratorContext) -> tuple[dict[str, float], float, float]:
        """
        Returns (weights_dict, uncertainty, thresh_delta).
        uncertainty ∈ [0, 1] — above 0.35 triggers SmolLM2 fallback.
        """
        vec = ctx.to_vector()[np.newaxis]          # (1, 20)
        out = self._session.run(None, {self._input_name: vec})[0][0]  # (6,)

        # softmax over the biometric outputs only. The exported MLP still emits a
        # 6-wide vector; slot [3] was the old 'behavior' weight, now dropped
        # (behaviour is a Risk signal, not a Trust signal — proposal §4.3). We
        # softmax over voice/face/finger and renormalise to sum to 1.
        w_raw  = out[:3]
        w_exp  = np.exp(w_raw - w_raw.max())
        w_norm = w_exp / w_exp.sum()

        weights = {
            "voice":    float(w_norm[0]),
            "face":     float(w_norm[1]),
            "finger":   float(w_norm[2]),
        }
        uncertainty  = float(1.0 / (1.0 + math.exp(-out[4])))  # sigmoid
        thresh_delta = float(math.tanh(out[5]) * 0.15)

        logger.debug(f"PolicyMLP: w={weights} unc={uncertainty:.3f} Δt={thresh_delta:+.3f}")
        return weights, uncertainty, thresh_delta


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 2 — SmolLM2-135M-Instruct (GPU, ~200MB INT4)
# ══════════════════════════════════════════════════════════════════════════════

class SmolLM2Orchestrator:
    """
    SmolLM2-135M-Instruct as a reasoning fallback when PolicyMLP is uncertain.
    
    On Jetson Orin 8GB:
      - Load via llama.cpp (Q4_K_M) for ~150MB VRAM + fast TensorRT path
      - OR via HuggingFace + transformers (INT4 bitsandbytes) ~200MB
    
    We use llama.cpp here via ctransformers / llama-cpp-python — this gives
    the best latency on Jetson's GPU without needing full PyTorch CUDA.
    
    Model: HuggingFaceTB/SmolLM2-135M-Instruct  (GGUF: smollm2-135m-instruct.Q4_K_M.gguf)
    Download: huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct-GGUF
    """

    # JSON output schema enforced via grammar (llama.cpp)
    _SCHEMA = {
        "type": "object",
        "properties": {
            "voice":       {"type": "number", "minimum": 0, "maximum": 1},
            "face":        {"type": "number", "minimum": 0, "maximum": 1},
            "finger":      {"type": "number", "minimum": 0, "maximum": 1},
            "behavior":    {"type": "number", "minimum": 0, "maximum": 1},
            "thresh_delta":{"type": "number", "minimum": -0.15, "maximum": 0.15},
            "reason":      {"type": "string"},
        },
        # 'behavior' no longer required — it's ignored by the Trust fusion (§4.3).
        "required": ["voice", "face", "finger", "thresh_delta"],
    }

    def __init__(self, llm):
        self._llm = llm   # llama_cpp.Llama instance

    @classmethod
    def load(cls, gguf_path: str, n_gpu_layers: int = 33) -> "SmolLM2Orchestrator | None":
        """
        n_gpu_layers=33 offloads all layers to GPU on Jetson Orin.
        Reduce to 0 for pure CPU (slower but frees GPU for LLM).
        """
        if not Path(gguf_path).exists():
            logger.warning(f"SmolLM2: GGUF not found at {gguf_path}")
            return None
        try:
            from llama_cpp import Llama, LlamaGrammar  # type: ignore
            llm = Llama(
                model_path   = gguf_path,
                n_gpu_layers = n_gpu_layers,
                n_ctx        = 512,    # small context — prompt is ~200 tokens
                n_threads    = 4,
                verbose      = False,
            )
            logger.info(f"SmolLM2-135M loaded from {gguf_path} (gpu_layers={n_gpu_layers})")
            return cls(llm)
        except Exception as exc:
            logger.warning(f"SmolLM2: load failed ({exc})")
            return None

    def infer(self, ctx: OrchestratorContext) -> tuple[dict[str, float], float]:
        """
        Returns (weights_dict, thresh_delta).
        Parses JSON from model output with grammar constraint.
        """
        t0 = time.perf_counter()
        prompt = (
            "<|im_start|>system\n"
            "You are a biometric fusion weight orchestrator for a vehicle AI. "
            "Given the driving context, return optimal fusion weights as JSON only.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{ctx.to_prompt()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            from llama_cpp import LlamaGrammar  # type: ignore
            import json as _json
            grammar = LlamaGrammar.from_json_schema(_json.dumps(self._SCHEMA))
            out = self._llm(
                prompt,
                max_tokens = 120,
                temperature= 0.1,    # near-deterministic
                grammar    = grammar,
            )
            raw   = out["choices"][0]["text"].strip()
            data  = json.loads(raw)
            lat   = (time.perf_counter() - t0) * 1000

            # Normalise weights to sum to 1 — biometric modalities only.
            # 'behavior' is ignored if the model still emits it (Risk signal, §4.3).
            w_raw = np.array([data["voice"], data["face"],
                              data["finger"]], dtype=np.float32)
            w_raw = np.clip(w_raw, 0.01, 1.0)
            w_norm = w_raw / w_raw.sum()

            weights = {
                "voice":    float(w_norm[0]),
                "face":     float(w_norm[1]),
                "finger":   float(w_norm[2]),
            }
            thresh_delta = float(np.clip(data.get("thresh_delta", 0.0), -0.15, 0.15))
            reason = data.get("reason", "")

            logger.info(f"SmolLM2 orchestrator: w={weights} Δt={thresh_delta:+.3f} "
                        f"reason='{reason[:60]}' lat={lat:.0f}ms")
            return weights, thresh_delta

        except Exception as exc:
            logger.error(f"SmolLM2.infer: {exc} — falling back to base weights")
            return dict(_BASE_W), 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  DynamicOrchestrator — top-level, used by BiometricGate
# ══════════════════════════════════════════════════════════════════════════════

class DynamicOrchestrator:
    """
    Drop-in replacement for the static _W_VOICE / _W_FACE / ... constants.
    
    Typical integration in BiometricGate:
        # Replace FusionScorer.fuse(voice_r, face_r, finger_r, beh_r)  with:
        ctx = gate.build_context(audio_np, vehicle_state)
        weights, thresh_delta = gate.orchestrator.get_weights(ctx)
        score, eff = FusionScorer.fuse_dynamic(voice_r, face_r, finger_r, beh_r, weights)
        decision = FusionScorer.route(score, thresh_delta)
    """

    def __init__(
        self,
        mlp:     PolicyMLP | None,
        llm:     SmolLM2Orchestrator | None,
        unc_thr: float = _UNCERTAINTY_THRESH,
    ):
        self._mlp    = mlp
        self._llm    = llm
        self._unc    = unc_thr
        self._cache: dict[str, Any] = {}    # LRU-1 cache — repeated same context → skip inference

    @classmethod
    def load(cls, store_dir: str) -> "DynamicOrchestrator":
        store = Path(store_dir)
        mlp_path  = store / "orchestrator_mlp.onnx"
        gguf_path = store / "smollm2-135m-instruct.Q4_K_M.gguf"

        mlp = PolicyMLP.load(str(mlp_path)) if mlp_path.exists() else None
        llm = SmolLM2Orchestrator.load(str(gguf_path)) if gguf_path.exists() else None

        if mlp is None and llm is None:
            logger.warning("DynamicOrchestrator: no models found — using static weights")
        return cls(mlp, llm)

    def get_weights(
        self,
        ctx: OrchestratorContext,
    ) -> tuple[dict[str, float], float]:
        """
        Returns (weights_dict, thresh_delta).
        
        Decision tree:
          1. No models → static base weights
          2. PolicyMLP only → MLP output
          3. MLP + SmolLM2 → MLP first; if uncertainty > threshold → SmolLM2
          4. SmolLM2 only → SmolLM2 directly
        """
        if self._mlp is None and self._llm is None:
            return dict(_BASE_W), 0.0

        if self._mlp is not None:
            weights, uncertainty, thresh_delta = self._mlp.infer(ctx)
            if uncertainty <= self._unc or self._llm is None:
                return weights, thresh_delta
            # Uncertain — escalate to SmolLM2
            logger.info(f"Orchestrator: MLP uncertainty={uncertainty:.3f} → SmolLM2")
            return self._llm.infer(ctx)

        # SmolLM2 only
        return self._llm.infer(ctx)


# ══════════════════════════════════════════════════════════════════════════════
#  Updated FusionScorer (drop-in for the static version in biometric_gate.py)
# ══════════════════════════════════════════════════════════════════════════════

class DynamicFusionScorer:
    """Extends FusionScorer with dynamic weight support."""

    @staticmethod
    def fuse_dynamic(
        voice:      Any,    # ModalityResult
        face:       Any,
        finger:     Any,
        behavioral: Any,
        weights:    dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        candidates: dict[str, tuple[float, float]] = {}

        def _add(name: str, res: Any, w: float) -> None:
            if res.score is not None and res.confident:
                candidates[name] = (res.score, w)

        _add("voice",    voice,      weights.get("voice",    0.30))
        _add("face",     face,       weights.get("face",     0.40))
        _add("finger",   finger,     weights.get("finger",   0.30))
        # NOTE (DriveAuth): behaviour deliberately NOT fused into Trust — it is a
        # Risk signal only (proposal §4.3). ``behavioral`` arg kept for signature
        # compatibility but ignored here.

        if not candidates:
            return 0.0, {}

        total_w = sum(w for _, w in candidates.values())
        eff: dict[str, float] = {}
        fused = 0.0
        for name, (score, w) in candidates.items():
            eff_w = w / total_w
            eff[name] = eff_w
            fused += eff_w * score

        return float(fused), eff

    @staticmethod
    def route(score: float, thresh_delta: float = 0.0) -> str:
        """
        DEPRECATED — retired in the DriveAuth migration. The live decision path
        is driveauth/policy_engine.py (tiered, versioned, deterministic; Trust
        and Risk are separate inputs there, not a single fused score). This
        method is kept only so any lingering importer doesn't hard-crash; it is
        not called anywhere in the live DriveAuthGate pipeline.

        Dual-mode import (see driveauth/gate.py for why): try the bare-after-
        sys.path style used by workers first, since that's the more common
        caller for anything still reaching this deprecated path, then fall back
        to a package-relative import.
        """
        try:
            from biometric_gate import THRESH_APPROVE, THRESH_STEP_UP
        except ImportError:
            from .biometric_gate import THRESH_APPROVE, THRESH_STEP_UP
        approve  = THRESH_APPROVE  + thresh_delta
        step_up  = THRESH_STEP_UP  + thresh_delta
        if score >= approve:
            return "pass"
        elif score >= step_up:
            return "step_up"
        else:
            return "deny"