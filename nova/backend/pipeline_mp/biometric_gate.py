"""
pipeline/biometric_gate.py
--------------------------
Multi-modal biometric verification gate for nova_loop.

Intercepts every stt.finalize() call before LLM dispatch.
Runs inside the stt_worker process — no extra subprocess needed.

Architecture (from diagram):
  STT finalizes → BiometricGate.intercept(transcript, audio_np, ...)
    ├── fast intent sniff  (payment keyword regex, ~0.1ms)
    ├── voice verifier     (ECAPA-TDNN, reuses pvad_firered embeddings)
    ├── behavioral score   (LSTM passive, always-on background feed)
    ├── face verifier      (MobileFaceNet INT8/ONNX, on-demand IR frame)
    └── finger verifier    (FingerNet-lite INT8/ONNX, on-demand only)
         ↓
    FusionScorer           (weighted: voice×0.25 + face×0.35 + finger×0.30 + behavior×0.10)
         ↓
    ConfidenceRouter
      ≥ 0.85 → PASS  (dispatch LLM normally)
      0.60–0.84 → STEP_UP  (TTS prompt → re-auth)
      < 0.60 → DENY  (TTS reject + rate-limit + log)

Secure template store: models/biometric_store/  (AES-256, same key as L-3)

Usage in stt_worker.py:
    gate = BiometricGate.load(l3_dir, store_dir)
    result = gate.intercept(transcript, audio_np, ws_out_queue, llm_in_queue)
    # result is "pass", "step_up", or "deny"
    # On "pass": the gate already put the message into llm_in_queue for you.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("BiometricGate")

# ── Confidence thresholds (mirror Image 1 / verify.py) ───────────────────────
THRESH_APPROVE  = float(os.getenv("NOVA_BIO_APPROVE",  "0.85"))
THRESH_STEP_UP  = float(os.getenv("NOVA_BIO_STEP_UP",  "0.60"))
# score ≥ THRESH_APPROVE → approve
# THRESH_STEP_UP ≤ score < THRESH_APPROVE → step-up
# score < THRESH_STEP_UP → deny

# ── Fusion weights ────────────────────────────────────────────────────────────
# RETIRED (DriveAuth migration): the old Trust fusion blended a behaviour score
# straight into the same Trust number as voice/face/finger. That conflation is
# exactly what proposal §4.3 warns against — behaviour/location/context must
# drive a separate Risk Score, never Trust. Trust fusion now lives in
# driveauth/trust_fusion.py (biometrics only); behaviour moved to
# driveauth/risk_model.py. These constants remain only so any external importer
# doesn't hard-crash, but nothing in the DriveAuth path reads them.
_W_VOICE    = 0.30   # deprecated — see driveauth/trust_fusion.py
_W_FACE     = 0.40   # deprecated
_W_FINGER   = 0.30   # deprecated
_W_BEHAVIOR = 0.0    # deprecated — behaviour is NOT a Trust input (proposal §4.3)

# ── Step-up retries ───────────────────────────────────────────────────────────
MAX_STEP_UP_RETRIES = 2
STEP_UP_TIMEOUT_S   = 30.0   # seconds to wait for re-auth audio

# ── Payment intent fast scan ──────────────────────────────────────────────────
_PAYMENT_RE = re.compile(
    r"\b(order|buy|purchase|pay|send money|transfer|checkout|coffee|burger"
    r"|pizza|food|latte|cappuccino|add to cart|top.?up|recharge)\b",
    re.IGNORECASE,
)

# ── Audio preprocessing (must match L-3/verify.py) ───────────────────────────
_PREEMPH   = 0.97
_RMS_TGT   = 0.08
_RMS_FLOOR = 1e-6


def _preprocess(audio: np.ndarray) -> np.ndarray:
    out = np.empty_like(audio, dtype=np.float32)
    out[0]  = audio[0]
    out[1:] = audio[1:] - _PREEMPH * audio[:-1]
    rms = float(np.sqrt(np.mean(out ** 2)))
    if rms > _RMS_FLOOR:
        out *= _RMS_TGT / rms
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Modality result containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityResult:
    score:     float | None   # None = not available / skipped
    confident: bool           # was the signal clean enough to trust?
    latency_ms: float = 0.0


@dataclass
class FusionResult:
    score:        float
    decision:     str          # "pass" | "step_up" | "deny"
    voice:        ModalityResult = field(default_factory=lambda: ModalityResult(None, False))
    face:         ModalityResult = field(default_factory=lambda: ModalityResult(None, False))
    finger:       ModalityResult = field(default_factory=lambda: ModalityResult(None, False))
    behavioral:   ModalityResult = field(default_factory=lambda: ModalityResult(None, False))
    is_payment:   bool = False
    effective_weights: dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  Voice Verifier — reuses ECAPA-TDNN from pvad/speaker_gate.py
# ══════════════════════════════════════════════════════════════════════════════

class VoiceVerifier:
    """
    ECAPA-TDNN cosine similarity against the enrolled voiceprint.
    Uses the same model and key as speaker_gate.py / L-3/verify.py.
    Operates on audio already captured by STT — no extra recording.
    """

    def __init__(self, ecapa_model, driver_embedding: np.ndarray | None, device: str):
        self._model    = ecapa_model
        self._emb      = driver_embedding   # shape (192,) normalised
        self._device   = device

    @classmethod
    def load(cls, l3_dir: str, driver_id: str = "driver1", device: str | None = None) -> "VoiceVerifier":
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        l3_path = Path(l3_dir)
        sys.path.insert(0, str(l3_path))

        # Load encrypted voiceprint
        driver_embedding: np.ndarray | None = None
        try:
            from crypto_utils import load_array  # type: ignore
            vp_path = l3_path / "data" / "voiceprints" / f"{driver_id}.enc"
            if vp_path.exists():
                emb  = load_array(vp_path)
                norm = np.linalg.norm(emb)
                driver_embedding = emb / norm if norm > 1e-8 else emb
                logger.info(f"VoiceVerifier: voiceprint loaded for '{driver_id}' dim={emb.shape}")
            else:
                logger.warning(f"VoiceVerifier: no voiceprint at {vp_path} — voice disabled")
        except Exception as exc:
            logger.error(f"VoiceVerifier: voiceprint load failed: {exc}")

        # Load ECAPA-TDNN (SpeechBrain)
        ecapa_model = None
        if driver_embedding is not None:
            try:
                import torch
                from speechbrain.pretrained import SpeakerRecognition  # type: ignore
                ecapa_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(l3_path / "data" / "ecapa_model"),
                    run_opts={"device": device},
                )
                logger.info("VoiceVerifier: ECAPA-TDNN loaded")
            except Exception as exc:
                logger.warning(f"VoiceVerifier: ECAPA load failed ({exc}) — voice will fail-open")
                driver_embedding = None

        return cls(ecapa_model, driver_embedding, device)

    def score(self, audio_f32: np.ndarray, sample_rate: int = 16_000) -> ModalityResult:
        """Cosine similarity against enrolled voiceprint. Returns 0.0–1.0."""
        t0 = time.perf_counter()

        if self._model is None or self._emb is None:
            return ModalityResult(score=None, confident=False)

        # Need at least 1 second of audio for ECAPA
        if len(audio_f32) < sample_rate:
            return ModalityResult(score=None, confident=False)

        try:
            import torch
            proc = _preprocess(audio_f32)
            wav  = torch.from_numpy(proc).unsqueeze(0).to(self._device)
            with torch.no_grad():
                emb = self._model.encode_batch(wav)
            live_emb = emb.squeeze().cpu().numpy()
            norm     = np.linalg.norm(live_emb)
            if norm > 1e-8:
                live_emb /= norm
            sim = float(np.dot(self._emb, live_emb))
            # cosine sim ∈ [-1, 1] → clip to [0, 1] for fusion
            sim = float(np.clip(sim, 0.0, 1.0))
            lat = (time.perf_counter() - t0) * 1000
            logger.debug(f"VoiceVerifier: score={sim:.3f} lat={lat:.1f}ms")
            return ModalityResult(score=sim, confident=True, latency_ms=lat)
        except Exception as exc:
            logger.error(f"VoiceVerifier.score: {exc}")
            return ModalityResult(score=None, confident=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Face Verifier — MobileFaceNet INT8 / ONNX  (IR camera frame)
# ══════════════════════════════════════════════════════════════════════════════

class FaceVerifier:
    """
    MobileFaceNet INT8 via ONNX Runtime.
    Captures an IR camera frame on demand (single shot).
    Falls back to OpenCV webcam if IR camera index not set.
    Set NOVA_IR_CAMERA_INDEX env var (default 0).
    Model path: models/biometric_store/mobilefacenet_int8.onnx
    Template: models/biometric_store/faces/{driver_id}.enc  (AES-256)
    """

    _FACE_SIZE = (112, 112)  # MobileFaceNet input

    def __init__(self, session, driver_embedding: np.ndarray | None, crypto_key: bytes | None):
        self._session = session
        self._emb     = driver_embedding
        self._key     = crypto_key
        self._cam_idx = int(os.getenv("NOVA_IR_CAMERA_INDEX", "0"))

    @classmethod
    def load(cls, store_dir: str, driver_id: str = "driver1") -> "FaceVerifier":
        store = Path(store_dir)
        session = None
        driver_embedding: np.ndarray | None = None
        crypto_key: bytes | None = None

        onnx_path = store / "mobilefacenet_int8.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
                session = ort.InferenceSession(str(onnx_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                logger.info("FaceVerifier: MobileFaceNet INT8 ONNX loaded")
            except Exception as exc:
                logger.warning(f"FaceVerifier: ONNX load failed ({exc})")
        else:
            logger.warning(f"FaceVerifier: model not found at {onnx_path}")

        face_enc = store / "faces" / f"{driver_id}.enc"
        if face_enc.exists() and session is not None:
            try:
                from cryptography.fernet import Fernet  # type: ignore
                key_path = store / ".bio_key"
                if key_path.exists():
                    crypto_key = key_path.read_bytes()
                    f = Fernet(crypto_key)
                    raw = f.decrypt(face_enc.read_bytes())
                    driver_embedding = np.frombuffer(raw, dtype=np.float32).copy()
                    norm = np.linalg.norm(driver_embedding)
                    if norm > 1e-8:
                        driver_embedding /= norm
                    logger.info(f"FaceVerifier: face template loaded for '{driver_id}'")
            except Exception as exc:
                logger.error(f"FaceVerifier: template load failed: {exc}")

        return cls(session, driver_embedding, crypto_key)

    def capture_and_score(self) -> ModalityResult:
        """Grab one IR frame, run MobileFaceNet, compare to enrolled template."""
        t0 = time.perf_counter()

        if self._session is None or self._emb is None:
            return ModalityResult(score=None, confident=False)

        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(self._cam_idx)
            if not cap.isOpened():
                logger.warning("FaceVerifier: camera not available")
                return ModalityResult(score=None, confident=False)
            # Discard first few frames (exposure settle)
            for _ in range(3):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return ModalityResult(score=None, confident=False)

            # Preprocess for MobileFaceNet
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, self._FACE_SIZE)
            blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis]  # NCHW

            input_name = self._session.get_inputs()[0].name
            emb = self._session.run(None, {input_name: blob})[0][0]
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb /= norm
            sim = float(np.clip(np.dot(self._emb, emb), 0.0, 1.0))
            lat = (time.perf_counter() - t0) * 1000
            logger.debug(f"FaceVerifier: score={sim:.3f} lat={lat:.1f}ms")
            return ModalityResult(score=sim, confident=True, latency_ms=lat)
        except Exception as exc:
            logger.error(f"FaceVerifier.capture_and_score: {exc}")
            return ModalityResult(score=None, confident=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Fingerprint Verifier — FingerNet-lite INT8 / ONNX  (on-demand only)
# ══════════════════════════════════════════════════════════════════════════════

class FingerVerifier:
    """
    FingerNet-lite INT8 via ONNX Runtime.
    Only activated for high-tier payment requests.
    Reads from steering-wheel or gear-shift sensor (USB HID or GPIO).
    Set NOVA_FINGER_DEVICE env var to sensor path.
    Model path: models/biometric_store/fingernet_lite_int8.onnx
    Template: models/biometric_store/fingers/{driver_id}.enc  (AES-256)
    """

    def __init__(self, session, driver_template: bytes | None, crypto_key: bytes | None):
        self._session  = session
        self._template = driver_template   # raw minutiae bytes
        self._key      = crypto_key
        self._device   = os.getenv("NOVA_FINGER_DEVICE", "/dev/input/fingerprint0")

    @classmethod
    def load(cls, store_dir: str, driver_id: str = "driver1") -> "FingerVerifier":
        store = Path(store_dir)
        session = None
        driver_template: bytes | None = None
        crypto_key: bytes | None = None

        onnx_path = store / "fingernet_lite_int8.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
                session = ort.InferenceSession(str(onnx_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                logger.info("FingerVerifier: FingerNet-lite INT8 loaded")
            except Exception as exc:
                logger.warning(f"FingerVerifier: ONNX load failed ({exc})")

        finger_enc = store / "fingers" / f"{driver_id}.enc"
        if finger_enc.exists():
            try:
                from cryptography.fernet import Fernet  # type: ignore
                key_path = store / ".bio_key"
                if key_path.exists():
                    crypto_key = key_path.read_bytes()
                    f = Fernet(crypto_key)
                    driver_template = f.decrypt(finger_enc.read_bytes())
                    logger.info(f"FingerVerifier: template loaded for '{driver_id}'")
            except Exception as exc:
                logger.error(f"FingerVerifier: template load failed: {exc}")

        return cls(session, driver_template, crypto_key)

    def capture_and_score(self) -> ModalityResult:
        """Read one scan from the hardware sensor and compare."""
        t0 = time.perf_counter()

        if self._session is None or self._template is None:
            return ModalityResult(score=None, confident=False)

        # ── Hardware read ──────────────────────────────────────────────────
        # Replace with actual sensor SDK / HID read for your hardware.
        # Here we check a UNIX socket / named pipe that your sensor daemon writes.
        scan_socket = os.getenv("NOVA_FINGER_SOCKET", "/tmp/nova_finger.sock")
        raw_scan: bytes | None = None
        try:
            import socket as sock
            s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect(scan_socket)
            # Protocol: send "SCAN\n", read 256×256 uint8 image
            s.sendall(b"SCAN\n")
            chunks = []
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            raw_scan = b"".join(chunks)
            s.close()
        except Exception as exc:
            logger.warning(f"FingerVerifier: sensor read failed ({exc})")
            return ModalityResult(score=None, confident=False)

        if not raw_scan or len(raw_scan) < 256 * 256:
            return ModalityResult(score=None, confident=False)

        try:
            img = np.frombuffer(raw_scan[:256*256], dtype=np.uint8).reshape(1, 1, 256, 256)
            blob = img.astype(np.float32) / 255.0
            input_name = self._session.get_inputs()[0].name
            minutiae = self._session.run(None, {input_name: blob})[0][0]

            # Compare to stored template (treated as embedding vector)
            tmpl = np.frombuffer(self._template, dtype=np.float32)
            if len(tmpl) != len(minutiae):
                return ModalityResult(score=None, confident=False)
            sim = float(np.clip(np.dot(minutiae, tmpl) /
                                (np.linalg.norm(minutiae) * np.linalg.norm(tmpl) + 1e-8),
                                0.0, 1.0))
            lat = (time.perf_counter() - t0) * 1000
            logger.debug(f"FingerVerifier: score={sim:.3f} lat={lat:.1f}ms")
            return ModalityResult(score=sim, confident=True, latency_ms=lat)
        except Exception as exc:
            logger.error(f"FingerVerifier.capture_and_score: {exc}")
            return ModalityResult(score=None, confident=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Behavioral Monitor — LSTM passive, always-on
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralMonitor:
    """
    Passive LSTM that scores driving style, seating posture, and pressure
    patterns from CAN-bus / seat sensor stream.

    Feed sensor data via .update(sensor_dict) from your vehicle data thread.
    The gate reads the rolling score via .get_score().

    Sensor dict keys (all optional — use whatever your car exposes):
      steering_torque_nm, brake_pressure_bar, throttle_pct,
      seat_pressure_kpa, lateral_accel_g, yaw_rate_dps
    """

    def __init__(self, session, driver_profile: np.ndarray | None, window: int = 50):
        self._session = session
        self._profile = driver_profile   # enrolled mean feature vector
        self._window  = window           # ~5s at 10Hz sensor rate
        self._buf: list[np.ndarray] = []
        self._score: float = 1.0         # start fail-open
        self._lock  = threading.Lock()

    @classmethod
    def load(cls, store_dir: str, driver_id: str = "driver1") -> "BehavioralMonitor":
        store = Path(store_dir)
        session = None
        driver_profile: np.ndarray | None = None

        onnx_path = store / "behavioral_lstm_int8.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
                session = ort.InferenceSession(str(onnx_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                logger.info("BehavioralMonitor: LSTM loaded")
            except Exception as exc:
                logger.warning(f"BehavioralMonitor: ONNX load failed ({exc})")

        profile_enc = store / "behavioral" / f"{driver_id}.enc"
        if profile_enc.exists():
            try:
                from cryptography.fernet import Fernet  # type: ignore
                key_path = store / ".bio_key"
                if key_path.exists():
                    f = Fernet(key_path.read_bytes())
                    raw = f.decrypt(profile_enc.read_bytes())
                    driver_profile = np.frombuffer(raw, dtype=np.float32).copy()
                    logger.info(f"BehavioralMonitor: profile loaded for '{driver_id}'")
            except Exception as exc:
                logger.error(f"BehavioralMonitor: profile load failed: {exc}")

        return cls(session, driver_profile)

    def update(self, sensor: dict[str, float]) -> None:
        """Call from vehicle sensor thread at ~10Hz."""
        vec = np.array([
            sensor.get("steering_torque_nm", 0.0),
            sensor.get("brake_pressure_bar", 0.0),
            sensor.get("throttle_pct",       0.0),
            sensor.get("seat_pressure_kpa",  0.0),
            sensor.get("lateral_accel_g",    0.0),
            sensor.get("yaw_rate_dps",       0.0),
        ], dtype=np.float32)
        with self._lock:
            self._buf.append(vec)
            if len(self._buf) > self._window:
                self._buf.pop(0)
            self._score = self._compute_score()

    def _compute_score(self) -> float:
        if self._session is None or self._profile is None or len(self._buf) < 5:
            return 1.0   # fail-open when not enough data
        try:
            seq = np.stack(self._buf[-self._window:], axis=0)[np.newaxis]  # (1, T, 6)
            input_name = self._session.get_inputs()[0].name
            out = self._session.run(None, {input_name: seq})[0][0]  # (embed_dim,)
            norm_out     = out / (np.linalg.norm(out) + 1e-8)
            norm_profile = self._profile / (np.linalg.norm(self._profile) + 1e-8)
            return float(np.clip(np.dot(norm_out, norm_profile), 0.0, 1.0))
        except Exception:
            return 1.0

    def get_score(self) -> ModalityResult:
        with self._lock:
            s = self._score
        confident = len(self._buf) >= 10   # at least 1s of data
        return ModalityResult(score=s, confident=confident)


# ══════════════════════════════════════════════════════════════════════════════
#  Fusion Scorer
# ══════════════════════════════════════════════════════════════════════════════

class FusionScorer:
    """
    DEPRECATED — retired in the DriveAuth migration.

    The Trust Score is now produced by driveauth/trust_fusion.py, which fuses
    ONLY voice/face/finger. Behaviour/location/context feed a separate Risk
    Score (driveauth/risk_model.py), per proposal §4.3. This class is kept as a
    thin, behaviour-free shim purely so any lingering importer keeps working; the
    live DriveAuthGate pipeline does not call it. The ``behavioral`` argument is
    accepted for signature compatibility and ignored.
    """

    @staticmethod
    def fuse(
        voice:    ModalityResult,
        face:     ModalityResult,
        finger:   ModalityResult,
        behavioral: ModalityResult | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Returns (fused_score, effective_weights) over biometrics only.
        ``behavioral`` is ignored — it is no longer a Trust input (§4.3).
        """
        candidates: dict[str, tuple[float, float]] = {}   # name → (score, base_weight)

        def _add(name: str, res: ModalityResult, base_w: float) -> None:
            if res.score is not None and res.confident:
                candidates[name] = (res.score, base_w)

        _add("voice",  voice,  _W_VOICE)
        _add("face",   face,   _W_FACE)
        _add("finger", finger, _W_FINGER)
        # behaviour intentionally NOT added — Risk signal, not Trust (§4.3)

        if not candidates:
            # No modalities available — cannot verify
            return 0.0, {}

        total_w = sum(w for _, w in candidates.values())
        eff: dict[str, float] = {}
        fused = 0.0
        for name, (score, base_w) in candidates.items():
            eff_w = base_w / total_w
            eff[name] = eff_w
            fused += eff_w * score

        return float(fused), eff

    @staticmethod
    def route(score: float) -> str:
        if score >= THRESH_APPROVE:
            return "pass"
        elif score >= THRESH_STEP_UP:
            return "step_up"
        else:
            return "deny"


# ══════════════════════════════════════════════════════════════════════════════
#  Rate limiter (deny + lock)
# ══════════════════════════════════════════════════════════════════════════════

class _RateLimiter:
    """Simple sliding-window deny rate limiter."""

    def __init__(self, window_s: float = 300.0, max_denies: int = 3):
        self._window   = window_s
        self._max      = max_denies
        self._denies:  list[float] = []
        self._locked_until: float  = 0.0
        self._lock     = threading.Lock()

    def record_deny(self) -> bool:
        """Record a deny. Returns True if now rate-limited (locked)."""
        now = time.time()
        with self._lock:
            # Prune old events
            self._denies = [t for t in self._denies if now - t < self._window]
            self._denies.append(now)
            if len(self._denies) >= self._max:
                self._locked_until = now + self._window
                logger.warning(f"BiometricGate: rate-limited until {self._locked_until:.0f}")
                return True
        return False

    def is_locked(self) -> bool:
        with self._lock:
            return time.time() < self._locked_until


# ══════════════════════════════════════════════════════════════════════════════
#  Audit Logger
# ══════════════════════════════════════════════════════════════════════════════

class _AuditLog:
    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, driver_id: str, score: float,
            decision: str, transcript: str) -> None:
        import json as _json
        entry = {
            "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event":      event,
            "driver_id":  driver_id,
            "score":      round(score, 4),
            "decision":   decision,
            "transcript": transcript[:120],
        }
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Gate
# ══════════════════════════════════════════════════════════════════════════════

class BiometricGate:
    """
    Drop-in interceptor for stt_worker.py.

    Typical usage
    -------------
    # At worker startup:
    gate = BiometricGate.load(l3_dir="nova/backend/nova-l7/L-3",
                               store_dir="models/biometric_store",
                               driver_id="driver1")

    # Inside the stt stop/finalize block, replacing the bare llm_in_queue.put():
    routed = gate.intercept(
        transcript   = transcript,
        audio_np     = audio_np,          # float32 PCM from STT session
        ws_out_queue = ws_out_queue,
        llm_in_queue = llm_in_queue,
    )
    # "routed" is "pass", "step_up", or "deny" — for logging only.
    # The gate has already dispatched to llm_in_queue on "pass".
    """

    def __init__(
        self,
        voice:     VoiceVerifier,
        face:      FaceVerifier,
        finger:    FingerVerifier,
        behavioral: BehavioralMonitor,
        driver_id: str,
        audit_log: _AuditLog,
        rate_limiter: _RateLimiter,
        enabled: bool = True,
        passive_only_threshold: float = 0.55,  # below this even passive check blocks
    ):
        self.voice      = voice
        self.face       = face
        self.finger     = finger
        self.behavioral = behavioral
        self.driver_id  = driver_id
        self._audit     = audit_log
        self._rl        = rate_limiter
        self._enabled   = enabled
        self._passive_thresh = passive_only_threshold
        # Pending step-up: stash original transcript while awaiting re-auth
        self._pending: dict[str, Any] | None = None
        self._pending_retries: int = 0

    @classmethod
    def load(
        cls,
        l3_dir:    str | None = None,
        store_dir: str | None = None,
        driver_id: str        = "driver1",
        enabled:   bool       = True,
    ) -> "BiometricGate":
        backend = Path(__file__).parent.parent
        if l3_dir is None:
            l3_dir = str(backend / "nova-l7" / "L-3")
        if store_dir is None:
            store_dir = str(backend / ".." / ".." / "models" / "biometric_store")

        store_path = Path(store_dir)
        logger.info(f"BiometricGate: loading (l3={l3_dir}, store={store_dir}, driver={driver_id})")

        # Ensure store key exists (shared with L-3 crypto key if present)
        key_path = store_path / ".bio_key"
        if not key_path.exists():
            try:
                from cryptography.fernet import Fernet  # type: ignore
                store_path.mkdir(parents=True, exist_ok=True)
                key_path.write_bytes(Fernet.generate_key())
                logger.info("BiometricGate: generated new bio key")
            except Exception as exc:
                logger.warning(f"BiometricGate: key gen failed: {exc}")

        voice  = VoiceVerifier.load(l3_dir, driver_id)
        face   = FaceVerifier.load(store_dir, driver_id)
        finger = FingerVerifier.load(store_dir, driver_id)
        beh    = BehavioralMonitor.load(store_dir, driver_id)

        audit = _AuditLog(store_path / "audit" / "biometric_events.jsonl")
        rl    = _RateLimiter()

        return cls(voice, face, finger, beh, driver_id, audit, rl, enabled)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_behavioral(self, sensor: dict[str, float]) -> None:
        """Feed vehicle sensor data — call from your sensor thread at ~10Hz."""
        self.behavioral.update(sensor)

    def require_auth(self, tier: str = "normal") -> FusionResult:
        """
        Called by llm_worker's _needs_tools() hook when payment intent fires.
        Runs full multi-modal verification synchronously.
        Returns FusionResult — caller decides what to do.
        """
        return self._run_full_auth(audio_np=None, tier=tier)

    def intercept(
        self,
        transcript:   str,
        audio_np:     np.ndarray,
        ws_out_queue: Any,   # multiprocessing.Queue
        llm_in_queue: Any,
    ) -> str:
        """
        Main intercept point — call instead of llm_in_queue.put() in stt_worker.

        Returns "pass", "step_up", or "deny".
        On "pass": dispatches to llm_in_queue internally.
        On "step_up"/"deny": sends TTS reply via ws_out_queue.
        """
        if not self._enabled:
            # Gate disabled — pass everything through
            llm_in_queue.put({"type": "text", "text": transcript,
                               "audio_data": audio_np.tolist()})
            return "pass"

        if self._rl.is_locked():
            self._tts_deny(ws_out_queue,
                "I'm sorry, too many failed verification attempts. "
                "Please wait a few minutes before trying again.")
            return "deny"

        # ── Check if this is a re-auth attempt for a pending step-up ────────
        if self._pending is not None:
            return self._handle_reauth(transcript, audio_np, ws_out_queue, llm_in_queue)

        is_payment = bool(_PAYMENT_RE.search(transcript))

        if is_payment:
            result = self._run_full_auth(audio_np, tier="payment")
        else:
            result = self._run_passive_check(audio_np)

        result.is_payment = is_payment
        decision = result.decision

        self._audit.log(
            event      = "payment_auth" if is_payment else "passive_check",
            driver_id  = self.driver_id,
            score      = result.score,
            decision   = decision,
            transcript = transcript,
        )

        if decision == "pass":
            logger.info(f"BiometricGate: PASS score={result.score:.3f} payment={is_payment}")
            llm_in_queue.put({
                "type":       "text",
                "text":       transcript,
                "audio_data": audio_np.tolist(),
                "bio_score":  result.score,
                "bio_pass":   True,
            })
            return "pass"

        elif decision == "step_up":
            logger.info(f"BiometricGate: STEP_UP score={result.score:.3f} — requesting re-auth")
            self._pending = {
                "transcript": transcript,
                "audio_data": audio_np.tolist(),
                "is_payment": is_payment,
            }
            self._pending_retries = 0
            self._tts_step_up(ws_out_queue, is_payment)
            return "step_up"

        else:  # deny
            logger.warning(f"BiometricGate: DENY score={result.score:.3f}")
            locked = self._rl.record_deny()
            if locked:
                self._tts_deny(ws_out_queue,
                    "I cannot verify your identity. "
                    "For safety, commands are paused for a few minutes.")
                ws_out_queue.put({"type": "security_alert",
                                  "reason": "biometric_deny_rate_limited",
                                  "score": result.score})
            else:
                self._tts_deny(ws_out_queue,
                    "I couldn't verify your identity. "
                    "Please try again or contact support.")
            return "deny"

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _run_passive_check(self, audio_np: np.ndarray) -> FusionResult:
        """
        Voice + behavioral only — fast, every utterance.
        Face and finger are not activated for non-payment requests.
        """
        voice_r = self.voice.score(audio_np)
        beh_r   = self.behavioral.get_score()
        # Skip face/finger for passive
        score, eff = FusionScorer.fuse(
            voice_r, ModalityResult(None, False), ModalityResult(None, False), beh_r)
        # Passive check uses a lower bar — only fully deny on strong signal
        # (the gate is more conservative for active payment auth)
        decision = FusionScorer.route(score)
        return FusionResult(score=score, decision=decision,
                            voice=voice_r, behavioral=beh_r,
                            effective_weights=eff)

    def _run_full_auth(self, audio_np: np.ndarray | None, tier: str = "payment") -> FusionResult:
        """
        Full multi-modal auth: voice + face + finger + behavioral.
        Runs modalities in parallel where possible.
        """
        results: dict[str, ModalityResult] = {}
        threads: list[threading.Thread] = []

        def _voice() -> None:
            if audio_np is not None:
                results["voice"] = self.voice.score(audio_np)
            else:
                results["voice"] = ModalityResult(None, False)

        def _face() -> None:
            results["face"] = self.face.capture_and_score()

        def _finger() -> None:
            results["finger"] = self.finger.capture_and_score()

        def _beh() -> None:
            results["behavior"] = self.behavioral.get_score()

        for fn in (_voice, _face, _finger, _beh):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=6.0)

        voice_r  = results.get("voice",    ModalityResult(None, False))
        face_r   = results.get("face",     ModalityResult(None, False))
        finger_r = results.get("finger",   ModalityResult(None, False))
        beh_r    = results.get("behavior", ModalityResult(None, False))

        score, eff = FusionScorer.fuse(voice_r, face_r, finger_r, beh_r)
        decision = FusionScorer.route(score)

        logger.info(
            f"FullAuth: score={score:.3f} decision={decision} "
            f"v={voice_r.score} f={face_r.score} fp={finger_r.score} b={beh_r.score}"
        )
        return FusionResult(score=score, decision=decision,
                            voice=voice_r, face=face_r,
                            finger=finger_r, behavioral=beh_r,
                            effective_weights=eff)

    def _handle_reauth(
        self,
        transcript:   str,
        audio_np:     np.ndarray,
        ws_out_queue: Any,
        llm_in_queue: Any,
    ) -> str:
        """Handle second (or Nth) attempt after a step-up challenge."""
        assert self._pending is not None
        pending = self._pending
        is_payment = pending.get("is_payment", False)

        result = self._run_full_auth(audio_np, tier="payment" if is_payment else "normal")
        decision = result.decision
        self._pending_retries += 1

        self._audit.log(
            event      = f"reauth_attempt_{self._pending_retries}",
            driver_id  = self.driver_id,
            score      = result.score,
            decision   = decision,
            transcript = transcript,
        )

        if decision == "pass":
            logger.info(f"BiometricGate: REAUTH PASS score={result.score:.3f}")
            self._pending = None
            self._pending_retries = 0
            # Dispatch the ORIGINAL transcript to LLM
            llm_in_queue.put({
                "type":       "text",
                "text":       pending["transcript"],
                "audio_data": pending["audio_data"],
                "bio_score":  result.score,
                "bio_pass":   True,
            })
            return "pass"

        elif self._pending_retries >= MAX_STEP_UP_RETRIES:
            logger.warning(f"BiometricGate: REAUTH MAX RETRIES EXCEEDED")
            self._pending = None
            self._pending_retries = 0
            locked = self._rl.record_deny()
            if locked:
                self._tts_deny(ws_out_queue,
                    "Too many failed verification attempts. Commands paused.")
                ws_out_queue.put({"type": "security_alert",
                                  "reason": "step_up_exhausted",
                                  "score": result.score})
            else:
                self._tts_deny(ws_out_queue,
                    "I still couldn't verify your identity. Request cancelled.")
            return "deny"
        else:
            logger.info(f"BiometricGate: REAUTH STEP_UP retry={self._pending_retries}")
            self._tts_step_up(ws_out_queue, is_payment, retry=True)
            return "step_up"

    def _tts_step_up(self, ws_out_queue: Any, is_payment: bool, retry: bool = False) -> None:
        if is_payment:
            if retry:
                msg = "I still need to verify you. Please say your name clearly, then look at the camera."
            else:
                msg = ("To authorise this payment, I need to verify your identity. "
                       "Please say your name clearly and look at the camera.")
        else:
            if retry:
                msg = "Please try again — speak clearly so I can verify you."
            else:
                msg = "Could you verify it's you? Please speak clearly."

        ws_out_queue.put({"type": "tts_speak", "text": msg})
        ws_out_queue.put({"type": "generation_start"})

    def _tts_deny(self, ws_out_queue: Any, message: str) -> None:
        ws_out_queue.put({"type": "tts_speak", "text": message})
        ws_out_queue.put({"type": "recording_stopped"})


# ══════════════════════════════════════════════════════════════════════════════
#  Enrollment helpers (called by /enroll/* endpoints)
# ══════════════════════════════════════════════════════════════════════════════

def enroll_voice(driver_id: str, audio_np: np.ndarray, l3_dir: str) -> bool:
    """
    Extend existing /enroll/voice-sample — saves ECAPA embedding to HSM.
    This is a thin wrapper around L-3/enroll.py's existing logic.
    """
    try:
        l3_path = Path(l3_dir)
        sys.path.insert(0, str(l3_path))
        from enroll import enroll_voice as _enroll  # type: ignore  (L-3)
        _enroll(driver_id=driver_id, audio_np=audio_np)
        return True
    except Exception as exc:
        logger.error(f"enroll_voice: {exc}")
        return False


def enroll_face(driver_id: str, store_dir: str, num_frames: int = 5) -> bool:
    """
    NEW /enroll/face-sample endpoint.
    Captures `num_frames` IR frames, averages MobileFaceNet embeddings,
    encrypts and saves to biometric_store/faces/{driver_id}.enc
    """
    store = Path(store_dir)
    onnx_path = store / "mobilefacenet_int8.onnx"
    if not onnx_path.exists():
        logger.error(f"enroll_face: model not found at {onnx_path}")
        return False

    try:
        import onnxruntime as ort  # type: ignore
        import cv2  # type: ignore
        from cryptography.fernet import Fernet  # type: ignore

        session  = ort.InferenceSession(str(onnx_path))
        cam_idx  = int(os.getenv("NOVA_IR_CAMERA_INDEX", "0"))
        cap      = cv2.VideoCapture(cam_idx)
        embs: list[np.ndarray] = []

        for i in range(num_frames):
            time.sleep(0.3)
            for _ in range(3):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                continue
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (112, 112))
            blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
            out  = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
            norm = np.linalg.norm(out)
            if norm > 1e-8:
                out /= norm
            embs.append(out)
            logger.info(f"enroll_face: captured frame {i+1}/{num_frames}")

        cap.release()
        if not embs:
            logger.error("enroll_face: no frames captured")
            return False

        mean_emb = np.mean(embs, axis=0).astype(np.float32)
        norm     = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb /= norm

        key_path = store / ".bio_key"
        if not key_path.exists():
            store.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(Fernet.generate_key())

        f        = Fernet(key_path.read_bytes())
        enc_data = f.encrypt(mean_emb.tobytes())
        out_path = store / "faces" / f"{driver_id}.enc"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(enc_data)
        logger.info(f"enroll_face: saved face template for '{driver_id}'")
        return True
    except Exception as exc:
        logger.error(f"enroll_face: {exc}")
        return False


def enroll_fingerprint(driver_id: str, store_dir: str, num_scans: int = 3) -> bool:
    """
    NEW /enroll/fingerprint endpoint.
    Reads `num_scans` finger scans, averages FingerNet minutiae vectors,
    encrypts and saves to biometric_store/fingers/{driver_id}.enc
    """
    store = Path(store_dir)
    onnx_path = store / "fingernet_lite_int8.onnx"
    if not onnx_path.exists():
        logger.error(f"enroll_fingerprint: model not found at {onnx_path}")
        return False

    try:
        import socket as sock
        import onnxruntime as ort  # type: ignore
        from cryptography.fernet import Fernet  # type: ignore

        session  = ort.InferenceSession(str(onnx_path))
        scan_socket = os.getenv("NOVA_FINGER_SOCKET", "/tmp/nova_finger.sock")
        embs: list[np.ndarray] = []

        for i in range(num_scans):
            logger.info(f"enroll_fingerprint: requesting scan {i+1}/{num_scans}")
            s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
            s.settimeout(10.0)
            s.connect(scan_socket)
            s.sendall(b"SCAN\n")
            chunks = []
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            s.close()

            if len(raw) < 256 * 256:
                continue
            img  = np.frombuffer(raw[:256*256], dtype=np.uint8).reshape(1, 1, 256, 256)
            blob = img.astype(np.float32) / 255.0
            out  = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
            embs.append(out)

        if not embs:
            logger.error("enroll_fingerprint: no scans captured")
            return False

        mean_emb = np.mean(embs, axis=0).astype(np.float32)

        key_path = store / ".bio_key"
        if not key_path.exists():
            store.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(Fernet.generate_key())

        f        = Fernet(key_path.read_bytes())
        enc_data = f.encrypt(mean_emb.tobytes())
        out_path = store / "fingers" / f"{driver_id}.enc"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(enc_data)
        logger.info(f"enroll_fingerprint: saved template for '{driver_id}'")
        return True
    except Exception as exc:
        logger.error(f"enroll_fingerprint: {exc}")
        return False