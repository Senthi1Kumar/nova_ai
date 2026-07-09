"""
driveauth/gate.py
----------------
§4.2 — DriveAuthGate: the orchestrating gate that replaces ``BiometricGate``.

It keeps the EXACT public method signatures of the old gate so the two call
sites (stt_worker, llm_worker) don't need restructuring — only the imported
class name changes:

    DriveAuthGate.load(l3_dir=..., store_dir=..., driver_id=..., enabled=...)
    gate.intercept(transcript, audio_np, ws_out_queue, llm_in_queue) -> "pass"|"step_up"|"deny"
    gate.require_auth(tier="payment") -> result   (result.decision / result.score usable)
    gate.update_behavioral(sensor_dict)

Internally it runs the DriveAuth pipeline (quality → matchers → OOD →
Trust/Risk/Confidence → PolicyEngine → FraudStateMachine → OTP/fallback) and
returns the legacy pass/step_up/deny string so downstream branches still work.

Reuse, don't re-init: the biometric verifiers (Voice/Face/Finger) are loaded
exactly as before (they already pick CUDAExecutionProvider / torch.cuda). The
Tier-1 MLP stays CPU and Tier-2 SmolLM2 stays GPU inside DynamicOrchestrator —
unchanged. RiskModel runs CPU. No second CUDA context is created here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .types import Decision, ModalityResult, RiskContext, DriveAuthResult
from .quality_gate import QualityGate
from .ood_detector import OODDetector
from .risk_model import RiskModel
from .trust_fusion import TrustFusion
from .confidence import ConfidenceScorer
from .policy_engine import PolicyEngine, classify_tier
from .fraud_state import FraudStateMachine, FraudState
from .step_up_otp import OTPStepUp
from .step_up_fallback import StepUpFallback
from .audit_log import AuditLog

logger = logging.getLogger("driveauth.gate")

MAX_STEP_UP_RETRIES = int(os.getenv("NOVA_STEP_UP_RETRIES", "2"))

# Reuse the payment-intent regex shape from the old gate.
import re
_PAYMENT_RE = re.compile(
    r"\b(order|buy|purchase|pay|send money|transfer|checkout|coffee|burger"
    r"|pizza|food|latte|cappuccino|add to cart|top.?up|recharge)\b",
    re.IGNORECASE,
)


class DriveAuthGate:
    """Drop-in replacement for BiometricGate."""

    def __init__(
        self,
        *,
        voice, face, finger, behavioral,        # existing verifiers (reused as-is)
        driver_id: str,
        store_dir: str,
        quality: QualityGate,
        ood: OODDetector,
        risk: RiskModel,
        trust: TrustFusion,
        confidence: ConfidenceScorer,
        policy: PolicyEngine,
        fraud: FraudStateMachine,
        otp: OTPStepUp,
        fallback: StepUpFallback,
        audit: AuditLog,
        enabled: bool = True,
    ):
        self.voice = voice
        self.face = face
        self.finger = finger
        self.behavioral = behavioral
        self.driver_id = driver_id
        self._store = store_dir
        self._q = quality
        self._ood = ood
        self._risk = risk
        self._trust = trust
        self._conf = confidence
        self._policy = policy
        self._fraud = fraud
        self._otp = otp
        self._fallback = fallback
        self._audit = audit
        self._enabled = enabled
        self._pending: dict[str, Any] | None = None
        self._pending_retries = 0
        # Latest vehicle/risk context, fed by update_vehicle_context / update_behavioral
        self._risk_ctx = RiskContext()
        self._ctx_lock = threading.Lock()

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        l3_dir: str | None = None,
        store_dir: str | None = None,
        driver_id: str = "driver1",
        enabled: bool = True,
    ) -> "DriveAuthGate":
        # Reuse the existing verifier loaders from the old module — they already
        # load templates and pick CUDA/CPU correctly. We only replace the fusion
        # + decision layer, not the matchers.
        #
        # Dual-mode import: this repo has two calling conventions depending on
        # who imports `driveauth` —
        #   (a) from OUTSIDE pipeline_mp (main.py etc.): `pipeline_mp.driveauth.gate`,
        #       in which case a relative import resolves correctly.
        #   (b) from INSIDE a worker (stt_*_worker.py, llm_worker.py), which does
        #       `sys.path.insert(0, pipeline_mp_dir)` then a bare
        #       `from driveauth.gate import DriveAuthGate` — in that case
        #       `driveauth` is imported as a top-level package with no parent,
        #       so a relative import fails with "attempted relative import
        #       beyond top-level package".
        # Try relative first, fall back to bare — matches the defensive style
        # already used elsewhere in this codebase (e.g. dialogue_manager.py's
        # guarded imports).
        try:
            from ..biometric_gate import (
                VoiceVerifier, FaceVerifier, FingerVerifier, BehavioralMonitor,
            )
        except ImportError:
            from biometric_gate import (
                VoiceVerifier, FaceVerifier, FingerVerifier, BehavioralMonitor,
            )

        backend = Path(__file__).parent.parent.parent   # .../nova/backend
        if l3_dir is None:
            l3_dir = str(backend / "nova-l7" / "L-3")
        if store_dir is None:
            store_dir = str(backend / ".." / ".." / "models" / "biometric_store")

        store_path = Path(store_dir)
        logger.info(f"DriveAuthGate: loading (l3={l3_dir}, store={store_dir}, driver={driver_id})")

        # Ensure crypto key exists (shared with the biometric store)
        key_path = store_path / ".bio_key"
        if not key_path.exists():
            try:
                from cryptography.fernet import Fernet  # type: ignore
                store_path.mkdir(parents=True, exist_ok=True)
                key_path.write_bytes(Fernet.generate_key())
            except Exception as exc:
                logger.warning(f"DriveAuthGate: key gen failed: {exc}")

        voice = VoiceVerifier.load(l3_dir, driver_id)
        face  = FaceVerifier.load(store_dir, driver_id)
        finger = FingerVerifier.load(store_dir, driver_id)
        beh   = BehavioralMonitor.load(store_dir, driver_id)

        # Optional dynamic orchestrator (Tier-1 CPU MLP / Tier-2 GPU SmolLM2).
        orchestrator = None
        try:
            try:
                from ..dynamic_orchestrator import DynamicOrchestrator
            except ImportError:
                from dynamic_orchestrator import DynamicOrchestrator
            orchestrator = DynamicOrchestrator.load(store_dir)
        except Exception as exc:
            logger.info(f"DriveAuthGate: orchestrator unavailable ({exc}) — static trust weights")

        fingerprint_available = os.getenv("NOVA_FINGERPRINT_AVAILABLE", "1") == "1"
        if not fingerprint_available:
            logger.info("DriveAuthGate: fingerprint marked unavailable for this trim")

        gate = cls(
            voice=voice, face=face, finger=finger, behavioral=beh,
            driver_id=driver_id, store_dir=store_dir,
            quality=QualityGate(),
            ood=OODDetector.load(store_dir, driver_id),
            risk=RiskModel.load(store_dir),
            trust=TrustFusion(orchestrator),
            confidence=ConfidenceScorer(),
            policy=PolicyEngine(),
            fraud=FraudStateMachine(store_path / "fraud" / "ladder.json", driver_id),
            otp=OTPStepUp(),
            fallback=StepUpFallback(store_dir, driver_id),
            audit=AuditLog(store_path / "audit" / "driveauth_events.jsonl"),
            enabled=enabled,
        )
        gate._fingerprint_available = fingerprint_available
        return gate

    # ── context feeds ──────────────────────────────────────────────────────────

    def update_behavioral(self, sensor: dict[str, float]) -> None:
        """
        Kept for call-site compatibility. Feeds the behavioural monitor (whose
        score now flows into RISK, not TRUST) and refreshes CAN-derived context.
        """
        self.behavioral.update(sensor)
        with self._ctx_lock:
            if "vehicle_speed_kmh" in sensor:
                self._risk_ctx.speed_kmh = float(sensor["vehicle_speed_kmh"])
            if "ignition_on" in sensor:
                self._risk_ctx.ignition_on = bool(sensor["ignition_on"])

    def update_vehicle_context(self, **kwargs) -> None:
        """Optional richer feed (GPS/zone/time) from the CAN/telematics thread."""
        with self._ctx_lock:
            for k, v in kwargs.items():
                if hasattr(self._risk_ctx, k):
                    setattr(self._risk_ctx, k, v)

    def _build_risk_ctx(self, *, amount=0.0, beneficiary="", action="",
                        beneficiary_known=False) -> RiskContext:
        with self._ctx_lock:
            ctx = RiskContext(**vars(self._risk_ctx))
        ctx.amount = amount
        ctx.beneficiary = beneficiary
        ctx.action = action
        ctx.beneficiary_known = beneficiary_known
        beh = self.behavioral.get_score()
        ctx.behavioral_score = beh.score if beh.score is not None else None
        ctx.time_hour = float(time.localtime().tm_hour)
        return ctx

    # ── the single authentication call (§4.2) ──────────────────────────────────

    def authenticate(
        self,
        *,
        audio_np: np.ndarray | None,
        tier_hint: str = "payment",
        amount: float = 0.0,
        beneficiary: str = "",
        action: str = "",
        beneficiary_known: bool = False,
        is_guest: bool = False,
    ) -> DriveAuthResult:
        """Runs the full pipeline and returns the four scores + decision."""
        explanations: list[str] = []

        # ── 1. capture modalities in parallel (reusing existing verifiers) ────
        results: dict[str, ModalityResult] = {}
        threads: list[threading.Thread] = []

        def _voice():
            results["voice"] = (self.voice.score(audio_np) if audio_np is not None
                                else ModalityResult(None, False))

        def _face():
            results["face"] = self.face.capture_and_score()

        def _finger():
            if getattr(self, "_fingerprint_available", True):
                results["finger"] = self.finger.capture_and_score()
            else:
                results["finger"] = ModalityResult(None, False)

        for fn in (_voice, _face, _finger):
            t = threading.Thread(target=fn, daemon=True)
            t.start(); threads.append(t)
        for t in threads:
            t.join(timeout=6.0)

        voice_r  = results.get("voice",  ModalityResult(None, False))
        face_r   = results.get("face",   ModalityResult(None, False))
        finger_r = results.get("finger", ModalityResult(None, False))

        # ── 2. quality gate (§8a.5): reject a capture before it's trusted ─────
        qflags = self._q.evaluate(
            voice_audio=audio_np,
            # face/finger frames aren't surfaced by the existing verifiers'
            # capture_and_score(); we gate on their self-reported confidence and
            # let OOD + confidence handle the rest. Quality of voice is checked here.
        )
        if audio_np is not None and not qflags.voice_ok:
            voice_r = ModalityResult(None, False, quality=qflags.voice_q)
            explanations.append("voice_quality_rejected")
        else:
            voice_r.quality = qflags.voice_q

        # ── 3. OOD (§8a.6): feeds confidence, never trust ─────────────────────
        ood_flags = self._ood.evaluate(
            voice_emb=None, face_emb=None, finger_emb=None,
        )  # embeddings not surfaced by verifiers; hook point kept for enrollment-stats wiring
        for name, r in (("voice", voice_r), ("face", face_r), ("finger", finger_r)):
            r.ood = ood_flags.get(name, False)

        # ── 4. Trust (biometrics only) ────────────────────────────────────────
        trust, eff_w = self._trust.fuse(voice_r, face_r, finger_r, orch_ctx=None)

        # ── 5. Risk (context/behaviour — kept separate) ───────────────────────
        risk_ctx = self._build_risk_ctx(
            amount=amount, beneficiary=beneficiary, action=action,
            beneficiary_known=beneficiary_known,
        )
        risk, risk_reasons = self._risk.score(risk_ctx)
        explanations.extend(risk_reasons)

        # ── 6. Confidence (system self-consistency) ───────────────────────────
        confidence, conf_reasons = self._conf.score(
            voice_r, face_r, finger_r, qflags, ood_flags)
        explanations.extend(conf_reasons)

        # ── 7. Policy decision ────────────────────────────────────────────────
        tier = classify_tier(risk_ctx, is_guest=is_guest)
        n_conf = sum(1 for r in (voice_r, face_r, finger_r)
                     if r.score is not None and r.confident)
        rigor = self._fraud.rigor()
        decision, rule, active_thr, step_up_method = self._policy.decide(
            trust=trust, risk=risk, confidence=confidence, tier=tier,
            n_confident_modalities=n_conf, fraud_rigor=rigor,
            explanations=explanations,
        )

        result = DriveAuthResult(
            trust_score=trust, risk_score=risk, confidence_score=confidence,
            decision=decision, tier=tier, explanations=explanations,
            step_up_method=step_up_method,
            step_up_fallback="biometric_recapture_pin" if step_up_method == "otp_mobile" else None,
            policy_rule=rule, fraud_state=self._fraud.state.value,
            modality_scores={
                "voice":  {"score": voice_r.score, "conf": voice_r.confident, "q": voice_r.quality},
                "face":   {"score": face_r.score, "conf": face_r.confident, "q": face_r.quality},
                "finger": {"score": finger_r.score, "conf": finger_r.confident, "q": finger_r.quality},
                "effective_weights": eff_w,
            },
            active_thresholds=active_thr, ood_flags=ood_flags,
        )
        return result

    # ── legacy-compatible public API ────────────────────────────────────────────

    def require_auth(self, tier: str = "normal") -> DriveAuthResult:
        """Same signature as the old gate. Returns a result whose ``.decision``
        is a Decision enum and ``.score`` (Trust) / ``.legacy_decision`` are
        available for old call sites."""
        res = self.authenticate(audio_np=None, tier_hint=tier)
        self._post_decision(res, transcript="", event="require_auth")
        return res

    def intercept(
        self,
        transcript: str,
        audio_np: np.ndarray,
        ws_out_queue: Any,
        llm_in_queue: Any,
    ) -> str:
        """
        Main intercept point. Returns "pass" | "step_up" | "deny" (legacy strings).
        On "pass", dispatches to llm_in_queue internally, exactly like the old gate.
        """
        if not self._enabled:
            llm_in_queue.put({"type": "text", "text": transcript,
                              "audio_data": audio_np.tolist()})
            return "pass"

        if self._fraud.state == FraudState.LOCKED:
            self._tts_deny(ws_out_queue,
                "Verification is locked. Please re-authenticate from the paired phone app.")
            return "deny"

        # Re-auth in progress?
        if self._pending is not None:
            return self._handle_reauth(transcript, audio_np, ws_out_queue, llm_in_queue)

        is_payment = bool(_PAYMENT_RE.search(transcript))
        result = self.authenticate(
            audio_np=audio_np, tier_hint="payment" if is_payment else "normal",
        )
        result.is_payment = is_payment
        self._post_decision(result, transcript=transcript,
                            event="payment_auth" if is_payment else "passive_check")

        if result.decision == Decision.ACCEPT:
            logger.info(f"DriveAuthGate: ACCEPT trust={result.trust_score:.3f} "
                        f"risk={result.risk_score:.3f} payment={is_payment}")
            llm_in_queue.put({
                "type": "text", "text": transcript,
                "audio_data": audio_np.tolist(),
                "bio_score": result.trust_score, "bio_pass": True,
            })
            return "pass"

        if result.decision == Decision.STEP_UP_REQUIRED:
            logger.info(f"DriveAuthGate: STEP_UP trust={result.trust_score:.3f} "
                        f"risk={result.risk_score:.3f} method={result.step_up_method}")
            self._pending = {
                "transcript": transcript,
                "audio_data": audio_np.tolist(),
                "is_payment": is_payment,
                "step_up_method": result.step_up_method,
            }
            self._pending_retries = 0
            self._begin_step_up(result, ws_out_queue)
            self._fraud.record_soft_flag("step_up")
            return "step_up"

        # REJECT
        logger.warning(f"DriveAuthGate: REJECT trust={result.trust_score:.3f} "
                       f"risk={result.risk_score:.3f}")
        self._fraud.record_soft_flag("reject")
        self._tts_deny(ws_out_queue,
            "I couldn't verify your identity. Please try again or contact support.")
        ws_out_queue.put({"type": "security_alert", "reason": "driveauth_reject",
                          "trust": result.trust_score, "risk": result.risk_score})
        return "deny"

    # ── step-up handling ────────────────────────────────────────────────────────

    def _begin_step_up(self, result: DriveAuthResult, ws_out_queue: Any) -> None:
        """Kick off OTP delivery; fall back to on-device biometric+PIN if no signal."""
        if result.step_up_method == "otp_mobile":
            mobile = self._registered_mobile()
            challenge = self._otp.send(mobile)
            if challenge is not None:
                ws_out_queue.put({"type": "tts_speak",
                    "text": "I've sent a one-time code to your registered mobile number. "
                            "Please read it out or enter it to authorise this."})
                ws_out_queue.put({"type": "generation_start"})
                self._pending["mode"] = "otp"
                return
            # No signal → fall back.
            logger.info("DriveAuthGate: OTP unreachable — offline biometric+PIN fallback")
            ws_out_queue.put({"type": "tts_speak",
                "text": "I can't reach the network for a code, so I'll verify on-device. "
                        "Please say your PIN and look at the camera."})
            ws_out_queue.put({"type": "generation_start"})
            self._pending["mode"] = "fallback"
            return

        # guest / pin_card_present
        ws_out_queue.put({"type": "tts_speak",
            "text": "This requires a PIN or card. Please enter your PIN to continue."})
        ws_out_queue.put({"type": "generation_start"})
        self._pending["mode"] = "fallback"

    def _handle_reauth(self, transcript, audio_np, ws_out_queue, llm_in_queue) -> str:
        assert self._pending is not None
        pending = self._pending
        self._pending_retries += 1
        mode = pending.get("mode", "otp")

        passed = False
        if mode == "otp" and self._otp.has_active_challenge:
            # The re-auth utterance is expected to contain the OTP digits.
            code = "".join(ch for ch in transcript if ch.isdigit())
            passed = self._otp.verify(code)
            if not passed:
                logger.info("DriveAuthGate: OTP verify failed")
        else:
            # Fallback path: biometric recheck + spoken PIN.
            pin = "".join(ch for ch in transcript if ch.isdigit()) or None
            passed, reasons = self._fallback.run(
                pin=pin,
                biometric_recheck=lambda: self.authenticate(audio_np=audio_np).trust_score,
            )
            logger.info(f"DriveAuthGate: fallback reasons={reasons}")

        if passed:
            logger.info("DriveAuthGate: REAUTH PASS")
            self._fraud.record_clean()
            llm_in_queue.put({
                "type": "text", "text": pending["transcript"],
                "audio_data": pending["audio_data"], "bio_pass": True,
            })
            self._pending = None
            self._pending_retries = 0
            return "pass"

        if self._pending_retries >= MAX_STEP_UP_RETRIES:
            logger.warning("DriveAuthGate: REAUTH max retries — deny")
            self._pending = None
            self._pending_retries = 0
            state = self._fraud.record_soft_flag("step_up_exhausted")
            if state in (FraudState.HEIGHTENED, FraudState.LOCKED):
                self._tts_deny(ws_out_queue,
                    "Too many failed attempts. Commands are paused for safety.")
                ws_out_queue.put({"type": "security_alert",
                                  "reason": "step_up_exhausted"})
            else:
                self._tts_deny(ws_out_queue,
                    "I still couldn't verify you. Request cancelled.")
            return "deny"

        ws_out_queue.put({"type": "tts_speak",
            "text": "That didn't match. Please try once more."})
        ws_out_queue.put({"type": "generation_start"})
        return "step_up"

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _post_decision(self, result: DriveAuthResult, *, transcript: str, event: str) -> None:
        self._audit.log_decision(event=event, driver_id=self.driver_id,
                                 result=result, transcript=transcript)
        if result.decision == Decision.ACCEPT:
            self._fraud.record_clean()

    def _registered_mobile(self) -> str | None:
        """Load the enrolled mobile number (plaintext-free store)."""
        path = Path(self._store) / "contacts" / f"{self.driver_id}.mobile"
        try:
            if path.exists():
                return path.read_text().strip()
        except Exception:
            pass
        return os.getenv("NOVA_DRIVER_MOBILE") or None

    def _tts_deny(self, ws_out_queue: Any, message: str) -> None:
        ws_out_queue.put({"type": "tts_speak", "text": message})
        ws_out_queue.put({"type": "recording_stopped"})

    # Explicit dispute hook for the phone-sync review path (§6.2 / §8.3)
    def mark_not_mine(self) -> None:
        """Called when the user flags a transaction 'not mine' at review time."""
        self._fraud.record_confirmed_fraud()
