"""
driveauth/step_up_fallback.py
----------------------------
§4.3a — No-signal fallback for step-up.

Invoked when ``step_up_otp.send()`` returns None (registered number unreachable
— tunnel, garage, rural fuel stop: exactly the scenarios DriveAuth Edge exists
for). Rather than block the transaction indefinitely on an OTP that may never
arrive, we do a full on-device re-check: strongest available biometric modality
re-captured, plus a locally-stored PIN.

This keeps the ACCEPT path reachable with radios disabled, which is the whole
point of the offline design. The PIN is stored as a salted hash in the encrypted
biometric store — never in the clear.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path

logger = logging.getLogger("driveauth.fallback")

_PIN_MIN_LEN = int(os.getenv("NOVA_PIN_MIN_LEN", "4"))


class StepUpFallback:
    """
    On-device biometric-recapture + local PIN verifier.

    ``biometric_recheck`` is a callable supplied by the gate that re-runs the
    strongest available modality and returns a fresh Trust Score — we don't
    duplicate the matcher wiring here, we reuse the gate's.
    """

    def __init__(self, store_dir: str, driver_id: str = "driver1"):
        self._store = Path(store_dir)
        self._driver = driver_id
        self._pin_hash, self._pin_salt = self._load_pin()

    def _load_pin(self) -> tuple[bytes | None, bytes | None]:
        pin_path = self._store / "pins" / f"{self._driver}.enc"
        if not pin_path.exists():
            logger.info("Fallback: no local PIN enrolled")
            return None, None
        try:
            from cryptography.fernet import Fernet  # type: ignore
            key_path = self._store / ".bio_key"
            if not key_path.exists():
                return None, None
            f = Fernet(key_path.read_bytes())
            raw = f.decrypt(pin_path.read_bytes())
            # stored as salt(16) || sha256-hmac(32)
            if len(raw) < 48:
                return None, None
            return raw[16:48], raw[:16]
        except Exception as exc:
            logger.error(f"Fallback: PIN load failed ({exc})")
            return None, None

    def verify_pin(self, pin: str) -> bool:
        if self._pin_hash is None or self._pin_salt is None:
            logger.warning("Fallback: PIN check requested but none enrolled")
            return False
        if len(pin) < _PIN_MIN_LEN:
            return False
        digest = hmac.new(self._pin_salt, pin.encode("utf-8"), hashlib.sha256).digest()
        return hmac.compare_digest(digest, self._pin_hash)

    def run(
        self,
        pin: str | None,
        biometric_recheck,          # callable() -> float trust score in [0,1]
        min_trust: float = 0.80,
    ) -> tuple[bool, list[str]]:
        """
        Returns (passed, reason_codes). Requires BOTH a fresh biometric re-check
        above ``min_trust`` AND a valid PIN — two independent factors, since this
        path is standing in for an OTP and should not be weaker than one.
        """
        reasons: list[str] = ["offline_fallback_used"]

        pin_ok = self.verify_pin(pin) if pin is not None else False
        if not pin_ok:
            reasons.append("pin_failed_or_missing")

        try:
            trust = float(biometric_recheck())
        except Exception as exc:
            logger.error(f"Fallback: biometric recheck failed ({exc})")
            trust = 0.0
        bio_ok = trust >= min_trust
        if not bio_ok:
            reasons.append("biometric_recheck_failed")

        passed = pin_ok and bio_ok
        reasons.append("fallback_passed" if passed else "fallback_failed")
        return passed, reasons


def enroll_pin(store_dir: str, driver_id: str, pin: str) -> bool:
    """Helper for the enrollment endpoint — stores a salted-hashed PIN."""
    import secrets
    if len(pin) < _PIN_MIN_LEN:
        logger.error("enroll_pin: PIN too short")
        return False
    try:
        from cryptography.fernet import Fernet  # type: ignore
        store = Path(store_dir)
        key_path = store / ".bio_key"
        if not key_path.exists():
            store.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(Fernet.generate_key())
        salt = secrets.token_bytes(16)
        digest = hmac.new(salt, pin.encode("utf-8"), hashlib.sha256).digest()
        f = Fernet(key_path.read_bytes())
        enc = f.encrypt(salt + digest)
        out = store / "pins" / f"{driver_id}.enc"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(enc)
        logger.info(f"enroll_pin: stored PIN for '{driver_id}'")
        return True
    except Exception as exc:
        logger.error(f"enroll_pin: {exc}")
        return False
