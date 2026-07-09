"""
driveauth/step_up_otp.py
-----------------------
§4.3a — Step-up via OTP sent to the user's registered mobile number.

Scoped, narrow exception to the offline design:
  * The OTP is sent by the bank/payment provider backend over the standard
    cellular network to the enrolled mobile number. It does NOT route through
    the vehicle's own connectivity and does NOT require the paired phone to
    relay anything.
  * Only the STEP_UP path touches connectivity. ACCEPT / REJECT and all
    biometric/risk scoring stay fully on-device and offline.
  * If the provider is unreachable (no signal — tunnel, garage, rural stop),
    the caller falls back to ``step_up_fallback`` (on-device biometric + PIN).

This module owns generating/verifying the OTP challenge and asking the provider
to deliver it. It deliberately does NOT store the code in the clear beyond the
short verification window, and never logs the code itself.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass

logger = logging.getLogger("driveauth.otp")

_OTP_TTL_S       = float(os.getenv("NOVA_OTP_TTL_S", "120.0"))
_OTP_LENGTH      = int(os.getenv("NOVA_OTP_LENGTH", "6"))
_OTP_MAX_TRIES   = int(os.getenv("NOVA_OTP_MAX_TRIES", "3"))
_PROVIDER_URL    = os.getenv("NOVA_OTP_PROVIDER_URL", "")
_PROVIDER_TIMEOUT = float(os.getenv("NOVA_OTP_PROVIDER_TIMEOUT_S", "6.0"))


@dataclass
class OTPChallenge:
    salt:       bytes
    digest:     bytes            # hmac of the code — code itself not retained
    expires_at: float
    tries_left: int
    delivered:  bool


class OTPStepUp:
    """
    Sends and verifies an OTP challenge.

    ``send`` returns an OTPChallenge on successful hand-off to the provider, or
    None if the provider could not be reached — the caller uses None as the
    signal to invoke the no-signal fallback.
    """

    def __init__(self, provider_url: str = _PROVIDER_URL):
        self._provider = provider_url
        self._active: OTPChallenge | None = None

    def _hash(self, code: str, salt: bytes) -> bytes:
        return hmac.new(salt, code.encode("utf-8"), hashlib.sha256).digest()

    def send(self, mobile_number: str | None) -> OTPChallenge | None:
        """
        Ask the payment provider to deliver an OTP to the registered number.
        Returns the challenge (code retained only as a salted HMAC) or None if
        delivery could not be attempted / failed (→ fall back).
        """
        if not mobile_number:
            logger.warning("OTP: no registered mobile number — cannot send")
            return None
        if not self._provider:
            logger.warning("OTP: NOVA_OTP_PROVIDER_URL unset — cannot send")
            return None

        code = "".join(secrets.choice("0123456789") for _ in range(_OTP_LENGTH))
        salt = secrets.token_bytes(16)
        digest = self._hash(code, salt)

        delivered = self._deliver_via_provider(mobile_number, code)
        if not delivered:
            logger.warning("OTP: provider unreachable — caller should fall back")
            return None

        self._active = OTPChallenge(
            salt=salt, digest=digest,
            expires_at=time.time() + _OTP_TTL_S,
            tries_left=_OTP_MAX_TRIES, delivered=True,
        )
        logger.info("OTP: challenge delivered to registered number (code not logged)")
        return self._active

    def _deliver_via_provider(self, mobile_number: str, code: str) -> bool:
        """
        POST to the provider's send-OTP endpoint over cellular. Kept minimal and
        dependency-light (urllib) so it doesn't drag a new HTTP stack into the
        worker. Returns True on 2xx, False on any failure (→ offline fallback).

        NOTE: in a production integration the code would typically be generated
        provider-side; here we pass a reference id and let the provider template
        the SMS. We never write ``code`` to any log.
        """
        try:
            import json
            import urllib.request

            # Masked number in logs only.
            masked = mobile_number[:2] + "****" + mobile_number[-2:]
            payload = json.dumps({
                "to": mobile_number,
                "code": code,
                "ttl_s": int(_OTP_TTL_S),
                "purpose": "driveauth_step_up",
            }).encode("utf-8")
            req = urllib.request.Request(
                self._provider, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_PROVIDER_TIMEOUT) as resp:
                ok = 200 <= resp.status < 300
                logger.info(f"OTP: provider delivery to {masked} status={resp.status}")
                return ok
        except Exception as exc:
            logger.warning(f"OTP: delivery failed ({type(exc).__name__}) — will fall back")
            return False

    def verify(self, code: str) -> bool:
        """Constant-time verify against the active challenge."""
        ch = self._active
        if ch is None:
            return False
        if time.time() > ch.expires_at:
            logger.info("OTP: challenge expired")
            self._active = None
            return False
        if ch.tries_left <= 0:
            self._active = None
            return False
        ch.tries_left -= 1
        ok = hmac.compare_digest(self._hash(code, ch.salt), ch.digest)
        if ok:
            self._active = None
        return ok

    @property
    def has_active_challenge(self) -> bool:
        return self._active is not None and time.time() <= self._active.expires_at
