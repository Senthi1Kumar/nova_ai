"""
driveauth/audit_log.py
----------------------
§8.3 — Audit logging, decoupled from the old ``biometric_gate._AuditLog``.

Each entry carries more than a bare accept/reject so disputes and the daily
Bluetooth-sync review have enough to reconstruct WHY a decision was made:

  * per-modality scores AND quality flags (which modality drove/weakened it)
  * the thresholds active at decision time (they change over time)
  * Trust / Risk / Confidence scores and OOD flags
  * fraud-ladder state, tier, policy rule, timestamp, vehicle-context snapshot

Explicitly excluded: raw biometric images/audio or embedding vectors. The log
is diagnostic metadata, never reconstructable biometric data.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from .types import DriveAuthResult

logger = logging.getLogger("driveauth.audit")


class AuditLog:
    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_decision(
        self,
        *,
        event: str,
        driver_id: str,
        result: DriveAuthResult,
        transcript: str = "",
    ) -> None:
        entry = {
            "ts":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event":         event,
            "driver_id":     driver_id,
            "decision":      result.decision.value,
            "tier":          result.tier,
            "trust_score":   round(result.trust_score, 4),
            "risk_score":    round(result.risk_score, 4),
            "confidence":    round(result.confidence_score, 4),
            "fraud_state":   result.fraud_state,
            "policy_rule":   result.policy_rule,
            "step_up_method": result.step_up_method,
            "modality_scores": result.modality_scores,
            "active_thresholds": result.active_thresholds,
            "ood_flags":     result.ood_flags,
            "explanations":  result.explanations,
            "is_payment":    result.is_payment,
            # transcript is truncated and kept for dispute context only
            "transcript":    transcript[:120],
        }
        line = json.dumps(entry)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        logger.debug(f"audit: {event} decision={result.decision.value} "
                     f"trust={result.trust_score:.2f} risk={result.risk_score:.2f}")
