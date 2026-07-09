# DriveAuth migration — changeset

Implements the Trust/Risk-separation rework described in the review note, on top
of the real GPU pipeline (`pipeline_mp/biometric_gate.py` + `dynamic_orchestrator.py`),
wired in at the two existing call sites (`stt_worker`, `llm_worker`).

The headline fix: **behaviour/location/context no longer enter the Trust Score.**
They drive a separate **Risk Score** (`driveauth/risk_model.py`), exactly as the
proposal's §4.3 requires. OTP step-up (§4.3a) is a bonus, not the main change.

## Files to delete / retire (steps 1–4)

| # | Instruction | Done |
|---|-------------|------|
| 1 | Retire `L-7/mock_order.py` | Moved to `nova-l7/_retired/mock_order.py.retired`. Its only importer (`dialogue_manager.py`) already guards the import with `try/except ImportError`, so removal is non-breaking. |
| 2 | Retire `layer3_main.authorize_payment()` + OTP helpers | Marked **deprecated in-place** (banner docstring + `DeprecationWarning`). Left resolvable because `dialogue_manager.py` imports the symbol directly under an import guard; deleting it would break that guard. New code must route through `DriveAuthGate`. |
| 3 | Remove `_W_VOICE/_W_FACE/_W_FINGER/_W_BEHAVIOR`, `FusionScorer.fuse()/.route()`, and `BehavioralMonitor`'s contribution to the fused score (keep `BehavioralMonitor` itself) | `_W_*` constants marked deprecated (`_W_BEHAVIOR=0.0`); `FusionScorer.fuse()` no longer adds behaviour; the old gate's `_run_passive_check`/`_run_full_auth` still pass a behaviour arg positionally but it is now ignored. `BehavioralMonitor` class kept intact. |
| 4 | Drop `behavior` from `DynamicFusionScorer.fuse_dynamic` and from the weights dict | Done in `dynamic_orchestrator.py`: `_BASE_W` is biometric-only; `PolicyMLP.infer` softmaxes over 3 outputs; `SmolLM2Orchestrator.infer` parses 3 weights; schema no longer requires `behavior`; `fuse_dynamic` drops the behaviour `_add`. |

## New files (steps 5–15) — package `pipeline_mp/driveauth/`

| # | File | Role |
|---|------|------|
| 5 | `__init__.py` | Package surface; re-exports `DriveAuthGate` + types. |
| 6 | `quality_gate.py` | §8a.5 pre-matching quality (SNR/clip/blur/occlusion/contact-area); hard-gates a capture before matching. |
| 7 | `ood_detector.py` | §8a.6 distance-to-enrollment-distribution per modality; feeds Confidence. |
| 8 | `risk_model.py` | **New Risk Score** — GPS/geofence/ignition+speed (CAN)/time/amount/beneficiary-novelty/behaviour → risk, kept fully separate from Trust. GBT/MLP if present, else transparent additive fallback. CPU. |
| 9 | `trust_fusion.py` | `FusionScorer` replacement — fuses **only** voice/face/finger. Reuses the orchestrator's Tier-1/Tier-2 weighting minus behaviour. |
| 10 | `confidence.py` | §4.3 step 6 — OOD flags + quality flags + modality variance → Confidence Score (distinct from Trust). |
| 11 | `policy_engine.py` | §8a.4/§8a.10 — versioned declarative tier rules (micro/standard/high_value/guest); Trust+Risk+Confidence → decision; deterministic, separate from ML. |
| 12 | `fraud_state.py` | §6.2 — Normal→Elevated→Heightened→Locked ladder (replaces the flat 3-strikes `_RateLimiter`). Persisted per driver. |
| 13 | `step_up_otp.py` | §4.3a — OTP to registered mobile over cellular via payment provider (not vehicle connectivity). Code never logged; stored only as salted HMAC. |
| 14 | `step_up_fallback.py` | §4.3a no-signal fallback — on-device biometric recapture + local PIN when the provider is unreachable. |
| 15 | `gate.py` | `DriveAuthGate` — replaces `BiometricGate` with identical `.load()/.intercept()/.require_auth(tier=...)` signatures; returns `ACCEPT/STEP_UP_REQUIRED/REJECT` mapped to the old `pass/step_up/deny` strings. |
|   | `audit_log.py` | (step 18) decoupled `_AuditLog` with the extra §8.3 fields. |
|   | `types.py` | shared dataclasses + `Decision` enum + legacy-compat shims (`.score`, `.legacy_decision`). |

## Wiring (steps 16–19)

| # | Instruction | Done |
|---|-------------|------|
| 16 | `stt_worker.py`: swap `BiometricGate` → `DriveAuthGate` | Import + `.load()` call swapped; same call shape, `intercept()` usage below untouched. |
| 17 | `llm_worker.py`: same swap for the `require_auth(tier="payment")` site | Swapped; uses `.legacy_decision` so the `!= "pass"` branch is unchanged in shape. |
| 18 | Extend `_AuditLog` with active thresholds, fraud-ladder state, Risk/Confidence, OOD flags; optionally move to `driveauth/audit_log.py` | Moved to `driveauth/audit_log.py`; logs all requested fields, excludes raw biometrics/embeddings. |
| 19 | New env vars in `.env.example` | Added: `NOVA_RISK_APPROVE`, `NOVA_RISK_REJECT`, `NOVA_OTP_PROVIDER_URL`, `NOVA_FRAUD_LADDER_DECAY_HOURS`, `NOVA_FINGERPRINT_AVAILABLE` (+ optional `NOVA_DRIVER_MOBILE`). |

## GPU/edge notes (steps 20–23) — honoured

- **20** Verifiers (`Voice/Face/Finger`) loaded exactly as before — they already pick `CUDAExecutionProvider`/`torch.cuda`. `gate.load()` reuses their existing loaders; no matcher changes.
- **21** Tier-1 MLP stays CPU, Tier-2 SmolLM2 stays GPU (`n_gpu_layers=33`) — orchestrator untouched except behaviour removal.
- **22** `risk_model.py` runs on **CPU** (`CPUExecutionProvider`, `intra_op_num_threads=2`) like Tier-1, preserving GPU headroom for STT/LLM/TTS.
- **23** `gate.py` creates **no new CUDA context** — it reuses the already-loaded verifier ONNX/torch sessions and the orchestrator's Tier-2 session rather than instantiating new GPU sessions per call.

## Validation

`py_compile` passes on all 13 new modules + all 5 modified files. A logic-level
smoke test (in the PR description) exercises Trust-is-biometric-only, Risk
monotonicity vs. context, Confidence dropping on disagreement/OOD, the full
policy accept/step-up/reject matrix incl. mandatory high-value OTP, and the full
fraud ladder incl. persistence + reset — all green. Matcher-dependent paths
(voice/face/finger ONNX) need the real models to run and were not exercised here.

## Open wiring hooks (deliberate TODOs)

- `gate.authenticate()` passes `voice_emb/face_emb/finger_emb=None` to the OOD
  detector because the existing verifiers' `capture_and_score()` returns a score,
  not the embedding. To activate OOD fully, have the verifiers also return the
  live embedding and populate the enrollment `ood_stats/*.npz` at enroll time.
- Face/finger quality currently rides on the verifiers' own `confident` flag;
  surfacing the raw frame/contact metric into `QualityGate` would make §8a.5 gate
  those two modalities as strictly as it already gates voice.
