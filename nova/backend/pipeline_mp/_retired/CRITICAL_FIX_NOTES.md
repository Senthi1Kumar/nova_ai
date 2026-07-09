# Critical-fix pass — restoring the app's ability to start, and actually wiring the gate

This addresses the three bugs surfaced during the full-pipeline architecture
review, in priority order.

## 1. `pipeline_mp/llm_worker.py` — the app couldn't start

**Before:** the file was a patch-generator script (its own docstring said so —
"PATCH: nova/backend/pipeline_mp/llm_worker.py") that never defined
`run_llm_worker`. `pipeline_mp/__init__.py` does `from .llm_worker import
run_llm_worker` unconditionally, so importing `pipeline_mp` raised:

```
ImportError: cannot import name 'run_llm_worker' from 'pipeline_mp.llm_worker'
```

`main.py` and `gateway_simple.py` both import `pipeline_mp` at the top of the
file, so the whole app failed before FastAPI even started.

**After:** `llm_worker.py` is a real worker. Verified fix:

```
from pipeline_mp import run_stt_worker, run_kws_worker, run_llm_worker, run_tts_worker, run_pvad_worker
# -> SUCCESS (previously: ImportError)
```

It loads a local model via the existing `llm_config.LOCAL_LLM_REGISTRY`,
streams `assistant_start` / `llm_token` / `llm_route` events matching the real
frontend contract (`nova/frontend/index.html`'s actual `case` handlers — not
guessed), and forwards sentence-chunked text to `tts_in_queue` as
`text_to_speak` / `eof`, matching `tts_worker.py`'s real input protocol (also
read directly from source, not assumed).

It also re-checks payment intents against `DriveAuthGate.require_auth()` at the
tool-use boundary — the second-layer gate the original (broken) patch-script
described but that was never live.

The old patch-script is preserved at
`pipeline_mp/_retired/llm_worker_patch_script.py.retired`.

## 2. The DriveAuth gate was orphaned — now spliced into the real dispatch sites

**Before:** none of the three real STT backends (`stt_qwen3_worker.py`,
`stt_moonshine_worker.py`, `stt_kyutai_worker.py`) called any gate. Each simply
did a bare `llm_in_queue.put(payload)`. `stt_worker.py`, which *did* contain the
gate-wiring instructions, was dead code — `pipeline_mp/__init__.py` dispatches
directly to the three backend files, never to `stt_worker.py`.

**After:** `DriveAuthGate.load()` is initialized once near the top of each
real worker (same place `stt_config` is imported), and `DriveAuthGate.intercept()`
is spliced into all four real dispatch sites:

| File | Dispatch site(s) |
|---|---|
| `stt_qwen3_worker.py` | `_emit_final()` |
| `stt_moonshine_worker.py` | main transcript-event handler; `external_eos` held-line flush |
| `stt_kyutai_worker.py` | `_emit_final`-equivalent inference-thread finalize |

Each site: if the gate loaded, `bio_gate.intercept(...)` handles dispatch to
`llm_in_queue` and sends `generation_start` itself (matching the gate's own
contract); if the gate failed to load (e.g. no biometric models on a dev
machine), falls back to the original bare dispatch so STT/chat still works —
logged loudly (`logger.error(...)`) so the missing gate isn't silent. This
fail-open-on-load-failure choice matches the convention the original (broken)
patch-script itself used ("failing open") — worth revisiting if you want a
stricter fail-closed policy for payment intents specifically.

`stt_worker.py` is now a documented stub (not a fake worker) pointing at the
three real files. The original is archived at
`pipeline_mp/_retired/stt_worker_patch_script.py.retired`.

## 3. Stale `from pipeline.X` imports — fixed everywhere

`from pipeline.biometric_gate import ...` / `from pipeline.dynamic_orchestrator
import ...` never resolved to anything real — there is no module or package
named `pipeline` anywhere in this repo. The actual codebase uses two
conventions depending on caller context:

- **From outside `pipeline_mp`** (`main.py`, `gateway_simple.py`): package-
  qualified, e.g. `from pipeline_mp.llm_config import ...`.
- **From inside a worker** (`stt_*_worker.py`, now `llm_worker.py`): each
  worker does `sys.path.insert(0, pipeline_mp_dir)` then bare imports, e.g.
  `from stt_config import ...`.

Fixed in:
- `driveauth/gate.py` — `DriveAuthGate.load()`'s imports of `VoiceVerifier`
  et al. and `DynamicOrchestrator` now try a relative import first
  (`from ..biometric_gate import ...`), falling back to the bare style
  (`from biometric_gate import ...`) — dual-mode, since `gate.py` itself gets
  imported both ways depending on the caller.
- `dynamic_orchestrator.py`'s deprecated `DynamicFusionScorer.route()` — same
  dual-mode fix, plus a docstring clarifying it's dead code superseded by
  `driveauth/policy_engine.py`.

Verified both import conventions resolve correctly post-fix (tested directly,
not just read).

## Validation performed

- `py_compile` on every `.py` file in `pipeline_mp/` (excluding `_retired/`) — clean.
- Actual `from pipeline_mp import run_stt_worker, run_kws_worker, run_llm_worker, run_tts_worker, run_pvad_worker` — succeeds (previously raised `ImportError`).
- Actual import of `driveauth.gate.DriveAuthGate` under both calling conventions — succeeds.
- Re-ran the DriveAuth logic smoke test (Trust fusion biometrics-only, Risk model, Policy engine accept/step-up/reject) — all still pass, confirming no regression from the import fixes.

## Still open (not in scope of this pass, flagged for visibility)

- `layer3_main.authorize_payment()` remains deprecated-in-place (not deleted) —
  two authorization code paths still coexist in the repo, distinguished only by
  a comment. See the earlier `nova-l7/_retired/README.md`.
- The fail-open-on-gate-load-failure choice in the STT workers matches the
  original design's own convention but is a real product decision — worth
  revisiting if payment intents should fail *closed* instead when the gate
  can't load at all.
- `run_llm_worker`'s tool-execution framework is intentionally minimal (a
  payment-intent check plus local text generation) since the original tool
  dispatch logic (search, calendar, etc. per earlier context) was never present
  in this repo snapshot to restore — it's a functioning reconstruction, not a
  recovered original.
