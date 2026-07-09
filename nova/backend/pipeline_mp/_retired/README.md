# Retired patch-scripts (critical bug fix)

## What was wrong

`llm_worker.py` and `stt_worker.py` were never live worker code. Their own
docstrings say so ("PATCH: nova/backend/pipeline_mp/llm_worker.py") — they were
patch-generator scripts: `BEFORE_CHANGE_*`/`AFTER_CHANGE_*` string pairs plus an
`if __name__ == "__main__":` block that applies text-replacement patches to a
*target* file when run directly.

Two consequences, one severe:

1. `stt_worker.py` was harmless dead code — nothing imports it.
   `pipeline_mp/__init__.py`'s `run_stt_worker()` dispatches directly to the
   real per-backend files (`stt_moonshine_worker.py`, `stt_kyutai_worker.py`,
   `stt_qwen3_worker.py`), never to `stt_worker.py`.

2. `llm_worker.py` was **not** harmless: `pipeline_mp/__init__.py` does
   `from .llm_worker import run_llm_worker` unconditionally at package import
   time, and the patch-script never defines that function. This raised
   `ImportError: cannot import name 'run_llm_worker' from 'pipeline_mp.llm_worker'`
   the moment anything imported `pipeline_mp` — which `main.py` and
   `gateway_simple.py` both do at the top of the file. **The whole application
   failed to start.**

A second-order consequence: because neither file was ever live, the
`DriveAuthGate` wiring described inside them (added during an earlier
DriveAuth-migration pass) was never actually in the request path. No
transaction was ever gated by Trust/Risk/Policy — the gate existed but was
never invoked by any real worker.

## What replaced them

- `llm_worker.py` is now a real, functioning worker: it loads a local model via
  `llm_config.LOCAL_LLM_REGISTRY`, streams `assistant_start`/`llm_token`/
  `llm_route` events matching the frontend's actual message contract
  (`nova/frontend/index.html`), forwards sentence-chunked text to
  `tts_in_queue` as `text_to_speak`/`eof` (matching `tts_worker.py`'s real
  input protocol), and re-checks payment intents against `DriveAuthGate`
  before treating them as tool calls — the second-layer gate the original
  design called for.
- `DriveAuthGate.intercept()` is now spliced directly into the four real
  dispatch sites across the three real STT backends
  (`stt_qwen3_worker.py`, `stt_moonshine_worker.py` ×2, `stt_kyutai_worker.py`),
  using the same `sys.path.insert(pipeline_mp_dir)` + bare-import convention
  those files already use for `stt_config`/`dynamic_orchestrator`.
- `stt_worker.py` is retired here since nothing ever used it and its presence
  was actively misleading (it looks like the STT worker but isn't).

These two `.retired` files are kept for reference/history, not for use.
