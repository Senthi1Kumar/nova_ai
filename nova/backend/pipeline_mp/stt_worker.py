"""
pipeline_mp/stt_worker.py
-------------------------
RETIRED — this file is not, and never was, a live worker.

`pipeline_mp/__init__.py`'s `run_stt_worker()` dispatches directly to one of the
three real per-backend files based on `NOVA_STT_BACKEND` / `STT_SETTINGS.active`:

    stt_moonshine_worker.py
    stt_kyutai_worker.py
    stt_qwen3_worker.py

Nothing imports `pipeline_mp.stt_worker`. This file previously held a
patch-generator script (BEFORE_CHANGE_*/AFTER_CHANGE_* string pairs meant to be
applied to a target file) that looked like a live worker but wasn't. Its
DriveAuthGate wiring was consequently never in any real request path — see
`_retired/stt_worker_patch_script.py.retired` and `_retired/README.md` for the
full history.

The DriveAuthGate.intercept() call is now spliced directly into the three real
files above, at their actual `llm_in_queue.put(...)` dispatch sites, using the
same `sys.path.insert(pipeline_mp_dir)` + bare-import convention those files
already use for `stt_config` / `dynamic_orchestrator`.

This stub is kept only so the filename doesn't silently disappear from history;
it defines nothing and is never imported.
"""
