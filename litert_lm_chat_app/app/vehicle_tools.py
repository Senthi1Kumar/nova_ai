"""In-vehicle assistant (IVA) tools — the toolbox Nova was fine-tuned on.

Nova v0.2 was trained against the toolbox produced by
`training/data_pipeline/intent_schema.build_toolbox(intents)` over
`config/intents_v0.2.yaml`. If we don't hand the model that SAME toolbox at
runtime, it correctly refuses every in-cabin request ("I can't control your
car") because, from its point of view, no such tool exists. This module rebuilds
that exact toolbox and wraps each entry as a LiteRT-LM `Tool` so the schema the
model sees is byte-for-byte identical to training (names, arg names, `action`
enums, descriptions) — not a lossy re-derivation from a Python signature.

Execution is stubbed: these return a success payload rather than actuating a real
vehicle. Wiring to CAN-bus / HAL is out of scope for the demo; what matters here
is that the model emits a well-formed call and receives a result it can confirm.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from litert_lm.interfaces import Tool, ToolEventHandler

log = logging.getLogger("litert_app.vehicle_tools")


class ToolCallCapture(ToolEventHandler):
    """Records tool calls so the app can recover a spoken confirmation.

    Nova was trained to emit the confirmation text AND the tool call in one
    assistant turn. But litert_lm's `automatic_tool_calling` classifies any
    chunk containing a tool call as "tool call only" and never yields its text
    (conversation.py) — so the confirmation is swallowed and the voice path goes
    silent. This handler observes each approved call (it fires just before the
    tool executes) so `_run_turn` can, when the model streams no text, speak a
    synthesized confirmation instead. `calls` is reset per turn by the caller.
    """

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        fn = tool_call.get("function", {}) or {}
        self.calls.append({"name": fn.get("name", ""),
                           "args": fn.get("arguments", {}) or {}})
        return True

    def process_tool_response(self, tool_response: dict[str, Any]) -> dict[str, Any]:
        return tool_response


def synthesize_confirmation(name: str, args: Mapping[str, Any]) -> str:
    """Build a short natural-language confirmation for a fire-and-confirm IVA
    call, used only when the model itself streamed no text. Deterministic and
    schema-agnostic: humanize the `action` plus its salient argument values."""
    def _num(v: Any) -> Any:
        # Speak whole-number floats as ints ("20" not "20.0").
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    args = {k: _num(v) for k, v in dict(args).items()}
    action = str(args.get("action", "")).replace("_", " ").strip()
    # Values worth speaking back; drop bookkeeping keys.
    vals = {k: v for k, v in args.items() if k not in ("action", "zone") and v not in (None, "")}
    # A couple of hand-tuned phrasings for the most common cabin actions.
    if args.get("action") == "set_temperature":
        temp = args.get("target_temperature", args.get("value", args.get("amount", "")))
        zone = args.get("zone")
        z = "" if not zone or zone == "all" else f" for the {zone} zone"
        return f"Setting the temperature to {temp} degrees{z}."
    if action and vals:
        detail = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in vals.items())
        return f"Okay — {action}: {detail}."
    if action:
        return f"Okay, {action}."
    return "Okay, done."

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INTENTS_YAML = _REPO_ROOT / "training" / "data_pipeline" / "config" / "intents_v0.2.yaml"
_PIPELINE_DIR = _REPO_ROOT / "training" / "data_pipeline"


class SchemaTool(Tool):
    """A LiteRT-LM Tool whose description is a fixed OpenAPI schema dict.

    Unlike `tool_from_function` (which introspects a Python signature), this
    passes a pre-built schema through unchanged — required so the toolbox the
    model sees matches the fine-tuning data exactly. `execute` runs a stub
    handler that reports success; swap in real actuation later.
    """

    def __init__(self, schema: dict[str, Any]):
        self._schema = schema
        self._name = schema["function"]["name"]

    def get_tool_description(self) -> dict[str, Any]:
        return self._schema

    def execute(self, param: Mapping[str, Any]) -> Any:
        # Simulated CAN/OBD: persistence-backed tools read/write the SQLite
        # vehicle store (app/vehicle_db.py) so state survives across turns
        # and vehicle_query returns what vehicle_command set. Tools without
        # a handler keep the original acknowledge-only stub.
        args = dict(param)
        from app.vehicle_db import execute_tool
        result = execute_tool(self._name, args)
        if result is not None:
            log.info("IVA tool executed (db): %s(%s)", self._name, args)
            return {"tool": self._name, **result}
        log.info("IVA tool executed (stub): %s(%s)", self._name, args)
        return {"status": "success", "tool": self._name, "applied": args}


def build_vehicle_tools(intents_path: Path | None = None) -> list[Tool]:
    """Load the v0.2 intents and return the trained toolbox as LiteRT Tools.

    Returns [] (with a warning) if the intents file or the pipeline's
    `intent_schema` module can't be loaded, so a missing training tree degrades
    to "web tools only" rather than crashing engine start.
    """
    path = intents_path or _INTENTS_YAML
    if not path.exists():
        log.warning("IVA intents not found at %s — vehicle tools disabled", path)
        return []

    # `intent_schema` lives in the (non-package) pipeline dir; make it importable.
    if str(_PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(_PIPELINE_DIR))
    try:
        import yaml  # PyYAML is a pipeline dep
        import intent_schema  # training/data_pipeline/intent_schema.py
    except Exception:
        log.exception("could not import intent_schema/yaml — vehicle tools disabled")
        return []

    try:
        intents = yaml.safe_load(path.read_text())["intents"]
        toolbox = intent_schema.build_toolbox(intents)  # exact training schema
    except Exception:
        log.exception("failed to build toolbox from %s — vehicle tools disabled", path)
        return []

    tools = [SchemaTool(t) for t in toolbox]
    log.info("IVA toolbox loaded: %d tools (%s)",
             len(tools), ", ".join(t._name for t in tools))
    return tools
