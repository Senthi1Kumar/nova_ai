"""Skill/system prompts surfaced via `tools_v2.add_skill`.

These prompts are NOT used by the main litert_lm_chat_app voice loop — that
flow uses `litert_system_prompt` from `app.config.Settings` (the Nova persona).

This module exists so `tools_v2.py` (a separate FastMCP server you may run
standalone for telematics workflows) is importable without crashing. Fill in
TELEMATICS_ASSISTANT_SYSTEM_MESSAGE with your real telematics-skill prompt
when you set that server up; the stub below is a placeholder.
"""
from __future__ import annotations

TELEMATICS_ASSISTANT_SYSTEM_MESSAGE = (
    "You are a telematics assistant. You can execute read-only SQL against the "
    "tc_position_attributes table to answer questions about vehicle fleet data: "
    "positions, speeds, fuel level, events, geofence breaches. Keep replies "
    "concise; round numbers; never invent device IDs."
)
