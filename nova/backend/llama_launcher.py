"""Spawn and shut down a llama-server child process from a YAML preset.

Preset selection (in order):
  1. NOVA_LLAMA_CONFIG  — absolute path to a YAML file
  2. NOVA_LLAMA_PRESET  — preset name under nova/backend/configs/llama/<name>.yaml
  3. default            — "gemma-4-e4b"

Disable auto-launch entirely with NOVA_DISABLE_LLAMA=1.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "configs" / "llama"
_DEFAULT_PRESET = "gemma-4-e4b"


def _resolve_config_path() -> Optional[Path]:
    """Pick the active preset YAML.

    Resolution order:
      1. NOVA_LLAMA_CONFIG (absolute path) wins if set.
      2. <preset>.yaml under configs/llama/ — host-specific copy (gitignored).
      3. <preset>.yaml.example — committed template; used only if no .yaml exists.
    """
    explicit = os.environ.get("NOVA_LLAMA_CONFIG")
    if explicit:
        return Path(explicit)
    preset = os.environ.get("NOVA_LLAMA_PRESET", _DEFAULT_PRESET)
    local = _CONFIG_DIR / f"{preset}.yaml"
    if local.is_file():
        return local
    example = _CONFIG_DIR / f"{preset}.yaml.example"
    if example.is_file():
        logger.warning(
            "Using committed template %s — copy it to %s and edit cwd/binary for this host",
            example.name, local.name,
        )
        return example
    return local  # caller will warn & skip when missing


def spawn() -> Optional["subprocess.Popen[bytes]"]:
    """Launch llama-server per the active preset. Returns None if skipped."""
    if os.environ.get("NOVA_DISABLE_LLAMA", "0").lower() in ("1", "true", "yes"):
        logger.info("NOVA_DISABLE_LLAMA set — skipping llama-server auto-launch")
        return None

    cfg_path = _resolve_config_path()
    if cfg_path is None or not cfg_path.is_file():
        logger.warning("llama preset not found: %s — skipping auto-launch", cfg_path)
        return None

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    cwd = cfg.get("cwd")
    binary = cfg.get("binary", "./build/bin/llama-server")
    args = [str(a) for a in (cfg.get("args") or [])]
    if not cwd or not Path(cwd).is_dir():
        logger.warning("llama-server cwd missing/invalid in %s: %r — skipping", cfg_path, cwd)
        return None
    resolved = binary if os.path.isabs(binary) else str(Path(cwd) / binary)
    if not (os.path.isfile(resolved) and os.access(resolved, os.X_OK)):
        logger.warning("llama-server binary not executable: %s — skipping", resolved)
        return None

    cmd = [binary, *args]
    logger.info("Launching llama-server [%s]: %s (cwd=%s)", cfg_path.name, " ".join(cmd), cwd)
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        start_new_session=True,  # own process group; we control its lifecycle
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def shutdown(proc: Optional["subprocess.Popen[bytes]"]) -> None:
    """Terminate the llama-server process group, escalating to SIGKILL if needed."""
    if proc is None or proc.poll() is not None:
        return
    logger.info("Stopping llama-server (pid=%s)…", proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("llama-server did not exit in 10s; sending SIGKILL")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("llama-server still alive after SIGKILL")
