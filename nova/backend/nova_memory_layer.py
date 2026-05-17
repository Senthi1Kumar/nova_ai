"""Persistent memory layer for Nova.

Three cooperating pieces, all in this file:

  - NovaMemoryLayer  — public class (instantiated once at startup, shared across
    Sessions). Wraps mem0 for semantic add/recall + owns the SessionJournal.
  - SessionJournal   — in-memory list of turn records for the active WS session;
    flushed to session_logs/*.json on close.
  - DiaryCompactor   — async function called from close_session(); summarises
    the session JSON via the local LLM and appends to daily_logs/YYYY-MM-DD.md.

Fail-open: if mem0 fails to import or initialize, the layer becomes a no-op
shell that still writes session JSON + daily diary entries (just no semantic
recall). Nova's existing MEMORY.md path keeps working regardless.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nova-memory")

# Disable mem0's posthog telemetry before the import — env var must be set first.
os.environ.setdefault("MEM0_TELEMETRY", "False")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class SessionJournal:
    """In-memory record of one WS session's turns. Flushed to disk on close."""

    def __init__(self, user_id: str, session_logs_dir: Path) -> None:
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.started_at = _utcnow_iso()
        self.ended_at: Optional[str] = None
        self.turns: list[dict] = []
        self._dir = session_logs_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, role: str, text: str, *, ts: Optional[str] = None,
            tool_calls: Optional[list] = None,
            metrics: Optional[dict] = None) -> None:
        rec: dict[str, Any] = {"t": ts or _utcnow_iso(), "role": role, "text": text}
        if tool_calls:
            rec["tool_calls"] = tool_calls
        if metrics:
            rec["metrics"] = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
        self.turns.append(rec)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at or _utcnow_iso(),
            "n_turns": len(self.turns),
            "turns": self.turns,
        }

    def flush(self) -> Path:
        """Write the journal JSON to session_logs/. Returns the written path."""
        self.ended_at = _utcnow_iso()
        ts_for_name = self.started_at.replace(":", "").replace("-", "")[:15]
        path = self._dir / f"session_{self.session_id}_{ts_for_name}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        return path


class NovaMemoryLayer:
    """Owner of the active SessionJournal + thin wrapper around mem0.

    All public methods are safe to call even if mem0 failed to initialize —
    they degrade to journal-only / no-op behavior.
    """

    def __init__(self, user_id: str = "driver") -> None:
        # `user_id` defaults so the no-arg `NovaMemoryLayer()` form keeps
        # working for legacy callers; nova_loop.py passes it explicitly.
        self.user_id = user_id
        self.disabled = os.getenv("NOVA_MEM0_DISABLED", "0") == "1"

        backend = Path(__file__).parent
        self.session_logs_dir = backend / "session_logs"
        self.daily_logs_dir = Path(os.getenv("NOVA_DAILY_LOG_DIR", str(backend / "daily_logs")))
        self.daily_logs_dir.mkdir(parents=True, exist_ok=True)

        self.journal = SessionJournal(user_id=user_id, session_logs_dir=self.session_logs_dir)
        self._mem: Any = None
        self._mem_ready = False

        if self.disabled:
            logger.info("NovaMemoryLayer: mem0 disabled via NOVA_MEM0_DISABLED=1 — journal-only mode")
        else:
            self._init_mem0()

        status = "on" if self._mem_ready else "off (journal-only)"
        logger.info(f"NovaMemoryLayer: ready (user_id={user_id}, mem0={status})")

    def start_new_session(self) -> None:
        """Reset the SessionJournal for a fresh WS connection. mem0 client is
        kept warm (it's a shared, expensive singleton)."""
        self.journal = SessionJournal(user_id=self.user_id, session_logs_dir=self.session_logs_dir)

    def _init_mem0(self) -> None:
        # Read these BEFORE doing anything to env or importing mem0.
        llm_model = os.getenv("NOVA_MEM0_LLM_MODEL") or os.getenv("NOVA_LLM_MODEL", "unsloth/gemma-4-E4B-it")
        llm_base_url = os.getenv("NOVA_MEM0_LLM_BASE_URL") or os.getenv("NOVA_LLM_BASE_URL", "http://localhost:8080/v1")
        embed_model = os.getenv("NOVA_MEM0_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Scrub any env that mem0's openai LLM wrapper might auto-detect
        # (it ignores config.openai_base_url in v2 and reads env instead;
        # also auto-uses OPENROUTER_API_KEY when present). Save the user's
        # original OPENROUTER_API_KEY only — Nova's own LLM class doesn't
        # touch OPENAI_* env vars (it builds its own httpx requests).
        self._saved_openrouter = os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "local-dummy"
        os.environ["OPENAI_BASE_URL"] = llm_base_url

        # Import mem0 AFTER env is fixed up so any module-level client init
        # inside mem0 sees our local llama-server config.
        try:
            from mem0 import Memory  # type: ignore
        except Exception as e:
            logger.warning(f"mem0 import failed: {e} — falling back to journal-only")
            return

        # ── Fully self-hosted / on-device config ─────────────────────────
        # LLM         → local llama.cpp via OpenAI-compatible /v1 (no cloud call)
        # Embedder    → HuggingFace sentence-transformers, loaded locally
        # Vector store → local Postgres with pgvector extension
        # mem0's "openai" provider name is misleading — it just speaks the
        # OpenAI HTTP wire format. We point it at our llama-server.

        # Postgres connection — defaults match the NOVA_DB_URL in .env.example.
        # Override via NOVA_MEM0_PG_* env vars (or NOVA_DB_URL parsing later).
        pg = {
            "dbname":   os.getenv("NOVA_MEM0_PG_DB",   "nova_db"),
            "user":     os.getenv("NOVA_MEM0_PG_USER", "nova"),
            "password": os.getenv("NOVA_MEM0_PG_PASSWORD", "nova_dev"),
            "host":     os.getenv("NOVA_MEM0_PG_HOST", "localhost"),
            "port":     int(os.getenv("NOVA_MEM0_PG_PORT", "5432")),
            "collection_name": os.getenv("NOVA_MEM0_PG_TABLE", "nova_memories"),
            # all-MiniLM-L6-v2 is 384-dim; keep this in sync if the embedder changes.
            "embedding_model_dims": int(os.getenv("NOVA_MEM0_EMBED_DIMS", "384")),
            "diskann": False,  # use ivfflat / hnsw via pgvector defaults
            "hnsw": True,
        }

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "openai_base_url": llm_base_url,   # local llama.cpp
                    "api_key": "local-dummy",
                    "temperature": 0.1,
                    "max_tokens": 256,
                },
            },
            "embedder": {"provider": "huggingface", "config": {"model": embed_model}},
            "vector_store": {"provider": "pgvector", "config": pg},
        }
        logger.info(
            f"mem0: llm={llm_model}@{llm_base_url} (local), "
            f"embedder={embed_model} (local), "
            f"store=pgvector@{pg['host']}:{pg['port']}/{pg['dbname']}.{pg['collection_name']}"
        )
        try:
            self._mem = Memory.from_config(config)
            self._mem_ready = True
        except Exception as e:
            logger.warning(f"mem0 init failed: {e} — falling back to journal-only "
                           f"(check pgvector extension is installed; see NOVA_LOOP.md)")
            self._mem = None
            self._mem_ready = False

    # ── legacy compat shims ────────────────────────────────────────────────
    # Earlier prototypes used a different surface (`.enabled` + `.search_block`).
    # Keep these aliases so any out-of-tree caller still works.

    @property
    def enabled(self) -> bool:
        """True when mem0 is loaded and ready (legacy alias for _mem_ready)."""
        return bool(self._mem_ready and not self.disabled)

    def search_block(self, query: str, k: int = 5) -> str:
        """Return recalled memories as a single newline-joined block. Older
        callers expected a string they could splice into a system prompt;
        new code should prefer .recall() which returns a list.
        """
        lines = self.recall(query, k=k)
        return "\n".join(f"- {ln}" for ln in lines) if lines else ""

    # ── public per-turn API ─────────────────────────────────────────────────

    def add_turn(self, role: str, text: str, *,
                 ts: Optional[str] = None,
                 tool_calls: Optional[list] = None,
                 metrics: Optional[dict] = None) -> None:
        """Append a turn to the journal. mem0 ingestion is fire-and-forget elsewhere."""
        if not text:
            return
        self.journal.add(role, text, ts=ts, tool_calls=tool_calls, metrics=metrics)

    def ingest_turn_pair(self, user_text: str, assistant_text: str) -> None:
        """Synchronously hand a complete user/assistant exchange to mem0 for
        fact extraction. Caller should run this on an executor — it makes one
        non-streaming LLM call internally and takes ~300-800 ms.
        """
        if not (self._mem_ready and self._mem) or not user_text:
            return
        try:
            msgs = [{"role": "user", "content": user_text}]
            if assistant_text:
                msgs.append({"role": "assistant", "content": assistant_text})
            self._mem.add(msgs, user_id=self.user_id)
        except Exception as e:
            logger.warning(f"mem0.add failed (non-fatal): {e}")

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Return up to k short bullet strings semantically related to `query`."""
        if not (self._mem_ready and self._mem) or not query:
            return []
        # mem0 v2 deprecated `user_id=` on .search() — must use filters={}.
        # Older versions accepted both, so try the new API first and fall back.
        for kwargs in (
            {"query": query, "filters": {"user_id": self.user_id}, "limit": k},
            {"query": query, "user_id": self.user_id, "limit": k},
        ):
            try:
                res = self._mem.search(**kwargs)
                items = res.get("results", res) if isinstance(res, dict) else res
                lines: list[str] = []
                for item in items[:k]:
                    memo = item.get("memory") if isinstance(item, dict) else None
                    if memo:
                        lines.append(str(memo).strip())
                return lines
            except TypeError:
                continue  # API mismatch — try the next kwargs shape
            except Exception as e:
                logger.warning(f"mem0.search failed (non-fatal): {e}")
                return []
        return []

    # ── session lifecycle ──────────────────────────────────────────────────

    def session_summary_json(self) -> dict:
        return self.journal.to_dict()

    async def close_session(self, llm_oneshot=None) -> None:
        """Flush the journal to disk and append a diary entry for today.

        `llm_oneshot` is an optional callable (prompt:str, max_tokens:int) -> str
        used to summarize the session into the diary. If absent or it raises,
        a deterministic fallback summary is written.
        """
        try:
            journal_path = self.journal.flush()
            logger.info(f"session journal flushed: {journal_path.name} ({len(self.journal.turns)} turns)")
        except Exception as e:
            logger.warning(f"session flush failed: {e}")
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._compact_to_diary, llm_oneshot)
        except Exception as e:
            logger.warning(f"diary compaction failed: {e}")

    def _compact_to_diary(self, llm_oneshot) -> None:
        snapshot = self.journal.to_dict()
        if not snapshot["turns"]:
            return

        # Build a compact transcript for the summarizer (cap at ~3000 chars so it
        # always fits in the LLM context regardless of session length).
        lines = []
        for t in snapshot["turns"]:
            who = "Driver" if t["role"] == "user" else "Nova"
            lines.append(f"- {who}: {t['text']}")
        transcript = "\n".join(lines)[-3000:]

        summary: str = ""
        if callable(llm_oneshot):
            try:
                raw = llm_oneshot(
                    "Summarise the following voice-agent session in 3-5 short bullet "
                    "points. Focus on durable facts the driver shared (location, "
                    "preferences, decisions made), topics discussed, and any actions "
                    "Nova took (web searches, refusals). Output bullets only, no "
                    "preamble.\n\n" + transcript,
                    max_tokens=240,
                )
                summary = str(raw) if raw else ""
            except Exception as e:
                logger.warning(f"LLM diary summary failed: {e}")

        summary_text: str = str(summary)
        if not summary_text.strip():
            # Deterministic fallback so the diary always gets *something* useful.
            summary_text = self._fallback_summary(snapshot)

        diary_path = self.daily_logs_dir / f"{_today_str()}.md"
        diary_path.parent.mkdir(parents=True, exist_ok=True)
        started = snapshot["started_at"]
        ended = snapshot["ended_at"]
        section = (
            f"\n## Session {started[11:16]} → {ended[11:16]} UTC  (`{snapshot['session_id'][:8]}`)\n"
            f"_{snapshot['n_turns']} turns · user={snapshot['user_id']}_\n\n"
            f"{summary_text.strip()}\n"
        )
        if not diary_path.exists():
            diary_path.write_text(f"# {_today_str()}\n{section}")
        else:
            with diary_path.open("a") as f:
                f.write(section)
        logger.info(f"diary appended: {diary_path.name}")

    def _fallback_summary(self, snapshot: dict) -> str:
        n_user = sum(1 for t in snapshot["turns"] if t["role"] == "user")
        n_tool = sum(1 for t in snapshot["turns"] if t.get("tool_calls"))
        first_user = next((t["text"] for t in snapshot["turns"] if t["role"] == "user"), "")
        return (
            f"- {n_user} user turn(s), {n_tool} tool invocation(s).\n"
            f"- Opening user message: \"{first_user[:120]}\"\n"
            f"- (LLM summarizer unavailable — see "
            f"session_logs/session_{snapshot['session_id']}_*.json for full transcript.)"
        )
