"""Persistent memory layer for the LiteRT chat app.

Three cooperating pieces, all in this file:

  - MemoryLayer    — public class (instantiated once at startup, shared across
    sessions). Wraps mem0 for semantic add/recall + owns the SessionJournal.
  - SessionJournal — in-memory list of turn records for the active session;
    flushed to runtime/session_logs/*.json on close.
  - _compact_to_diary — LLM-summarises the session into 3-5 bullets and
    appends to runtime/daily_logs/YYYY-MM-DD.md.

Fail-open: if mem0 fails to import or initialize (Postgres down, pgvector
missing, OpenAI shim unreachable), the layer becomes a no-op shell that
still writes session JSON + daily diary entries. The chat-app continues
fully functional, just without semantic recall.

Adapted from Nova's `nova/backend/nova_memory_layer.py` — same shape, paths
relocated under `runtime/` so the chat app stays self-contained.
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

logger = logging.getLogger("litert_app.memory")

# Disable mem0's posthog telemetry before any mem0 import — env-var must be set first.
os.environ.setdefault("MEM0_TELEMETRY", "False")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── SessionJournal ─────────────────────────────────────────────────────────

class SessionJournal:
    """In-memory record of one session's turns. Flushed to disk on close."""

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


# ── MemoryLayer (owns journal + mem0) ──────────────────────────────────────

class MemoryLayer:
    """Owner of the active SessionJournal + thin wrapper around mem0.

    Safe to call all public methods even when mem0 is unavailable — they
    degrade to journal-only / no-op behaviour.
    """

    def __init__(
        self,
        user_id: str = "user",
        runtime_dir: Optional[Path] = None,
    ) -> None:
        # Read the disabled flag via Settings (which loads .env) — falling
        # back to os.getenv for shell-only overrides. pydantic-settings does
        # NOT push .env values into os.environ, so reading os alone misses
        # anything that lives only in .env. Same pattern as Tavily/Mapbox keys.
        from app.config import get_settings
        cfg = get_settings()
        env_disabled = os.getenv("NOVA_MEM0_DISABLED", "").strip().strip('"').strip("'")
        if env_disabled:
            self.disabled = env_disabled == "1"
        else:
            self.disabled = bool(getattr(cfg, "nova_mem0_disabled", False))

        self.user_id = user_id

        runtime_dir = runtime_dir or Path(__file__).resolve().parent.parent / "runtime"
        self.session_logs_dir = runtime_dir / "session_logs"
        self.daily_logs_dir = Path(
            os.getenv("NOVA_DAILY_LOG_DIR", str(runtime_dir / "daily_logs"))
        )
        self.daily_logs_dir.mkdir(parents=True, exist_ok=True)

        self.journal = SessionJournal(user_id=user_id, session_logs_dir=self.session_logs_dir)
        self._mem: Any = None
        self._mem_ready = False

        if self.disabled:
            logger.info("MemoryLayer: mem0 disabled via NOVA_MEM0_DISABLED=1 — journal-only mode")
        else:
            self._init_mem0()

        status = "on" if self._mem_ready else "off (journal + diary recall)"
        logger.info("MemoryLayer ready (user_id=%s, mem0=%s)", user_id, status)

    def start_new_session(self) -> None:
        """Reset the SessionJournal for a fresh session. mem0 client stays warm."""
        self.journal = SessionJournal(user_id=self.user_id, session_logs_dir=self.session_logs_dir)

    # ── mem0 init ──────────────────────────────────────────────────────────

    def _init_mem0(self) -> None:
        llm_base_url = os.getenv("NOVA_MEM0_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        llm_model    = os.getenv("NOVA_MEM0_LLM_MODEL", "litertlm-local")
        embed_model  = os.getenv("NOVA_MEM0_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Scrub env that mem0's openai LLM wrapper might auto-detect.
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "local-dummy"
        os.environ["OPENAI_BASE_URL"] = llm_base_url

        try:
            from mem0 import Memory  # type: ignore
        except Exception as exc:
            logger.warning("mem0 import failed: %s — journal-only mode", exc)
            return

        pg = {
            "dbname":   os.getenv("NOVA_MEM0_PG_DB",   "litert_chat_db"),
            "user":     os.getenv("NOVA_MEM0_PG_USER", "nova"),
            "password": os.getenv("NOVA_MEM0_PG_PASSWORD", "nova_dev"),
            "host":     os.getenv("NOVA_MEM0_PG_HOST", "localhost"),
            "port":     int(os.getenv("NOVA_MEM0_PG_PORT", "5432")),
            "collection_name":       os.getenv("NOVA_MEM0_PG_TABLE", "litert_chat_memories"),
            "embedding_model_dims":  int(os.getenv("NOVA_MEM0_EMBED_DIMS", "384")),
            "diskann": False,
            "hnsw": True,
        }

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "openai_base_url": llm_base_url,
                    "api_key": "local-dummy",
                    "temperature": 0.1,
                    "max_tokens": 256,
                },
            },
            # Pin embedder to CPU — GPU is reserved for LiteRT-LM's Gemma-4
            # (4 GB VRAM is fully booked). all-MiniLM-L6-v2 on CPU embeds in
            # ~10-30 ms, negligible compared to mem0's LLM call.
            "embedder": {"provider": "huggingface",
                         "config": {"model": embed_model,
                                    "model_kwargs": {"device": "cpu"}}},
            "vector_store": {"provider": "pgvector", "config": pg},
        }
        logger.info(
            "mem0: llm=%s@%s, embedder=%s, store=pgvector@%s:%s/%s.%s",
            llm_model, llm_base_url, embed_model,
            pg["host"], pg["port"], pg["dbname"], pg["collection_name"],
        )
        try:
            self._mem = Memory.from_config(config)
            self._mem_ready = True
        except Exception as exc:
            logger.warning(
                "mem0 init failed: %s — journal-only mode (check pgvector + LLM shim)", exc
            )
            self._mem = None
            self._mem_ready = False

    # ── per-turn API ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self._mem_ready and not self.disabled)

    def add_turn(self, role: str, text: str, *,
                 ts: Optional[str] = None,
                 tool_calls: Optional[list] = None,
                 metrics: Optional[dict] = None) -> None:
        """Append a turn to the session journal. Fast/sync."""
        if not text:
            return
        self.journal.add(role, text, ts=ts, tool_calls=tool_calls, metrics=metrics)

    def ingest_turn_pair(self, user_text: str, assistant_text: str) -> None:
        """Hand a complete user/assistant exchange to mem0 for fact extraction.

        Internally makes one non-streaming LLM call via the OpenAI shim
        (~300-800 ms). Caller should run this on an executor / background task
        so it doesn't block the next user turn.
        """
        if not self.enabled or not user_text:
            return
        try:
            msgs = [{"role": "user", "content": user_text}]
            if assistant_text:
                msgs.append({"role": "assistant", "content": assistant_text})
            self._mem.add(msgs, user_id=self.user_id)
        except Exception as exc:
            logger.warning("mem0.add failed (non-fatal): %s", exc)

    def batch_ingest_session(self) -> int:
        """Hand every user/assistant turn-pair in the current journal to mem0
        in one batch. Caller MUST ensure the LiteRT-LM engine has no active
        chat conversation (it only supports one conversation at a time, and
        the OpenAI shim spins up its own throwaway one per call).

        Returns the number of pairs successfully ingested.
        """
        if not self.enabled or not self.journal.turns:
            return 0
        pairs: list[tuple[str, str]] = []
        last_user: Optional[str] = None
        for t in self.journal.turns:
            if t["role"] == "user":
                last_user = t["text"]
            elif t["role"] == "assistant" and last_user:
                pairs.append((last_user, t["text"]))
                last_user = None
        ok = 0
        for u, a in pairs:
            try:
                self.ingest_turn_pair(u, a)
                ok += 1
            except Exception as exc:
                logger.warning("batch ingest pair failed (non-fatal): %s", exc)
        logger.info("mem0 batch-ingested %d/%d turn pairs from session %s",
                    ok, len(pairs), self.journal.session_id[:8])
        return ok

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Return up to k short bullet strings semantically related to `query`."""
        if not self.enabled or not query:
            return []
        # mem0 v2 deprecated `user_id=` on .search() — use filters={}. Older
        # versions accepted both; try the new API first and fall back.
        for kwargs in (
            {"query": query, "filters": {"user_id": self.user_id}, "limit": k},
            {"query": query, "user_id": self.user_id, "limit": k},
        ):
            try:
                res = self._mem.search(**kwargs)
                items = res.get("results", res) if isinstance(res, dict) else res
                out: list[str] = []
                for item in items[:k]:
                    memo = item.get("memory") if isinstance(item, dict) else None
                    if memo:
                        out.append(str(memo).strip())
                return out
            except TypeError:
                continue
            except Exception as exc:
                logger.warning("mem0.search failed (non-fatal): %s", exc)
                return []
        return []

    def recall_block(self, query: str, k: int = 5) -> str:
        """Newline-joined `- ...` block ready to splice into a system prompt."""
        lines = self.recall(query, k=k)
        return "\n".join(f"- {ln}" for ln in lines) if lines else ""

    def recall_recent_diary(self, days: int = 3, max_chars: int = 1600) -> str:
        """Read the last `days` diary files and return a single text block
        suitable for splicing into a system prompt. Sessions within each day
        are ordered NEWEST FIRST (the diary file appends chronologically, so
        we reverse on read). The combined body is truncated from the END so
        the freshest sessions across all days survive truncation."""
        from datetime import date, timedelta
        import re
        today = date.today()
        chunks: list[str] = []
        for delta in range(days):
            d = today - timedelta(days=delta)
            path = self.daily_logs_dir / f"{d.isoformat()}.md"
            if not path.exists():
                continue
            try:
                text = path.read_text().strip()
            except Exception as exc:
                logger.warning("diary read failed for %s: %s", path.name, exc)
                continue
            if not text:
                continue
            # Split on session headers and reverse so newest session in the
            # day comes first. Header line is "## Session HH:MM → HH:MM UTC".
            parts = re.split(r"(?m)^(?=## Session )", text)
            day_header = parts[0].strip()        # e.g. "# 2026-06-05"
            sessions = [p.strip() for p in parts[1:] if p.strip()]
            sessions.reverse()
            day_body = day_header + "\n\n" + "\n\n".join(sessions) if day_header else "\n\n".join(sessions)
            chunks.append(day_body)
        if not chunks:
            return ""
        body = "\n\n".join(chunks)
        if len(body) > max_chars:
            body = body[:max_chars].rsplit("\n", 1)[0] + "\n…(older entries truncated)"
        return body

    # ── session lifecycle ──────────────────────────────────────────────────

    def session_summary_json(self) -> dict:
        return self.journal.to_dict()

    async def close_session(self, llm_oneshot=None) -> None:
        """Flush the journal and append today's diary entry."""
        try:
            journal_path = self.journal.flush()
            logger.info("session journal flushed: %s (%d turns)",
                        journal_path.name, len(self.journal.turns))
        except Exception as exc:
            logger.warning("session flush failed: %s", exc)
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._compact_to_diary, llm_oneshot)
        except Exception as exc:
            logger.warning("diary compaction failed: %s", exc)

    def _compact_to_diary(self, llm_oneshot) -> None:
        snapshot = self.journal.to_dict()
        if not snapshot["turns"]:
            return

        lines = []
        for t in snapshot["turns"]:
            who = "User" if t["role"] == "user" else "Nova"
            lines.append(f"- {who}: {t['text']}")
            # Render any tool calls attached to this turn so the LLM
            # summariser can populate the TOOLS: section accurately.
            # Without this it sees only user/assistant text and reports
            # "no tools used" even on tool-heavy sessions.
            for ev in (t.get("tool_calls") or []):
                kind = ev.get("kind", "")
                name = ev.get("name", "")
                if kind == "tool_call":
                    args_repr = str(ev.get("args", ""))[:180]
                    lines.append(f"  - tool_call: {name}({args_repr})")
                elif kind == "tool_result":
                    summary = (ev.get("summary") or "")[:180]
                    ok = ev.get("ok", True)
                    lines.append(f"  - tool_result: {name} ok={ok} → {summary}")
                else:
                    # Forward-compat for legacy/alternate event shapes.
                    lines.append(f"  - tool: {name} {str(ev)[:160]}")
        transcript = "\n".join(lines)[-3000:]

        summary: str = ""
        if callable(llm_oneshot):
            try:
                raw = llm_oneshot(
                    "You are writing Nova's persistent memory of one voice session "
                    "with this user. This memory will be read back to Nova in future "
                    "sessions so she can recall what she knows about the user.\n\n"
                    "Output four labeled sections, in this exact order, NO preamble, "
                    "NO closing line, NO mention of Nova's actions or refusals:\n"
                    "USER: who the user is + stable preferences they revealed "
                    "(one line; omit entirely if nothing new this session)\n"
                    "FACTS: concrete things established this session — names, "
                    "numbers, decisions, dates, items the user owns or chose. "
                    "Bullets. Omit section if empty.\n"
                    "TOOLS: tool calls + one-line takeaway from each result "
                    "(bullets; omit section if no tools were used)\n"
                    "OPEN: the user's current goal + any unanswered question "
                    "(one line; omit if session was casual chit-chat)\n\n"
                    "Rules: NEVER write 'Nova said', 'Nova stated', 'Nova "
                    "refused' — those describe the assistant's behavior, not "
                    "the user. Write facts as standalone statements the future-"
                    "Nova can act on. If the user shared NOTHING worth "
                    "remembering, output the single line: NOTHING NOTABLE.\n\n"
                    "Transcript:\n" + transcript,
                    300,
                )
                summary = str(raw) if raw else ""
            except Exception as exc:
                logger.warning("LLM diary summary failed: %s", exc)

        summary_text = summary.strip() or self._fallback_summary(snapshot)
        # Skip writing entries that contain no user-relevant content — they
        # otherwise pollute PRIOR DAYS recall with self-referential noise like
        # "Nova said it had no memory", which reinforces denial behavior.
        if summary_text.strip().upper().startswith("NOTHING NOTABLE"):
            logger.info("diary skip: session had no notable content (session=%s)",
                        snapshot['session_id'][:8])
            return
        diary_path = self.daily_logs_dir / f"{_today_str()}.md"
        diary_path.parent.mkdir(parents=True, exist_ok=True)
        started, ended = snapshot["started_at"], snapshot["ended_at"]
        section = (
            f"\n## Session {started[11:16]} → {ended[11:16]} UTC  "
            f"(`{snapshot['session_id'][:8]}`)\n"
            f"_{snapshot['n_turns']} turns · user={snapshot['user_id']}_\n\n"
            f"{summary_text.strip()}\n"
        )
        if not diary_path.exists():
            diary_path.write_text(f"# {_today_str()}\n{section}")
        else:
            with diary_path.open("a") as f:
                f.write(section)
        logger.info("diary appended: %s", diary_path.name)

    def _fallback_summary(self, snapshot: dict) -> str:
        n_user = sum(1 for t in snapshot["turns"] if t["role"] == "user")
        n_tool = sum(1 for t in snapshot["turns"] if t.get("tool_calls"))
        first_user = next((t["text"] for t in snapshot["turns"] if t["role"] == "user"), "")
        return (
            f"- {n_user} user turn(s), {n_tool} tool invocation(s).\n"
            f"- Opening user message: \"{first_user[:120]}\"\n"
            f"- (LLM summariser unavailable — see "
            f"session_logs/session_{snapshot['session_id']}_*.json for full transcript.)"
        )
