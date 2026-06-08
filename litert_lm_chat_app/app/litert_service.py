from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

from app.config import Settings
from app.profiling import nvtx_range
from app.tools import (
    add_numbers,
    web_search,
)
# Dormant fallbacks — re-import + add to DEFAULT_TOOLS to re-enable.
# Small models (Gemma-4-E2B) tool-call more reliably with ONE web tool than
# two; google_search lives at app.tools.google_search and on the sidecar but
# is unregistered here so Gemma doesn't have to choose between them.
# from app.tools import google_search, tavily_search, tavily_extract, tavily_research
# from app.misc.mapbox_tools import (
#     mapbox_search_and_geocode, mapbox_reverse_geocode, mapbox_ground_location,
#     mapbox_place_details, mapbox_directions, mapbox_isochrone, mapbox_matrix,
#     mapbox_category_search, mapbox_static_map, mapbox_optimize_route,
#     mapbox_map_match,
# )

log = logging.getLogger("litert_app.service")
_DEBUG_CHUNKS = os.environ.get("LITERT_DEBUG_CHUNKS", "").lower() in ("1", "true", "yes")

try:
    import litert_lm
except Exception:
    litert_lm = None  # type: ignore


class LiteRTNotReady(RuntimeError):
    pass


DEFAULT_TOOLS = [
    # Single web tool: small models (Gemma-4-E2B) malform tool calls less
    # often when they don't have to choose between web_search and
    # google_search. Brave snippets via the IResearcher FastMCP sidecar
    # (start with `python -m app.tools_v2`); google_search still lives on
    # the sidecar for other clients and can be re-added here when needed.
    web_search,
    add_numbers,
]


@dataclass
class TurnRecord:
    """One round-trip in the conversation, in serializable form.

    `user_text` captures whatever entered the prompt (text, audio-hint, etc.);
    audio / image bytes are NOT preserved because we can't replay them — the
    compactor summarizes from text only. `assistant_text` is the full reply
    accumulated across stream chunks. `tool_chars` accumulates the byte length
    of tool_call args + tool_result content for THIS turn so the compaction
    estimator sees real budget pressure (without it, search-heavy turns blow
    the 4096-token wall silently because the engine holds tool messages we
    don't tally).
    """
    user_text: str
    assistant_text: str = ""
    tool_chars: int = 0


@dataclass
class ConversationHandle:
    session_id: str
    conversation: object
    lock: threading.Lock
    turns: list[TurnRecord] = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.turns is None:
            self.turns = []


class LiteRTChatService:
    """Single LiteRT-LM engine with multiple lightweight conversations.

    Engine is initialized once with backend=GPU, audio_backend=CPU, MTP on.
    Each browser tab/session gets its own Conversation so history is isolated.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine: Optional[object] = None
        self._engine_cm = None
        self._conversations: Dict[str, ConversationHandle] = {}
        self._global_lock = threading.Lock()
        self.ready_error: Optional[str] = None
        # Optional memory layer (SessionJournal + mem0). Wired by main.py
        # after the engine + shim are ready. None = memory disabled.
        self._memory = None

    def attach_memory(self, memory) -> None:
        """Hand the chat service a MemoryLayer instance. Called from
        app.main.lifespan after the LLM shim is configured."""
        self._memory = memory
        log.info("memory layer attached: %s",
                 "enabled" if getattr(memory, "enabled", False) else "journal-only")

    def _build_system_prompt(self) -> str:
        """Compose the system prompt for a fresh conversation.

        Base persona + a PRIOR DAYS block recalled from the on-disk diary
        (`runtime/daily_logs/YYYY-MM-DD.md`). When mem0 is enabled we ALSO
        splice in its semantic-recall block; when mem0 is off (the default
        on 4 GB devices), diary recency carries cross-session memory alone.
        """
        base = self.settings.litert_system_prompt
        if self._memory is None:
            return base

        parts = [base]

        # Diary recall — always available when MemoryLayer is constructed
        # (writes happen at session close regardless of mem0 state).
        try:
            diary = self._memory.recall_recent_diary(days=3, max_chars=1600)
        except Exception:
            log.exception("diary recall failed; continuing without it")
            diary = ""
        if diary:
            parts.append(
                "PRIOR DAYS — these are YOUR OWN notes from sessions with this "
                "user over the last 3 days. They are real memory: if the user "
                "asks 'what did we discuss', 'do you remember', 'what's my "
                "name', or anything about prior conversations, CONSULT THIS "
                "BLOCK and answer from it. Do NOT say 'I have no memory' or "
                "'I can't access previous conversations' — you have these "
                "notes. Don't bring them up unprompted, but use them whenever "
                "asked.\n\n" + diary
            )
            log.info("system prompt: PRIOR DAYS block included (%d chars)", len(diary))
        else:
            log.info("system prompt: no diary content (fresh install or empty logs)")

        # Semantic recall via mem0 — only if it's actually enabled.
        if getattr(self._memory, "enabled", False):
            try:
                k = int(os.environ.get("NOVA_MEM_RECALL_K", "5"))
                recall = self._memory.recall_block(
                    "user context preferences recent topics", k=k
                )
                if recall:
                    parts.append(
                        "PRIOR CONTEXT (durable facts from previous sessions):\n"
                        + recall
                    )
            except Exception:
                log.exception("mem0 recall at session start failed; skipping")

        return "\n\n".join(parts)

    def _backend(self, name: str):
        name = (name or "CPU").upper().strip()
        ctor = getattr(litert_lm.Backend, name, None)
        if ctor is None:
            raise LiteRTNotReady(f"Unknown LiteRT backend: {name}")
        return ctor() if callable(ctor) else ctor

    def start(self) -> None:
        if litert_lm is None:
            self.ready_error = (
                "litert_lm could not be imported. Install dependencies with: "
                "pip install -r requirements.txt"
            )
            return

        if not self.settings.litert_model_path:
            self.ready_error = "LITERT_MODEL_PATH is not set. Copy .env.example to .env and set your model path."
            return

        try:
            litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
            kwargs = {
                "backend": self._backend(self.settings.litert_backend),
                "audio_backend": self._backend(self.settings.litert_audio_backend),
                "vision_backend": self._backend(self.settings.litert_vision_backend),
                "enable_speculative_decoding": self.settings.litert_enable_speculative,
                "max_num_tokens": int(self.settings.litert_max_num_tokens),
            }
            if self.settings.litert_cache_dir:
                Path(self.settings.litert_cache_dir).mkdir(parents=True, exist_ok=True)
                kwargs["cache_dir"] = self.settings.litert_cache_dir
            log.info("LiteRT Engine init: backend=%s audio=%s vision=%s MTP=%s max_tokens=%d",
                     self.settings.litert_backend, self.settings.litert_audio_backend,
                     self.settings.litert_vision_backend,
                     self.settings.litert_enable_speculative,
                     int(self.settings.litert_max_num_tokens))

            self._engine_cm = litert_lm.Engine(self.settings.litert_model_path, **kwargs)
            self.engine = self._engine_cm.__enter__()
            self.ready_error = None
        except Exception as exc:
            self.ready_error = f"Failed to initialize LiteRT-LM engine: {exc}"
            self.engine = None

    def close_all_sessions(self) -> None:
        """Close every active conversation but keep the engine loaded.

        LiteRT-LM only supports one conversation per engine — calling this
        before mem0 ingest / diary compaction frees the engine so those paths
        can spin up their own throwaway conversations without hitting
        FAILED_PRECONDITION: A session already exists.
        """
        for handle in list(self._conversations.values()):
            cm = getattr(handle.conversation, "_litert_context_manager", None)
            if cm:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
        self._conversations.clear()

    def stop(self) -> None:
        self.close_all_sessions()
        if self._engine_cm is not None:
            try:
                self._engine_cm.__exit__(None, None, None)
            except Exception:
                pass
        self._engine_cm = None
        self.engine = None

    def health(self) -> dict:
        return {
            "engine_loaded": self.engine is not None,
            "model_path": self.settings.litert_model_path,
            "backend": self.settings.litert_backend,
            "audio_backend": self.settings.litert_audio_backend,
            "vision_backend": self.settings.litert_vision_backend,
            "speculative_decoding": self.settings.litert_enable_speculative,
            "max_num_tokens": int(self.settings.litert_max_num_tokens),
            "error": self.ready_error,
            "active_conversations": len(self._conversations),
        }

    def new_session(self) -> str:
        if self.engine is None:
            raise LiteRTNotReady(self.ready_error or "LiteRT-LM engine is not loaded")

        session_id = str(uuid.uuid4())
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._build_system_prompt()}],
            }
        ]
        conversation_cm = self.engine.create_conversation(messages=messages, tools=DEFAULT_TOOLS, enable_constrained_decoding=True)
        conversation = conversation_cm.__enter__()
        setattr(conversation, "_litert_context_manager", conversation_cm)
        self._conversations[session_id] = ConversationHandle(session_id, conversation, threading.Lock())
        return session_id

    def close_session(self, session_id: str) -> None:
        handle = self._conversations.pop(session_id, None)
        if not handle:
            return
        cm = getattr(handle.conversation, "_litert_context_manager", None)
        if cm:
            cm.__exit__(None, None, None)

    def _get_or_create(self, session_id: Optional[str]) -> ConversationHandle:
        if self.engine is None:
            raise LiteRTNotReady(self.ready_error or "LiteRT-LM engine is not loaded")
        if not session_id or session_id not in self._conversations:
            session_id = self.new_session()
        return self._conversations[session_id]

    def send_sync(self, message: str, session_id: Optional[str] = None) -> tuple[str, str]:
        handle = self._get_or_create(session_id)
        with handle.lock:
            response = handle.conversation.send_message(message)
        return handle.session_id, extract_text(response)

    def send_stream(self, message: str, session_id: Optional[str] = None) -> tuple[str, Iterator[dict]]:
        handle = self._get_or_create(session_id)
        return handle.session_id, self._run_turn(handle, user_label=message, sender=lambda c: c.send_message_async(message))

    def send_audio_stream(
        self,
        audio_path: str | Path,
        session_id: Optional[str] = None,
        prompt_hint: str = "",
        image_path: Optional[str | Path] = None,
    ) -> tuple[str, Iterator[dict]]:
        handle = self._get_or_create(session_id)
        hint = prompt_hint or self.settings.audio_prompt_hint
        label = "[voice]" + (" [image]" if image_path else "")

        def _send(conv):
            parts: list = [hint]
            if image_path:
                parts.append(litert_lm.Content.ImageFile(
                    absolute_path=str(Path(image_path).resolve())))
            parts.append(litert_lm.Content.AudioFile(
                absolute_path=str(Path(audio_path).resolve())))
            return conv.send_message_async(litert_lm.Contents.of(*parts))

        return handle.session_id, self._run_turn(handle, user_label=label, sender=_send)

    def send_text_with_image_stream(
        self,
        message: str,
        image_path: str | Path,
        session_id: Optional[str] = None,
    ) -> tuple[str, Iterator[dict]]:
        handle = self._get_or_create(session_id)
        label = message + " [image]"

        def _send(conv):
            return conv.send_message_async(litert_lm.Contents.of(
                message,
                litert_lm.Content.ImageFile(
                    absolute_path=str(Path(image_path).resolve())),
            ))

        return handle.session_id, self._run_turn(handle, user_label=label, sender=_send)

    # ---------- shared streaming + history tracking ----------

    # ---------- compaction ----------

    _KEEP_RECENT_TURNS = 2
    # Trigger compaction when token count > this * max_num_tokens. Kept low
    # (0.50) because each tool-call turn adds ~500-800 tokens (search result +
    # assistant prose), and we want headroom for the NEXT turn's generation
    # before we hit the 4096-token wall. Heavy-tool sessions silently stall
    # mid-stream once the prompt+history exceeds the budget.
    _COMPACT_RATIO = 0.50
    _SUMMARY_TARGET_TOKENS = 250

    def _count_tokens(self, text: str) -> int:
        """Count tokens via engine.tokenize. Falls back to chars/4 on error."""
        if self.engine is None:
            return len(text) // 4
        try:
            return len(self.engine.tokenize(text))
        except Exception:
            return len(text) // 4

    def _serialize_turns(self, turns: list["TurnRecord"]) -> str:
        """Render turns to a flat string used for token counting + summarization.
        Voice-turn user_text is stored as the literal '[voice]' (Gemma-4
        interprets audio natively, no STT transcript exists), so for the
        summariser we replace it with a label that signals "user spoke" — the
        assistant_text below carries the semantic content the summariser uses
        to extract facts."""
        lines = []
        for t in turns:
            user_repr = t.user_text
            if user_repr.startswith("[voice]"):
                user_repr = "(user voice message)"
            lines.append(f"User: {user_repr}")
            if t.assistant_text:
                lines.append(f"Assistant: {t.assistant_text}")
        return "\n".join(lines)

    def _conversation_token_count(self, handle: "ConversationHandle") -> int:
        sys_prompt = self.settings.litert_system_prompt
        body = self._serialize_turns(handle.turns)
        # tool_chars (call args + result content) live in the engine's history
        # but aren't in user/assistant_text. Estimate at ~4 chars/token to
        # reflect their real budget weight in compaction decisions.
        tool_tokens = sum(t.tool_chars for t in handle.turns) // 4
        return self._count_tokens(sys_prompt + "\n" + body) + tool_tokens

    def _summarize(self, text: str) -> str:
        """Run a throwaway summarizer conversation. Returns a short summary or
        an empty string on failure (the caller falls back to keeping context)."""
        if self.engine is None:
            return ""
        sys_msg = (
            "You are a context-compactor for an on-device voice assistant. "
            f"Condense the conversation below into under {self._SUMMARY_TARGET_TOKENS} tokens. "
            "Output four labeled sections, in this exact order, no preamble, no closing line:\n"
            "USER: who the user is + stable preferences (one line, omit if nothing known)\n"
            "FACTS: established facts, named entities, numbers, decisions (bullets)\n"
            "TOOLS: tool calls made + the one-line takeaway from each result (bullets)\n"
            "OPEN: the user's current goal + any unanswered question (one line)\n"
            "Drop pleasantries, narration, and anything the assistant already said back. "
            "Write as the assistant's working memory, not as a transcript."
        )
        try:
            cm = self.engine.create_conversation(
                messages=[{"role": "system", "content": [{"type": "text", "text": sys_msg}]}],
                tools=[],
            )
            conv = cm.__enter__()
            try:
                response = conv.send_message(text)
                return extract_text(response).strip()
            finally:
                cm.__exit__(None, None, None)
        except Exception as exc:
            log.warning("summarizer failed: %s", exc)
            return ""

    def _maybe_compact(self, handle: "ConversationHandle") -> Optional[dict]:
        """Check usage; if over threshold, summarize old turns and rebuild the
        conversation. Returns a synthetic SSE-shaped chunk describing the
        compaction so the caller can pass it through to the client, or None."""
        if self.engine is None or len(handle.turns) <= self._KEEP_RECENT_TURNS + 1:
            return None
        budget = int(self.settings.litert_max_num_tokens)
        used = self._conversation_token_count(handle)
        if used < int(budget * self._COMPACT_RATIO):
            return None

        log.info("compaction triggered: used=%d / budget=%d (%.0f%%)",
                 used, budget, 100 * used / budget)
        old_turns = handle.turns[:-self._KEEP_RECENT_TURNS]
        recent_turns = handle.turns[-self._KEEP_RECENT_TURNS:]
        summary = self._summarize(self._serialize_turns(old_turns))
        if not summary:
            log.warning("compaction: summarizer returned empty; aborting")
            return None

        # Rebuild conversation with the summary as a pseudo-assistant turn.
        new_messages = [
            {"role": "system",
             "content": [{"type": "text", "text": self.settings.litert_system_prompt}]},
            {"role": "assistant",
             "content": [{"type": "text",
                          "text": f"(memory of prior conversation)\n{summary}"}]},
        ]
        for t in recent_turns:
            # Voice turns store "[voice]" as user_text (Gemma-4 interprets the
            # audio natively, so there's no transcript). Replaying that literal
            # string after compaction feeds the model noise. Substitute a brief
            # placeholder that conveys "user spoke; my reply below carries the
            # substance" — the assistant_text holds the actual semantic content.
            user_repr = t.user_text
            if user_repr.startswith("[voice]"):
                user_repr = "(voice message from the user — see my reply for context)"
            new_messages.append({"role": "user",
                                 "content": [{"type": "text", "text": user_repr}]})
            if t.assistant_text:
                new_messages.append({"role": "assistant",
                                     "content": [{"type": "text", "text": t.assistant_text}]})

        old_cm = getattr(handle.conversation, "_litert_context_manager", None)
        try:
            new_cm = self.engine.create_conversation(messages=new_messages, tools=DEFAULT_TOOLS)
            new_conv = new_cm.__enter__()
            setattr(new_conv, "_litert_context_manager", new_cm)
        except Exception:
            log.exception("compaction: failed to build new conversation; keeping old one")
            return None

        if old_cm is not None:
            try:
                old_cm.__exit__(None, None, None)
            except Exception:
                pass

        handle.conversation = new_conv
        compacted_count = len(old_turns)
        handle.turns = list(recent_turns)
        new_used = self._conversation_token_count(handle)
        log.info("compaction done: %d turns -> summary (%d -> %d tokens, freed %d)",
                 compacted_count, used, new_used, used - new_used)

        # Synthetic chunk passed through the SSE generator unchanged — the
        # /api/voice/chat/stream layer surfaces it as a `compaction` event.
        return {
            "_synthetic": "compaction",
            "compacted_turns": compacted_count,
            "tokens_before": used,
            "tokens_after": new_used,
            "tokens_saved": used - new_used,
            "summary_preview": summary[:160] + ("…" if len(summary) > 160 else ""),
        }

    def _run_turn(self, handle: "ConversationHandle", user_label: str, sender) -> Iterator[dict]:
        """Drive one chunk-iterator turn, accumulating the assistant text into
        the handle's history. Pre-turn AND post-turn compaction guard the
        4096-token wall: pre-turn protects this turn's generation, post-turn
        sets up headroom for the next one."""
        # Pre-turn compaction: if the conversation is already over the
        # threshold, the incoming turn will likely overflow the budget
        # mid-generation and silently stall. Compact first.
        pre_event = None
        try:
            with nvtx_range("engine.pre_compact"):
                pre_event = self._maybe_compact(handle)
        except Exception:
            log.exception("pre-turn compaction failed; continuing without it")

        turn = TurnRecord(user_text=user_label[:2000])  # cap; not persisted
        handle.turns.append(turn)

        # Record user side now (memory is journal-first; mem0 ingestion
        # happens after assistant text is final).
        if self._memory is not None:
            try:
                self._memory.add_turn("user", turn.user_text)
            except Exception:
                log.exception("memory.add_turn(user) failed; non-fatal")

        def iterator() -> Iterator[dict]:
            if pre_event is not None:
                yield pre_event
            with handle.lock, nvtx_range("engine.send_and_stream"):
                first_token_seen = False
                for chunk in sender(handle.conversation):
                    txt = extract_text(chunk)
                    if txt:
                        if not first_token_seen:
                            # Mark TTFT — gap between this range's start and
                            # engine.send_and_stream's start = prefill time.
                            with nvtx_range("engine.first_token"):
                                pass
                            first_token_seen = True
                        turn.assistant_text += txt
                    # Accumulate tool weight so compaction sees real budget use.
                    for kind, payload in extract_tool_events(chunk):
                        if kind == "tool_call":
                            turn.tool_chars += len(str(payload.get("args") or "")) + 40
                        elif kind == "tool_result":
                            turn.tool_chars += len(str(payload.get("content") or payload.get("result") or "")) + 20
                    yield chunk
            # Memory: persist assistant text to the in-memory journal only.
            # mem0 fact-extraction is DEFERRED to session close — LiteRT-LM
            # only allows one conversation per engine, so we can't spin up a
            # throwaway conversation for mem0 while the user's chat conv is
            # open. See MemoryLayer.batch_ingest() + close_all_sessions().
            if self._memory is not None and turn.assistant_text:
                try:
                    self._memory.add_turn("assistant", turn.assistant_text)
                except Exception:
                    log.exception("memory.add_turn(assistant) failed; non-fatal")
            # After stream drain — check budget, compact if needed.
            try:
                with nvtx_range("engine.post_compact"):
                    event = self._maybe_compact(handle)
                if event is not None:
                    yield event
            except Exception:
                log.exception("compaction failed; leaving conversation as-is")

        return iterator()


def extract_text(payload: dict) -> str:
    """Concatenate all text parts from a LiteRT-LM response/chunk."""
    if not payload:
        return ""
    parts = []
    for item in payload.get("content", []) or []:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts)


_TOOL_CALL_TYPES = {"tool_call", "function_call", "toolCall", "functionCall"}
_TOOL_RESULT_TYPES = {"tool_result", "tool_response", "function_response",
                      "toolResult", "functionResponse"}


def extract_synthetic_event(chunk: dict) -> Optional[tuple[str, dict]]:
    """Return ('compaction', payload) for the marker chunk _run_turn emits
    after a successful context-compaction, else None. Lets the SSE layer
    forward a `compaction` event to the browser."""
    if isinstance(chunk, dict) and chunk.get("_synthetic") == "compaction":
        return ("compaction", {
            "compacted_turns": chunk.get("compacted_turns"),
            "tokens_before": chunk.get("tokens_before"),
            "tokens_after": chunk.get("tokens_after"),
            "tokens_saved": chunk.get("tokens_saved"),
            "summary_preview": chunk.get("summary_preview", ""),
        })
    return None


def extract_tool_events(chunk: dict) -> list[tuple[str, dict]]:
    """Yield (kind, payload) for tool/function call+result items in a chunk.

    Covers the variant shapes Gemma/LiteRT have used: snake_case (`tool_call`,
    `tool_result`) and Gemma's `function_call` / `function_response`. Unknown
    shapes log once (when LITERT_DEBUG_CHUNKS=1) and are skipped.
    """
    out: list[tuple[str, dict]] = []
    if not chunk:
        return out
    if _DEBUG_CHUNKS:
        log.debug("raw chunk: %s", chunk)
    for item in chunk.get("content", []) or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in _TOOL_CALL_TYPES:
            payload = {
                "name": item.get("name") or item.get("tool_name") or item.get("function_name") or "",
                "args": item.get("args") or item.get("arguments") or item.get("input") or {},
            }
            log.info("tool_call extracted: %s args=%s", payload["name"], payload["args"])
            out.append(("tool_call", payload))
        elif t in _TOOL_RESULT_TYPES:
            summary = (item.get("result") or item.get("output")
                       or item.get("response") or item.get("text") or "")
            payload = {
                "name": item.get("name") or item.get("tool_name") or item.get("function_name") or "",
                "ok": bool(item.get("ok", True)),
                "summary": summary if isinstance(summary, str) else str(summary),
            }
            log.info("tool_result extracted: %s ok=%s summary_len=%d",
                     payload["name"], payload["ok"], len(payload["summary"]))
            out.append(("tool_result", payload))
    return out
