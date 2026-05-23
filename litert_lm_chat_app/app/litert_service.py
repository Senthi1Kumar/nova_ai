from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

from app.config import Settings
from app.tools import (
    add_numbers,
    mapbox_category_search,
    mapbox_directions,
    mapbox_ground_location,
    mapbox_isochrone,
    mapbox_map_match,
    mapbox_matrix,
    mapbox_optimize_route,
    mapbox_place_details,
    mapbox_reverse_geocode,
    mapbox_search_and_geocode,
    mapbox_static_map,
    tavily_extract,
    tavily_research,
    tavily_search,
)

log = logging.getLogger("litert_app.service")
_DEBUG_CHUNKS = os.environ.get("LITERT_DEBUG_CHUNKS", "").lower() in ("1", "true", "yes")

try:
    import litert_lm
except Exception:
    litert_lm = None  # type: ignore


class LiteRTNotReady(RuntimeError):
    pass


DEFAULT_TOOLS = [
    # Web
    tavily_search,
    tavily_extract,
    tavily_research,
    # Maps (Mapbox MCP — only effective if MAPBOX_ACCESS_TOKEN is set)
    mapbox_search_and_geocode,
    mapbox_reverse_geocode,
    mapbox_ground_location,
    mapbox_place_details,
    mapbox_directions,
    mapbox_isochrone,
    mapbox_matrix,
    mapbox_category_search,
    mapbox_static_map,
    mapbox_optimize_route,
    mapbox_map_match,
    # Trivial
    add_numbers,
]


@dataclass
class ConversationHandle:
    session_id: str
    conversation: object
    lock: threading.Lock


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
            }
            if self.settings.litert_cache_dir:
                kwargs["cache_dir"] = self.settings.litert_cache_dir

            self._engine_cm = litert_lm.Engine(self.settings.litert_model_path, **kwargs)
            self.engine = self._engine_cm.__enter__()
            self.ready_error = None
        except Exception as exc:
            self.ready_error = f"Failed to initialize LiteRT-LM engine: {exc}"
            self.engine = None

    def stop(self) -> None:
        for handle in list(self._conversations.values()):
            cm = getattr(handle.conversation, "_litert_context_manager", None)
            if cm:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
        self._conversations.clear()
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
                "content": [{"type": "text", "text": self.settings.litert_system_prompt}],
            }
        ]
        conversation_cm = self.engine.create_conversation(messages=messages, tools=DEFAULT_TOOLS)
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

        def iterator() -> Iterator[dict]:
            with handle.lock:
                for chunk in handle.conversation.send_message_async(message):
                    yield chunk

        return handle.session_id, iterator()

    def send_audio_stream(
        self,
        audio_path: str | Path,
        session_id: Optional[str] = None,
        prompt_hint: str = "",
        image_path: Optional[str | Path] = None,
    ) -> tuple[str, Iterator[dict]]:
        handle = self._get_or_create(session_id)
        hint = prompt_hint or self.settings.audio_prompt_hint

        def iterator() -> Iterator[dict]:
            with handle.lock:
                parts: list = [hint]
                if image_path:
                    parts.append(litert_lm.Content.ImageFile(
                        absolute_path=str(Path(image_path).resolve())))
                parts.append(litert_lm.Content.AudioFile(
                    absolute_path=str(Path(audio_path).resolve())))
                contents = litert_lm.Contents.of(*parts)
                for chunk in handle.conversation.send_message_async(contents):
                    yield chunk

        return handle.session_id, iterator()

    def send_text_with_image_stream(
        self,
        message: str,
        image_path: str | Path,
        session_id: Optional[str] = None,
    ) -> tuple[str, Iterator[dict]]:
        handle = self._get_or_create(session_id)

        def iterator() -> Iterator[dict]:
            with handle.lock:
                contents = litert_lm.Contents.of(
                    message,
                    litert_lm.Content.ImageFile(
                        absolute_path=str(Path(image_path).resolve())),
                )
                for chunk in handle.conversation.send_message_async(contents):
                    yield chunk

        return handle.session_id, iterator()


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
