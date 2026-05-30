"""Minimal OpenAI-compatible /v1 wrapper around the existing LiteRT-LM engine.

Why this exists: mem0's `openai` LLM provider speaks the OpenAI HTTP wire
format. We don't want to run a second LLM process (llama-server) alongside
LiteRT-LM just so mem0 can extract facts — that would double VRAM use. So
we expose a tiny shim that takes OpenAI chat-completion requests, runs them
through the already-loaded LiteRT-LM engine via a throwaway conversation,
and returns the result in OpenAI's response shape.

Only the endpoints mem0 actually needs are implemented:
  - GET  /v1/models                  (probe)
  - POST /v1/chat/completions        (non-streaming, single response)

Embeddings come from sentence-transformers loaded inside mem0 directly —
they don't hit this shim.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("litert_app.shim")

router = APIRouter(prefix="/v1", tags=["openai-shim"])

# Set externally by app.main during startup.
_chat_service = None


def configure(chat_service) -> None:
    """Hand the shim a reference to the singleton LiteRTChatService."""
    global _chat_service
    _chat_service = chat_service


# ── OpenAI request/response shapes ─────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "litertlm-local"
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    # Other OpenAI fields are accepted but ignored.

    class Config:
        extra = "allow"


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "litertlm-local", "object": "model", "owned_by": "local"},
        ],
    }


@router.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """Run a one-shot completion against the LiteRT-LM engine.

    Implementation: build a throwaway Conversation seeded with the request's
    system messages (if any), then send the concatenated user/assistant
    history as a single message. Reply is returned in OpenAI shape.
    """
    if _chat_service is None or _chat_service.engine is None:
        raise HTTPException(status_code=503, detail="LiteRT-LM engine not loaded")

    # mem0 typically sends a short fact-extraction prompt as the LAST message
    # with the conversation context above it. We honour the LiteRT-LM message
    # shape: extract the trailing user message and stuff the rest into system.
    sys_parts: list[str] = []
    history_parts: list[str] = []
    user_msg: str = ""

    for i, m in enumerate(req.messages):
        if i == len(req.messages) - 1 and m.role in ("user", "human"):
            user_msg = m.content
        elif m.role == "system":
            sys_parts.append(m.content)
        else:
            history_parts.append(f"{m.role.upper()}: {m.content}")

    if history_parts:
        # Fold prior history into the user message so the model sees it.
        user_msg = "\n".join(history_parts + [f"USER: {user_msg}" if user_msg else ""])
    if not user_msg.strip():
        # Some mem0 paths send messages without a trailing user role.
        user_msg = "\n".join(f"{m.role.upper()}: {m.content}" for m in req.messages)

    sys_text = "\n\n".join(sys_parts) if sys_parts else "Be concise."

    log.info("shim: chat_completions (msgs=%d user_chars=%d)",
             len(req.messages), len(user_msg))

    # Throwaway conversation — no tools, no history, no system-prompt baggage
    # from the active chat. Just a single call to the engine.
    try:
        import litert_lm
        cm = _chat_service.engine.create_conversation(
            messages=[{"role": "system", "content": [{"type": "text", "text": sys_text}]}],
            tools=[],
        )
        conv = cm.__enter__()
        try:
            response = conv.send_message(user_msg)
        finally:
            cm.__exit__(None, None, None)
    except Exception as exc:
        log.exception("shim: completion failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Extract text from LiteRT-LM response shape.
    reply_text = ""
    for item in (response.get("content", []) or []):
        if item.get("type") == "text":
            reply_text += item.get("text", "")

    # Return in OpenAI chat-completion shape.
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,        # not exposed by LiteRT-LM
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }
