from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Quiet third-party HTTP + MCP transport chatter. We still get our own
# litert_app.mcp / litert_app.tools INFO lines; just the per-request
# heartbeats from httpx and streamable-http negotiation are hidden.
for _noisy in ("httpx", "httpcore", "mcp", "mcp.client.streamable_http"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.clause_splitter import maybe_flush_clause
from app.config import get_settings
from app.profiling import nvtx_range
from app.litert_service import (
    LiteRTChatService,
    LiteRTNotReady,
    extract_synthetic_event,
    extract_text,
    extract_tool_events,
)
from app.mcp_bridge import MCPBridge, MCPNotReady, set_bridge
from app.voice_service import PocketTTSService, VoiceNotReady

# Memory layer (SessionJournal + DiaryCompactor + optional mem0 semantic recall)
# and the OpenAI-shim that lets mem0 talk to LiteRT-LM for fact extraction.
from app.memory_layer import MemoryLayer
from app import llm_shim

# Optional local extensions under app/misc/. App runs fine without them.
try:
    import app.misc  # noqa: F401
    _MISC_AVAILABLE = True
except ImportError:
    _MISC_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

settings = get_settings()
chat_service = LiteRTChatService(settings)
tts_service = PocketTTSService(settings)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Memory layer singleton — instantiated in lifespan once the engine is up
# (mem0 init needs the OpenAI shim, which needs chat_service).
memory: MemoryLayer | None = None


def _llm_oneshot_for_diary(prompt: str, max_tokens: int = 240) -> str:
    """Used by MemoryLayer.close_session() to summarise the day. Returns
    a short string; raises if the engine isn't loaded so the caller can
    fall back to deterministic summary."""
    if not chat_service.engine:
        raise RuntimeError("engine not loaded")
    import litert_lm
    cm = chat_service.engine.create_conversation(
        messages=[{"role": "system",
                   "content": [{"type": "text",
                                "text": "Be concise. Output only bullet points."}]}],
        tools=[],
    )
    conv = cm.__enter__()
    try:
        response = conv.send_message(prompt)
    finally:
        cm.__exit__(None, None, None)
    out = ""
    for item in (response.get("content", []) or []):
        if item.get("type") == "text":
            out += item.get("text", "")
    return out


_mcp_bridge: MCPBridge | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_bridge
    # Connect to the IResearcher FastMCP sidecar (started in another terminal
    # via `python -m app.tools_v2`). If the sidecar isn't running we log a
    # warning and continue — web_search / google_search will return an
    # "unavailable" message until the sidecar comes online.
    bridge = MCPBridge(url=settings.iresearcher_mcp_url, name="iresearcher")
    try:
        bridge.start(connect_timeout=10.0)
        _mcp_bridge = bridge
        set_bridge(bridge)
    except MCPNotReady as exc:
        logging.getLogger("litert_app.startup").warning(
            "IResearcher MCP sidecar not reachable at %s — start it with "
            "`python -m app.tools_v2`. (%s)",
            settings.iresearcher_mcp_url, exc,
        )

    chat_service.start()

    # Wire the OpenAI shim to the now-loaded engine BEFORE mem0 tries to use it.
    llm_shim.configure(chat_service)

    # Memory layer — opens its own DB connection inside mem0. Fail-open: if
    # Postgres/mem0 isn't available the layer logs a warning and runs in
    # journal-only mode.
    global memory
    cfg = get_settings()
    memory_enabled = cfg.nova_memory and os.environ.get("NOVA_MEMORY", "1") != "0"
    if memory_enabled:
        try:
            user_id = os.environ.get("NOVA_MEMORY_USER_ID") or cfg.nova_memory_user_id
            memory = MemoryLayer(user_id=user_id)
            chat_service.attach_memory(memory)
        except Exception as exc:
            logging.getLogger("litert_app.startup").warning(
                "MemoryLayer init crashed: %s — continuing without memory", exc)
            memory = None
    else:
        logging.getLogger("litert_app.startup").info(
            "MemoryLayer disabled (NOVA_MEMORY=0)")

    # Pre-warm Pocket TTS so the first voice turn doesn't pay the ~5-7s
    # cold-load (438 MB safetensors + sentencepiece + voice-state encode).
    # Runs in a thread to keep startup non-blocking for /api/health probes.
    import threading
    def _tts_warmup():
        t0 = time.perf_counter()
        try:
            tts_service.warmup()
            logging.getLogger("litert_app.startup").info(
                "Pocket TTS pre-warmed in %.1fs", time.perf_counter() - t0)
        except Exception as exc:
            logging.getLogger("litert_app.startup").warning(
                "Pocket TTS pre-warm failed (non-fatal): %s", exc)
    threading.Thread(target=_tts_warmup, daemon=True, name="tts-warmup").start()

    yield

    # Memory close pipeline on shutdown:
    #   1. close active chat conversations so the engine is free
    #   2. batch-ingest journal into mem0 (uses the OpenAI shim → engine)
    #   3. flush journal JSON + append today's diary entry
    #   4. tear down the engine
    if memory is not None:
        try:
            chat_service.close_all_sessions()
            await asyncio.get_running_loop().run_in_executor(
                None, memory.batch_ingest_session
            )
            await memory.close_session(llm_oneshot=_llm_oneshot_for_diary)
        except Exception as exc:
            logging.getLogger("litert_app.shutdown").warning(
                "MemoryLayer close failed: %s", exc)

    chat_service.stop()
    if _mcp_bridge is not None:
        _mcp_bridge.stop()
        set_bridge(None)
        _mcp_bridge = None


app = FastAPI(title="LiteRT-LM Native Voice Chat", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/audio", StaticFiles(directory=str(ROOT_DIR / settings.tts_output_dir)), name="audio")

# OpenAI-compatible /v1 endpoints used by mem0's fact-extraction LLM client.
# Routes are no-op until app.llm_shim.configure(chat_service) is called in
# the lifespan (after the engine loads).
app.include_router(llm_shim.router)

# /maps/ only mounted when the optional misc/ extensions are present.
if _MISC_AVAILABLE:
    _MAPS_DIR = ROOT_DIR / settings.maps_output_dir
    _MAPS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/maps", StaticFiles(directory=str(_MAPS_DIR)), name="maps")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


_FFMPEG = shutil.which("ffmpeg")


def _transcode_to_wav16k(src: Path) -> Path:
    """Transcode any browser-supplied audio (webm/opus/ogg/etc.) to 16 kHz mono WAV.

    Gemma-4-E4B's bundled miniaudio decoder only accepts WAV/FLAC/MP3/Vorbis —
    not the webm/opus containers MediaRecorder produces — so we normalise here.
    Also re-samples already-WAV inputs that aren't 16 kHz mono.
    """
    if _FFMPEG is None:
        raise VoiceNotReady(
            "ffmpeg not found on PATH. Install ffmpeg (sudo pacman -S ffmpeg / "
            "brew install ffmpeg) — required to decode browser audio."
        )
    # Always write to a fresh tempfile so a .wav input doesn't collide with
    # itself as the output (ffmpeg rc=234 "Output same as input" otherwise).
    fd, out_str = tempfile.mkstemp(suffix=".16k.wav")
    os.close(fd)
    out = Path(out_str)
    proc = subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-ac", "1", "-ar", "16000", str(out)],
        capture_output=True,
    )
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise VoiceNotReady(
            f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.decode('utf-8', 'ignore')[:300]}"
        )
    return out


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_FRIENDLY_ERRORS = (
    ("Failed to parse tool calls",
     "I tried to use a tool but mangled the call. Try rephrasing — usually fewer special characters help."),
    ("Failed to parse FC tool calls",
     "I tried to use a tool but mangled the call. Try rephrasing — usually fewer special characters help."),
    ("Input token ids are too long",
     "I've hit the context limit for this conversation. Tap ↻ New chat to clear and continue."),
    ("INVALID_ARGUMENT: Audio backend constraint",
     "Engine audio backend mismatch — check ⚙ Settings, Gemma-4 requires audio=CPU."),
    ("VK_ERROR_OUT_OF_DEVICE_MEMORY",
     "GPU ran out of memory. Lower ctx in ⚙ Settings or switch a backend to CPU."),
    ("miniaudio decoder",
     "Couldn't decode the audio you sent. Try recording again."),
)


def _friendly_error(exc: Exception) -> str:
    raw = str(exc)
    logging.getLogger("litert_app.error").error("turn failed: %s", raw)
    for needle, friendly in _FRIENDLY_ERRORS:
        if needle in raw:
            return friendly
    # Truncate and strip template-looking tokens (`<|...|>`) so they don't
    # leak into the caption.
    cleaned = raw.replace("<|", "").replace("|>", "").strip()
    return cleaned[:200] + ("…" if len(cleaned) > 200 else "")


# ── Tool-call salvage ─────────────────────────────────────────────────────
# When Gemma-4-E2B fails to emit the tool-call special tokens correctly, it
# spells them out as literal text like  query:<|"|>...<|"|>.  LiteRT-LM's
# grammar parser rejects this and raises INVALID_ARGUMENT. The model HAS
# correctly figured out the query — we just need to extract it and call the
# tool ourselves, then stream the snippets back as a normal reply.

_SALVAGE_QUERY_RE = re.compile(r'query\s*:\s*<\|"\|>(.+?)<\|"\|>', re.DOTALL)
# Fallback for the bare-value malformation: {query:current news...}  or
# {query:"current news..."} — capture everything until the next comma or
# closing brace, then strip optional surrounding quotes.
_SALVAGE_BARE_QUERY_RE = re.compile(r'query\s*:\s*([^,}]+?)\s*[,}]', re.IGNORECASE)
_SALVAGE_TOOL_RE  = re.compile(r'call:(\w+)\s*\{', re.IGNORECASE)


def _salvage_from_parse_error(exc_str: str) -> tuple[str, str] | None:
    """Return (tool_name, query) if we recognise the leaked template pattern."""
    if "Failed to parse" not in exc_str:
        return None
    tool_m = _SALVAGE_TOOL_RE.search(exc_str)
    if not tool_m:
        return None
    q_m = _SALVAGE_QUERY_RE.search(exc_str) or _SALVAGE_BARE_QUERY_RE.search(exc_str)
    if not q_m:
        return None
    query = q_m.group(1).strip().strip('"').strip("'")
    # Gemma sometimes emits a doubled {query:query:...} — strip a leaked
    # leading "query:" so we don't send it on to Brave verbatim.
    while query.lower().startswith("query:"):
        query = query[len("query:"):].strip()
    return tool_m.group(1).strip(), query


def _format_search_for_speech(search_json_or_text: str) -> str:
    """Turn a Brave/Serper JSON result blob into 2-3 spoken sentences.
    Synthesises the FACTS — does NOT paste titles + raw snippets verbatim
    (that was the old behavior: the model heard the literal article
    headlines and quoted them whole). Instead, pull the most
    fact-bearing sub-sentence from each top result and stitch them with
    light connective tissue."""
    try:
        data = json.loads(search_json_or_text)
    except Exception:
        return "I searched but couldn't parse the result. Try rephrasing."
    if isinstance(data, dict) and data.get("error"):
        return f"The search failed. {data['error']}"
    results = (data.get("results") if isinstance(data, dict) else None) or []
    if not results:
        return "I searched but didn't find anything useful. Try rephrasing the question."

    # Extract the first fact-shaped sentence from each top result. Brave
    # descriptions are usually one or two snippets glued with " ... " — the
    # first half tends to carry the lede.
    facts: list[str] = []
    for r in results[:3]:
        desc = (r.get("description") or r.get("snippet") or "").strip()
        if not desc:
            continue
        # First " ... " separator marks the snippet boundary; keep only
        # the first clause.
        head = desc.split("…")[0].split("...")[0].strip()
        # Cap any individual fact at ~180 chars so TTS doesn't blow the
        # 50-token chunk limit you saw warn.
        if len(head) > 180:
            head = head[:180].rsplit(" ", 1)[0] + "."
        # Drop trailing punctuation noise + ensure terminator.
        head = head.rstrip(",;: ")
        if head and head[-1] not in ".!?":
            head += "."
        facts.append(head)
    if not facts:
        return "I searched but the results didn't say anything useful. Try rephrasing."

    parts = ["Here's the gist."] + facts[:3]
    return " ".join(parts)


def _emit_clause(text: str, idx: int, tts: PocketTTSService) -> Iterator[str]:
    yield sse("clause", {"index": idx, "text": text})
    seq = 0
    try:
        for pcm_bytes, sample_rate in tts.synthesize_stream(text):
            b64 = base64.b64encode(pcm_bytes).decode("ascii")
            yield sse("audio_chunk", {
                "clause": idx,
                "seq": seq,
                "sample_rate": sample_rate,
                "format": "pcm_s16le",
                "b64": b64,
            })
            seq += 1
    except VoiceNotReady as exc:
        yield sse("warning", {"warning": str(exc)})
    yield sse("clause_end", {"clause": idx})


def _metrics(t0: float, first_token_ts: float | None, first_audio_ts: float | None,
             tokens: int, audio_dur_ms: int | None = None) -> dict:
    now = time.perf_counter()
    ttft = int((first_token_ts - t0) * 1000) if first_token_ts else None
    ttfb = int((first_audio_ts - t0) * 1000) if first_audio_ts else None
    decode_window = (now - first_token_ts) if first_token_ts else None
    tps = (tokens / decode_window) if decode_window and decode_window > 0 else None
    return {
        "audio_dur_ms": audio_dur_ms,
        "llm_ttft_ms": ttft,
        "total_ttfb_ms": ttfb,
        "tts_ttfb_ms": (ttfb - ttft) if (ttft is not None and ttfb is not None) else None,
        "tokens": tokens,
        "decode_tok_per_s": round(tps, 1) if tps else None,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    data = chat_service.health()
    data["tts"] = tts_service.health()
    return data


@app.get("/api/system/stats")
def system_stats():
    """Host CPU / RAM / GPU usage for the UI SYSTEM panel. GPU comes from
    nvidia-smi (the LiteRT OpenCL backend runs on the NVIDIA card); fields
    are null when a source is unavailable so the UI can show em-dashes."""
    out = {"cpu_pct": None, "ram_used_gb": None, "ram_total_gb": None,
           "gpu": {"used_mb": None, "total_mb": None, "util_pct": None}}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["ram_used_gb"] = round((vm.total - vm.available) / 1e9, 1)
        out["ram_total_gb"] = round(vm.total / 1e9, 1)
    except Exception:
        try:  # stdlib fallback
            out["cpu_pct"] = round(os.getloadavg()[0] / (os.cpu_count() or 1) * 100, 1)
            mem = {l.split(":")[0]: int(l.split()[1])
                   for l in open("/proc/meminfo") if ":" in l}
            out["ram_total_gb"] = round(mem["MemTotal"] * 1024 / 1e9, 1)
            out["ram_used_gb"] = round(
                (mem["MemTotal"] - mem["MemAvailable"]) * 1024 / 1e9, 1)
        except Exception:
            pass
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        used, total, util = (int(x) for x in q.stdout.split("\n")[0].split(","))
        out["gpu"] = {"used_mb": used, "total_mb": total, "util_pct": util}
    except Exception:
        pass
    return out


@app.get("/api/vehicle/state")
def vehicle_state():
    """Simulated CAN/OBD snapshot + reminders + calendar from the SQLite
    store the IVA tools read/write (app/vehicle_db.py). Backs the upcoming
    dashboard strip in the UI; also handy for curl during voice testing."""
    from app.vehicle_db import get_events, get_reminders, get_state, init_db
    init_db()
    return {"state": get_state(), "reminders": get_reminders(),
            "events": get_events()}


@app.get("/api/debug/system_prompt", response_class=PlainTextResponse)
def debug_system_prompt():
    """Return what Gemma actually sees as the system prompt right now.
    Useful for verifying PRIOR DAYS diary recall is being spliced. Hit it
    from a browser tab or curl: `curl localhost:8000/api/debug/system_prompt`."""
    return chat_service._build_system_prompt()


class DebugToolReq(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)


@app.post("/api/debug/tool")
def debug_tool(req: DebugToolReq):
    """Invoke a single MCP tool by name with args, bypass the model entirely,
    return the raw FastMCP response (post truncation cap). Lets you exercise
    web_search, google_search, geocode, execute_sql_query, add_skill in
    isolation to see what Gemma would see for a given query.

    Examples:
      curl -X POST localhost:8000/api/debug/tool -H 'content-type: application/json' \
        -d '{"name":"web_search","args":{"query":"latest AI news","search_type":"news","num_results":5}}'
      curl -X POST localhost:8000/api/debug/tool -H 'content-type: application/json' \
        -d '{"name":"geocode","args":{"location":"Chennai"}}'
    """
    from app.mcp_bridge import get_bridge
    bridge = get_bridge()
    if bridge is None or not bridge.ready:
        raise HTTPException(status_code=503, detail="MCP sidecar not connected. Start with: python -m app.tools_v2")
    try:
        raw = bridge.call_tool(req.name, req.args, timeout=30.0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"tool failed: {exc}") from exc
    return {"name": req.name, "args": req.args, "result_chars": len(raw), "result": raw}


class EvalRawRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256


@app.post("/api/eval/raw")
async def eval_raw(req: EvalRawRequest):
    """Eval-harness only: raw first-turn generation, bypassing Nova's system
    prompt/memory/tools wrapper entirely (spec §5). The device receives the
    final host-rendered prompt string verbatim via a raw litert_lm Session
    (apply_prompt_template=False) — see LiteRTChatService.generate_raw_text.
    Closes any active chat session first (single-tenant engine)."""
    try:
        result = await asyncio.to_thread(
            chat_service.generate_raw_text, req.prompt, req.max_new_tokens)
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


class ReconfigRequest(BaseModel):
    backend: str | None = None
    audio_backend: str | None = None
    vision_backend: str | None = None
    max_num_tokens: int | None = Field(default=None, ge=1024, le=65536)
    enable_speculative: bool | None = None


@app.post("/api/engine/reconfigure")
def reconfigure(req: ReconfigRequest):
    """Tear down and rebuild the engine with new backend / KV-cache settings.

    All active conversations are invalidated. Returns the new health snapshot
    on success, or a 503 with the error message if init failed.
    """
    # Push field-level overrides into the cached Settings instance.
    if req.backend:           settings.litert_backend = req.backend
    if req.audio_backend:     settings.litert_audio_backend = req.audio_backend
    if req.vision_backend:    settings.litert_vision_backend = req.vision_backend
    if req.max_num_tokens:    settings.litert_max_num_tokens = int(req.max_num_tokens)
    if req.enable_speculative is not None:
        settings.litert_enable_speculative = bool(req.enable_speculative)

    chat_service.stop()
    chat_service.start()
    health_data = chat_service.health()
    if not health_data.get("engine_loaded"):
        raise HTTPException(
            status_code=503,
            detail=health_data.get("error") or "engine failed to restart",
        )
    return {"ok": True, **health_data}


@app.post("/api/session")
def create_session():
    try:
        return {"session_id": chat_service.new_session()}
    except LiteRTNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/api/session/{session_id}")
async def close_session(session_id: str):
    # Step 1: tear down the user's chat conversation so the engine becomes
    # free for mem0's throwaway conversations + the diary summariser.
    chat_service.close_session(session_id)
    if memory is not None:
        try:
            # Step 2: batch-ingest the just-closed session's journal into mem0.
            # Runs on a thread executor because mem0.add is sync + blocking.
            await asyncio.get_running_loop().run_in_executor(
                None, memory.batch_ingest_session
            )
            # Step 3: flush journal JSON + append today's diary.
            await memory.close_session(llm_oneshot=_llm_oneshot_for_diary)
        except Exception as exc:
            logging.getLogger("litert_app.memory").warning(
                "memory close pipeline failed: %s", exc)
        # Step 4: start a fresh journal for the next session.
        memory.start_new_session()
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        session_id, text = chat_service.send_sync(req.message, req.session_id)
        return {"session_id": session_id, "message": text}
    except LiteRTNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


@app.post("/api/chat/stream")
async def chat_stream(
    message: str = Form(...),
    session_id: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
):
    """Text chat, optionally with an attached image. Multipart in, SSE out."""
    image_path: Path | None = None
    if image is not None and image.filename:
        suffix = Path(image.filename).suffix or ".jpg"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_path = Path(tmp.name)
            tmp.write(await image.read())

    logging.getLogger("litert_app.http").info(
        "/api/chat/stream entry: msg_chars=%d image=%s session=%s",
        len(message or ""), bool(image_path), session_id or "<new>")

    try:
        if image_path is not None:
            active_sid, chunk_iterator = chat_service.send_text_with_image_stream(
                message, image_path, session_id)
        else:
            active_sid, chunk_iterator = chat_service.send_stream(message, session_id)
    except LiteRTNotReady as exc:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def event_stream():
        yield sse("session", {"session_id": active_sid})
        try:
            for chunk in chunk_iterator:
                ev = extract_synthetic_event(chunk)
                if ev is not None:
                    yield sse(ev[0], ev[1])
                    continue
                tok = extract_text(chunk)
                if tok:
                    yield sse("token", {"text": tok})
                for kind, payload in extract_tool_events(chunk):
                    yield sse(kind, payload)
            yield sse("done", {})
        except Exception as exc:
            yield sse("error", {"error": _friendly_error(exc)})
        finally:
            if image_path is not None:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/tts")
def text_to_speech(req: ChatRequest):
    try:
        audio_url = tts_service.synthesize(req.message)
        return {"audio_url": audio_url}
    except VoiceNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/voice/chat/stream")
async def voice_chat_stream(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    notts: int = 0,
):
    """Native audio -> Gemma-4 -> clause-split -> Pocket-TTS streaming PCM.

    Optional `image` multipart field attaches a vision frame to the turn.
    `?notts=1` query param skips the TTS synthesis step entirely — useful
    for batch eval where you only care about the text reply + tool calls,
    not the spoken audio. Cuts wall-clock per turn 3-5x.
    """
    suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(await audio.read())

    image_path: Path | None = None
    if image is not None and image.filename:
        img_suffix = Path(image.filename).suffix or ".jpg"
        with NamedTemporaryFile(delete=False, suffix=img_suffix) as itmp:
            image_path = Path(itmp.name)
            itmp.write(await image.read())

    logging.getLogger("litert_app.http").info(
        "/api/voice/chat/stream entry: audio=%dB image=%s session=%s",
        temp_path.stat().st_size if temp_path.exists() else -1,
        bool(image_path), session_id or "<new>")

    cfg = get_settings()

    def event_stream():
        t0 = time.perf_counter()
        clause_idx = 0
        buf: list[str] = []
        first_audio_ts: float | None = None
        first_token_ts: float | None = None
        active_sid = session_id
        token_count = 0
        wav_path: Path | None = None

        try:
            yield sse("status", {"stage": "transcode"})
            with nvtx_range("voice.transcode"):
                wav_path = _transcode_to_wav16k(temp_path)
            yield sse("status", {"stage": "encode"})
            with nvtx_range("voice.engine_send"):
                active_sid, chunk_iter = chat_service.send_audio_stream(
                    wav_path, active_sid, image_path=image_path)
            yield sse("session", {"session_id": active_sid})

            for chunk in chunk_iter:
                ev = extract_synthetic_event(chunk)
                if ev is not None:
                    yield sse(ev[0], ev[1])
                    continue
                for kind, payload in extract_tool_events(chunk):
                    yield sse(kind, payload)

                tok = extract_text(chunk)
                if not tok:
                    continue
                if first_token_ts is None:
                    first_token_ts = time.perf_counter()
                token_count += 1
                yield sse("token", {"text": tok})
                buf.append(tok)
                clause, buf = maybe_flush_clause(
                    buf,
                    min_chars=cfg.clause_min_chars,
                    comma_min_chars=cfg.clause_comma_min_chars,
                )
                if clause:
                    if notts:
                        yield sse("clause", {"index": clause_idx, "text": clause})
                    else:
                        with nvtx_range(
                            "voice.tts_first_clause" if clause_idx == 0 else "voice.tts_clause"
                        ):
                            yield from _emit_clause(clause, clause_idx, tts_service)
                    if first_audio_ts is None:
                        first_audio_ts = time.perf_counter()
                    clause_idx += 1

            tail = "".join(buf).strip()
            if tail:
                if notts:
                    yield sse("clause", {"index": clause_idx, "text": tail})
                else:
                    yield from _emit_clause(tail, clause_idx, tts_service)
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter()

            yield sse("turn_metrics",
                      _metrics(t0, first_token_ts, first_audio_ts, token_count))
            yield sse("done", {})
        except (LiteRTNotReady, VoiceNotReady) as exc:
            yield sse("error", {"error": _friendly_error(exc)})
        except Exception as exc:
            # Salvage path: model botched tool-call sentinels. If we can
            # extract the intended query, run the tool ourselves and speak
            # the result so the user gets an answer instead of an error.
            raw = str(exc)
            salvaged = _salvage_from_parse_error(raw)
            if salvaged:
                tool_name, query = salvaged
                logging.getLogger("litert_app.error").warning(
                    "salvaging botched %s call with query=%r", tool_name, query)
                yield sse("tool_call", {"name": tool_name + " (salvaged)", "args": {"query": query}})
                try:
                    from app.tools import _mcp_proxy
                    raw_result = _mcp_proxy(
                        "web_search",
                        {"query": query, "search_type": "web", "num_results": 5},
                        label="Web search",
                    )
                    yield sse("tool_result", {"name": tool_name, "ok": True, "summary": raw_result[:200]})
                    speech = _format_search_for_speech(raw_result)
                    yield sse("token", {"text": speech})
                    # Stream the salvaged response through TTS clause-by-clause.
                    clause_buf: list[str] = list(speech)
                    cidx = 0
                    while clause_buf:
                        clause, clause_buf = maybe_flush_clause(
                            clause_buf,
                            min_chars=cfg.clause_min_chars,
                            comma_min_chars=cfg.clause_comma_min_chars,
                        )
                        if clause:
                            yield from _emit_clause(clause, cidx, tts_service)
                            cidx += 1
                        else:
                            tail = "".join(clause_buf).strip()
                            if tail:
                                yield from _emit_clause(tail, cidx, tts_service)
                            break
                    yield sse("done", {})
                    return
                except Exception as salvage_exc:
                    logging.getLogger("litert_app.error").exception(
                        "salvage path also failed: %s", salvage_exc)
                    # fall through to friendly error
            yield sse("error", {"error": _friendly_error(exc)})
        finally:
            for p in (temp_path, wav_path, image_path):
                if p is not None:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/voice/chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
):
    """Non-streaming fallback: audio in, single WAV URL out."""
    suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
    temp_path = None
    wav_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(await audio.read())

        wav_path = _transcode_to_wav16k(temp_path)
        active_sid, chunk_iter = chat_service.send_audio_stream(wav_path, session_id)
        text_parts: list[str] = []
        for chunk in chunk_iter:
            text_parts.append(extract_text(chunk))
        response_text = "".join(text_parts).strip()

        audio_url, tts_error = None, None
        if response_text:
            try:
                audio_url = tts_service.synthesize(response_text)
            except VoiceNotReady as exc:
                tts_error = str(exc)

        return {
            "session_id": active_sid,
            "message": response_text,
            "audio_url": audio_url,
            "tts_error": tts_error,
        }
    except LiteRTNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        for p in (temp_path, wav_path):
            if p is not None and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
