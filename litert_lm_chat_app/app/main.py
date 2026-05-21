from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.clause_splitter import maybe_flush_clause
from app.config import get_settings
from app.litert_service import (
    LiteRTChatService,
    LiteRTNotReady,
    extract_text,
    extract_tool_events,
)
from app.voice_service import PocketTTSService, VoiceNotReady

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

settings = get_settings()
chat_service = LiteRTChatService(settings)
tts_service = PocketTTSService(settings)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_service.start()
    yield
    chat_service.stop()


app = FastAPI(title="LiteRT-LM Native Voice Chat", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/audio", StaticFiles(directory=str(ROOT_DIR / settings.tts_output_dir)), name="audio")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


_FFMPEG = shutil.which("ffmpeg")


def _transcode_to_wav16k(src: Path) -> Path:
    """Transcode any browser-supplied audio (webm/opus/ogg/etc.) to 16 kHz mono WAV.

    Gemma-4-E4B's bundled miniaudio decoder only accepts WAV/FLAC/MP3/Vorbis —
    not the webm/opus containers MediaRecorder produces — so we normalise here.
    """
    if _FFMPEG is None:
        raise VoiceNotReady(
            "ffmpeg not found on PATH. Install ffmpeg (sudo pacman -S ffmpeg / "
            "brew install ffmpeg) — required to decode browser audio."
        )
    out = src.with_suffix(".wav")
    proc = subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-ac", "1", "-ar", "16000", str(out)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise VoiceNotReady(
            f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.decode('utf-8', 'ignore')[:300]}"
        )
    return out


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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


@app.post("/api/session")
def create_session():
    try:
        return {"session_id": chat_service.new_session()}
    except LiteRTNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/api/session/{session_id}")
def close_session(session_id: str):
    chat_service.close_session(session_id)
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
def chat_stream(req: ChatRequest):
    try:
        session_id, chunk_iterator = chat_service.send_stream(req.message, req.session_id)
    except LiteRTNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def event_stream():
        yield sse("session", {"session_id": session_id})
        try:
            for chunk in chunk_iterator:
                tok = extract_text(chunk)
                if tok:
                    yield sse("token", {"text": tok})
                for kind, payload in extract_tool_events(chunk):
                    yield sse(kind, payload)
            yield sse("done", {})
        except Exception as exc:
            yield sse("error", {"error": str(exc)})

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
):
    """Native audio -> Gemma-4 -> clause-split -> Pocket-TTS streaming PCM."""
    suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(await audio.read())

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
            wav_path = _transcode_to_wav16k(temp_path)
            yield sse("status", {"stage": "encode"})
            active_sid, chunk_iter = chat_service.send_audio_stream(wav_path, active_sid)
            yield sse("session", {"session_id": active_sid})

            for chunk in chunk_iter:
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
                    yield from _emit_clause(clause, clause_idx, tts_service)
                    if first_audio_ts is None:
                        first_audio_ts = time.perf_counter()
                    clause_idx += 1

            tail = "".join(buf).strip()
            if tail:
                yield from _emit_clause(tail, clause_idx, tts_service)
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter()

            yield sse("turn_metrics",
                      _metrics(t0, first_token_ts, first_audio_ts, token_count))
            yield sse("done", {})
        except (LiteRTNotReady, VoiceNotReady) as exc:
            yield sse("error", {"error": str(exc)})
        except Exception as exc:
            yield sse("error", {"error": f"Voice streaming failed: {exc}"})
        finally:
            for p in (temp_path, wav_path):
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
