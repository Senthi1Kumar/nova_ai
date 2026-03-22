import asyncio
import json
import logging
import os
import sys
import psutil
import pynvml
import time
import multiprocessing as mp
from pathlib import Path
from contextlib import asynccontextmanager
from queue import Empty
import numpy as np
import asyncpg
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastrtc import AsyncStreamHandler, Stream, wait_for_item
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydub import AudioSegment
from dotenv import load_dotenv

# Import our worker runners
from pipeline_mp import run_stt_worker, run_kws_worker, run_llm_worker, run_tts_worker

# Monkeypatch torchaudio for speechbrain compatibility
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"] # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Nova-Gateway")

# Suppress noisy httpx/aioice telemetry logs from FastRTC/Gradio/HF
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

load_dotenv()

# Disable HF/Gradio analytics telemetry
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

# Global state for multiprocessing pipeline
pipeline_state = {
    "stop_event": None,
    "kws_in_queue": None,
    "stt_in_queue": None,
    "llm_in_queue": None,
    "tts_in_queue": None,
    "ws_out_queue": None,
    "processes": {}, # Now a dict: component -> process
    "fsm_state": "IDLE",
    "current_llm": "auto",  # "auto" = query router decides local vs cloud per-query
    "current_voice": "alba",
    "current_tts_engine": "pocket-tts",  # default low-VRAM; "faster-qwen3" loaded on demand
    # Echo suppression: when to stop blocking mic input
    # Computed as: last_audio_out_time + remaining_playback_duration + tail
    "tts_suppress_until": 0.0,
    # Running count of 24kHz int16 samples buffered but not yet "played" by browser
    "_tts_pending_samples": 0,
    # When the first audio_out chunk of the current TTS response arrived
    "_tts_start_time": 0.0,
    # Total samples in the current TTS response
    "_tts_total_samples": 0,
    "_tts_last_chunk_time": 0.0,
    # E2E latency: start timer on KWS/PTT, inject into tts_metrics
    "e2e_start_time": 0.0,
}

# Enrollment status for progress tracking
enrollment_status = {"state": "idle", "message": ""}
# Accumulated voice fingerprints for dedicated enrollment
_voice_fp_pending: list = []

# Path constants for enrollment checks
KWS_MODELS_DIR = Path(__file__).parent / "kws" / "models"
VOICEPRINT_DIR  = Path(__file__).parent / "nova-l7" / "L-3" / "data" / "voiceprints"

active_websockets = set()
active_rtc_handlers = set()

# Postgres connection pool (initialized in lifespan)
nova_db: asyncpg.Pool | None = None

# Per-turn accumulator for DB writes
_pending_turn: dict = {}
_turn_index: int = 0
_current_session_id: str | None = None

# NVML for metrics
try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    nvml_handle = None

def get_metrics():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_mem_used = gpu_util = 0
    gpu_mem_total_mb = 0

    # Map component name to its PID
    pid_map = {name: p.pid for name, p in pipeline_state["processes"].items() if p and p.pid}
    gpu_metrics = {}

    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_mem_used = float(info.used) / 1024 ** 2
            gpu_mem_total_mb = float(info.total) / 1024 ** 2
            gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu)

            # NVML compute processes (misses ONNX-RT which uses cudaMallocAsync)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(nvml_handle)
            for p in procs:
                for comp_name, pid in pid_map.items():
                    if p.pid == pid:
                        gpu_metrics[comp_name] = p.usedGpuMemory / 1024 ** 2
        except Exception:
            pass

    # Fallback: nvidia-smi per-process query catches ONNX-RT allocations NVML misses
    if pid_map and len(gpu_metrics) < len(pid_map):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(", ")
                if len(parts) == 2:
                    try:
                        smi_pid, smi_mem = int(parts[0]), float(parts[1])
                        for comp_name, pid in pid_map.items():
                            if smi_pid == pid and comp_name not in gpu_metrics:
                                gpu_metrics[comp_name] = smi_mem
                    except ValueError:
                        pass
        except Exception:
            pass

    # Determine if using OpenRouter (API) or local LLM
    current_llm = pipeline_state.get("current_llm", "auto")
    using_openrouter = not current_llm.startswith("local:") and current_llm != "auto"
    if current_llm == "auto":
        using_openrouter = os.environ.get("OPENROUTER_API_KEY") is not None
    current_voice = pipeline_state.get("current_voice", "alba")

    def get_comp_info(comp_key, display_name, is_gpu_component):
        gpu_vram = gpu_metrics.get(comp_key, 0)
        actual_is_gpu = gpu_vram > 10  # More than 10MB means it's using GPU
        vram_display = gpu_vram if actual_is_gpu else 0
        
        # For LLM, show 0 VRAM if using API
        if comp_key == "llm" and using_openrouter:
            vram_display = 0
            actual_is_gpu = False
        
        return {
            "name": display_name,
            "vram_mb": round(vram_display, 1),
            "load": "GPU" if actual_is_gpu else "CPU",
            "is_gpu": actual_is_gpu
        }

    current_tts_engine = pipeline_state.get("current_tts_engine", "pocket-tts")
    if current_tts_engine == "faster-qwen3":
        tts_display = "FasterQwen3TTS"
    else:
        tts_display = f"Pocket-TTS ({current_voice})"

    return {
        "cpu": cpu_usage,
        "ram": ram_usage,
        "gpu_mem": round(gpu_mem_used, 1),
        "gpu_mem_total": round(gpu_mem_total_mb, 1),
        "gpu_util": round(gpu_util, 1),
        "tts_engine": current_tts_engine,
        "components": {
            "stt": get_comp_info("stt", "Moonshine (Medium)", True),
            "llm": get_comp_info("llm", current_llm, not using_openrouter),
            "tts": get_comp_info("tts", tts_display, True),
            "kws": get_comp_info("kws", "StreamingKWS V2", True),
        },
    }


async def _open_session(driver_id: str = "driver1") -> str | None:
    """Insert a new session row and return its session_id."""
    global _current_session_id, _turn_index
    if nova_db is None:
        return None
    try:
        row = await nova_db.fetchrow(
            "INSERT INTO sessions (driver_id) VALUES ($1) RETURNING session_id::text",
            driver_id,
        )
        _current_session_id = row["session_id"]
        _turn_index = 0
        logger.info(f"[DB] Session opened: {_current_session_id}")
        return _current_session_id
    except Exception as e:
        logger.warning(f"[DB] open_session failed: {e}")
        return None


async def _close_session() -> None:
    global _current_session_id
    if nova_db is None or _current_session_id is None:
        return
    try:
        await nova_db.execute(
            "UPDATE sessions SET ended_at = NOW() WHERE session_id = $1",
            _current_session_id,
        )
        logger.info(f"[DB] Session closed: {_current_session_id}")
    except Exception as e:
        logger.warning(f"[DB] close_session failed: {e}")
    _current_session_id = None


async def _flush_turn() -> None:
    """Write the accumulated _pending_turn to the turns table."""
    global _pending_turn, _turn_index, _current_session_id
    if nova_db is None or not _pending_turn:
        return
    # Open a session on first turn if none exists
    if _current_session_id is None:
        await _open_session()
    try:
        import json as _json
        entities = _pending_turn.get("entities")
        await nova_db.execute(
            """INSERT INTO turns
               (session_id, turn_index, user_text, intent, entities,
                nova_response, fsm_state, routing, stt_latency_ms)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
            _current_session_id,
            _turn_index,
            _pending_turn.get("user_text"),
            _pending_turn.get("intent"),
            _json.dumps(entities) if entities else None,
            _pending_turn.get("nova_response"),
            _pending_turn.get("fsm_state"),
            _pending_turn.get("routing"),
            _pending_turn.get("stt_latency_ms"),
        )
        _turn_index += 1
        logger.info(f"[DB] Turn {_turn_index} written: intent={_pending_turn.get('intent')}")
    except Exception as e:
        logger.warning(f"[DB] flush_turn failed: {e}")
    _pending_turn = {}


async def ws_queue_reader():
    """Background task to read from ws_out_queue and broadcast to WebSockets."""
    loop = asyncio.get_running_loop()
    while not pipeline_state["stop_event"].is_set():
        try:
            # We use an executor to avoid blocking the asyncio loop while waiting on the mp.Queue
            msg = await loop.run_in_executor(None, pipeline_state["ws_out_queue"].get, True, 0.1)
            
            if isinstance(msg, dict):
                msg_type = msg.get("type")
                if msg_type not in ("audio_out",):
                    logger.info(f"[ws_queue] {msg_type} | fsm={pipeline_state['fsm_state']} | ws_clients={len(active_websockets)}")

                # Update gateway state based on worker events
                if msg.get("type") == "kws_detected":
                    if pipeline_state["fsm_state"] == "GENERATING":
                        # Hard interrupt: stop TTS immediately, clear buffered audio
                        pipeline_state["tts_interrupt_event"].set()
                        pipeline_state["stt_in_queue"].put({"type": "tts_unmute"})
                        for h in list(active_rtc_handlers):
                            h.audio_queue = asyncio.Queue()
                        pipeline_state["_tts_pending_samples"] = 0
                        pipeline_state["_tts_start_time"] = 0.0
                        pipeline_state["_tts_total_samples"] = 0
                        pipeline_state["_tts_last_chunk_time"] = 0.0
                        pipeline_state["tts_suppress_until"] = 0.0
                    pipeline_state["fsm_state"] = "LISTENING"
                    pipeline_state["e2e_start_time"] = time.time()
                    pipeline_state["stt_in_queue"].put({"type": "start"})
                    pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "LISTENING"})

                elif msg.get("type") == "ptt_started":
                    if pipeline_state["fsm_state"] == "GENERATING":
                        pipeline_state["tts_interrupt_event"].set()
                        pipeline_state["stt_in_queue"].put({"type": "tts_unmute"})
                        for h in list(active_rtc_handlers):
                            h.audio_queue = asyncio.Queue()
                        pipeline_state["_tts_pending_samples"] = 0
                        pipeline_state["_tts_start_time"] = 0.0
                        pipeline_state["_tts_total_samples"] = 0
                        pipeline_state["_tts_last_chunk_time"] = 0.0
                        pipeline_state["tts_suppress_until"] = 0.0
                    pipeline_state["fsm_state"] = "LISTENING"
                    pipeline_state["e2e_start_time"] = time.time()
                    # NOTE: Do NOT send start to STT here — the WS/POST handler
                    # already sent start(ptt=True) directly.  Sending a redundant
                    # start(ptt=False) here races with the PTT session and can
                    # cause the session to die after PTT release.
                    pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "LISTENING"})

                elif msg.get("type") == "speech_started":
                    if pipeline_state["fsm_state"] in ("IDLE", "LISTENING"):
                        pipeline_state["fsm_state"] = "LISTENING"
                        pipeline_state["e2e_start_time"] = time.time()
                        pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "LISTENING"})
                        # Forward speech_started to UI (handled there directly)

                elif msg.get("type") == "transcript":
                    latency = msg.get("latency", {})
                    stt_ms = int(latency.get("stt_ttfb", 0) * 1000) or None
                    _pending_turn["user_text"] = msg.get("data")
                    _pending_turn["stt_latency_ms"] = stt_ms

                elif msg.get("type") == "dm_state":
                    d = msg.get("data", {})
                    _pending_turn.update({
                        "intent":       d.get("intent"),
                        "entities":     d.get("entities"),
                        "nova_response": d.get("nova_says"),
                        "fsm_state":    d.get("fsm_state"),
                        "routing":      d.get("routing"),
                    })

                elif msg.get("type") == "generation_start":
                    pipeline_state["fsm_state"] = "GENERATING"
                    # NOTE: Do NOT send "stop" to STT here — STT already pauses
                    # itself in on_line_completed.  Sending a redundant stop races
                    # with start_ptt and kills PTT sessions.
                    # Mute mic capture while TTS will be playing (echo suppression)
                    pipeline_state["stt_in_queue"].put({"type": "tts_mute"})
                    # KWS remains listening in background for Wake Word Interrupt
                    pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "GENERATING"})

                elif msg.get("type") == "tts_metrics":
                    # Inject E2E latency (KWS/PTT detection → first TTS audio)
                    if pipeline_state["e2e_start_time"] > 0:
                        msg["data"] = msg.get("data") or {}
                        msg["data"]["e2e"] = time.time() - pipeline_state["e2e_start_time"]
                        pipeline_state["e2e_start_time"] = 0.0

                elif msg.get("type") == "generation_done":
                    asyncio.ensure_future(_flush_turn())
                    # Only honour if we're still in GENERATING — ignore stale done from an interrupted TTS
                    if pipeline_state["fsm_state"] == "GENERATING":
                        # Calculate remaining browser playback time before unmuting STT.
                        # Audio chunks are buffered in the browser — unmuting too early
                        # causes moonshine to transcribe TTS speaker output (echo).
                        now = time.time()
                        remaining_play = 0.0
                        if pipeline_state["_tts_start_time"] > 0:
                            total_dur = pipeline_state["_tts_total_samples"] / 24000.0
                            elapsed = now - pipeline_state["_tts_start_time"]
                            remaining_play = max(0.0, total_dur - elapsed)
                        # Minimum 1.5s suppress: short DM responses ("Done. AC on.")
                        # finish TTS fast but browser still plays buffered audio.
                        suppress_delay = max(remaining_play + 0.4, 1.5)
                        pipeline_state["_tts_pending_samples"] = 0
                        pipeline_state["_tts_start_time"] = 0.0
                        pipeline_state["_tts_total_samples"] = 0
                        pipeline_state["_tts_last_chunk_time"] = 0.0
                        pipeline_state["tts_suppress_until"] = now + suppress_delay
                        # Schedule unmute + STT restart after playback finishes
                        async def _delayed_unmute_and_restart(delay: float) -> None:
                            await asyncio.sleep(delay)
                            pipeline_state["stt_in_queue"].put({"type": "tts_unmute"})
                            if os.environ.get("NOVA_NO_KWS") == "1":
                                pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})
                        asyncio.ensure_future(_delayed_unmute_and_restart(suppress_delay))
                        pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "IDLE"})
                        if os.environ.get("NOVA_NO_KWS") == "1":
                            pipeline_state["fsm_state"] = "LISTENING"
                        else:
                            pipeline_state["fsm_state"] = "IDLE"
                            pipeline_state["stt_in_queue"].put({"type": "stop"})

                elif msg.get("type") == "recording_stopped":
                    # STT found nothing — return to IDLE/LISTENING.
                    # Also escape if somehow stuck in GENERATING (e.g. empty transcript
                    # was sent to LLM which never responded with generation_done).
                    if pipeline_state["fsm_state"] != "GENERATING":
                        pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "IDLE"})
                        if os.environ.get("NOVA_NO_KWS") == "1":
                            pipeline_state["fsm_state"] = "LISTENING"
                            pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})
                        else:
                            pipeline_state["fsm_state"] = "IDLE"

                elif msg.get("type") == "stt_variant_changed":
                    pipeline_state["current_stt_variant"] = msg.get("data", {}).get("variant", "small")

                elif msg.get("type") == "interrupted":
                    pipeline_state["stt_in_queue"].put({"type": "tts_unmute"})
                    pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "IDLE"})
                    if os.environ.get("NOVA_NO_KWS") == "1":
                        pipeline_state["fsm_state"] = "LISTENING"
                        pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})
                    else:
                        pipeline_state["fsm_state"] = "IDLE"
                        pipeline_state["stt_in_queue"].put({"type": "stop"})

                if msg.get("type") == "audio_out":
                    audio_bytes: bytes | None = msg.get("bytes")
                    if audio_bytes is None:
                        continue
                    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16)
                    # Echo suppression: track total TTS duration and when it started.
                    # remaining_play = total_duration - elapsed_since_first_chunk (accurate even
                    # when generation_done fires mid-stream while browser is still playing).
                    now = time.time()
                    new_samples = len(audio_arr)
                    if pipeline_state["_tts_start_time"] == 0.0:
                        pipeline_state["_tts_start_time"] = now
                    pipeline_state["_tts_total_samples"] += new_samples
                    total_dur = pipeline_state["_tts_total_samples"] / 24000.0
                    elapsed = now - pipeline_state["_tts_start_time"]
                    remaining = max(0.0, total_dur - elapsed)
                    pipeline_state["_tts_last_chunk_time"] = now
                    pipeline_state["tts_suppress_until"] = now + remaining + 0.5
                    to_remove = set()
                    for ws in active_websockets:
                        try:
                            await ws.send_bytes(audio_bytes)
                        except Exception:
                            to_remove.add(ws)
                    active_websockets.difference_update(to_remove)
                    # Push to WebRTC audio queues (emit() drains these)
                    for handler in list(active_rtc_handlers):
                        try:
                            handler.audio_queue.put_nowait((24000, audio_arr))
                        except Exception:
                            pass
                    continue

                # Broadcast JSON to all clients
                json_str = json.dumps(msg)
                to_remove = set()
                for ws in active_websockets:
                    try:
                        await ws.send_text(json_str)
                    except Exception:
                        to_remove.add(ws)
                active_websockets.difference_update(to_remove)
                # Forward JSON events over WebRTC data channel
                for handler in list(active_rtc_handlers):
                    try:
                        await handler.send_message(json_str)
                    except Exception:
                        pass

            elif isinstance(msg, bytes):
                # We expect this is an audio_out payload from TTS (we wrapped it in a dict in TTS worker, wait, let's double check!)
                # In TTS worker, we put {"type": "audio_out", "bytes": pcm_bytes}, so it will be a dict. 
                pass

        except Empty:
            pass
        except Exception as e:
            logger.error(f"[ws_queue] Error processing message: {e}", exc_info=True)

class NovaRTCHandler(AsyncStreamHandler):
    """FastRTC/WebRTC audio bridge into the multiprocessing pipeline."""

    def copy(self):
        return NovaRTCHandler()

    async def start_up(self):
        active_rtc_handlers.add(self)
        kws_ok   = (KWS_MODELS_DIR / "mlp_weights.pth").exists()
        voice_ok = (VOICEPRINT_DIR / "driver1.enc").exists()
        pipeline_state["ws_out_queue"].put({
            "type": "enrollment_check",
            "kws_enrolled": kws_ok,
            "voice_enrolled": voice_ok,
            "fully_enrolled": kws_ok and voice_ok
        })

    def shutdown(self) -> None:
        active_rtc_handlers.discard(self)

    def __init__(self):
        # FastRTC resamples WebRTC 48kHz Opus → 16kHz for us (input_sample_rate).
        # We emit 24kHz PCM so FastRTC upsamples to 48kHz for the browser (output_sample_rate).
        super().__init__(expected_layout="mono", input_sample_rate=16000, output_sample_rate=24000)
        self.audio_queue = asyncio.Queue()

    async def receive(self, frame: tuple[int, np.ndarray]):
        # Echo suppression: suppress mic input until TTS audio finishes playing in browser.
        # tts_suppress_until is set to: last_chunk_time + remaining_playback + 500ms reverb tail.
        if time.time() < pipeline_state["tts_suppress_until"]:
            return
        sr, audio_arr = frame
        # FastRTC delivers int16 at input_sample_rate; convert to bytes for workers.
        audio_bytes = audio_arr.flatten().astype(np.int16).tobytes()
        if not pipeline_state["kws_in_queue"]:
            return
        state = pipeline_state["fsm_state"]
        if state == "IDLE":
            pipeline_state["kws_in_queue"].put(audio_bytes)
        elif state == "LISTENING":
            pipeline_state["stt_in_queue"].put(audio_bytes)
        elif state == "GENERATING":
            pipeline_state["kws_in_queue"].put(audio_bytes)  # wake-word interrupt

    async def emit(self):
        return await wait_for_item(self.audio_queue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nova_db
    db_url = os.environ.get("NOVA_DB_URL")
    if db_url:
        try:
            nova_db = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
            logger.info("Postgres pool connected.")
        except Exception as e:
            logger.warning(f"Postgres unavailable — conversation storage disabled: {e}")
            nova_db = None

    mp.set_start_method("spawn", force=True)
    
    pipeline_state["stop_event"] = mp.Event()
    pipeline_state["tts_interrupt_event"] = mp.Event()
    pipeline_state["kws_in_queue"] = mp.Queue()
    pipeline_state["stt_in_queue"] = mp.Queue()
    pipeline_state["llm_in_queue"] = mp.Queue()
    pipeline_state["tts_in_queue"] = mp.Queue()
    pipeline_state["ws_out_queue"] = mp.Queue()
    
    bypass_kws = os.environ.get("NOVA_NO_KWS") == "1"

    # Spawn KWS Worker
    pipeline_state["processes"]["kws"] = mp.Process(target=run_kws_worker, args=(
        pipeline_state["kws_in_queue"],
        pipeline_state["ws_out_queue"],
        pipeline_state["stop_event"],
        bypass_kws
    ))

    # Spawn STT Worker
    pipeline_state["processes"]["stt"] = mp.Process(target=run_stt_worker, args=(
        pipeline_state["stt_in_queue"],
        pipeline_state["llm_in_queue"],
        pipeline_state["ws_out_queue"],
        pipeline_state["stop_event"]
    ))

    # Spawn LLM Worker
    pipeline_state["processes"]["llm"] = mp.Process(target=run_llm_worker, args=(
        pipeline_state["llm_in_queue"],
        pipeline_state["tts_in_queue"],
        pipeline_state["ws_out_queue"],
        pipeline_state["stop_event"],
        pipeline_state["tts_interrupt_event"]
    ))

    # Spawn TTS Worker
    pipeline_state["processes"]["tts"] = mp.Process(target=run_tts_worker, args=(
        pipeline_state["tts_in_queue"],
        pipeline_state["ws_out_queue"],
        pipeline_state["stop_event"],
        pipeline_state["tts_interrupt_event"]
    ))

    for p in pipeline_state["processes"].values():
        p.start()
        
    logger.info("All Multi-Processing Workers Started.")
    
    # Start the async reader (no reference kept — event loop holds it alive)
    asyncio.create_task(ws_queue_reader())

    if bypass_kws:
        logger.info("NOVA_NO_KWS is set. Starting STT in continuous VAD mode.")
        pipeline_state["fsm_state"] = "LISTENING"
        pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})

    yield

    logger.info("Shutting down workers...")
    pipeline_state["stop_event"].set()
    for p in pipeline_state["processes"].values():
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
    await _close_session()
    if nova_db:
        await nova_db.close()
    logger.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# Mount WebRTC stream — exposes /webrtc/offer for SDP exchange
rtc_stream = Stream(handler=NovaRTCHandler(), modality="audio", mode="send-receive")
rtc_stream.mount(app)

app.mount("/static", StaticFiles(directory="nova/frontend"), name="static")

@app.get("/")
async def get_index():
    with open("nova/frontend/nova-dashboard.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/config")
async def get_config():
    """Public frontend config — only expose keys that are safe to be client-side."""
    return {"maps_api_key": os.getenv("MAPS_DEMO_KEY", "")}

@app.post("/ui-event")
async def handle_ui_event(req: Request):
    """Control endpoint for WebRTC clients (PTT, voice/LLM change)."""
    data = await req.json()
    msg_type = data.get("type")
    if msg_type == "start_ptt":
        pipeline_state["fsm_state"] = "LISTENING"
        pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "LISTENING"})
        # Stop first to clear any active session (avoids stale start guard)
        pipeline_state["stt_in_queue"].put({"type": "stop"})
        pipeline_state["stt_in_queue"].put({"type": "start", "ptt": True})
        pipeline_state["ws_out_queue"].put({"type": "ptt_started"})
    elif msg_type == "stop_ptt":
        pipeline_state["kws_in_queue"].put({"type": "stop_ptt"})
        pipeline_state["stt_in_queue"].put({"type": "tts_unmute"})
        pipeline_state["stt_in_queue"].put({"type": "stop"})
    elif msg_type == "change_voice":
        pipeline_state["current_voice"] = data.get("data")
        pipeline_state["tts_in_queue"].put({"type": "change_voice", "data": data.get("data")})
    elif msg_type == "change_tts_engine":
        pipeline_state["current_tts_engine"] = data.get("data")
        pipeline_state["tts_in_queue"].put({"type": "change_tts_engine", "data": data.get("data")})
    elif msg_type == "change_llm":
        pipeline_state["current_llm"] = data.get("data")
        pipeline_state["llm_in_queue"].put({"type": "change_llm", "data": data.get("data")})
    elif msg_type == "change_stt_variant":
        pipeline_state["current_stt_variant"] = data.get("data")
        pipeline_state["stt_in_queue"].put({"type": "change_stt_variant", "data": data.get("data")})
    return {"status": "ok"}

@app.get("/metrics")
async def metrics_endpoint():
    return get_metrics()

@app.get("/tts/info")
async def tts_info():
    """Return available TTS engines, voices, VRAM estimates, and current selection."""
    from pipeline_mp.llm_config import TTS_VRAM_ESTIMATES

    engines = []
    # Discover voice refs in voices/ dir
    voices_dir = Path(__file__).parent / "voices"
    qwen_voices = []
    if voices_dir.exists():
        for wav in sorted(voices_dir.glob("*.wav")):
            qwen_voices.append({"id": wav.stem, "name": wav.stem.replace("_", " ").title()})

    engines.append({
        "id": "faster-qwen3",
        "name": "FasterQwen3TTS",
        "vram_mb": TTS_VRAM_ESTIMATES.get("faster-qwen3", 2750),
        "voices": qwen_voices,
    })

    pocket_voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    engines.append({
        "id": "pocket-tts",
        "name": "Pocket-TTS",
        "vram_mb": TTS_VRAM_ESTIMATES.get("pocket-tts", 150),
        "voices": [{"id": v, "name": v.title()} for v in pocket_voices],
    })

    # GPU total VRAM for budget calculations
    gpu_total_mb = 0
    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_total_mb = round(float(info.total) / 1024 ** 2)
        except Exception:
            pass

    return {
        "current_engine": pipeline_state.get("current_tts_engine", "pocket-tts"),
        "current_voice": pipeline_state.get("current_voice", "alba"),
        "engines": engines,
        "gpu_total_mb": gpu_total_mb,
    }


@app.get("/llm/info")
async def llm_info():
    """Return available LLM backends (cloud + local) with VRAM estimates."""
    from pipeline_mp.llm_config import LOCAL_LLM_REGISTRY, BASELINE_VRAM_MB

    has_api_key = os.environ.get("OPENROUTER_API_KEY") is not None

    cloud_models = []
    if has_api_key:
        cloud_models = [
            {"id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "name": "Nemotron 49B", "vram_mb": 0},
            {"id": "qwen/qwen3.5-9b", "name": "Qwen 3.5 9B", "vram_mb": 0},
            {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "vram_mb": 0},
            {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B", "vram_mb": 0},
        ]

    local_models = []
    for key, cfg in LOCAL_LLM_REGISTRY.items():
        local_models.append({
            "id": f"local:{key}",
            "name": cfg.display_name,
            "vram_mb": cfg.vram_mb,
            "supports_tools": cfg.supports_tools,
        })

    # GPU total for client-side budget math
    gpu_total_mb = 0
    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_total_mb = round(float(info.total) / 1024 ** 2)
        except Exception:
            pass

    return {
        "has_api_key": has_api_key,
        "current_llm": pipeline_state.get("current_llm", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
        "cloud_models": cloud_models,
        "local_models": local_models,
        "gpu_total_mb": gpu_total_mb,
        "baseline_vram_mb": BASELINE_VRAM_MB,
    }

@app.get("/stt/info")
async def stt_info():
    """Return available STT variants with VRAM estimates and current selection."""
    from pipeline_mp.stt_config import STT_VARIANT_REGISTRY

    variants = []
    for key, cfg in STT_VARIANT_REGISTRY.items():
        variants.append({
            "id": key,
            "name": cfg.display_name,
            "vram_mb": cfg.vram_mb,
            "rtf_target": cfg.rtf_target,
        })

    gpu_total_mb = 0
    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_total_mb = round(float(info.total) / 1024 ** 2)
        except Exception:
            pass

    return {
        "current_variant": pipeline_state.get("current_stt_variant", "small"),
        "variants": variants,
        "gpu_total_mb": gpu_total_mb,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    # Immediately inform this client about enrollment state
    kws_ok   = (KWS_MODELS_DIR / "mlp_weights.pth").exists()
    voice_ok = (VOICEPRINT_DIR / "driver1.enc").exists()
    await websocket.send_text(json.dumps({
        "type": "enrollment_check",
        "kws_enrolled": kws_ok,
        "voice_enrolled": voice_ok,
        "fully_enrolled": kws_ok and voice_ok
    }))
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                audio_bytes = data["bytes"]
                
                # If we are IDLE, feed KWS
                if pipeline_state["fsm_state"] == "IDLE":
                    if os.environ.get("NOVA_NO_KWS") == "1":
                        pass # MicTranscriber handles VAD locally; ignore websocket audio bytes
                    else:
                        pipeline_state["kws_in_queue"].put(audio_bytes)
                # If we are LISTENING, feed STT
                elif pipeline_state["fsm_state"] == "LISTENING":
                    pipeline_state["stt_in_queue"].put(audio_bytes)
                # If GENERATING, still feed KWS to allow Wake Word Interruption!
                elif pipeline_state["fsm_state"] == "GENERATING":
                    pipeline_state["kws_in_queue"].put(audio_bytes)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg["type"] == "start_ptt":
                    # Hard override to listening
                    pipeline_state["fsm_state"] = "LISTENING"
                    pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "LISTENING"})
                    # Stop first to clear any active session (avoids stale start guard)
                    pipeline_state["stt_in_queue"].put({"type": "stop"})
                    pipeline_state["stt_in_queue"].put({"type": "start", "ptt": True})
                    pipeline_state["ws_out_queue"].put({"type": "ptt_started"})

                elif msg["type"] == "stop_ptt":
                    pipeline_state["kws_in_queue"].put({"type": "stop_ptt"})
                    pipeline_state["stt_in_queue"].put({"type": "tts_unmute"})
                    pipeline_state["stt_in_queue"].put({"type": "stop"})
                    # STT worker will emit on_line_completed, then we transition
                    
                elif msg["type"] == "change_voice":
                    pipeline_state["current_voice"] = msg["data"]
                    pipeline_state["tts_in_queue"].put({"type": "change_voice", "data": msg["data"]})
                    
                elif msg["type"] == "change_tts_engine":
                    pipeline_state["current_tts_engine"] = msg["data"]
                    pipeline_state["tts_in_queue"].put({"type": "change_tts_engine", "data": msg["data"]})

                elif msg["type"] == "change_llm":
                    pipeline_state["current_llm"] = msg["data"]
                    pipeline_state["llm_in_queue"].put({"type": "change_llm", "data": msg["data"]})

    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WebSocket closed: {e}")
    finally:
        active_websockets.discard(websocket)

@app.post("/enroll/upload")
async def upload_enrollment(
    file: UploadFile = File(...),
    sample_type: str = Form(...),
    index: str = Form(...),
):
    refs_dir  = Path(__file__).parent / "kws" / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"{sample_type}_{index}.wav"
    file_path = refs_dir / filename
    content   = await file.read()
    import io
    try:
        audio = AudioSegment.from_file(io.BytesIO(content))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(file_path, format="wav")
        logger.info(f"Converted and saved enrollment sample: {filename}")
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        file_path.write_bytes(content)
    return {"status": "success", "filename": filename}

@app.post("/enroll/face")
async def trigger_face_enroll():
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "nova-l7", "L-3"))
        from enroll import enroll_face
        driver_id = "driver1"
        success = await asyncio.to_thread(enroll_face, driver_id, False)
        if success:
            return {"status": "success", "message": "Face ID enrolled successfully"}
        else:
            return {"status": "error", "error": "Could not extract face or no camera found"}
    except Exception as e:
        logger.error(f"Face enrollment failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

@app.post("/enroll/pin")
async def trigger_pin_enroll(pin: str = Form(...)):
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "nova-l7", "L-3"))
        from enroll import enroll_pin
        driver_id = "driver1"
        success = await asyncio.to_thread(enroll_pin, driver_id, pin)
        if success:
            return {"status": "success", "message": "PIN enrolled successfully"}
        else:
            return {"status": "error", "error": "Could not enroll PIN"}
    except Exception as e:
        logger.error(f"PIN enrollment failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

async def _run_enrollment_task():
    global enrollment_status
    enrollment_status = {"state": "running", "message": "Generating synthetic KWS data..."}
    logger.info("Triggering synthetic data generation for KWS hardening...")
    proc = None
    try:
        synth_script = Path(__file__).parent / "kws" / "generate_synthetic_data.py"
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(synth_script)
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        logger.warning("Synthetic data generation timed out after 120s — skipping.")
        try:
            if proc is not None:
                proc.kill()
        except:  # noqa: E722
            pass
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}", exc_info=True)

    # 1. Update KWS Engine (with datetime version tag for rollback)
    from datetime import datetime
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    bypass_kws = os.environ.get("NOVA_NO_KWS") == "1"
    if bypass_kws:
        # KWS worker is in sleep loop when bypassed — run enrollment directly
        try:
            from kws.kws_engine_v2 import StreamingKWSv2, GoogleEmbeddingModel
            kws_model_path = str(Path(__file__).parent / "kws" / "google_speech_embedding.onnx")
            kws_engine = StreamingKWSv2(model_path=None)
            kws_engine.model = GoogleEmbeddingModel(kws_model_path)
            refs_dir = Path(__file__).parent / "kws" / "refs"
            ref_paths = sorted(str(p) for p in refs_dir.glob("nova_*.wav"))
            noise_paths = sorted(str(p) for p in refs_dir.glob("noise_*.wav"))
            if ref_paths:
                logger.info("Running KWS enrollment directly (KWS worker bypassed)...")
                await asyncio.to_thread(kws_engine.enroll, ref_paths, noise_paths, version_tag)
                logger.info("KWS enrollment completed directly.")
        except Exception as e:
            logger.error(f"Direct KWS enrollment failed: {e}", exc_info=True)
    elif pipeline_state["kws_in_queue"]:
        pipeline_state["kws_in_queue"].put({"type": "enroll", "version_tag": version_tag})

    # 2. KWS enrollment done — skip L-3 voiceprint here; use /enroll/voice-sample instead.
    # (KWS samples are short "Nova" utterances — poor for speaker verification.)
    enrollment_status = {"state": "done", "message": "KWS enrollment complete. Now enroll your voice fingerprint."}
    # Notify connected clients
    pipeline_state["ws_out_queue"].put({"type": "enrollment_done"})
    logger.info("KWS enrollment complete. Voice fingerprint enrollment pending via /enroll/voice-sample.")

@app.post("/enroll/re-enroll")
async def trigger_re_enroll():
    global enrollment_status
    enrollment_status = {"state": "running", "message": "Starting KWS enrollment..."}
    asyncio.create_task(_run_enrollment_task())
    return {"status": "processing", "message": "Enrollment and synthetic generation started in background."}

@app.get("/enroll/status")
async def enroll_status():
    return enrollment_status

@app.post("/enroll/voice-sample")
async def enroll_voice_sample(
    file: UploadFile = File(...),
    index: int = Form(...),
    total: int = Form(5),
    driver_id: str = Form("driver1"),
):
    """
    Dedicated voice fingerprint enrollment using longer speech samples.
    Accepts `total` WAV uploads (index 1..total); on the last one, computes
    the mean ECAPA-TDNN embedding and saves it as the driver's encrypted voiceprint.
    driver_id defaults to 'driver1' (owner); use 'driver2' for a family member.
    Max 2 drivers supported.
    """
    global _voice_fp_pending

    # Pause STT on first sample so moonshine doesn't transcribe enrollment phrases
    if index == 1 and pipeline_state.get("stt_in_queue"):
        pipeline_state["stt_in_queue"].put({"type": "stop"})
        logger.info("Paused STT for voice enrollment.")

    if driver_id not in ("driver1", "driver2"):
        return {"status": "error", "error": "Only driver1 and driver2 are supported."}
    import io

    content = await file.read()
    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(content))
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
        tmp_path = Path(__file__).parent / "kws" / "refs" / f"_vp_tmp_{index}.wav"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        audio_seg.export(tmp_path, format="wav")
    except Exception as e:
        return {"status": "error", "error": f"Audio conversion failed: {e}"}

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "nova-l7", "L-3"))
        from verify import extract_live_fingerprint
        fp = await asyncio.to_thread(extract_live_fingerprint, tmp_path)
        _voice_fp_pending.append(fp)
        logger.info(f"Voice fingerprint sample {index}/{total} extracted.")
    except Exception as e:
        logger.error(f"Fingerprint extraction failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        tmp_path.unlink(missing_ok=True)

    if index >= total:
        try:
            from crypto_utils import save_array
            master = np.mean(_voice_fp_pending, axis=0)
            master = master / np.linalg.norm(master)
            vp_dir = Path(__file__).parent / "nova-l7" / "L-3" / "data" / "voiceprints"
            vp_dir.mkdir(parents=True, exist_ok=True)
            save_array(master, vp_dir / f"{driver_id}.enc")
            _voice_fp_pending.clear()
            logger.info(f"Voice fingerprint enrolled and saved for {driver_id}.")
            # Resume STT after enrollment completes
            if pipeline_state.get("stt_in_queue") and os.environ.get("NOVA_NO_KWS") == "1":
                pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})
                pipeline_state["fsm_state"] = "LISTENING"
                logger.info("Resumed STT after voice enrollment.")
            return {"status": "enrolled", "message": f"Voice fingerprint saved for {driver_id}."}
        except Exception as e:
            logger.error(f"Failed to save voiceprint: {e}", exc_info=True)
            # Resume STT even on failure
            if pipeline_state.get("stt_in_queue") and os.environ.get("NOVA_NO_KWS") == "1":
                pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})
                pipeline_state["fsm_state"] = "LISTENING"
            return {"status": "error", "error": str(e)}

    return {"status": "ok", "collected": index, "total": total}


@app.get("/enroll/check")
async def enrollment_check():
    """Returns whether KWS and voice fingerprint for driver1 are enrolled."""
    kws_ok   = (KWS_MODELS_DIR / "mlp_weights.pth").exists()
    voice_ok = (VOICEPRINT_DIR / "driver1.enc").exists()
    return {"kws_enrolled": kws_ok, "voice_enrolled": voice_ok, "fully_enrolled": kws_ok and voice_ok}


@app.get("/enroll/drivers")
async def list_enrolled_drivers():
    """List all enrolled driver IDs (by scanning voiceprints directory)."""
    if not VOICEPRINT_DIR.exists():
        return {"drivers": []}
    drivers = sorted(p.stem for p in VOICEPRINT_DIR.glob("*.enc"))
    return {"drivers": drivers}


@app.delete("/enroll/drivers/{driver_id}")
async def remove_driver(driver_id: str):
    """Remove a driver's voice fingerprint. Owner (driver1) cannot be removed."""
    from fastapi import HTTPException
    if driver_id == "driver1":
        raise HTTPException(status_code=400, detail="Cannot remove the owner account.")
    path = VOICEPRINT_DIR / f"{driver_id}.enc"
    if path.exists():
        path.unlink()
    return {"status": "removed", "driver_id": driver_id}


@app.get("/enroll/kws-models")
async def list_kws_versions():
    """List saved KWS model versions (datetime-stamped directories)."""
    versions_dir = KWS_MODELS_DIR / "versions"
    if not versions_dir.exists():
        return {"versions": [], "active_version": None}
    versions = sorted(
        (d.name for d in versions_dir.iterdir() if d.is_dir() and (d / "mlp_weights.pth").exists()),
        reverse=True
    )
    # Read the persisted active version tag
    active_tag_file = KWS_MODELS_DIR / "active_version.txt"
    active_version = active_tag_file.read_text().strip() if active_tag_file.exists() else None
    # Fall back to most recent if tag not recorded or no longer exists
    if active_version not in versions and versions:
        active_version = versions[0]
    return {"versions": versions, "active_version": active_version}


@app.post("/enroll/kws-activate/{version}")
async def activate_kws_version(version: str):
    """Switch the active KWS model to a previously saved version."""
    import shutil
    v_dir = KWS_MODELS_DIR / "versions" / version
    if not (v_dir / "mlp_weights.pth").exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Version {version} not found.")

    # Persist the active version tag so /enroll/kws-models can report it
    (KWS_MODELS_DIR / "active_version.txt").write_text(version)

    bypass_kws = os.environ.get("NOVA_NO_KWS") == "1"
    if bypass_kws:
        # KWS worker is in sleep loop — it won't read the queue.
        # Copy the version files directly to the active model path.
        for fname in ("mlp_weights.pth", "refs.pkl"):
            src = v_dir / fname
            if src.exists():
                shutil.copy2(src, KWS_MODELS_DIR / fname)
        # Notify all clients immediately (no worker round-trip needed)
        msg = json.dumps({"type": "kws_version_activated", "version": version, "success": True})
        for ws in list(active_websockets):
            try:
                await ws.send_text(msg)
            except Exception:
                pass
        return {"status": "activated", "version": version}

    if pipeline_state["kws_in_queue"]:
        pipeline_state["kws_in_queue"].put({"type": "activate_version", "version": version})
    return {"status": "activating", "version": version}


@app.delete("/enroll/kws-models/{version}")
async def delete_kws_version(version: str):
    """Delete a saved KWS model version."""
    import shutil
    v_dir = KWS_MODELS_DIR / "versions" / version
    if v_dir.exists():
        shutil.rmtree(v_dir)
    return {"status": "deleted", "version": version}


if __name__ == "__main__":
    import uvicorn
    if "--no-kws" in sys.argv:
        os.environ["NOVA_NO_KWS"] = "1"
        sys.argv.remove("--no-kws")
    uvicorn.run(app, host="0.0.0.0", port=8000)
