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
    torchaudio.list_audio_backends = lambda: ["ffmpeg"] # Dummy backend to satisfy speechbrain

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
    "current_llm": "qwen/qwen3.5-9b",
    "current_voice": "alba",
    # Echo suppression: when to stop blocking mic input
    # Computed as: last_audio_out_time + remaining_playback_duration + tail
    "tts_suppress_until": 0.0,
    # Running count of 24kHz int16 samples buffered but not yet "played" by browser
    "_tts_pending_samples": 0,
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

# NVML for metrics
try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    nvml_handle = None

def get_metrics():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_mem_total = gpu_util = 0

    # Map component name to its PID
    pid_map = {name: p.pid for name, p in pipeline_state["processes"].items() if p and p.pid}
    gpu_metrics = {}

    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_mem_total = info.used / 1024 ** 2
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu

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
    using_openrouter = os.environ.get("OPENROUTER_API_KEY") is not None
    current_llm = pipeline_state.get("current_llm", "qwen/qwen3.5-9b")
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

    return {
        "cpu": cpu_usage,
        "ram": ram_usage,
        "gpu_mem": round(gpu_mem_total, 1),
        "gpu_util": round(gpu_util, 1),
        "components": {
            "stt": get_comp_info("stt", "Moonshine (Medium)", True),
            "llm": get_comp_info("llm", current_llm, not using_openrouter),
            "tts": get_comp_info("tts", f"Pocket-TTS ({current_voice})", True),
            "kws": get_comp_info("kws", "StreamingKWS V2", True),
        },
    }


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
                    # Only honour if we're still in GENERATING — ignore stale done from an interrupted TTS
                    if pipeline_state["fsm_state"] == "GENERATING":
                        # Calculate remaining browser playback time before unmuting STT.
                        # Audio chunks are buffered in the browser — unmuting too early
                        # causes moonshine to transcribe TTS speaker output (echo).
                        now = time.time()
                        remaining_play = 0.0
                        if pipeline_state["_tts_last_chunk_time"] > 0:
                            elapsed = now - pipeline_state["_tts_last_chunk_time"]
                            pending = max(0, pipeline_state["_tts_pending_samples"] - int(elapsed * 24000))
                            remaining_play = pending / 24000.0
                        suppress_delay = remaining_play + 1.0  # +1s safety margin for speaker reverb
                        pipeline_state["_tts_pending_samples"] = 0
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
                    # STT found nothing — always return to IDLE (no interrupt scenario)
                    if pipeline_state["fsm_state"] != "GENERATING":
                        pipeline_state["kws_in_queue"].put({"type": "set_state", "state": "IDLE"})
                        if os.environ.get("NOVA_NO_KWS") == "1":
                            pipeline_state["fsm_state"] = "LISTENING"
                            pipeline_state["stt_in_queue"].put({"type": "start", "ptt": False})
                        else:
                            pipeline_state["fsm_state"] = "IDLE"

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
                    audio_bytes = msg.get("bytes")
                    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16)
                    # Echo suppression: track how long audio will play in browser.
                    # TTS is 24kHz int16 (2 bytes/sample).  Accumulate pending samples
                    # and recompute suppress_until = now + remaining_play_time + 500ms tail.
                    now = time.time()
                    new_samples = len(audio_arr)
                    elapsed = now - pipeline_state["_tts_last_chunk_time"] if pipeline_state["_tts_last_chunk_time"] else 0.0
                    played_samples = int(elapsed * 24000)
                    pending = max(0, pipeline_state["_tts_pending_samples"] - played_samples) + new_samples
                    pipeline_state["_tts_pending_samples"] = pending
                    pipeline_state["_tts_last_chunk_time"] = now
                    pipeline_state["tts_suppress_until"] = now + (pending / 24000.0) + 0.5
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
    logger.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# Mount WebRTC stream — exposes /webrtc/offer for SDP exchange
rtc_stream = Stream(handler=NovaRTCHandler(), modality="audio", mode="send-receive")
rtc_stream.mount(app)

app.mount("/static", StaticFiles(directory="nova/frontend"), name="static")

@app.get("/")
async def get_index():
    with open("nova/frontend/index.html") as f:
        return HTMLResponse(content=f.read())

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
    elif msg_type == "change_llm":
        pipeline_state["current_llm"] = data.get("data")
        pipeline_state["llm_in_queue"].put({"type": "change_llm", "data": data.get("data")})
    return {"status": "ok"}

@app.get("/metrics")
async def metrics_endpoint():
    return get_metrics()

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
    if pipeline_state["kws_in_queue"]:
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
        return {"versions": []}
    versions = sorted(
        (d.name for d in versions_dir.iterdir() if d.is_dir() and (d / "mlp_weights.pth").exists()),
        reverse=True
    )
    return {"versions": versions}


@app.post("/enroll/kws-activate/{version}")
async def activate_kws_version(version: str):
    """Switch the active KWS model to a previously saved version."""
    v_dir = KWS_MODELS_DIR / "versions" / version
    if not (v_dir / "mlp_weights.pth").exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Version {version} not found.")
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
