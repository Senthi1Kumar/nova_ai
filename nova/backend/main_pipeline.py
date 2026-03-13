import io
import asyncio
import json
import logging
import time
import os
import sys
import threading
import random
from pathlib import Path
from contextlib import asynccontextmanager
from warnings import filterwarnings
from dotenv import load_dotenv

import numpy as np
import torch
import psutil
import pynvml
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from pocket_tts import TTSModel
from queue import Queue

filterwarnings("ignore")
load_dotenv()

# Add L7 path
L7_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova-l7", "L-7")
sys.path.append(L7_PATH)
try:
    from dialogue_manager import DialogueManager
except ImportError as e:
    DialogueManager = None

# Add L3 path
L3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nova-l7", "L-3")
sys.path.append(L3_PATH)
try:
    from crypto_utils import save_array
    from verify import extract_live_fingerprint
    from enroll import enroll_face, enroll_pin
except ImportError as e:
    pass

from pipeline.thread_manager import ThreadManager
from pipeline.vad_handler import VADHandler
from pipeline.stt_handler import STTHandler
from pipeline.llm_handler import LLMHandler
from pipeline.tts_handler import TTSHandler
from kws.kws_engine import GoogleEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-backend")

stt_model = None
llm_model = None
llm_tokenizer = None
openai_client = None
tts_model = None
voice_catalog = {}
current_voice_name = "alba"
current_llm_name = "qwen/qwen3.5-9b"
shared_vad_model = None
shared_kws_model = None

# We only allow 1 active session in this edge architecture for now
active_pipeline = None
active_ws = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        stt_model, \
        llm_model, \
        llm_tokenizer, \
        openai_client, \
        tts_model, \
        voice_catalog, \
        shared_vad_model, \
        shared_kws_model

    primary_model = "Qwen/Qwen3.5-0.8B"
    fallback_model = "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit"

    logger.info("Loading models for pipeline...")

    stt_model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info("OpenRouter initialized.")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(primary_model)
            llm_model = AutoModelForCausalLM.from_pretrained(
                primary_model, quantization_config=bnb_config, device_map="auto"
            )
        except Exception:
            llm_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            llm_model = AutoModelForCausalLM.from_pretrained(
                fallback_model, quantization_config=bnb_config, device_map="auto"
            )

    tts_model = TTSModel.load_model().to("cuda" if torch.cuda.is_available() else "cpu")
    for voice in ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]:
        try:
            state = tts_model.get_state_for_audio_prompt(voice)
            for module_name, module_state in state.items():
                for key, tensor in module_state.items():
                    if isinstance(tensor, torch.Tensor):
                        state[module_name][key] = tensor.to(tts_model.device)
            voice_catalog[voice] = state
        except Exception:
            pass

    kws_model_path = os.path.join(os.path.dirname(__file__), "kws", "google_speech_embedding.onnx")
    shared_kws_model = GoogleEmbeddingModel(kws_model_path)
    shared_vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")

    logger.info("Models loaded.")
    yield
    if active_pipeline:
        active_pipeline.stop()

app = FastAPI(lifespan=lifespan)

try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    nvml_handle = None

def get_metrics():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_mem_used = gpu_util = 0
    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_mem_used = info.used / 1024 ** 2
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu
        except Exception:
            pass

    return {
        "cpu": cpu_usage,
        "ram": ram_usage,
        "gpu_mem": gpu_mem_used,
        "gpu_util": gpu_util,
        "components": {
            "stt": {"name": "Faster-Whisper", "vram_mb": 480, "load": "Med", "wer": "4.5%"},
            "llm": {"name": "API" if openai_client else "Local", "vram_mb": 0 if openai_client else 600, "load": "High", "accuracy": "N/A", "f1": "N/A"},
            "tts": {"name": "Pocket-TTS", "vram_mb": 1200, "load": "Very High"},
            "kws": {"name": "DTW + MLP", "vram_mb": 15, "load": "Ultra-Low", "far": "0.01%", "frr": "1.5%", "trr": "99.99%"},
        },
    }

def build_threaded_pipeline(ws_queue):
    stop_event = threading.Event()
    q_vad_in = Queue()
    q_stt_in = Queue()
    q_llm_in = Queue()
    q_tts_in = Queue()

    dm = DialogueManager() if DialogueManager else None

    vad = VADHandler(stop_event, q_vad_in, q_stt_in, setup_args=(ws_queue, shared_vad_model, shared_kws_model))
    stt = STTHandler(stop_event, q_stt_in, q_llm_in, setup_args=(ws_queue, stt_model))
    llm = LLMHandler(stop_event, q_llm_in, q_tts_in, setup_args=(ws_queue, dm, llm_model, llm_tokenizer, openai_client))
    tts = TTSHandler(stop_event, q_tts_in, ws_queue, setup_args=(ws_queue, tts_model, voice_catalog))

    manager = ThreadManager([vad, stt, llm, tts])
    manager.start()
    return manager, q_vad_in, stop_event

async def websocket_sender(websocket, ws_queue, stop_event, q_vad_in):
    while not stop_event.is_set():
        try:
            # We use a non-blocking get to allow async sleep and check for stop_event
            msg = await asyncio.to_thread(ws_queue.get, True, 0.1)
            
            if isinstance(msg, bytes):
                await websocket.send_bytes(msg)
            elif isinstance(msg, dict):
                if msg.get("type") == "recording_stopped":
                    q_vad_in.put({"type": "generation_done", "next_state": "IDLE"})
                elif msg.get("type") == "ptt_started":
                    q_vad_in.put({"type": "generation_done", "next_state": "LISTENING"})
                
                await websocket.send_text(json.dumps(msg))
                
        except Exception as e:
            if "Empty" not in str(type(e)):
                pass
        await asyncio.sleep(0.001)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_pipeline, active_ws
    await websocket.accept()
    
    if active_pipeline:
        active_pipeline.stop()
        
    active_ws = websocket
    ws_queue = Queue()
    
    # 1. Build and start the pipeline threads
    active_pipeline, q_vad_in, stop_event = build_threaded_pipeline(ws_queue)
    
    # 2. Start the async sender task to push queue data to frontend
    sender_task = asyncio.create_task(websocket_sender(websocket, ws_queue, stop_event, q_vad_in))
    
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                # Instantly drop audio bytes into the threaded VAD queue
                q_vad_in.put(data["bytes"])
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg["type"] == "start_ptt":
                    q_vad_in.put({"type": "start_ptt"})
                elif msg["type"] == "stop_ptt":
                    q_vad_in.put({"type": "stop_ptt"})
                elif msg["type"] == "change_voice":
                    # We can pass config changes via queue
                    q_vad_in.put(msg)
    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket closed")
    finally:
        stop_event.set()
        if active_pipeline:
            active_pipeline.stop()
        sender_task.cancel()

app.mount("/static", StaticFiles(directory="nova/frontend"), name="static")

@app.get("/")
async def get_index():
    with open("nova/frontend/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/metrics")
async def metrics_endpoint():
    return get_metrics()

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    import sys

    if "--no-kws" in sys.argv:
        os.environ["NOVA_NO_KWS"] = "1"
        sys.argv.remove("--no-kws")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)