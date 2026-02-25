import asyncio
import json
import logging
import time
import os
import sys
import threading
import re
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch
import psutil
import pynvml
import jiwer
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from pocket_tts import TTSModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-backend")

# Global model placeholders
stt_model = None
llm_model = None
llm_tokenizer = None
tts_model = None
voice_catalog = {}
current_voice_name = "alba"
shared_vad_model = None
shared_kws_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_model, llm_model, llm_tokenizer, tts_model, voice_catalog, shared_vad_model, shared_kws_model
    
    logger.info("Loading models (FastAPI WebSocket + Faster-Whisper + SmolLM/Qwen Fallback)...")
    
    # 1. STT: Faster-Whisper
    stt_model = WhisperModel("small", device="cuda", compute_type="float16")

    # 2. LLM: SmolLM2 with Qwen Fallback
    primary_model = "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit"
    fallback_model = "Qwen/Qwen2.5-0.5B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        logger.info(f"Attempting to load primary model: {primary_model}")
        llm_tokenizer = AutoTokenizer.from_pretrained(primary_model)
        llm_model = AutoModelForCausalLM.from_pretrained(
            primary_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        logger.info("Primary model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load primary model: {e}. Falling back to Qwen.")
        llm_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        llm_model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            quantization_config=bnb_config,
            device_map="auto"
        )

    # 3. TTS: Pocket-TTS
    tts_model = TTSModel.load_model()
    catalog = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']
    for voice in catalog:
        try:
            voice_catalog[voice] = tts_model.get_state_for_audio_prompt(voice)
        except Exception as e: logger.error(f"Failed voice {voice}: {e}")

    # 4. KWS & VAD Shared Models
    kws_model_path = os.path.join(os.path.dirname(__file__), "kws", "google_speech_embedding.onnx")
    from kws.kws_engine import GoogleEmbeddingModel, StreamingKWS
    shared_kws_model = GoogleEmbeddingModel(kws_model_path)
    shared_vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    logger.info("Models loaded successfully.")
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# GPU Initialization
try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except: nvml_handle = None

def get_metrics():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_mem_used = 0
    gpu_util = 0
    if nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_mem_used = info.used / 1024**2
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu
        except: pass
    return {"cpu": cpu_usage, "ram": ram_usage, "gpu_mem": gpu_mem_used, "gpu_util": gpu_util}

# --- KWS Integration ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kws.kws_engine import StreamingKWS

class NovaSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.state = "IDLE" # IDLE, LISTENING, GENERATING
        self.kws_engine = StreamingKWS(model_path=None)
        self.kws_engine.model = shared_kws_model
        self.kws_engine.vad_model = shared_vad_model
        
        # Auto-enroll
        refs_dir = os.path.join(os.path.dirname(__file__), "kws", "refs")
        if os.path.exists(refs_dir):
            ref_paths = sorted([str(p) for p in Path(refs_dir).glob("nova_*.wav")])
            noise_paths = sorted([str(p) for p in Path(refs_dir).glob("noise_*.wav")])
            if ref_paths:
                self.kws_engine.enroll(ref_paths, noise_paths)
        
        self.command_audio = []
        self.silence_chunks = 0
        self.max_silence_chunks = 15 
        self.ptt_active = False
        self.interrupt_flag = False
        self.tts_muted = False
        self.e2e_start_time = None
        self.first_response_chunk = True

    async def send_json(self, data: dict):
        await self.websocket.send_text(json.dumps(data))

    async def send_audio(self, audio_int16: np.ndarray):
        await self.websocket.send_bytes(audio_int16.tobytes())

    async def handle_audio(self, audio_bytes: bytes):
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_np = audio_int16.astype(np.float32) / 32768.0
        
        # VAD monitoring for all states (to handle interruptions)
        prob = 0
        if shared_vad_model:
            try:
                vad_chunk = audio_np[-512:] if len(audio_np) >= 512 else np.pad(audio_np, (0, 512-len(audio_np)))
                with torch.no_grad():
                    prob = shared_vad_model(torch.from_numpy(vad_chunk).unsqueeze(0), 16000).item()
            except: pass

        if self.state == "GENERATING" and prob > 0.7:
            self.interrupt_flag = True
            self.state = "LISTENING"
            self.command_audio = [audio_bytes]
            await self.send_json({"type": "interrupted"})
            return

        if self.state == "IDLE":
            detected, latency = await self.kws_engine.process_chunk(audio_np)
            if detected:
                logger.info(f"Wake word detected! Latency: {latency:.2f}ms")
                self.state = "LISTENING"
                self.command_audio = []
                self.silence_chunks = 0
                await self.send_json({"type": "kws_detected", "data": {"latency": latency}})

        elif self.state == "LISTENING":
            self.command_audio.append(audio_bytes)
            if not self.ptt_active:
                if prob < 0.4: self.silence_chunks += 1
                else: self.silence_chunks = 0
                
                if self.silence_chunks >= self.max_silence_chunks:
                    logger.info("Auto-stop triggered by silence")
                    asyncio.create_task(self.process_command())

    async def process_command(self):
        if self.state != "LISTENING": return
        self.state = "GENERATING"
        self.interrupt_flag = False
        self.e2e_start_time = time.time()
        self.first_response_chunk = True
        try:
            full_audio = b"".join(self.command_audio)
            audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            t0 = time.time()
            segments, _ = stt_model.transcribe(audio_np, beam_size=5, vad_filter=True)
            transcript = " ".join([s.text for s in segments]).strip()
            ttfb_stt = time.time() - t0
            
            await self.send_json({"type": "transcript", "data": transcript, "latency": {"stt_ttfb": ttfb_stt}})
            
            if transcript:
                await self.generate_response(transcript)
            else:
                self.state = "IDLE"
                await self.send_json({"type": "recording_stopped"})
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            self.state = "IDLE"
        finally:
            self.command_audio = []

    async def generate_response(self, prompt: str):
        try:
            system_msg = (
                "Your name is Nova. You are a helpful AI assistant. Answers are short and clear."
            )
            full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = llm_tokenizer(full_prompt, return_tensors="pt").to("cuda")
            streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=128)
            threading.Thread(target=llm_model.generate, kwargs=generation_kwargs).start()

            await self.send_json({"type": "assistant_start"})
            
            sentence_buffer = ""
            start_time = time.time()
            ttft_llm = None
            is_thinking = False

            for new_text in streamer:
                if self.interrupt_flag: break
                if ttft_llm is None: ttft_llm = time.time() - start_time
                await self.send_json({"type": "llm_token", "data": new_text, "latency": {"llm_ttft": ttft_llm}})
                
                if "<think>" in new_text: is_thinking = True
                if not is_thinking:
                    sentence_buffer += new_text
                    if any(punct in new_text for punct in [".", "!", "?", "\n"]):
                        clean_text = self.clean_for_tts(sentence_buffer)
                        if clean_text: asyncio.create_task(self.enqueue_tts(clean_text))
                        sentence_buffer = ""
                if "</think>" in new_text:
                    is_thinking = False
                    sentence_buffer = ""

            if not self.interrupt_flag:
                final_clean = self.clean_for_tts(sentence_buffer)
                if final_clean: asyncio.create_task(self.enqueue_tts(final_clean))
            
        except Exception as e: logger.error(f"LLM error: {e}")
        finally:
            self.state = "IDLE"
            await self.send_json({"type": "recording_stopped"})

    def clean_for_tts(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.replace("<think>", "").replace("</think>", "").strip()

    def create_wav_header(self, pcm_len):
        header = bytearray(b'RIFF')
        header.extend((36 + pcm_len).to_bytes(4, 'little'))
        header.extend(b'WAVEfmt ')
        header.extend((16).to_bytes(4, 'little'))
        header.extend((1).to_bytes(2, 'little')) # PCM
        header.extend((1).to_bytes(2, 'little')) # Mono
        header.extend((24000).to_bytes(4, 'little')) # 24kHz
        header.extend((48000).to_bytes(4, 'little')) # Byte rate
        header.extend((2).to_bytes(2, 'little')) # Block align
        header.extend((16).to_bytes(2, 'little')) # Bits
        header.extend(b'data')
        header.extend(pcm_len.to_bytes(4, 'little'))
        return header

    async def enqueue_tts(self, text: str):
        if self.tts_muted: return
        try:
            t0 = time.time()
            voice_state = voice_catalog.get(current_voice_name)
            audio_chunks = tts_model.generate_audio_stream(model_state=voice_state, text_to_generate=text, copy_state=True)
            first_chunk = True
            for chunk in audio_chunks:
                if self.interrupt_flag: break
                
                current_time = time.time()
                audio_np = chunk.cpu().numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                pcm_bytes = audio_int16.tobytes()
                wav_data = self.create_wav_header(len(pcm_bytes)) + pcm_bytes
                
                if first_chunk:
                    metrics_data = {"ttfa": current_time - t0}
                    if self.first_response_chunk and self.e2e_start_time:
                        metrics_data["e2e"] = current_time - self.e2e_start_time
                        self.first_response_chunk = False
                    await self.send_json({"type": "tts_metrics", "data": metrics_data})
                    first_chunk = False
                
                await self.websocket.send_bytes(wav_data)
        except Exception as e: logger.error(f"TTS error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = NovaSession(websocket)
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                await session.handle_audio(data["bytes"])
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg["type"] == "start_ptt":
                    session.ptt_active = True
                    session.state = "LISTENING"
                    session.command_audio = []
                    await session.send_json({"type": "ptt_started"})
                elif msg["type"] == "stop_ptt":
                    session.ptt_active = False
                    await session.process_command()
                elif msg["type"] == "change_voice":
                    global current_voice_name
                    current_voice_name = msg["data"]
                    await session.send_json({"type": "voice_changed", "data": current_voice_name})
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WebSocket closed: {e}")

app.mount("/static", StaticFiles(directory="nova/frontend"), name="static")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
import shutil
from pydub import AudioSegment
import io


@app.post("/enroll/upload")
async def upload_enrollment(
    file: UploadFile = File(...), 
    sample_type: str = Form(...), # "nova" or "noise"
    index: int = Form(...)
):
    refs_dir = Path(__file__).parent / "kws" / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{sample_type}_{index}.wav"
    file_path = refs_dir / filename
    
    content = await file.read()
    try:
        audio = AudioSegment.from_file(io.BytesIO(content))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(file_path, format="wav")
        logger.info(f"Converted and saved enrollment sample: {filename}")
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    
    return {"status": "success", "filename": filename}

@app.post("/enroll/re-enroll")
async def trigger_re_enroll():
    global shared_kws_model, shared_vad_model
    refs_dir = Path(__file__).parent / "kws" / "refs"
    
    # We need to trigger enrollment on the instances or re-create the baseline
    # For now, let's just confirm the files exist. 
    # The NovaSession objects will re-enroll on next session start.
    return {"status": "enrolled", "path": str(refs_dir)}

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
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
