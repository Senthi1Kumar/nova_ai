import logging
import numpy as np
import torch
import time
import os
from pipeline.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class VADHandler(BaseHandler):
    """
    Handles Voice Activity Detection, Silence detection, and Wake Word (KWS) integration.
    Reads from: `queue_in` (raw audio chunks from WebSocket)
    Writes to: `queue_out` (accumulated full audio buffer to STT)
    """

    def setup(self, ws_queue, vad_model, kws_model):
        self.ws_queue = ws_queue
        self.vad_model = vad_model
        self.kws_model = kws_model
        
        self.state = "IDLE"  # IDLE | LISTENING | GENERATING
        self.command_audio = []
        self.silence_chunks = 0
        self.max_silence_chunks = 4
        self.ptt_active = False
        
        # KWS Engine instance
        from kws.kws_engine import StreamingKWS
        self.kws_engine = StreamingKWS(model_path=None)
        self.kws_engine.model = self.kws_model
        self.kws_engine.vad_model = self.vad_model
        
        self.bypass_kws = os.environ.get("NOVA_NO_KWS") == "1"

    def process(self, audio_bytes):
        # We expect raw bytes from the WebSocket
        if isinstance(audio_bytes, dict) and audio_bytes.get("type") == "stop_ptt":
            self.ptt_active = False
            if self.state == "LISTENING" and self.command_audio:
                logger.info("VAD: PTT stopped, dispatching audio.")
                yield self._dispatch_audio()
            return
            
        if isinstance(audio_bytes, dict) and audio_bytes.get("type") == "start_ptt":
            self.ptt_active = True
            self.state = "LISTENING"
            self.command_audio = []
            self.ws_queue.put({"type": "ptt_started"})
            return

        if isinstance(audio_bytes, dict) and audio_bytes.get("type") == "interrupted":
            self.state = "IDLE"
            self.command_audio = []
            return

        if isinstance(audio_bytes, dict) and audio_bytes.get("type") == "generation_done":
            self.state = audio_bytes.get("next_state", "IDLE")
            if self.state == "LISTENING":
                self.command_audio = []
                self.silence_chunks = 0
            return

        if not isinstance(audio_bytes, bytes):
            return

        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_np = audio_int16.astype(np.float32) / 32768.0

        # Run VAD for interruption checking
        prob = 0.0
        if self.vad_model:
            try:
                vad_chunk = audio_np[-512:] if len(audio_np) >= 512 else np.pad(audio_np, (0, 512 - len(audio_np)))
                with torch.no_grad():
                    prob = self.vad_model(torch.from_numpy(vad_chunk).unsqueeze(0), 16000).item()
            except Exception:
                pass

        # Handle Interruption logic (if we are generating, and someone starts speaking loudly)
        if self.state == "GENERATING" and prob > 0.7:
            # We must signal the TTS/LLM to stop
            self.ws_queue.put({"type": "interrupted", "trigger": "vad"})
            self.kws_engine.unmute()
            self.state = "LISTENING"
            self.command_audio = [audio_bytes]
            return

        if self.state == "IDLE":
            if self.bypass_kws:
                self.state = "LISTENING"
                self.command_audio = [audio_bytes]
                self.silence_chunks = 0
                self.ws_queue.put({"type": "kws_detected", "data": {"latency": 0.0}})
            else:
                detected, latency = self.kws_engine.process_chunk(audio_np) # We need to await this if async, but we are in a thread!
                # Wait, kws_engine.process_chunk is an async method in main.py, but in the thread we need sync.
                # Actually, in the kws_engine it's probably sync or we can run it.
                # Assuming we refactored KWS to be synchronous or we wrap it.
                if detected:
                    logger.info(f"Wake word detected! Latency: {latency:.2f}ms")
                    self.state = "LISTENING"
                    self.command_audio = []
                    self.silence_chunks = 0
                    self.ws_queue.put({"type": "kws_detected", "data": {"latency": latency}})

        elif self.state == "LISTENING":
            self.command_audio.append(audio_bytes)
            if not self.ptt_active:
                if prob < 0.4:
                    self.silence_chunks += 1
                else:
                    self.silence_chunks = 0
                    
                if self.silence_chunks >= self.max_silence_chunks:
                    logger.info("VAD: Auto-stop triggered by silence.")
                    yield self._dispatch_audio()

    def _dispatch_audio(self):
        full_audio = b"".join(self.command_audio)
        self.state = "GENERATING"
        self.command_audio = []
        return full_audio
