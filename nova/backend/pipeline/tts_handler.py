import logging
import time
import numpy as np
import torch
from pipeline.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class TTSHandler(BaseHandler):
    """
    Handles Text-to-Speech generation using Pocket-TTS.
    Reads from: `queue_in` (text sentences from LLM)
    Writes to: `queue_out` (int16 PCM audio bytes to WebSocket Streamer)
    """

    def setup(self, ws_queue, tts_model, voice_catalog):
        self.ws_queue = ws_queue
        self.tts_model = tts_model
        self.voice_catalog = voice_catalog
        
        self.current_voice_name = "alba"
        self.tts_muted = False
        
        # Audio chunks are pushed back to the websocket queue
        # For simplicity, we assume ws_queue acts as our audio output queue, 
        # but in a pure HF pipeline, we'd yield them to the send_audio_chunks_queue
        # We will yield raw bytes.

    def process(self, message):
        if not isinstance(message, dict):
            return
            
        if message.get("type") == "interrupted":
            # TTS should stop its generator immediately.
            # BaseHandler takes care of reading from the queue, 
            # so if we receive interrupt here, we just ignore any further TTS until the next command.
            return
            
        if message.get("type") == "trigger_listen":
            # Pass state change to UI
            self.ws_queue.put({"type": "ptt_started"})
            return

        if message.get("type") == "text_to_speak":
            text = message.get("text")
            if not text or self.tts_muted:
                return
                
            yield from self._generate_audio_stream(text)

    def _generate_audio_stream(self, text: str):
        try:
            t0 = time.time()
            voice_state = self.voice_catalog.get(self.current_voice_name)
            
            # This is a synchronous generator
            audio_gen = self.tts_model.generate_audio_stream(
                model_state=voice_state, text_to_generate=text, copy_state=True
            )
            
            first_chunk = True
            
            # Important: The HF repo yields Numpy arrays or Bytes
            for chunk in audio_gen:
                if self.stop_event.is_set():
                    break
                    
                now = time.time()
                audio_np = chunk.cpu().numpy()
                audio_i16 = (audio_np * 32767).astype(np.int16)
                pcm_bytes = audio_i16.tobytes()
                
                if first_chunk:
                    chunk_dur = len(pcm_bytes) / 2 / 24000.0
                    ttfa = now - t0
                    tts_rtf = ttfa / chunk_dur if chunk_dur > 0 else 0
                    self.ws_queue.put({"type": "tts_metrics", "data": {"ttfa": ttfa, "tts_rtf": tts_rtf}})
                    first_chunk = False
                    
                # We yield the bytes so the pipeline can send it to the WebSocketStreamer
                yield pcm_bytes

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
