import logging
import time
import numpy as np
from pipeline.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class STTHandler(BaseHandler):
    """
    Handles Speech-to-Text inference using Faster-Whisper.
    Reads from: `queue_in` (full accumulated audio buffer from VAD)
    Writes to: `queue_out` (transcript text to LLM)
    """

    def setup(self, ws_queue, stt_model):
        self.ws_queue = ws_queue
        self.stt_model = stt_model

    def process(self, full_audio_bytes):
        # Allow passing custom instructions/interrupts through the queue
        if isinstance(full_audio_bytes, dict):
            # If interrupted, pass the message down the chain
            if full_audio_bytes.get("type") == "interrupted":
                yield full_audio_bytes
            return

        logger.info("STT: Processing audio chunk...")
        
        # Convert bytes to numpy float32
        audio_np = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        t0 = time.time()
        # Run Faster-Whisper transcribe
        try:
            segments, _ = self.stt_model.transcribe(
                audio_np,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 1000, "speech_pad_ms": 400},
                language="en",
            )
            segments = list(segments)
        except Exception as e:
            logger.error(f"STT: Transcription error: {e}")
            return
            
        ttfb_stt = time.time() - t0
        audio_dur = len(audio_np) / 16000.0
        stt_rtf = ttfb_stt / audio_dur if audio_dur > 0 else 0

        # Hallucination guard: If it's mostly silence, don't pass an empty string
        avg_no_speech = (
            sum(s.no_speech_prob for s in segments) / len(segments)
            if segments else 1.0
        )
        
        transcript = (
            " ".join(s.text for s in segments).strip()
            if avg_no_speech < 0.6
            else ""
        )

        logger.info(f"STT: Extracted Text: '{transcript}'")
        
        # Send STT latency metrics to the UI immediately
        self.ws_queue.put({
            "type": "transcript",
            "data": transcript,
            "latency": {"stt_ttfb": ttfb_stt, "stt_rtf": stt_rtf},
        })

        # Yield a tuple containing the transcript AND the original audio array (Layer 3 biometric needs the raw audio)
        yield {
            "type": "text", 
            "text": transcript, 
            "audio_np": audio_np
        }
