import sys
import os
import logging
import time
import multiprocessing as mp
import multiprocessing.synchronize
from queue import Empty

# Monkeypatch torchaudio for speechbrain compatibility
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]  # type: ignore[attr-defined]

logger = logging.getLogger("STTWorker")

def run_stt_worker(
    stt_in_queue: mp.Queue,  # type: ignore[type-arg]
    llm_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
):
    """
    Multiprocessing worker for Moonshine STT using MicTranscriber.
    audio -> mic capture -> VAD -> speaker identification -> STT -> app action
    """
    import setproctitle
    setproctitle.setproctitle("nova-stt-worker")

    # Setup paths for intent_classifier
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    l7_path = os.path.join(backend_dir, "nova-l7", "L-7")
    sys.path.append(l7_path)

    try:
        from moonshine_voice import (
            Transcriber,
            TranscriptEventListener,
            get_model_for_language,
            ModelArch,
        )
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        logger.error(f"Failed to import moonshine_voice or dependencies: {e}")
        return

    logger.info("Initializing Moonshine STT (Medium Streaming)...")

    try:
        model_path, model_arch = get_model_for_language("en", ModelArch.MEDIUM_STREAMING)
        transcriber = Transcriber(model_path=model_path, model_arch=model_arch, options={"return_audio_data": "1"})
        mic_stream = transcriber.create_stream(0.5)

        # Intent recognition is handled by DialogueManager in LLM worker
        # (IntentClassifier loads gemma-300m there; no need for a second copy here)

        logger.info("Moonshine STT initialized and waiting for commands.")
    except Exception as e:
        logger.error(f"Failed to load Moonshine models: {e}")
        return

    state = {
        "session_active": False,
        "is_ptt": False,
        "should_listen": False,
        "tts_muted": False,       # Echo suppression: mute mic while TTS is playing
        "stream_started": False,   # mic_stream started once, never stopped (C lib doesn't support restart)
        "ptt_stopping": False,     # PTT released — waiting for on_line_completed or timeout
        "ptt_stop_time": 0.0,      # When PTT stop was requested (for timeout)
        "session_start_time": 0.0, # When session was started (to reject stale stops)
    }

    PTT_DRAIN_TIMEOUT = 3.0  # Max seconds to wait for moonshine to finish after PTT release

    def audio_callback(in_data, _frames, _time, _status):
        if not state["should_listen"] or state["tts_muted"]:
            return
        if in_data is not None:
            try:
                audio_data = in_data.astype(np.float32).flatten()
                mic_stream.add_audio(audio_data, 16000)
            except Exception as e:
                error_msg = str(e)
                if "VAD is not active" in error_msg or "moonshine-c-api" in error_msg.lower():
                    pass
                else:
                    logger.debug(f"Audio callback error: {e}")

    sd_stream = sd.InputStream(
        samplerate=16000,
        blocksize=1024,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    sd_stream.start()

    def _ensure_stream_started():
        """Start mic_stream once. Never stop it — the C library doesn't support restart."""
        if not state["stream_started"]:
            try:
                mic_stream.start()
                state["stream_started"] = True
                logger.info("Moonshine mic_stream started (persistent).")
            except Exception as e:
                logger.error(f"Failed to start mic_stream: {e}")

    def _end_session():
        """End the current STT session without touching mic_stream."""
        state["session_active"] = False
        state["should_listen"] = False
        state["is_ptt"] = False
        state["ptt_stopping"] = False

    class STTListener(TranscriptEventListener):
        def on_line_started(self, _event):
            if state["session_active"]:
                ws_out_queue.put({"type": "speech_started"})

        def on_line_text_changed(self, event):
            if state["session_active"] and event.line.text:
                ws_out_queue.put({
                    "type": "transcript_partial",
                    "data": event.line.text
                })

        def on_line_completed(self, event):
            if not state["session_active"]:
                return

            transcript = event.line.text.strip()

            latency_ms = event.line.last_transcription_latency_ms
            duration = event.line.duration
            rtf = (latency_ms / 1000.0) / duration if duration > 0 else 0

            speaker_info = f"[Speaker #{event.line.speaker_index}] " if event.line.has_speaker_id else ""
            logger.info(f"STT Final: {speaker_info}'{transcript}' | Latency: {latency_ms/1000.0:.2f}s | RTF: {rtf:.2f}")

            # Send transcript and metrics
            ws_out_queue.put({
                "type": "transcript",
                "data": transcript,
                "latency": {"stt_ttfb": latency_ms / 1000.0, "stt_rtf": rtf}
            })

            # Send to LLM worker with audio data for Layer 3
            llm_payload = {
                "type": "text",
                "text": transcript,
            }
            if event.line.audio_data:
                llm_payload["audio_data"] = event.line.audio_data

            if transcript or (event.line.audio_data and not transcript):
                llm_in_queue.put(llm_payload)
                ws_out_queue.put({"type": "generation_start"})
            else:
                ws_out_queue.put({"type": "recording_stopped"})

            # In PTT mode, on_line_completed means we're done — end the session
            if state["is_ptt"]:
                logger.info("PTT: line completed, ending session.")
                _end_session()
            # In auto-listen mode, keep session alive — just pause listening
            # until generation_done restarts it via "start" message
            else:
                logger.info("Auto-listen: line completed, pausing until generation_done.")
                state["should_listen"] = False
                state["session_active"] = False

    listener = STTListener()
    mic_stream.add_listener(listener)

    while not stop_event.is_set():
        try:
            msg = stt_in_queue.get(timeout=0.05)
        except Empty:
            # Check PTT drain timeout
            if state["ptt_stopping"] and (time.time() - state["ptt_stop_time"]) > PTT_DRAIN_TIMEOUT:
                logger.warning("PTT drain timeout — forcing session end.")
                _end_session()
                ws_out_queue.put({"type": "recording_stopped"})
            continue

        if isinstance(msg, dict):
            if msg.get("type") == "start":
                if not state["session_active"]:
                    state["session_active"] = True
                    state["is_ptt"] = msg.get("ptt", False)
                    state["ptt_stopping"] = False
                    state["session_start_time"] = time.time()
                    logger.info(f"STT Session Started (PTT: {state['is_ptt']})")
                    _ensure_stream_started()
                    state["should_listen"] = True

            elif msg.get("type") == "stop":
                if state["session_active"]:
                    # Reject stale stops that arrive within 0.5s of session start.
                    # These are leftovers from previous flows racing through mp.Queue.
                    age = time.time() - state["session_start_time"]
                    if age < 0.5 and not state["is_ptt"]:
                        logger.info(f"Ignoring stale stop (session age={age:.2f}s < 0.5s)")
                        continue
                    if state["is_ptt"]:
                        # PTT release: stop feeding audio but let moonshine drain
                        # on_line_completed will end the session, or timeout will
                        logger.info("PTT released — draining buffered audio...")
                        state["should_listen"] = False
                        state["ptt_stopping"] = True
                        state["ptt_stop_time"] = time.time()
                    else:
                        # Auto-listen stop from gateway (silence timeout or explicit stop)
                        logger.info("STT Session Stopped (auto-listen).")
                        _end_session()
                        # Notify gateway so it can restart the listen cycle
                        ws_out_queue.put({"type": "recording_stopped"})

            elif msg.get("type") == "interrupted":
                if state["session_active"]:
                    logger.info("STT Session Interrupted")
                    _end_session()

            elif msg.get("type") == "tts_mute":
                state["tts_muted"] = True

            elif msg.get("type") == "tts_unmute":
                state["tts_muted"] = False

        elif isinstance(msg, bytes):
            pass

    try:
        sd_stream.stop()
        sd_stream.close()
        mic_stream.close()
        transcriber.close()
    except Exception:
        pass

    logger.info("STT Worker exiting.")
