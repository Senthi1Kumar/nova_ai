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

    Supports runtime variant switching via "change_stt_variant" queue message.
    Falls back to CPU automatically when CUDA is unavailable.
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
        import torch
    except ImportError as e:
        logger.error(f"Failed to import moonshine_voice or dependencies: {e}")
        return

    # Add pipeline_mp to sys.path so stt_config is importable
    pipeline_mp_dir = os.path.dirname(os.path.abspath(__file__))
    if pipeline_mp_dir not in sys.path:
        sys.path.insert(0, pipeline_mp_dir)

    from stt_config import STT_SETTINGS, STT_VARIANT_REGISTRY

    # ── Device detection ──────────────────────────────────────────────────────
    has_cuda = torch.cuda.is_available()
    device_label = "CUDA" if has_cuda else "CPU"
    if not has_cuda:
        logger.warning(
            "CUDA not available — Moonshine will run on CPU. "
            "Expect higher latency. Consider using 'tiny' variant for better RTF."
        )

    # ── Model loader / unloader ───────────────────────────────────────────────
    _transcriber_state: dict = {
        "transcriber": None,
        "mic_stream":  None,
        "variant":     STT_SETTINGS.active,
        "listener":    None,
    }

    def _load_variant(variant_key: str):
        """Load a Moonshine variant, replacing any currently loaded model."""
        if variant_key not in STT_VARIANT_REGISTRY:
            logger.warning(f"Unknown STT variant '{variant_key}', keeping current.")
            return

        cfg = STT_VARIANT_REGISTRY[variant_key]
        arch = getattr(ModelArch, cfg.model_arch_name)

        ws_out_queue.put({
            "type": "stt_variant_loading",
            "data": {"variant": variant_key, "status": "loading", "device": device_label}
        })
        logger.info(f"Loading Moonshine {cfg.display_name} on {device_label} …")

        try:
            model_path, model_arch = get_model_for_language(cfg.language, arch)
            new_transcriber = Transcriber(
                model_path=model_path,
                model_arch=model_arch,
                options={"return_audio_data": "1"},
            )
            new_mic_stream = new_transcriber.create_stream(cfg.vad_threshold)
            _transcriber_state["transcriber"] = new_transcriber
            _transcriber_state["mic_stream"]  = new_mic_stream
            _transcriber_state["variant"]     = variant_key

            # Re-attach the listener if one already exists
            if _transcriber_state["listener"] is not None:
                new_mic_stream.add_listener(_transcriber_state["listener"])

            logger.info(f"Moonshine {cfg.display_name} ready on {device_label}.")
            ws_out_queue.put({
                "type": "stt_variant_loading",
                "data": {"variant": variant_key, "status": "ready", "device": device_label}
            })
            ws_out_queue.put({
                "type": "stt_variant_changed",
                "data": {"variant": variant_key, "display_name": cfg.display_name}
            })
        except Exception as e:
            logger.error(f"Failed to load Moonshine {variant_key}: {e}")
            ws_out_queue.put({
                "type": "stt_variant_loading",
                "data": {"variant": variant_key, "status": "error", "error": str(e)}
            })

    # Initial load
    _load_variant(STT_SETTINGS.active)

    if _transcriber_state["transcriber"] is None:
        logger.error("STT: initial model load failed — worker exiting.")
        return

    # ── Audio preprocessing ──────────────────────────────────────────────────
    # Pre-emphasis boosts high-frequency energy (consonants, fricatives) that
    # are critical for speech recognition but often attenuated by cheap mics
    # and in-car noise environments.  Coefficient 0.97 is the standard choice.
    _preemph_coeff = 0.97
    _preemph_prev = np.float32(0.0)   # carry-over sample across chunks
    # RMS normalization target: -22 dBFS ≈ 0.08 RMS.  Ensures consistent
    # input level regardless of mic gain or user distance from mic.
    _rms_target = 0.08
    _rms_floor  = 1e-6   # avoid divide-by-zero on silence

    def preprocess_audio(pcm_f32: np.ndarray) -> np.ndarray:
        """Pre-emphasis filter + RMS normalization on a float32 PCM chunk."""
        nonlocal _preemph_prev
        # Pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        out = np.empty_like(pcm_f32)
        if len(pcm_f32) == 0:
            return pcm_f32
        out[0] = pcm_f32[0] - _preemph_coeff * _preemph_prev
        out[1:] = pcm_f32[1:] - _preemph_coeff * pcm_f32[:-1]
        _preemph_prev = pcm_f32[-1]

        # RMS normalization — scale chunk so its RMS matches the target level.
        rms = np.sqrt(np.mean(out ** 2))
        if rms > _rms_floor:
            out = out * (_rms_target / rms)
            # Soft-clip to [-1, 1] to avoid digital clipping after gain
            np.clip(out, -1.0, 1.0, out=out)
        return out

    # ── Session state ─────────────────────────────────────────────────────────
    state = {
        "session_active": False,
        "is_ptt": False,
        "should_listen": False,
        "tts_muted": False,
        "stream_started": False,
        "ptt_stopping": False,
        "ptt_stop_time": 0.0,
        "session_start_time": 0.0,
    }

    PTT_DRAIN_TIMEOUT = 3.0
    # Safety timeout: if KWS triggers but no speech is ever detected,
    # end the session after this many seconds to avoid hanging in LISTENING.
    NO_SPEECH_TIMEOUT = 8.0

    def _ensure_stream_started():
        if not state["stream_started"]:
            mic_stream = _transcriber_state["mic_stream"]
            if mic_stream is None:
                return
            try:
                mic_stream.start()
                state["stream_started"] = True
                logger.info("Moonshine mic_stream started (persistent).")
            except Exception as e:
                logger.error(f"Failed to start mic_stream: {e}")

    def _end_session():
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

            ws_out_queue.put({
                "type": "transcript",
                "data": transcript,
                "latency": {"stt_ttfb": latency_ms / 1000.0, "stt_rtf": rtf}
            })

            llm_payload = {
                "type": "text",
                "text": transcript,
            }
            if event.line.audio_data:
                llm_payload["audio_data"] = event.line.audio_data

            if transcript:
                llm_in_queue.put(llm_payload)
                ws_out_queue.put({"type": "generation_start"})
            else:
                ws_out_queue.put({"type": "recording_stopped"})

            if state["is_ptt"]:
                logger.info("PTT: line completed, ending session.")
                _end_session()
            else:
                logger.info("Auto-listen: line completed, pausing until generation_done.")
                state["should_listen"] = False
                state["session_active"] = False

    listener = STTListener()
    _transcriber_state["listener"] = listener
    _transcriber_state["mic_stream"].add_listener(listener)

    # Report available variants and current config to gateway
    ws_out_queue.put({
        "type": "stt_variants_available",
        "data": {
            "variants": {
                k: {"display_name": v.display_name, "vram_mb": v.vram_mb}
                for k, v in STT_VARIANT_REGISTRY.items()
            },
            "active": _transcriber_state["variant"],
            "device": device_label,
        }
    })

    while not stop_event.is_set():
        try:
            msg = stt_in_queue.get(timeout=0.05)
        except Empty:
            if state["ptt_stopping"] and (time.time() - state["ptt_stop_time"]) > PTT_DRAIN_TIMEOUT:
                logger.warning("PTT drain timeout — forcing session end.")
                _end_session()
                ws_out_queue.put({"type": "recording_stopped"})
            elif (
                state["session_active"]
                and not state["is_ptt"]
                and not state["ptt_stopping"]
                and state["session_start_time"] > 0
                and (time.time() - state["session_start_time"]) > NO_SPEECH_TIMEOUT
            ):
                logger.warning("No speech detected within timeout — ending session.")
                _end_session()
                ws_out_queue.put({"type": "recording_stopped"})
            continue

        if isinstance(msg, bytes):
            if state["should_listen"] and not state["tts_muted"]:
                mic_stream = _transcriber_state["mic_stream"]
                if mic_stream is not None:
                    audio = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                    audio = preprocess_audio(audio)
                    try:
                        mic_stream.add_audio(audio, 16000)
                    except Exception as e:
                        if "VAD is not active" not in str(e):
                            logger.debug(f"add_audio error: {e}")
            continue

        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type")

        if msg_type == "change_stt_variant":
            requested = msg.get("data", "")
            if requested == _transcriber_state["variant"]:
                continue
            if state["session_active"]:
                logger.warning("Cannot switch STT variant mid-session — ignoring.")
                continue
            # Reset stream_started so the new stream can be started fresh
            state["stream_started"] = False
            _load_variant(requested)

        elif msg_type == "start":
            if not state["session_active"]:
                state["session_active"] = True
                state["is_ptt"] = msg.get("ptt", False)
                state["ptt_stopping"] = False
                state["session_start_time"] = time.time()
                logger.info(f"STT Session Started (PTT: {state['is_ptt']})")
                _ensure_stream_started()
                state["should_listen"] = True

        elif msg_type == "stop":
            if state["session_active"]:
                age = time.time() - state["session_start_time"]
                if age < 0.5 and not state["is_ptt"]:
                    logger.info(f"Ignoring stale stop (session age={age:.2f}s < 0.5s)")
                    continue
                if state["is_ptt"]:
                    logger.info("PTT released — draining buffered audio...")
                    state["should_listen"] = False
                    state["ptt_stopping"] = True
                    state["ptt_stop_time"] = time.time()
                else:
                    logger.info("STT Session Stopped (auto-listen).")
                    _end_session()
                    ws_out_queue.put({"type": "recording_stopped"})

        elif msg_type == "interrupted":
            if state["session_active"]:
                logger.info("STT Session Interrupted")
                _end_session()

        elif msg_type == "tts_mute":
            state["tts_muted"] = True

        elif msg_type == "tts_unmute":
            state["tts_muted"] = False

    try:
        mic_stream = _transcriber_state["mic_stream"]
        transcriber = _transcriber_state["transcriber"]
        if mic_stream:
            mic_stream.close()
        if transcriber:
            transcriber.close()
    except Exception:
        pass

    logger.info("STT Worker exiting.")
