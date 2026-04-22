"""
Qwen3-ASR worker (in-process) for Nova pipeline.

Qwen3-ASR-0.6B is an offline encoder-decoder ASR model (Qwen3 backbone, Apache-2.0).
It has no native PyTorch streaming API — only its vLLM backend streams.
This worker approximates streaming for Nova by:

    1. Accumulating the session's 16 kHz int16 audio into a ring buffer.
    2. Running webrtcvad on every 20 ms frame to track speech / silence.
    3. While speech is active, every PARTIAL_INTERVAL re-transcribing the
       growing buffer via `model.transcribe((np_audio, 16000))` and emitting
       `transcript_partial` to the frontend.
    4. On END_OF_SPEECH_SILENCE seconds of contiguous trailing silence,
       running one final transcribe and emitting `transcript` + pushing to
       `llm_in_queue` — same contract as stt_moonshine_worker.py.

Env vars:
    NOVA_STT_VARIANT                qwen3_asr_0_6b (or any backend=qwen3 entry)
    NOVA_QWEN3_DEVICE               default "cuda"
    NOVA_QWEN3_DTYPE                default "bfloat16" — bfloat16 | float16 | float32
    NOVA_QWEN3_PARTIAL_INTERVAL_MS  default 800  — min ms between partials
    NOVA_QWEN3_VAD_AGGRESSIVENESS   default 3    — webrtcvad 0..3 (3 = strictest)
    NOVA_QWEN3_END_SILENCE_MS       default 800  — trailing silence → final
    NOVA_QWEN3_PARTIAL_WINDOW_SEC   default 5.0  — rolling window for partials
                                                   (final still uses full utterance)
    NOVA_QWEN3_STABILITY_TICKS      default 3    — consecutive identical partials → final
                                                   (fallback when VAD keeps re-triggering
                                                    on breath/mouth noise)
    NOVA_QWEN3_MAX_SEG_SEC          default 12   — hard cap on accumulated buffer
    NOVA_QWEN3_MAX_NEW_TOKENS       default 256
    NOVA_QWEN3_LANGUAGE             default ""   — passed to model.transcribe(language=...)
                                                   empty → auto-detect

Contract (matches stt_moonshine_worker / stt_kyutai_worker):
    Input queue  (stt_in_queue)  : bytes (PCM 16 kHz int16) + control dicts
                                    {type: start|stop|interrupted|tts_mute|tts_unmute|change_stt_variant}
    Output queue (ws_out_queue)  : speech_started, transcript_partial, transcript,
                                    recording_stopped, generation_start,
                                    stt_variant_loading, stt_variant_changed, stt_variants_available
    LLM queue    (llm_in_queue)  : {type: "text", text, audio_data?}
"""

from __future__ import annotations

import logging
import os
import sys
import time
import multiprocessing as mp
import multiprocessing.synchronize
from queue import Empty

logger = logging.getLogger("Qwen3STTWorker")


NOVA_SAMPLE_RATE = 16_000
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = NOVA_SAMPLE_RATE * VAD_FRAME_MS // 1000  # 320 samples


def run_stt_worker(
    stt_in_queue: mp.Queue,  # type: ignore[type-arg]
    llm_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
):
    import setproctitle
    setproctitle.setproctitle("nova-qwen3-stt")

    pipeline_mp_dir = os.path.dirname(os.path.abspath(__file__))
    if pipeline_mp_dir not in sys.path:
        sys.path.insert(0, pipeline_mp_dir)

    try:
        import numpy as np
        import torch
        import webrtcvad
    except ImportError as e:
        logger.error(f"Qwen3 STT worker missing base deps: {e}")
        ws_out_queue.put({
            "type": "stt_variant_loading",
            "data": {"variant": "qwen3", "status": "error", "error": str(e)},
        })
        return

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        logger.error(
            f"qwen-asr not installed: {e}. "
            f"Install with: uv pip install qwen-asr"
        )
        ws_out_queue.put({
            "type": "stt_variant_loading",
            "data": {"variant": "qwen3", "status": "error", "error": f"qwen-asr missing: {e}"},
        })
        return

    from stt_config import STT_SETTINGS, STT_VARIANT_REGISTRY

    # ── Config ────────────────────────────────────────────────────────────────
    device = os.getenv("NOVA_QWEN3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = os.getenv("NOVA_QWEN3_DTYPE", "bfloat16")
    partial_interval_s = max(0.1, int(os.getenv("NOVA_QWEN3_PARTIAL_INTERVAL_MS", "800")) / 1000.0)
    vad_aggressiveness = max(0, min(3, int(os.getenv("NOVA_QWEN3_VAD_AGGRESSIVENESS", "3"))))
    end_silence_s = max(0.2, int(os.getenv("NOVA_QWEN3_END_SILENCE_MS", "800")) / 1000.0)
    partial_window_s = max(1.0, float(os.getenv("NOVA_QWEN3_PARTIAL_WINDOW_SEC", "5.0")))
    max_seg_s = max(2.0, float(os.getenv("NOVA_QWEN3_MAX_SEG_SEC", "12")))
    # Silent-frame counter threshold: N contiguous silent frames before end-of-utterance
    end_silence_frames = max(1, int(round(end_silence_s * 1000 / VAD_FRAME_MS)))
    # Speech-start hysteresis: require N contiguous voiced frames before emitting speech_started
    speech_start_frames = max(1, int(os.getenv("NOVA_QWEN3_SPEECH_START_FRAMES", "3")))
    # Stability-based end-of-utterance: N consecutive identical partial transcripts
    # triggers finalize even when webrtcvad keeps reporting speech on breath noise.
    stability_ticks = max(1, int(os.getenv("NOVA_QWEN3_STABILITY_TICKS", "3")))
    max_new_tokens = int(os.getenv("NOVA_QWEN3_MAX_NEW_TOKENS", "256"))

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

    preemph_enabled = os.getenv("NOVA_QWEN3_PREEMPH", "1") == "1"
    rms_norm_enabled = os.getenv("NOVA_QWEN3_RMS_NORM", "1") == "1"

    active_variant = STT_SETTINGS.active
    cfg = STT_VARIANT_REGISTRY.get(active_variant)
    if cfg is None or cfg.backend != "qwen3":
        logger.error(f"Active variant '{active_variant}' is not a qwen3 variant.")
        return

    # Qwen3-ASR takes full English language names ("English", "Chinese", ...).
    # Map our ISO codes from stt_config so auto-detect doesn't drift to
    # Mandarin/Malay on noisy audio.
    _QWEN3_LANG_MAP = {
        "en": "English", "zh": "Chinese", "yue": "Cantonese",
        "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
        "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
        "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
        "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
        "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
        "pl": "Polish", "cs": "Czech", "fil": "Filipino", "fa": "Persian",
        "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
    }
    _env_lang = os.getenv("NOVA_QWEN3_LANGUAGE", "").strip()
    _raw_lang = _env_lang or cfg.language.strip()
    language_override = _QWEN3_LANG_MAP.get(_raw_lang.lower(), _raw_lang) or None
    logger.info(f"Qwen3: language pin = {language_override!r}")

    # ── Model load ────────────────────────────────────────────────────────────
    ws_out_queue.put({
        "type": "stt_variant_loading",
        "data": {"variant": active_variant, "status": "loading", "device": device},
    })
    logger.info(f"Loading Qwen3-ASR {cfg.qwen3_hf_repo} on {device} ({dtype_name}) …")

    try:
        model = Qwen3ASRModel.from_pretrained(
            cfg.qwen3_hf_repo,
            dtype=torch_dtype,
            device_map=device,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        logger.error(f"Qwen3-ASR load failed: {e}")
        ws_out_queue.put({
            "type": "stt_variant_loading",
            "data": {"variant": active_variant, "status": "error", "error": str(e)},
        })
        return

    logger.info(f"Qwen3-ASR {cfg.display_name} ready on {device}.")
    ws_out_queue.put({
        "type": "stt_variant_loading",
        "data": {"variant": active_variant, "status": "ready", "device": device},
    })
    ws_out_queue.put({
        "type": "stt_variant_changed",
        "data": {"variant": active_variant, "display_name": cfg.display_name},
    })
    ws_out_queue.put({
        "type": "stt_variants_available",
        "data": {
            "variants": {
                k: {"display_name": v.display_name, "vram_mb": v.vram_mb, "backend": v.backend}
                for k, v in STT_VARIANT_REGISTRY.items()
            },
            "active": active_variant,
            "device": device,
        },
    })

    vad = webrtcvad.Vad(vad_aggressiveness)

    # ── Audio preprocessing (same stage as Moonshine worker) ──────────────────
    _preemph_coeff = np.float32(0.97)
    _preemph_prev = np.float32(0.0)
    _rms_target = 0.08
    _rms_floor = 1e-6

    def preprocess(pcm_f32: "np.ndarray") -> "np.ndarray":
        nonlocal _preemph_prev
        if len(pcm_f32) == 0:
            return pcm_f32
        out = np.empty_like(pcm_f32)
        if preemph_enabled:
            out[0] = pcm_f32[0] - _preemph_coeff * _preemph_prev
            out[1:] = pcm_f32[1:] - _preemph_coeff * pcm_f32[:-1]
            _preemph_prev = pcm_f32[-1]
        else:
            out[:] = pcm_f32
        if rms_norm_enabled:
            rms = float(np.sqrt(np.mean(out ** 2)))
            if rms > _rms_floor:
                out = out * (_rms_target / rms)
                np.clip(out, -1.0, 1.0, out=out)
        return out

    # ── Session state ─────────────────────────────────────────────────────────
    state = {
        "session_active": False,
        "is_ptt": False,
        "should_listen": False,
        "tts_muted": False,
        "session_start_time": 0.0,
    }

    # Rolling buffers for the current utterance
    utter_f32: list[np.ndarray] = []      # preprocessed float32 samples (for Qwen3)
    utter_raw_f32: list[np.ndarray] = []  # raw (un-preemph'd) float32 — for ECAPA voice verify
    utter_i16_tail: bytearray = bytearray()  # leftover int16 bytes < VAD frame
    last_partial_time = 0.0
    last_partial_text = ""
    speech_started_emitted = False
    # Frame-counter hysteresis replaces wall-clock last_speech_time:
    # resistant to single noise blips that used to hold the session open.
    silent_frame_count = 0
    speech_frame_count = 0
    # Stability-based fallback: counts consecutive partials that returned the
    # same text. When webrtcvad stays "speech" on breath noise, this is what
    # actually triggers finalize.
    stable_partial_count = 0

    NO_SPEECH_TIMEOUT = float(os.getenv("NOVA_QWEN3_NO_SPEECH_TIMEOUT", "2.5"))

    def _reset_utterance():
        nonlocal last_partial_time, last_partial_text, speech_started_emitted
        nonlocal silent_frame_count, speech_frame_count, stable_partial_count
        utter_f32.clear()
        utter_raw_f32.clear()
        utter_i16_tail.clear()
        last_partial_time = 0.0
        last_partial_text = ""
        speech_started_emitted = False
        silent_frame_count = 0
        speech_frame_count = 0
        stable_partial_count = 0

    def _end_session():
        state["session_active"] = False
        state["should_listen"] = False
        state["is_ptt"] = False

    def _maybe_emit_partial(now_ts: float) -> bool:
        """
        Run a partial transcribe on the rolling window, emit if text changed,
        and track stability. Returns True when N consecutive partials returned
        the identical text — the caller should finalize.
        """
        nonlocal last_partial_text, last_partial_time, stable_partial_count
        audio_np = _current_audio_np(window_seconds=partial_window_s)
        if audio_np.size <= NOVA_SAMPLE_RATE // 4:  # < 250 ms
            last_partial_time = now_ts
            return False
        text, _ = _transcribe(audio_np)
        last_partial_time = now_ts
        if not text:
            return False
        if text == last_partial_text:
            stable_partial_count += 1
        else:
            stable_partial_count = 0
            last_partial_text = text
            ws_out_queue.put({"type": "transcript_partial", "data": text})
        return stable_partial_count >= stability_ticks

    def _current_audio_np(window_seconds: float | None = None) -> "np.ndarray":
        """Concatenated utterance audio. If window_seconds is set, return only the tail."""
        if not utter_f32:
            return np.zeros(0, dtype=np.float32)
        full = np.concatenate(utter_f32, dtype=np.float32)
        if window_seconds is None:
            return full
        window_samples = int(window_seconds * NOVA_SAMPLE_RATE)
        if full.size <= window_samples:
            return full
        return full[-window_samples:]

    def _transcribe(audio_np: "np.ndarray") -> tuple[str, str | None]:
        """Run Qwen3-ASR. Returns (text, detected_language)."""
        if audio_np.size == 0:
            return "", None
        try:
            results = model.transcribe(
                audio=[(audio_np, NOVA_SAMPLE_RATE)],
                language=[language_override] if language_override else None,
            )
        except Exception as e:
            logger.warning(f"Qwen3-ASR transcribe error: {e}")
            return "", None
        if not results:
            return "", None
        r = results[0]
        text = (getattr(r, "text", "") or "").strip()
        lang = getattr(r, "language", None)
        return text, lang

    def _emit_final():
        nonlocal last_partial_text
        audio_np = _current_audio_np()
        if audio_np.size == 0:
            ws_out_queue.put({"type": "recording_stopped"})
            _reset_utterance()
            return
        t0 = time.time()
        text, _ = _transcribe(audio_np)
        dt = time.time() - t0
        duration = audio_np.size / NOVA_SAMPLE_RATE
        rtf = dt / duration if duration > 0 else 0.0
        logger.info(f"Qwen3 Final: '{text}' | Latency: {dt:.2f}s | RTF: {rtf:.2f}")
        ws_out_queue.put({
            "type": "transcript",
            "data": text,
            "latency": {"stt_ttfb": dt, "stt_rtf": rtf},
        })
        if text:
            payload: dict = {"type": "text", "text": text}
            if utter_raw_f32:
                raw_concat = np.concatenate(utter_raw_f32, dtype=np.float32)
                payload["audio_data"] = raw_concat.tolist()
            llm_in_queue.put(payload)
            ws_out_queue.put({"type": "generation_start"})
        else:
            ws_out_queue.put({"type": "recording_stopped"})
        last_partial_text = ""
        _reset_utterance()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while not stop_event.is_set():
        try:
            msg = stt_in_queue.get(timeout=0.05)
        except Empty:
            # Idle checks: end-of-speech silence, no-speech timeout
            if state["session_active"] and not state["is_ptt"]:
                now = time.time()
                # No-speech timeout: session opened but nothing voiced yet
                if (
                    not speech_started_emitted
                    and state["session_start_time"] > 0
                    and (now - state["session_start_time"]) > NO_SPEECH_TIMEOUT
                ):
                    logger.warning(f"No speech detected within {NO_SPEECH_TIMEOUT}s — ending session.")
                    _end_session()
                    ws_out_queue.put({"type": "recording_stopped"})
                    _reset_utterance()
                    continue
                # End-of-utterance: speech started, N contiguous silent frames
                if (
                    speech_started_emitted
                    and silent_frame_count >= end_silence_frames
                ):
                    silence_ms = silent_frame_count * VAD_FRAME_MS
                    logger.info(f"Qwen3: end-of-utterance ({silence_ms}ms silence, {silent_frame_count} frames).")
                    _emit_final()
                    _end_session()
                    continue
                # Partial emission tick (speech active, no new audio this tick)
                if (
                    speech_started_emitted
                    and last_partial_time > 0
                    and (now - last_partial_time) >= partial_interval_s
                ):
                    if _maybe_emit_partial(now):
                        logger.info(
                            f"Qwen3: stability-triggered finalize "
                            f"({stable_partial_count} identical partials)."
                        )
                        _emit_final()
                        _end_session()
                        continue
            continue

        if isinstance(msg, bytes):
            if not state["should_listen"] or state["tts_muted"]:
                continue
            # Decode int16 → float32 for model, keep int16 bytes for VAD
            pcm_i16 = np.frombuffer(msg, dtype=np.int16)
            if pcm_i16.size == 0:
                continue
            pcm_f32_raw = pcm_i16.astype(np.float32) / 32768.0
            utter_raw_f32.append(pcm_f32_raw.copy())
            pcm_f32 = preprocess(pcm_f32_raw)
            utter_f32.append(pcm_f32)

            # Cap buffer to max_seg_s to keep re-transcribe cost bounded
            total_samples = sum(chunk.size for chunk in utter_f32)
            if total_samples > int(max_seg_s * NOVA_SAMPLE_RATE):
                logger.warning(f"Qwen3: buffer exceeded {max_seg_s}s, forcing final.")
                _emit_final()
                if not state["is_ptt"]:
                    _end_session()
                continue

            # VAD on original int16 (webrtcvad expects int16 PCM mono)
            utter_i16_tail.extend(msg)
            now = time.time()
            while len(utter_i16_tail) >= VAD_FRAME_SAMPLES * 2:
                frame = bytes(utter_i16_tail[: VAD_FRAME_SAMPLES * 2])
                del utter_i16_tail[: VAD_FRAME_SAMPLES * 2]
                try:
                    is_speech = vad.is_speech(frame, NOVA_SAMPLE_RATE)
                except Exception:
                    is_speech = False
                if is_speech:
                    silent_frame_count = 0
                    speech_frame_count += 1
                    if (
                        not speech_started_emitted
                        and speech_frame_count >= speech_start_frames
                    ):
                        speech_started_emitted = True
                        last_partial_time = now
                        ws_out_queue.put({"type": "speech_started"})
                else:
                    speech_frame_count = 0
                    if speech_started_emitted:
                        silent_frame_count += 1

            # Opportunistic partial while audio is streaming in
            if (
                speech_started_emitted
                and (now - last_partial_time) >= partial_interval_s
            ):
                if _maybe_emit_partial(now):
                    logger.info(
                        f"Qwen3: stability-triggered finalize "
                        f"({stable_partial_count} identical partials)."
                    )
                    _emit_final()
                    if not state["is_ptt"]:
                        _end_session()
                    continue
            continue

        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type")

        if msg_type == "start":
            if not state["session_active"]:
                state["session_active"] = True
                state["is_ptt"] = msg.get("ptt", False)
                state["session_start_time"] = time.time()
                state["tts_muted"] = False
                state["should_listen"] = True
                _reset_utterance()
                logger.info(f"Qwen3 STT Session Started (PTT: {state['is_ptt']})")

        elif msg_type == "stop":
            if state["session_active"]:
                age = time.time() - state["session_start_time"]
                if age < 0.5 and not state["is_ptt"]:
                    logger.info(f"Ignoring stale stop (age={age:.2f}s)")
                    continue
                logger.info("Qwen3 STT Session Stop — flushing final.")
                state["should_listen"] = False
                _emit_final()
                _end_session()

        elif msg_type == "interrupted":
            if state["session_active"]:
                logger.info("Qwen3 STT Session Interrupted")
                _end_session()
                _reset_utterance()

        elif msg_type == "tts_mute":
            state["tts_muted"] = True

        elif msg_type == "tts_unmute":
            state["tts_muted"] = False

        elif msg_type == "change_stt_variant":
            requested = msg.get("data", "")
            if requested == active_variant:
                continue
            requested_cfg = STT_VARIANT_REGISTRY.get(requested)
            if requested_cfg is None:
                logger.warning(f"Unknown variant '{requested}'.")
                continue
            if requested_cfg.backend != "qwen3":
                logger.warning(
                    f"Cross-backend switch rejected (qwen3 → {requested_cfg.backend}). "
                    f"Restart with NOVA_STT_VARIANT={requested}."
                )
                ws_out_queue.put({
                    "type": "stt_variant_loading",
                    "data": {
                        "variant": requested,
                        "status": "error",
                        "error": "cross-backend switch requires restart",
                    },
                })
                continue
            if state["session_active"]:
                logger.warning("Cannot switch variant mid-session.")
                continue
            logger.info(f"Qwen3: in-backend switch not yet implemented ({active_variant} → {requested}).")

    logger.info("Qwen3 STT Worker exiting.")
