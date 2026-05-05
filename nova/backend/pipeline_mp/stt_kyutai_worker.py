"""
Kyutai STT worker (in-process) for Nova pipeline.

Loads the Kyutai STT model via `moshi.models.loaders.CheckpointInfo.from_hf_repo`
and runs a persistent streaming session on GPU in a background thread. Nova
feeds 16 kHz int16 bytes through stt_in_queue; the worker resamples to 24 kHz,
encodes with mimi, decodes with LMGen, and emits the same event contract as
stt_moonshine_worker so main.py needs no backend-specific code.

Env vars:
    NOVA_STT_VARIANT        kyutai_stt_1b_en_fr | kyutai_stt_2_6b_en
    NOVA_KYUTAI_EMA_THRESH  default 0.5 — EMA above this = end-of-turn
    NOVA_KYUTAI_HEAD_IDX    default 2   — VAD head (2 = 2s pause)
    NOVA_KYUTAI_DEVICE      default cuda
"""

from __future__ import annotations

import logging
import math
import os
import queue as _queue
import sys
import threading
import time
import multiprocessing as mp
import multiprocessing.synchronize
from queue import Empty

logger = logging.getLogger("KyutaiSTTWorker")


NOVA_SAMPLE_RATE = 16_000
N_STEPS_TO_WAIT = 12   # first few steps' prs are noisy (matches Unmute)


class _PauseEMA:
    """Minimal EMA mirroring Unmute's pause-prediction tracker."""
    def __init__(self, attack_time: float, release_time: float, initial: float):
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = initial

    def update(self, dt: float, new_value: float) -> None:
        tau = self.release_time if new_value > self.value else self.attack_time
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
        self.value += alpha * (new_value - self.value)

    def reset(self, value: float = 1.0) -> None:
        self.value = value


def run_stt_worker(
    stt_in_queue: mp.Queue,  # type: ignore[type-arg]
    llm_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
):
    import setproctitle
    setproctitle.setproctitle("nova-kyutai-stt")

    pipeline_mp_dir = os.path.dirname(os.path.abspath(__file__))
    if pipeline_mp_dir not in sys.path:
        sys.path.insert(0, pipeline_mp_dir)

    try:
        import numpy as np
        import torch
        import julius
        import moshi.models
        from moshi.models.loaders import CheckpointInfo
    except ImportError as e:
        logger.error(f"Kyutai STT worker missing deps: {e}")
        ws_out_queue.put({
            "type": "stt_variant_loading",
            "data": {"variant": "kyutai", "status": "error", "error": str(e)}
        })
        return

    from stt_config import STT_SETTINGS, STT_VARIANT_REGISTRY

    ema_threshold = float(os.getenv("NOVA_KYUTAI_EMA_THRESH", "0.7"))
    # Trailing-silence finalize used when the variant has no semantic VAD heads
    # (e.g. kyutai_stt_2_6b_en). Time since last new text token after which we
    # treat the turn as ended.
    trailing_silence_s = float(os.getenv("NOVA_KYUTAI_TRAILING_SILENCE_S", "0.8"))
    vad_head_idx = int(os.getenv("NOVA_KYUTAI_HEAD_IDX", "2"))
    device = os.getenv("NOVA_KYUTAI_DEVICE", "cuda")
    preemph_enabled = os.getenv("NOVA_KYUTAI_PREEMPH", "1") == "1"
    rms_norm_enabled = os.getenv("NOVA_KYUTAI_RMS_NORM", "1") == "1"

    active_variant = STT_SETTINGS.active
    cfg = STT_VARIANT_REGISTRY.get(active_variant)
    if cfg is None or cfg.backend != "kyutai":
        logger.error(f"Active variant '{active_variant}' is not a Kyutai variant.")
        return

    # ── Model load ────────────────────────────────────────────────────────────
    ws_out_queue.put({
        "type": "stt_variant_loading",
        "data": {"variant": active_variant, "status": "loading", "device": device},
    })
    try:
        info = CheckpointInfo.from_hf_repo(cfg.kyutai_hf_repo)
        mimi = info.get_mimi(device=device)
        tokenizer = info.get_text_tokenizer()
        lm = info.get_moshi(device=device, dtype=torch.bfloat16)
        lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
        audio_silence_prefix_seconds = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)
        padding_token_id = info.raw_config.get("text_padding_token_id", 3)
        kyutai_sr = int(mimi.sample_rate)
        frame_size = int(mimi.frame_size)
        frame_rate = float(mimi.frame_rate)
    except Exception as e:
        logger.exception(f"Failed to load Kyutai model '{cfg.kyutai_hf_repo}'")
        ws_out_queue.put({
            "type": "stt_variant_loading",
            "data": {"variant": active_variant, "status": "error", "error": str(e)},
        })
        return

    logger.info(
        f"{cfg.display_name} loaded on {device} — sr={kyutai_sr} frame={frame_size} "
        f"rate={frame_rate:.1f} silence_prefix={audio_silence_prefix_seconds}s "
        f"delay={audio_delay_seconds}s"
    )
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
        }
    })

    # ── Audio preprocessing (pre-emphasis + RMS norm at 16 kHz) ──────────────
    _preemph_coeff = 0.97
    _rms_target = 0.08
    _rms_floor = 1e-6
    _preemph_prev = np.float32(0.0)

    def preprocess_audio(pcm_f32):
        nonlocal _preemph_prev
        if len(pcm_f32) == 0:
            return pcm_f32
        if preemph_enabled:
            out = np.empty_like(pcm_f32)
            out[0] = pcm_f32[0] - _preemph_coeff * _preemph_prev
            out[1:] = pcm_f32[1:] - _preemph_coeff * pcm_f32[:-1]
            _preemph_prev = pcm_f32[-1]
        else:
            out = pcm_f32
        if rms_norm_enabled:
            rms = float(np.sqrt(np.mean(out ** 2)))
            if rms > _rms_floor:
                out = out * (_rms_target / rms)
                np.clip(out, -1.0, 1.0, out=out)
        return out

    # ── Session state (mirrors Moonshine worker) ─────────────────────────────
    state = {
        "session_active": False,
        "is_ptt": False,
        "should_listen": False,
        "tts_muted": False,
        "ptt_stopping": False,
        "ptt_stop_time": 0.0,
        "session_start_time": 0.0,
    }
    PTT_DRAIN_TIMEOUT = 3.0
    # Kyutai's semantic-VAD end-of-turn handles the "stop listening when user
    # finished speaking" case via EMA pause-prediction heads — that's the
    # primary close mechanism. NO_SPEECH_TIMEOUT only fires when the user
    # never speaks at all (KWS false-fired on noise OR user paused too long
    # after wake-word before forming a sentence). 8s gives generous think-time.
    NO_SPEECH_TIMEOUT = float(os.getenv("NOVA_KYUTAI_NO_SPEECH_TIMEOUT", "8.0"))

    # Smart-Turn v3 — optional semantic gate on Kyutai's EMA end-of-turn.
    # The EMA heads predict acoustic pause; smart-turn adds semantic completion.
    # Stacking them: EMA fires → smart-turn confirms → finalize.
    from smart_turn import (  # noqa: E402 (local import)
        SmartTurnPredictor,
        is_enabled as _smart_turn_enabled,
        threshold as _smart_turn_threshold,
    )
    smart_turn = SmartTurnPredictor() if _smart_turn_enabled() else None
    smart_turn_threshold = _smart_turn_threshold()
    smart_turn_max_wait_s = max(
        0.5, int(os.getenv("NOVA_SMART_TURN_MAX_WAIT_MS", "3000")) / 1000.0
    )
    if smart_turn is not None:
        logger.info(
            f"Kyutai: Smart-Turn v3 enabled (threshold={smart_turn_threshold:.2f}, "
            f"max_wait={smart_turn_max_wait_s:.1f}s)"
        )
        # Eager warmup so the first finalize doesn't pay the ONNX load cost.
        try:
            import numpy as _np
            smart_turn.predict(_np.zeros(int(0.5 * NOVA_SAMPLE_RATE), dtype=_np.float32))
            logger.info("Kyutai: Smart-Turn v3 warmup complete.")
        except Exception as _e:
            logger.warning(f"Kyutai: Smart-Turn warmup failed (non-fatal): {_e}")

    # Shared between main thread and inference thread.
    audio_q: _queue.Queue = _queue.Queue(maxsize=256)   # np.float32 16kHz chunks
    ctrl_q: _queue.Queue = _queue.Queue()               # ("reset",None) | ("flush",reason) | ("shutdown",None)

    # Inference-thread-owned state
    inf_state = {
        "words": [],           # list[str]
        "pending_piece": "",   # buffer for sub-word pieces (id_to_piece)
        "ema": _PauseEMA(0.15, 0.05, 1.0),
        "steps_seen": 0,
        "t0": 0.0,
        "samples_in": 0,
        "emitted_final": False,
        "speech_started_sent": False,
        "audio_buf": [],       # list[np.ndarray] — raw 16kHz float chunks for voice verify
        "last_token_at": 0.0,  # wall-clock of most recent text token (trailing-silence finalize)
    }
    inf_lock = threading.Lock()

    def _reset_inf_state():
        with inf_lock:
            inf_state["words"] = []
            inf_state["pending_piece"] = ""
            inf_state["ema"].reset(1.0)
            inf_state["steps_seen"] = 0
            inf_state["t0"] = 0.0
            inf_state["samples_in"] = 0
            inf_state["emitted_final"] = False
            inf_state["speech_started_sent"] = False
            inf_state["audio_buf"] = []
            inf_state["last_token_at"] = 0.0
        # Drain any stale audio from a previous session
        try:
            while True:
                audio_q.get_nowait()
        except _queue.Empty:
            pass

    def _flush_final(reason: str):
        with inf_lock:
            if inf_state["emitted_final"]:
                return
            inf_state["emitted_final"] = True
            transcript = " ".join(inf_state["words"]).strip()
            transcript = " ".join(transcript.split())
            duration_s = inf_state["samples_in"] / NOVA_SAMPLE_RATE if inf_state["samples_in"] else 0.0
            latency_s = max(1e-3, time.time() - inf_state["t0"]) if inf_state["t0"] else 0.0
            rtf = latency_s / duration_s if duration_s > 0 else 0.0
            audio_list = None
            if inf_state["audio_buf"]:
                import numpy as np
                audio_concat = np.concatenate(inf_state["audio_buf"])
                audio_list = audio_concat.tolist()
        logger.info(f"Kyutai Final [{reason}]: '{transcript}' | lat={latency_s:.2f}s | RTF={rtf:.2f}")
        ws_out_queue.put({
            "type": "transcript",
            "data": transcript,
            "latency": {"stt_ttfb": latency_s, "stt_rtf": rtf},
        })
        if transcript:
            payload = {"type": "text", "text": transcript}
            if audio_list is not None:
                payload["audio_data"] = audio_list
            llm_in_queue.put(payload)
            ws_out_queue.put({"type": "generation_start"})
        else:
            ws_out_queue.put({"type": "recording_stopped"})

    # ── Inference thread ──────────────────────────────────────────────────────
    def _inference_thread():
        try:
            with torch.inference_mode(), mimi.streaming(1), lm_gen.streaming(1):
                # Prime streaming state with silence prefix (once).
                silence = torch.zeros((1, 1, frame_size), dtype=torch.float32, device=device)
                n_prefix = int(math.ceil(audio_silence_prefix_seconds * frame_rate))
                for _ in range(n_prefix):
                    _ = _decode_chunk(silence, is_silence=True)

                sample_buffer = np.zeros(0, dtype=np.float32)

                while not stop_event.is_set():
                    # Drain control messages first
                    try:
                        while True:
                            kind, payload = ctrl_q.get_nowait()
                            if kind == "shutdown":
                                return
                            if kind == "reset":
                                sample_buffer = np.zeros(0, dtype=np.float32)
                            if kind == "flush":
                                # Pad audio_delay_seconds of silence so trailing words emerge.
                                n_suffix = int(math.ceil(audio_delay_seconds * frame_rate))
                                for _ in range(n_suffix):
                                    _decode_chunk(silence, is_silence=True)
                                _flush_final(payload or "flush")
                    except _queue.Empty:
                        pass

                    # Pull audio (blocking with timeout so we can re-check ctrl)
                    try:
                        chunk16k = audio_q.get(timeout=0.05)
                    except _queue.Empty:
                        continue
                    if chunk16k is None:
                        continue

                    with inf_lock:
                        inf_state["samples_in"] += len(chunk16k)

                    # Resample 16 → kyutai_sr (24k typical) via julius
                    x16 = torch.from_numpy(chunk16k).to(device)
                    x24 = julius.resample_frac(x16, NOVA_SAMPLE_RATE, kyutai_sr)

                    # Accumulate into buffer, then emit whole mimi frames
                    x24_np = x24.detach().cpu().numpy().astype(np.float32, copy=False)
                    sample_buffer = np.concatenate([sample_buffer, x24_np])
                    n_full = (len(sample_buffer) // frame_size) * frame_size
                    if n_full == 0:
                        continue
                    frames_np = sample_buffer[:n_full]
                    sample_buffer = sample_buffer[n_full:]

                    frames_t = torch.from_numpy(frames_np).to(device)
                    frames_t = frames_t.view(1, 1, -1)
                    for i in range(0, n_full, frame_size):
                        chunk = frames_t[:, :, i:i + frame_size]
                        _decode_chunk(chunk, is_silence=False)
        except Exception:
            logger.exception("Kyutai inference thread crashed")

    def _decode_chunk(audio_chunk, is_silence: bool):
        """Run one mimi frame through LMGen; handle text tokens + VAD head."""
        audio_tokens = mimi.encode(audio_chunk)
        if cfg.kyutai_use_semantic_vad:
            text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)
        else:
            text_tokens = lm_gen.step(audio_tokens)
            vad_heads = None

        if text_tokens is None:
            return

        with inf_lock:
            inf_state["steps_seen"] += 1
            steps = inf_state["steps_seen"]

        # Decode text piece
        tok_id = int(text_tokens[0, 0, 0].cpu().item())
        if tok_id != 0 and tok_id != padding_token_id:
            piece = tokenizer.id_to_piece(tok_id)  # type: ignore[attr-defined]
            with inf_lock:
                inf_state["last_token_at"] = time.time()
            # sentencepiece uses ▁ as word boundary
            if piece.startswith("▁"):
                with inf_lock:
                    if inf_state["pending_piece"]:
                        inf_state["words"].append(inf_state["pending_piece"])
                    inf_state["pending_piece"] = piece[1:]
                    was_first = not inf_state["speech_started_sent"]
                    if was_first:
                        inf_state["speech_started_sent"] = True
                    partial = " ".join(inf_state["words"] + [inf_state["pending_piece"]]).strip()
                if was_first:
                    ws_out_queue.put({"type": "speech_started"})
                if partial:
                    ws_out_queue.put({"type": "transcript_partial", "data": partial})
            else:
                with inf_lock:
                    inf_state["pending_piece"] += piece
                    partial = " ".join(inf_state["words"] + [inf_state["pending_piece"]]).strip()
                if partial:
                    ws_out_queue.put({"type": "transcript_partial", "data": partial})

        # Semantic VAD EMA
        if cfg.kyutai_use_semantic_vad and vad_heads and not is_silence:
            try:
                pr = float(vad_heads[vad_head_idx][0, 0, 0].cpu().item())
            except Exception:
                pr = 0.0
            dt = 1.0 / frame_rate
            with inf_lock:
                if steps >= N_STEPS_TO_WAIT:
                    inf_state["ema"].update(dt=dt, new_value=pr)
                ema_val = inf_state["ema"].value
                have_words = bool(inf_state["words"]) or bool(inf_state["pending_piece"])

            if (
                state["session_active"]
                and not state["is_ptt"]
                and have_words
                and steps >= N_STEPS_TO_WAIT
                and ema_val > ema_threshold
            ):
                # Smart-Turn gate: confirm semantic completion before finalizing.
                if smart_turn is not None:
                    with inf_lock:
                        audio_for_st = (
                            np.concatenate(inf_state["audio_buf"])
                            if inf_state["audio_buf"]
                            else np.zeros(0, dtype=np.float32)
                        )
                    prob = smart_turn.predict(audio_for_st)
                    if prob < smart_turn_threshold:
                        # User's still talking — hold, but enforce a max-wait cap
                        # so long hesitation can't stall forever.
                        now_ts = time.time()
                        pending = inf_state.get("smart_turn_pending_since", 0.0)
                        if pending == 0.0:
                            inf_state["smart_turn_pending_since"] = now_ts
                            logger.info(
                                f"Kyutai: holding finalize "
                                f"(ema={ema_val:.2f}, smart-turn={prob:.2f} < "
                                f"{smart_turn_threshold:.2f})"
                            )
                            # Reset EMA so it has to rebuild before re-firing;
                            # prevents per-frame re-entry into this branch.
                            inf_state["ema"].reset(0.0)
                            return
                        if (now_ts - pending) < smart_turn_max_wait_s:
                            inf_state["ema"].reset(0.0)
                            return
                        logger.info(
                            f"Kyutai: smart-turn timeout ({smart_turn_max_wait_s:.1f}s) — forcing finalize."
                        )
                        inf_state["smart_turn_pending_since"] = 0.0
                    else:
                        inf_state["smart_turn_pending_since"] = 0.0
                        logger.info(
                            f"Kyutai end-of-turn (ema={ema_val:.2f}) [smart-turn={prob:.2f} ≥ {smart_turn_threshold:.2f}]."
                        )
                else:
                    logger.info(f"Kyutai semantic VAD fired end-of-turn (ema={ema_val:.2f}).")
                # Drain audio_delay_seconds of silence so the last text tokens
                # (which lag audio by this much) emerge before we finalize.
                silence_pad = torch.zeros((1, 1, frame_size), dtype=torch.float32, device=device)
                n_tail = int(math.ceil(audio_delay_seconds * frame_rate))
                for _ in range(n_tail):
                    audio_tokens = mimi.encode(silence_pad)
                    if cfg.kyutai_use_semantic_vad:
                        text_tokens, _ = lm_gen.step_with_extra_heads(audio_tokens)
                    else:
                        text_tokens = lm_gen.step(audio_tokens)
                    if text_tokens is None:
                        continue
                    tok_id = int(text_tokens[0, 0, 0].cpu().item())
                    if tok_id != 0 and tok_id != padding_token_id:
                        piece = tokenizer.id_to_piece(tok_id)  # type: ignore[attr-defined]
                        with inf_lock:
                            if piece.startswith("▁"):
                                if inf_state["pending_piece"]:
                                    inf_state["words"].append(inf_state["pending_piece"])
                                inf_state["pending_piece"] = piece[1:]
                            else:
                                inf_state["pending_piece"] += piece
                with inf_lock:
                    if inf_state["pending_piece"]:
                        inf_state["words"].append(inf_state["pending_piece"])
                        inf_state["pending_piece"] = ""
                _flush_final("semantic_vad")
                state["should_listen"] = False
                state["session_active"] = False
                _reset_inf_state()

    inf_thread = threading.Thread(target=_inference_thread, name="kyutai-infer", daemon=True)
    inf_thread.start()

    def _end_session():
        state["session_active"] = False
        state["should_listen"] = False
        state["is_ptt"] = False
        state["ptt_stopping"] = False
        _reset_inf_state()
        try:
            ctrl_q.put_nowait(("reset", None))
        except Exception:
            pass

    # ── Main loop (sync, same shape as Moonshine worker) ─────────────────────
    while not stop_event.is_set():
        try:
            msg = stt_in_queue.get(timeout=0.05)
        except Empty:
            if state["ptt_stopping"] and (time.time() - state["ptt_stop_time"]) > PTT_DRAIN_TIMEOUT:
                logger.warning("PTT drain timeout — forcing session end.")
                _flush_final("ptt_timeout")
                _end_session()
            elif (
                state["session_active"]
                and not state["is_ptt"]
                and not state["ptt_stopping"]
                and not cfg.kyutai_use_semantic_vad
                and inf_state["last_token_at"] > 0.0
                and (inf_state["words"] or inf_state["pending_piece"])
                and (time.time() - inf_state["last_token_at"]) > trailing_silence_s
            ):
                logger.info(
                    f"Kyutai trailing-silence end-of-turn "
                    f"(silence={time.time() - inf_state['last_token_at']:.2f}s ≥ "
                    f"{trailing_silence_s:.2f}s)"
                )
                try:
                    ctrl_q.put_nowait(("flush", "trailing_silence"))
                except _queue.Full:
                    pass
                state["should_listen"] = False
                state["session_active"] = False
            elif (
                state["session_active"]
                and not state["is_ptt"]
                and not state["ptt_stopping"]
                and state["session_start_time"] > 0
                and (time.time() - state["session_start_time"]) > NO_SPEECH_TIMEOUT
                and not inf_state["words"]
                and not inf_state["pending_piece"]
            ):
                cc = inf_state.get("chunk_count", 0)
                sa = inf_state.get("sum_abs", 0.0)
                db = inf_state.get("dropped_bytes", 0)
                avg_abs = (sa / cc) if cc else 0.0
                logger.warning(
                    f"No speech within {NO_SPEECH_TIMEOUT}s — ending session. "
                    f"chunks={cc} avg|x|={avg_abs:.4f} dropped_bytes={db} "
                    f"should_listen={state['should_listen']} tts_muted={state['tts_muted']}"
                )
                _end_session()
                ws_out_queue.put({"type": "recording_stopped"})
            continue

        if isinstance(msg, bytes):
            if state["should_listen"] and not state["tts_muted"] and state["session_active"]:
                try:
                    import numpy as np
                    raw_f32 = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                    with inf_lock:
                        inf_state["audio_buf"].append(raw_f32.copy())
                        inf_state["chunk_count"] = inf_state.get("chunk_count", 0) + 1
                        inf_state["sum_abs"] = inf_state.get("sum_abs", 0.0) + float(np.abs(raw_f32).mean())
                    audio_f32 = preprocess_audio(raw_f32)
                    try:
                        audio_q.put_nowait(audio_f32)
                    except _queue.Full:
                        logger.debug("audio queue full — dropping chunk")
                except Exception as e:
                    logger.debug(f"audio chunk error: {e}")
            else:
                with inf_lock:
                    inf_state["dropped_bytes"] = inf_state.get("dropped_bytes", 0) + 1
            continue

        if not isinstance(msg, dict):
            continue
        mtype = msg.get("type")

        if mtype == "change_stt_variant":
            requested = msg.get("data", "")
            if requested == active_variant:
                continue
            target = STT_VARIANT_REGISTRY.get(requested)
            if target is None or target.backend != "kyutai":
                logger.warning(f"Cross-backend switch to '{requested}' requires worker restart.")
                ws_out_queue.put({
                    "type": "stt_variant_loading",
                    "data": {"variant": requested, "status": "error",
                             "error": "cross-backend switch requires NOVA_STT_VARIANT restart"},
                })
                continue
            logger.warning(
                "Within-Kyutai variant swap requires worker restart (model weights change)."
            )
            ws_out_queue.put({
                "type": "stt_variant_loading",
                "data": {"variant": requested, "status": "error",
                         "error": "Kyutai variant swap requires worker restart"},
            })

        elif mtype == "start":
            if not state["session_active"]:
                state["session_active"] = True
                state["is_ptt"] = msg.get("ptt", False)
                state["ptt_stopping"] = False
                state["session_start_time"] = time.time()
                state["tts_muted"] = False
                _reset_inf_state()
                inf_state["t0"] = time.time()
                state["should_listen"] = True
                logger.info(f"Kyutai STT Session Started (PTT: {state['is_ptt']})")

        elif mtype == "stop":
            if state["session_active"]:
                age = time.time() - state["session_start_time"]
                if age < 0.5 and not state["is_ptt"]:
                    logger.info(f"Ignoring stale stop (age={age:.2f}s < 0.5s)")
                    continue
                if state["is_ptt"]:
                    logger.info("PTT released — draining...")
                    state["should_listen"] = False
                    state["ptt_stopping"] = True
                    state["ptt_stop_time"] = time.time()
                    # Tell inference thread to pad + flush
                    ctrl_q.put(("flush", "ptt_stop"))
                    # Wait up to PTT_DRAIN_TIMEOUT for final emission
                    deadline = time.time() + PTT_DRAIN_TIMEOUT
                    while time.time() < deadline and not inf_state["emitted_final"]:
                        time.sleep(0.02)
                    _end_session()
                else:
                    logger.info("Kyutai STT Session Stopped (auto).")
                    _end_session()
                    ws_out_queue.put({"type": "recording_stopped"})

        elif mtype == "external_eos":
            # Gateway VAD says end-of-speech — pad+flush so trailing words
            # emerge, then close the session. Same shape as PTT release.
            if state["session_active"] and not state["is_ptt"]:
                state["should_listen"] = False
                ctrl_q.put(("flush", msg.get("reason", "external_eos")))
                deadline = time.time() + PTT_DRAIN_TIMEOUT
                while time.time() < deadline and not inf_state["emitted_final"]:
                    time.sleep(0.02)
                _end_session()

        elif mtype == "interrupted":
            if state["session_active"]:
                logger.info("Kyutai STT Session Interrupted")
                _end_session()

        elif mtype == "tts_mute":
            state["tts_muted"] = True

        elif mtype == "tts_unmute":
            state["tts_muted"] = False

    # Shutdown
    try:
        ctrl_q.put(("shutdown", None))
    except Exception:
        pass
    logger.info("Kyutai STT Worker exiting.")
