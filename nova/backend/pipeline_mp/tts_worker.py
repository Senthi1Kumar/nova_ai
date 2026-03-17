"""
TTS Worker — FasterQwen3TTS (primary, CUDA graphs) with Pocket-TTS fallback.

Voice reference:
  Primary:  nova/backend/voices/nova_ref.wav   (record any clear 5-10s sample)
  Fallback: first available KWS enrollment sample

Qwen3-TTS outputs 12 kHz audio; this worker resamples to 24 kHz before
sending so the rest of the pipeline stays unchanged.
"""
import os
import time
import wave
import struct
import multiprocessing as mp
import multiprocessing.synchronize
import logging
import numpy as np
from pathlib import Path
from queue import Empty

import re
logger = logging.getLogger("TTSWorker")
_re_word = re.compile(r'[a-zA-Z]{2,}')

_BACKEND_DIR = Path(__file__).parent.parent
_VOICES_DIR  = _BACKEND_DIR / "voices"
_NOVA_REF    = _VOICES_DIR / "nova_ref.wav"

# helpers

def _find_ref_audio() -> str | None:
    if _NOVA_REF.exists():
        return str(_NOVA_REF)
    kws_samples = _BACKEND_DIR / "kws" / "samples"
    if kws_samples.exists():
        wavs = sorted(kws_samples.glob("*.wav"))
        if wavs:
            logger.info(f"Using KWS sample as voice reference: {wavs[0].name}")
            return str(wavs[0])
    return None


def _make_silence_wav(path: Path, duration_s: float = 3.0, sr: int = 24000) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return str(path)


def _resample_12k_to_24k(audio: np.ndarray) -> np.ndarray:
    """Upsample 12 kHz → 24 kHz using scipy polyphase (anti-aliased)."""
    try:
        from scipy.signal import resample_poly
        return resample_poly(audio, up=2, down=1).astype(np.float32)
    except ImportError:
        # Fallback: simple linear interpolation
        return np.interp(
            np.linspace(0, len(audio) - 1, len(audio) * 2),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


# worker 

def run_tts_worker(
    tts_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
    tts_interrupt_event: multiprocessing.synchronize.Event | None = None,
):
    import setproctitle
    setproctitle.setproctitle("nova-tts-worker")

    # Silence sox "not found" warning — sox CLI is optional for qwen-tts audio utils
    import logging as _logging
    _logging.getLogger("sox").setLevel(_logging.ERROR)

    use_qwen   = False
    qwen_model = None
    ref_audio  = None

    # Try FasterQwen3TTS
    try:
        from faster_qwen3_tts import FasterQwen3TTS

        logger.info("Loading FasterQwen3TTS (Qwen/Qwen3-TTS-12Hz-0.6B-Base) …")
        qwen_model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

        ref_audio = _find_ref_audio()
        if ref_audio is None:
            ref_audio = _make_silence_wav(_NOVA_REF)
            logger.warning(
                "No reference audio found — silence placeholder created at "
                f"{_NOVA_REF}. "
                "For a proper Nova voice, record a clear 5-10s sample and save "
                f"it to {_NOVA_REF}"
            )

        use_qwen = True
        logger.info(f"FasterQwen3TTS ready. Voice ref: {ref_audio}")

        # Warm up CUDA graphs at init — avoids ~60s delay on first real request
        logger.info("Warming up CUDA graphs (silent dummy generation)...")
        try:
            for _chunk, _sr, _timing in qwen_model.generate_voice_clone_streaming(
                text="Hello.",
                language="English",
                ref_audio=ref_audio,
                ref_text="",
                chunk_size=4,
            ):
                break  # one chunk is enough to trigger graph capture
            logger.info("CUDA graph warmup complete.")
        except Exception as warm_e:
            logger.warning(f"CUDA graph warmup failed (non-fatal): {warm_e}")

    except Exception as e:
        logger.warning(f"FasterQwen3TTS unavailable ({e}) — falling back to Pocket-TTS.")

    # Pocket-TTS fallback
    pocket_model  = None
    voice_catalog: dict = {}

    if not use_qwen:
        try:
            import torch
            from pocket_tts import TTSModel
            # from huggingface_hub import login

            # hf_token = os.environ.get("HF_TOKEN")
            # if hf_token:
            #     login(token=hf_token)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Pocket-TTS on {device} …")
            pocket_model = TTSModel.load_model().to(device)

            for voice in ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]:
                try:
                    state = pocket_model.get_state_for_audio_prompt(voice)
                    for mod, mod_state in state.items():
                        for k, v in mod_state.items():
                            if isinstance(v, torch.Tensor):
                                state[mod][k] = v.to(device)
                    voice_catalog[voice] = state
                except Exception as ve:
                    logger.error(f"Failed voice {voice}: {ve}")

            logger.info("Pocket-TTS loaded.")
        except Exception as e:
            logger.error(f"Failed to load Pocket-TTS: {e}")

    # main loop
    current_voice = "alba"
    # Use mp.Event for interrupt signaling (avoids queue reordering from get_nowait)
    _interrupt = tts_interrupt_event  # alias for readability

    while not stop_event.is_set():
        try:
            msg = tts_in_queue.get(timeout=0.05)
        except Empty:
            continue

        if not isinstance(msg, dict):
            continue

        if msg.get("type") == "change_voice":
            current_voice = msg.get("data", current_voice)
            continue

        if msg.get("type") == "eof":
            interrupted = _interrupt and _interrupt.is_set()
            logger.info(f"TTS: received 'eof' | interrupted={interrupted}")
            if not interrupted:
                logger.info("TTS: sending 'generation_done' to ws_out_queue")
                ws_out_queue.put({"type": "generation_done"})
            else:
                logger.info("TTS: skipping generation_done (was interrupted)")
            # Clear interrupt flag — new generation cycle starts fresh
            if _interrupt:
                _interrupt.clear()
            continue

        if msg.get("type") != "text_to_speak":
            continue

        if _interrupt and _interrupt.is_set():
            logger.info(f"TTS: skipping text_to_speak (interrupted): '{msg.get('text', '')[:40]}'")
            continue

        text = msg.get("text", "").strip()
        if not text:
            continue
        # Skip garbage fragments: must have at least one real word (2+ letters)
        if not _re_word.search(text):
            logger.info(f"TTS: skipping non-speakable text: '{text[:40]}'")
            continue
        logger.info(f"TTS: generating audio for: '{text[:60]}'")

        try:
            t0          = time.time()
            first_chunk = True

            if use_qwen and qwen_model is not None:
                # FasterQwen3TTS streaming (CUDA graphs, 12 kHz → 24 kHz)
                # Timeout: ~1s of TTS output per word, min 10s.  Prevents hangs on garbage.
                word_count = len(text.split())
                tts_timeout = max(10.0, word_count * 1.0)
                gen = qwen_model.generate_voice_clone_streaming(
                    text=text,
                    language="English",
                    ref_audio=ref_audio,
                    ref_text="",        # xvec_only=True (default) — ref_text unused
                    chunk_size=4,       # 4 steps ≈ 333 ms/chunk for low latency
                )
                gen_start = time.time()
                for audio_chunk, qsr, _timing in gen:
                    if stop_event.is_set() or (_interrupt and _interrupt.is_set()):
                        break
                    if time.time() - gen_start > tts_timeout:
                        logger.warning(f"TTS: generation timeout ({tts_timeout:.0f}s) for: '{text[:40]}' — aborting")
                        break

                    audio_np = np.asarray(audio_chunk, dtype=np.float32)

                    # Resample 12 kHz → 24 kHz (main_mp hardcodes 24 kHz)
                    if qsr != 24000:
                        audio_np = _resample_12k_to_24k(audio_np)

                    audio_np  = np.clip(audio_np, -1.0, 1.0)
                    audio_i16 = (audio_np * 32767).astype(np.int16)
                    pcm_bytes = audio_i16.tobytes()

                    if first_chunk:
                        ttfa      = time.time() - t0
                        chunk_dur = len(audio_np) / 24000.0
                        ws_out_queue.put({"type": "tts_metrics",
                                          "data": {"ttfa": ttfa,
                                                   "tts_rtf": ttfa / chunk_dur if chunk_dur else 0}})
                        first_chunk = False

                    ws_out_queue.put({"type": "audio_out", "bytes": pcm_bytes})

            elif pocket_model is not None:
                # ── Pocket-TTS streaming (24 kHz) ─────────────────────────────
                import torch
                voice_state = voice_catalog.get(current_voice)
                audio_gen   = pocket_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=text, copy_state=True
                )
                for chunk in audio_gen:
                    if stop_event.is_set() or (_interrupt and _interrupt.is_set()):
                        break

                    audio_np  = chunk.cpu().numpy()
                    audio_i16 = (audio_np * 32767).astype(np.int16)
                    pcm_bytes = audio_i16.tobytes()

                    if first_chunk:
                        ttfa      = time.time() - t0
                        chunk_dur = len(pcm_bytes) / 2 / 24000.0
                        ws_out_queue.put({"type": "tts_metrics",
                                          "data": {"ttfa": ttfa,
                                                   "tts_rtf": ttfa / chunk_dur if chunk_dur else 0}})
                        first_chunk = False

                    ws_out_queue.put({"type": "audio_out", "bytes": pcm_bytes})

        except Exception as e:
            logger.error(f"TTS streaming error: {e}", exc_info=True)

    logger.info("TTS Worker exiting.")
