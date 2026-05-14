"""Nova — single-process voice gateway (VAD + AEC fix).

Changes from previous version:
- VAD threshold raised to 0.65 (car-noise tuned).
- RMS energy gate: windows below -45 dBFS are ignored before VAD runs.
- AEC guard dropped from 500 ms → 100 ms so barge-in is immediate.
- AEC reference buffer is size-capped to prevent overflow/underrun.
- Noise-floor tracker rejects sustained low-energy false positives.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv  # type: ignore
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

logger = logging.getLogger("nova-simple")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

SR = 16_000
VAD_WIN = 512
APM_FRAME = 160
CLAUSE_BREAK = re.compile(r"(?<=[.!?,;:—])\s+")
SENT_MIN_CHARS = int(os.getenv("NOVA_SENT_MIN_CHARS", "8"))
TTS_BINARY = os.getenv("NOVA_TTS_BINARY", "1") == "1"

VAD_THRESHOLD = float(os.getenv("NOVA_VAD_THRESHOLD", "0.65"))  # ↑ was 0.5
VAD_MIN_RMS_DB = float(os.getenv("NOVA_VAD_MIN_RMS_DB", "-45"))  # NEW: energy floor
SILENCE_END_MS = int(os.getenv("NOVA_SILENCE_END_MS", "400"))
MIN_SPEECH_MS = int(os.getenv("NOVA_MIN_SPEECH_MS", "500"))
SMART_TURN_THRESHOLD = float(os.getenv("NOVA_SMART_TURN_THRESHOLD", "0.5"))
# When smart-turn rejects ("not done yet"), keep the turn open and demand this
# many extra ms of silence before re-checking, instead of discarding the audio.
SMART_TURN_REPRIEVE_MS = int(os.getenv("NOVA_SMART_TURN_REPRIEVE_MS", "1500"))
# Cap reprieves per turn — after this many, force-finalize regardless.
SMART_TURN_MAX_REPRIEVES = int(os.getenv("NOVA_SMART_TURN_MAX_REPRIEVES", "2"))
# Hard cap on total speech duration before we finalize, smart-turn or not.
MAX_UTTERANCE_S = float(os.getenv("NOVA_MAX_UTTERANCE_S", "12.0"))
AEC_GUARD_MS = int(os.getenv("NOVA_AEC_GUARD_MS", "100"))  # ↓ was 500
BARGE_IN_THRESHOLD = float(os.getenv("NOVA_BARGE_IN_THRESHOLD", "0.35"))
BARGE_IN_FRAMES = int(os.getenv("NOVA_BARGE_IN_FRAMES", "2"))

SYSTEM_PROMPT = os.getenv(
    "NOVA_SYSTEM_PROMPT",
    # ── CRITICAL RULES (placed first for small-model attention) ──
    "CRITICAL RULES:\n"
    "1. Your name is Nova. The driver's name is NOT Nova. Address the driver as 'you' or by their actual name.\n"
    "2. NEVER start a reply with 'Nova' or 'Nova,'. NEVER call the driver Nova.\n"
    "3. NEVER invent facts. If you don't know something, say 'I'm not sure' — do not guess.\n"
    "4. NEVER mention Tesla or any car brand unless the driver explicitly brings it up.\n"
    "5. Keep every reply under 3 short sentences. Be direct. No filler.\n\n"
    # ── PERSONALITY ──
    "You are Nova, a voice assistant in the driver's car. "
    "You are helpful, brief, and slightly casual. "
    "You are NOT the driver — you are the assistant. "
    "Always reply in English.\n\n"
    # ── TRANSCRIPTION ──
    "The driver's speech comes from speech-to-text and may have errors. "
    "If a transcript is garbled, guess the meaning rather than asking to repeat. "
    "If the driver's message ends abruptly mid-sentence, reply with 'Go on?' or 'What were you saying?'\n\n"
    # ── TOOLS ──
    "Web search: use ONLY for explicit questions about current events, weather, news, or prices. "
    "NEVER search for general conversation, introductions, or things the driver tells you about themselves. "
    "After a search, give 1-2 plain spoken sentences — no source names or URLs.\n\n"
    # ── SAFETY ──
    "Vehicle controls are hardware-managed. Do not simulate controlling them. "
    "If you don't know something, just say so honestly."
)

LATIN_ONLY = os.getenv("NOVA_LATIN_ONLY", "1") == "1"


def _is_mostly_latin(text: str) -> bool:
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if ord(c) < 0x0250)
    return latin / len(letters) >= 0.8


MEMORY_ENABLED = os.getenv("NOVA_MEMORY", "1") == "1"
MEM_DIR = Path(os.getenv("NOVA_MEM_DIR", str(Path(__file__).parent / "persona")))
USER_MD = MEM_DIR / "USER.md"
MEMORY_MD = MEM_DIR / "MEMORY.md"
MEMORY_CONSOLIDATE_EVERY = int(os.getenv("NOVA_MEMORY_CONSOLIDATE_EVERY", "5"))


def _read_text(p: Path) -> str:
    try:
        return p.read_text().strip() if p.exists() else ""
    except Exception:
        return ""


def load_persona() -> str:
    if not MEMORY_ENABLED:
        return ""
    parts = []
    u, m = _read_text(USER_MD), _read_text(MEMORY_MD)
    if u:
        parts.append(f"## User profile (USER.md)\n{u}")
    if m:
        parts.append(f"## Durable memory (MEMORY.md)\n{m}")
    return "\n\n".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rms_dbfs(samples_f32: np.ndarray) -> float:
    """Return RMS energy in dBFS. 0 dBFS = full-scale sine wave."""
    if samples_f32.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(samples_f32.astype(np.float64) ** 2))
    # Avoid log(0)
    if rms < 1e-10:
        return -120.0
    return 20.0 * np.log10(rms + 1e-10)


# ── Model wrappers ────────────────────────────────────────────────────────────

def make_stt():
    backend = os.getenv("NOVA_STT_BACKEND", "nemotron_streaming").lower()
    if backend in ("qwen3_streaming", "qwen3_stream"):
        return Qwen3StreamingSTT()
    if backend == "qwen3_0_6b":
        return Qwen3STT()
    if backend == "qwen3_trt":
        return Qwen3TRTSTT()
    if backend == "kyutai_1b":
        return Kyutai1BSTT()
    if backend == "moonshine":
        return MoonshineSTT()
    if backend == "nemotron_streaming":
        return NemotronStreamingSTT()
    return Qwen3StreamingSTT()


class MoonshineSTT:
    def __init__(self) -> None:
        from moonshine_voice import Transcriber, get_model_for_language  # type: ignore
        path, arch = get_model_for_language("en")
        self.t = Transcriber(model_path=str(path), model_arch=arch)
        logger.info("STT backend: moonshine")

    def transcribe(self, audio_f32: np.ndarray) -> str:
        lines = self.t.transcribe_without_streaming(audio_f32.tolist(), SR).lines
        return " ".join(l.text for l in lines if l.text).strip()


class Qwen3STT:
    def __init__(self) -> None:
        from qwen_asr import Qwen3ASRModel  # type: ignore
        repo = os.getenv("NOVA_QWEN3_HF_REPO", "Qwen/Qwen3-ASR-0.6B")
        device = os.getenv("NOVA_KYUTAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _LANG_MAP = {
            "en": "English", "fr": "French", "es": "Spanish", "de": "German",
            "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
            "hi": "Hindi", "tr": "Turkish", "th": "Thai", "vi": "Vietnamese",
            "id": "Indonesian", "ms": "Malay", "sv": "Swedish", "da": "Danish",
            "fi": "Finnish", "pl": "Polish", "cs": "Czech", "el": "Greek",
            "ro": "Romanian", "hu": "Hungarian", "fa": "Persian",
        }
        raw = (os.getenv("NOVA_QWEN3_LANGUAGE") or "English").strip()
        self.lang = _LANG_MAP.get(raw.lower(), raw if raw[:1].isupper() else raw.title())
        self.model = Qwen3ASRModel.from_pretrained(
            repo, dtype=dtype, device_map=device,
            max_new_tokens=int(os.getenv("NOVA_QWEN3_MAX_NEW_TOKENS", "200")),
        )
        if type(self) is Qwen3STT:
            logger.info(f"STT backend: qwen3_0_6b ({repo}, {device})")

    def transcribe(self, audio_f32: np.ndarray) -> str:
        if audio_f32.size == 0:
            return ""
        try:
            results = self.model.transcribe(
                audio=[(audio_f32, SR)],
                language=[self.lang] if self.lang else None,
            )
        except Exception as e:
            logger.warning(f"qwen3 transcribe failed: {e}")
            return ""
        return (results[0].text if results else "").strip()


class Qwen3StreamingSTT(Qwen3STT):
    def __init__(self) -> None:
        super().__init__()
        self._partial_interval_s = max(
            0.1, int(os.getenv("NOVA_QWEN3_PARTIAL_INTERVAL_MS", "400")) / 1000.0
        )
        self._partial_window_s = max(
            1.0, float(os.getenv("NOVA_QWEN3_PARTIAL_WINDOW_S", "3.0"))
        )
        self._stability_ticks = max(
            1, int(os.getenv("NOVA_QWEN3_STABILITY_TICKS", "2"))
        )
        self._max_utterance_s = max(
            2.0, float(os.getenv("NOVA_QWEN3_MAX_UTTERANCE_S", "12.0"))
        )

        import threading
        from concurrent.futures import ThreadPoolExecutor

        self._lock = threading.Lock()
        self._infer_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="qwen3-stt")

        self._utt_buf: list[np.ndarray] = []
        self._last_partial_at: float = 0.0
        self._last_partial_text: str = ""
        self._stable_count: int = 0
        self._inflight = None
        logger.info(
            f"STT backend: qwen3_streaming "
            f"(partial_interval={self._partial_interval_s*1000:.0f}ms, "
            f"window={self._partial_window_s:.1f}s, "
            f"stability={self._stability_ticks})"
        )

    def begin_utterance(self) -> None:
        with self._lock:
            self._utt_buf.clear()
            self._last_partial_at = 0.0
            self._last_partial_text = ""
            self._stable_count = 0
            self._inflight = None

    def feed_chunk(self, chunk_f32: np.ndarray):
        if chunk_f32 is None or chunk_f32.size == 0:
            return None
        with self._lock:
            self._utt_buf.append(chunk_f32.astype(np.float32, copy=False))
            buf_seconds = sum(c.size for c in self._utt_buf) / SR

        if buf_seconds > self._max_utterance_s:
            return None

        new_text = self._poll_inflight()

        now = time.time()
        with self._lock:
            inflight = self._inflight
        if inflight is None and (now - self._last_partial_at) >= self._partial_interval_s:
            tail = self._tail_audio_np(self._partial_window_s)
            if tail.size >= SR // 4:
                self._last_partial_at = now
                with self._lock:
                    self._inflight = self._infer_pool.submit(
                        self._transcribe_blocking, tail)
        return new_text

    def is_settled(self) -> bool:
        return self._stable_count >= self._stability_ticks and bool(self._last_partial_text)

    def finalize(self) -> str:
        with self._lock:
            inflight = self._inflight
        if inflight is not None:
            try:
                text = inflight.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"qwen3-stream inflight failed: {e}")
                text = ""
            with self._lock:
                self._inflight = None
                if text:
                    if text == self._last_partial_text:
                        self._stable_count += 1
                    else:
                        self._stable_count = 0
                        self._last_partial_text = text

        if self.is_settled():
            return self._last_partial_text

        full = self._full_audio_np()
        if full.size == 0:
            return self._last_partial_text
        final_text = self._transcribe_blocking(full)
        return final_text or self._last_partial_text

    def _poll_inflight(self):
        with self._lock:
            inflight = self._inflight
        if inflight is None or not inflight.done():
            return None
        try:
            text = inflight.result()
        except Exception as e:
            logger.warning(f"qwen3-stream partial failed: {e}")
            text = ""
        with self._lock:
            self._inflight = None
            if not text:
                return None
            if text == self._last_partial_text:
                self._stable_count += 1
                return None
            self._stable_count = 0
            self._last_partial_text = text
            return text

    def _tail_audio_np(self, window_seconds: float) -> np.ndarray:
        with self._lock:
            if not self._utt_buf:
                return np.zeros(0, dtype=np.float32)
            full = np.concatenate(self._utt_buf, dtype=np.float32)
        n = int(window_seconds * SR)
        return full if full.size <= n else full[-n:]

    def _full_audio_np(self) -> np.ndarray:
        with self._lock:
            if not self._utt_buf:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._utt_buf, dtype=np.float32)

    def _transcribe_blocking(self, audio_f32: np.ndarray) -> str:
        return super().transcribe(audio_f32)


class Qwen3TRTSTT:
    def __init__(self) -> None:
        self.engine_dir = Path(os.getenv(
            "NOVA_TRT_ENGINE_DIR",
            str(Path.home() / "tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines"),
        ))
        self.tokenizer_dir = Path(os.getenv(
            "NOVA_TRT_TOKENIZER_DIR",
            str(self.engine_dir / "llm"),
        ))
        self.trt_root = Path(os.getenv(
            "NOVA_TRT_EDGE_ROOT",
            str(Path.home() / "TensorRT-Edge-LLM"),
        ))
        if not (self.engine_dir / "llm").exists() or not (self.engine_dir / "audio").exists():
            raise RuntimeError(
                f"TRT engines not found under {self.engine_dir}. Build with "
                f"`./build/examples/llm/llm_build` and `audio_build` first."
            )

        self._runner = None
        try:
            import tensorrt_edgellm as trtelm  # type: ignore
            self._runner = trtelm.AudioLLMRunner(
                llm_engine_dir=str(self.engine_dir / "llm"),
                audio_engine_dir=str(self.engine_dir / "audio"),
                tokenizer_dir=str(self.tokenizer_dir),
            )
            self._mode = "py"
            logger.info(f"STT backend: qwen3_trt (Python bindings, engines={self.engine_dir})")
        except Exception as e:
            logger.info(f"TRT Python bindings unavailable ({e}); falling back to CLI.")
            self._mode = "cli"
            self._cli = self.trt_root / "build/examples/llm/llm_inference"
            self._preproc = "tensorrt_edgellm.scripts.preprocess_audio"
            if not self._cli.exists():
                raise RuntimeError(
                    f"Neither tensorrt_edgellm Python bindings nor CLI binary "
                    f"({self._cli}) found. Set NOVA_TRT_EDGE_ROOT."
                )
            logger.info(f"STT backend: qwen3_trt (CLI subprocess, cli={self._cli})")

    def transcribe(self, audio_f32: np.ndarray) -> str:
        if audio_f32.size == 0:
            return ""
        if self._mode == "py":
            try:
                return self._runner.transcribe(audio_f32, sample_rate=SR).strip()
            except Exception as e:
                logger.warning(f"trt-py transcribe failed: {e}")
                return ""
        return self._transcribe_cli(audio_f32)

    def _transcribe_cli(self, audio_f32: np.ndarray) -> str:
        import subprocess
        import tempfile
        import wave
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            wav_path = tdp / "in.wav"
            mel_path = tdp / "in.safetensors"
            in_json = tdp / "input.json"
            out_json = tdp / "output.json"

            i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(i16.tobytes())

            subprocess.run([
                "python", "-m", self._preproc,
                "--input", str(wav_path),
                "--output", str(mel_path),
            ], cwd=self.trt_root, check=True, capture_output=True)

            sys_prompt = (
                "Transcribe the following speech segment in English into "
                "English text. Only output the transcription, with no newlines."
            )
            in_json.write_text(json.dumps({
                "batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
                "max_generate_length": 256,
                "requests": [{
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": [
                            {"type": "audio", "audio": str(mel_path)}
                        ]},
                    ],
                }],
            }))

            subprocess.run([
                str(self._cli),
                "--engineDir", str(self.engine_dir / "llm"),
                "--multimodalEngineDir", str(self.engine_dir / "audio"),
                "--inputFile", str(in_json),
                "--outputFile", str(out_json),
            ], cwd=self.trt_root, check=True, capture_output=True)

            data = json.loads(out_json.read_text())
            try:
                text = data["responses"][0]["output_text"].strip()
            except (KeyError, IndexError):
                return ""
            text = re.sub(r"^language\s+\w+\b[\s,:-]*", "", text, flags=re.IGNORECASE)
            return text.strip()


class Kyutai1BSTT:
    def __init__(self) -> None:
        import math
        import moshi.models  # type: ignore
        from moshi.models.loaders import CheckpointInfo  # type: ignore
        repo = os.getenv("NOVA_KYUTAI_HF_REPO", "kyutai/stt-1b-en_fr")
        self.device = os.getenv("NOVA_KYUTAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        info = CheckpointInfo.from_hf_repo(repo)
        self.silence_prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        self.delay_s = info.stt_config.get("audio_delay_seconds", 0.5)
        self.mimi = info.get_mimi(device=self.device)
        _lm = info.get_moshi(device=self.device, dtype=torch.bfloat16)
        self.lm_gen = moshi.models.LMGen(_lm, temp=0, temp_text=0.0)
        self.tokenizer = info.get_text_tokenizer()
        self.padding_token_id = info.raw_config.get("text_padding_token_id", 3)
        self.kyutai_sr = int(self.mimi.sample_rate)
        self.frame_size = int(self.mimi.frame_size)
        self.frame_rate = float(self.mimi.frame_rate)
        self._math = math
        logger.info(
            f"STT backend: kyutai_1b ({repo}, {self.device}) "
            f"sr={self.kyutai_sr} frame={self.frame_size}"
        )

    def transcribe(self, audio_f32: np.ndarray) -> str:
        import julius  # type: ignore
        if audio_f32.size == 0:
            return ""
        with torch.inference_mode(), self.mimi.streaming(1), self.lm_gen.streaming(1):
            silence = torch.zeros((1, 1, self.frame_size), dtype=torch.float32, device=self.device)
            for _ in range(int(self._math.ceil(self.silence_prefix_s * self.frame_rate))):
                self._step(silence)

            x16 = torch.from_numpy(audio_f32).to(self.device)
            x_kr = julius.resample_frac(x16, SR, self.kyutai_sr).cpu().numpy()
            n_full = (len(x_kr) // self.frame_size) * self.frame_size
            if n_full == 0:
                return ""
            frames = torch.from_numpy(x_kr[:n_full]).to(self.device).view(1, 1, -1)

            words: list[str] = []
            pending = ""
            for i in range(0, n_full, self.frame_size):
                tok = self._step(frames[:, :, i:i + self.frame_size])
                pending, words = self._consume(tok, pending, words)
            for _ in range(int(self._math.ceil(self.delay_s * self.frame_rate))):
                tok = self._step(silence)
                pending, words = self._consume(tok, pending, words)
        if pending:
            words.append(pending)
        return " ".join(words).strip()

    def _step(self, audio_chunk):
        tokens = self.mimi.encode(audio_chunk)
        return self.lm_gen.step(tokens)

    def _consume(self, text_tokens, pending: str, words: list[str]) -> tuple[str, list[str]]:
        if text_tokens is None:
            return pending, words
        tok_id = int(text_tokens[0, 0, 0].cpu().item())
        if tok_id == 0 or tok_id == self.padding_token_id:
            return pending, words
        piece = self.tokenizer.id_to_piece(tok_id)
        if piece.startswith("▁"):
            if pending:
                words.append(pending)
            pending = piece[1:]
        else:
            pending += piece
        return pending, words


class NemotronStreamingSTT:
    """
    True cache-aware streaming STT for nvidia/nemotron-speech-streaming-en-0.6b.

    Mirrors the pattern in NeMo's speech_to_text_cache_aware_streaming_infer.py:
    each new audio chunk is encoded once via `conformer_stream_step`, with the
    encoder cache (last-channel, last-time) and RNNT decoder state
    (`previous_hypotheses`, `pred_out_stream`) threaded across calls. Cost per
    partial is O(chunk), not O(utterance).
    """

    def __init__(self) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        repo = "nvidia/nemotron-speech-streaming-en-0.6b"

        logger.info(f"Loading Nemotron STT ({repo}) …")
        self.model = ASRModel.from_pretrained(
            model_name=repo,
            map_location=self.device,
        ).to(self.device).eval()

        # Optional override of att_context_size, e.g. "70,13" (left, right).
        # Right context dictates streaming latency vs accuracy. Choices for
        # this model: {0, 1, 6, 13}.
        att_env = os.getenv("NOVA_NEMOTRON_ATT_CONTEXT")
        if att_env:
            try:
                parts = [int(x) for x in att_env.split(",")]
                if hasattr(self.model.encoder, "set_default_att_context_size"):
                    self.model.encoder.set_default_att_context_size(att_context_size=parts)
                    logger.info(f"Nemotron: att_context_size={parts}")
            except Exception as e:
                logger.warning(f"Nemotron: att_context override failed: {e}")

        if hasattr(self.model, "setup_streaming_params"):
            try:
                self.model.setup_streaming_params()
            except Exception as e:
                logger.warning(f"Nemotron: setup_streaming_params failed: {e}")

        # RNNT greedy decoding so `previous_hypotheses` is honored.
        try:
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
            dec_cfg = RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy")
            if hasattr(self.model, "change_decoding_strategy"):
                self.model.change_decoding_strategy(dec_cfg)
        except Exception as e:
            logger.warning(f"Nemotron: change_decoding_strategy failed: {e}")

        # Derive raw-audio samples per streaming chunk from the encoder's
        # streaming config: chunk_size is in *encoder* (post-subsampling) frames.
        scfg = self.model.encoder.streaming_cfg
        chunk_frames = getattr(scfg, "chunk_size", None)
        if isinstance(chunk_frames, (list, tuple)):
            chunk_frames = chunk_frames[0]
        subsampling = (
            getattr(self.model.encoder, "subsampling_factor", None)
            or getattr(getattr(self.model, "cfg", None), "subsampling_factor", None)
            or 8
        )
        try:
            win_stride = float(self.model.cfg.preprocessor.window_stride)
        except Exception:
            win_stride = 0.01
        if chunk_frames and chunk_frames > 0:
            self._chunk_samples = int(round(chunk_frames * subsampling * win_stride * SR))
        else:
            self._chunk_samples = int(0.08 * SR)
        if self._chunk_samples <= 0:
            self._chunk_samples = int(0.08 * SR)

        import threading
        self._lock = threading.Lock()
        self._reset_state()

        logger.info(
            f"STT backend: nemotron_streaming "
            f"(cache-aware, chunk_samples={self._chunk_samples} "
            f"≈{self._chunk_samples*1000.0/SR:.0f}ms, device={self.device})"
        )

    def _reset_state(self) -> None:
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = self.model.encoder.get_initial_cache_state(batch_size=1)
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_text = ""

    def begin_utterance(self) -> None:
        with self._lock:
            self._reset_state()

    def feed_chunk(self, chunk_f32: np.ndarray):
        if chunk_f32 is None or chunk_f32.size == 0:
            return None
        with self._lock:
            self._pending = np.concatenate(
                [self._pending, chunk_f32.astype(np.float32, copy=False)]
            )
            new_text = None
            while self._pending.size >= self._chunk_samples:
                slab = self._pending[: self._chunk_samples]
                self._pending = self._pending[self._chunk_samples:]
                text = self._stream_step(slab, last=False)
                if text and text != self._last_text:
                    self._last_text = text
                    new_text = text
            return new_text

    def is_settled(self) -> bool:
        return bool(self._last_text)

    def finalize(self) -> str:
        with self._lock:
            if self._pending.size > 0:
                slab = self._pending
                if slab.size < self._chunk_samples:
                    slab = np.pad(slab, (0, self._chunk_samples - slab.size))
                self._pending = np.zeros(0, dtype=np.float32)
                text = self._stream_step(slab, last=True)
                if text:
                    self._last_text = text
            return self._last_text

    def _stream_step(self, audio_chunk: np.ndarray, last: bool) -> str:
        import torch
        try:
            with torch.inference_mode():
                wav = torch.from_numpy(audio_chunk).unsqueeze(0).to(self.device)
                length = torch.tensor([audio_chunk.shape[0]], device=self.device)
                processed_signal, processed_signal_length = self.model.preprocessor(
                    input_signal=wav, length=length
                )
                drop_extra = (
                    0
                    if self._step_num == 0
                    else getattr(self.model.encoder.streaming_cfg, "drop_extra_pre_encoded", 0)
                )
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=last,
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
            self._cache_last_channel = cache_last_channel
            self._cache_last_time = cache_last_time
            self._cache_last_channel_len = cache_last_channel_len
            self._previous_hypotheses = previous_hypotheses
            self._pred_out_stream = pred_out_stream
            self._step_num += 1
            return self._extract_text(transcribed_texts)
        except Exception as e:
            logger.warning(f"nemotron stream_step failed: {e}")
            return ""

    @staticmethod
    def _extract_text(hyps) -> str:
        if not hyps:
            return ""
        h = hyps[0] if isinstance(hyps, (list, tuple)) else hyps
        if isinstance(h, str):
            return h.strip()
        if hasattr(h, "text"):
            return (h.text or "").strip()
        if isinstance(h, dict):
            return (h.get("text") or "").strip()
        return ""


class PocketTTS:
    def __init__(self) -> None:
        import torch
        from pocket_tts import TTSModel  # type: ignore

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Pocket-TTS (device={self.device}) …")
        self.model = TTSModel.load_model()
        if self.device == "cuda":
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                logger.warning(f"Pocket-TTS: CUDA move failed ({e}), staying on CPU")
                self.device = "cpu"

        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self._voice_states: dict[str, Any] = {}
        self._current_voice = os.getenv("NOVA_POCKET_VOICE", "alba")
        self._ensure_voice(self._current_voice)

        try:
            _ = self._generate_sync("Ready.")
            logger.info("Pocket-TTS: warmup done")
        except Exception as e:
            logger.warning(f"Pocket-TTS warmup failed: {e}")

    def _ensure_voice(self, name: str) -> None:
        if name in self._voice_states:
            return
        p = Path(name)
        if p.exists() and p.suffix in (".safetensors", ".wav"):
            logger.info(f"Pocket-TTS: loading voice from file '{name}'")
            self._voice_states[name] = self.model.get_state_for_audio_prompt(str(p))
        else:
            logger.info(f"Pocket-TTS: loading built-in voice '{name}'")
            self._voice_states[name] = self.model.get_state_for_audio_prompt(name)

    def set_voice(self, name: str) -> None:
        self._current_voice = name
        self._ensure_voice(name)
        logger.info(f"Pocket-TTS: voice set to '{name}'")

    def _generate_sync(self, text: str) -> np.ndarray:
        state = self._voice_states.get(self._current_voice)
        if state is None:
            self._ensure_voice(self._current_voice)
            state = self._voice_states[self._current_voice]
        audio = self.model.generate_audio(state, text)
        if hasattr(audio, "numpy"):
            audio = audio.cpu().numpy()
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.5:
                audio = np.clip(audio, -1.0, 1.0)
        return audio

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        return self._generate_sync(text), self.sample_rate

    async def synth_stream(self, text: str):
        loop = asyncio.get_running_loop()
        samples = await loop.run_in_executor(None, self._generate_sync, text)
        sr = self.sample_rate
        chunk_samples = int(os.getenv("NOVA_POCKET_CHUNK", str(max(sr // 12, 2048))))
        for i in range(0, len(samples), chunk_samples):
            yield samples[i:i + chunk_samples].copy(), sr
            await asyncio.sleep(0)

    async def warmup_stream(self) -> None:
        try:
            async for _ in self.synth_stream("Ready."):
                pass
        except Exception as e:
            logger.warning(f"Pocket-TTS stream warmup failed: {e}")


class LLM:
    def __init__(self) -> None:
        self.backend = os.getenv("NOVA_LLM_BACKEND", "openrouter").lower()
        self.max_tokens = int(os.getenv("NOVA_LLM_MAX_TOKENS", "200"))
        self.tool_max_tokens = int(os.getenv("NOVA_TOOL_MAX_TOKENS", "120"))
        if self.backend == "local":
            self.base_url = os.getenv("NOVA_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
            self.api_key = os.getenv("NOVA_LLM_API_KEY", "EMPTY")
            self.model = os.getenv("NOVA_LLM_MODEL", "google/gemma-2-2b-it")
        else:
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.model = os.getenv("NOVA_LLM_MODEL", "openai/gpt-4o-mini")
            if not self.api_key:
                logger.warning(
                    "OPENROUTER_API_KEY is not set — LLM calls will fail. "
                    "Either set it, or run `NOVA_LLM_BACKEND=local` against a "
                    "local OpenAI-compatible server."
                )
        logger.info(f"LLM backend: {self.backend} ({self.model} @ {self.base_url})")

    def stream(self, messages: list[dict], max_tokens: int | None = None) -> Iterator[str]:
        yield from self._stream_request(messages, tools=None, max_tokens=max_tokens)

    def _stream_request(self, messages: list[dict], tools: list[dict] | None, max_tokens: int | None = None) -> Iterator[str]:
        import requests
        mt = max_tokens if max_tokens is not None else self.max_tokens
        body: dict = {"model": self.model, "messages": messages, "stream": True, "max_tokens": mt}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        with requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body, stream=True, timeout=60,
        ) as r:
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int | None = None) -> dict:
        import requests
        mt = max_tokens if max_tokens is not None else self.tool_max_tokens
        body: dict = {"model": self.model, "messages": messages, "max_tokens": mt}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body, timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


# ── Serper tool calling ───────────────────────────────────────────────────────

SERPER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web (Google) for current facts, news, prices, "
            "weather snippets, sports scores, or anything time-sensitive. "
            "Use ONLY when the user asks about something the assistant cannot "
            "answer from its own knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Concise web-search query."},
                "max_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
        },
    },
}]

_NEWSY = ("news", "headline", "latest", "breaking", "today",
          "current events", "what happened")
_TOOL_TRIGGERS = re.compile(
    r"\b(news|weather|price|stock|score|score of|who won|when did|"
    r"how many|how much|latest|today|tomorrow|yesterday|currently|"
    r"right now|search|look up|google|find out|forecast|"
    r"what happened|headlines|update on|tell me about|did .* happen|is .* true)\b",
    re.IGNORECASE,
)
_CONVERSATIONAL = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|cool|nice|great|sure|"
    r"bye|goodbye|see you|later|stop|wait|pause|continue|go on|repeat that)\b",
    re.IGNORECASE,
)


def _needs_tools(prompt: str) -> bool:
    s = prompt.strip()
    if not s or _CONVERSATIONAL.match(s):
        return False
    # Require at least 4 words AND either a question mark or explicit trigger
    if len(s.split()) < 4:
        return False
    # Sentence fragments (no sentence-ending punctuation) are never tool triggers
    if not s.endswith((".", "?", "!")):
        return False
    return bool(_TOOL_TRIGGERS.search(s))


def _extract_tool_calls_from_text(text: str) -> list[dict]:
    pattern = re.compile(r'<\|tool_call>call:(\w+)\{(.*?)<tool_call\|>', re.DOTALL)
    calls = []
    for name, raw_args in pattern.findall(text):
        cleaned = re.sub(r'<\|"\|>(.*?)<\|"\|>', r'"\1"', raw_args)
        cleaned = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', cleaned)
        try:
            args = json.loads("{" + cleaned + "}")
        except Exception:
            m = re.search(r'"query"\s*:\s*"([^"]+)"', cleaned)
            args = {"query": m.group(1)} if m else {}
        calls.append({
            "id": f"fallback_{len(calls)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return calls


def _serper_search(query: str, max_results: int = 3) -> str:
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY not set", "query": query})
    import urllib.request
    import urllib.error
    is_news = any(kw in query.lower() for kw in _NEWSY)
    endpoint = "https://google.serper.dev/news" if is_news else "https://google.serper.dev/search"
    try:
        payload = json.dumps({"q": query, "num": max(1, min(max_results, 5))}).encode()
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        results = []
        if is_news:
            for r in data.get("news", [])[:max_results]:
                results.append({k: r.get(k, "") for k in ("title", "snippet", "source", "date")})
        else:
            kg = data.get("knowledgeGraph", {})
            if kg.get("description"):
                results.append({"title": kg.get("title", ""), "snippet": kg["description"], "source": "knowledge_graph"})
            for r in data.get("organic", [])[:max_results]:
                results.append({"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")})
        if not results:
            return json.dumps({"error": "No results", "query": query})
        return json.dumps(results[:max_results], ensure_ascii=False)
    except (urllib.error.URLError, Exception) as e:
        logger.warning(f"Serper search failed: {e}")
        return json.dumps({"error": str(e), "query": query})


def _exec_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return _serper_search(args.get("query", ""), int(args.get("max_results", 3)))
    return json.dumps({"error": f"unknown tool: {name}"})


# ── Audio front-end (APM + VAD, inline) ───────────────────────────────────────

try:
    from pipeline_mp.pvad_firered import FireRedPVAD
    _PVAD_AVAILABLE = True
except ImportError:
    _PVAD_AVAILABLE = False

class AudioFE:
    def __init__(self) -> None:
        from livekit.rtc import AudioFrame  # type: ignore
        from livekit.rtc.apm import AudioProcessingModule  # type: ignore
        from silero_vad import load_silero_vad  # type: ignore

        self.AudioFrame = AudioFrame
        self.apm = AudioProcessingModule(echo_cancellation=True, noise_suppression=True)
        self.vad = load_silero_vad(onnx=True)
        self.ref: deque[int] = deque()
        self.vad_buf = np.zeros(0, dtype=np.float32)
        # NEW: cap reference buffer to ~3 s of 16k audio to prevent overflow
        self._ref_max = SR * 3

    def push_ref(self, samples: np.ndarray, sr: int) -> None:
        n_in = samples.size
        n_out = int(round(n_in * SR / sr))
        if n_in == n_out:
            resampled = samples.astype(np.float32)
        else:
            x_in = np.linspace(0, n_in - 1, n_in, dtype=np.float32)
            x_out = np.linspace(0, n_in - 1, n_out, dtype=np.float32)
            resampled = np.interp(x_out, x_in, samples.astype(np.float32))
        self.ref.extend((resampled * 32767).clip(-32768, 32767).astype(np.int16).tolist())
        # Drop oldest if we exceed cap
        while len(self.ref) > self._ref_max:
            self.ref.popleft()

    def vad_prob_clean(self, mic_i16: np.ndarray) -> float:
        cleaned_chunks = []
        for i in range(0, len(mic_i16), APM_FRAME):
            mic_f = mic_i16[i:i + APM_FRAME]
            if len(mic_f) < APM_FRAME:
                mic_f = np.pad(mic_f, (0, APM_FRAME - len(mic_f)))
            ref_f = np.zeros(APM_FRAME, dtype=np.int16)
            for j in range(APM_FRAME):
                if not self.ref:
                    break
                ref_f[j] = self.ref.popleft()
            self.apm.process_reverse_stream(self.AudioFrame(
                ref_f.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=APM_FRAME))
            mic_frame = self.AudioFrame(
                mic_f.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=APM_FRAME)
            self.apm.process_stream(mic_frame)
            cleaned_chunks.append(np.frombuffer(bytes(mic_frame.data), dtype=np.int16))
        cleaned = np.concatenate(cleaned_chunks).astype(np.float32) / 32768.0
        self.vad_buf = np.concatenate([self.vad_buf, cleaned])
        last_p = 0.0
        while self.vad_buf.size >= VAD_WIN:
            w = self.vad_buf[:VAD_WIN]
            self.vad_buf = self.vad_buf[VAD_WIN:]
            last_p = float(self.vad(torch.from_numpy(w), SR).item())
        return last_p

    def reset(self) -> None:
        self.ref.clear()
        self.vad_buf = np.zeros(0, dtype=np.float32)


# ── Per-connection session ────────────────────────────────────────────────────

class Session:
    def __init__(self, ws: WebSocket, models: dict, executor) -> None:
        self.ws = ws
        self.stt = models["stt"]
        self.tts = models["tts"]
        self.llm: LLM = models["llm"]
        self.smart_turn = models["smart_turn"]
        self.history: list[dict] = []
        self.audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.executor = executor
        self.fe = AudioFE()
        self.in_turn = False
        self.silent_chunks = 0
        self.silence_limit = max(1, int(SILENCE_END_MS / (VAD_WIN / SR * 1000)))
        self.utt_buf: list[np.ndarray] = []
        self.silero = models["silero"]
        self.tts_playing = False
        self.tts_started_at = 0.0
        self.tts_drain_until = 0.0
        self.barge_in = asyncio.Event()
        self.barge_streak = 0
        self.last_assistant: str = ""
        self.turn_started_at: float = 0.0
        self.turn_metrics: dict[str, float] = {}
        # Smart-turn reprieve state: when smart-turn rejects mid-thought, we
        # keep the turn open instead of discarding. Force-accept on next
        # silence_end once we've used up the reprieve budget.
        self._smart_turn_reprieves: int = 0
        self._force_finalize: bool = False
        # NEW: noise-floor tracker for RMS gating
        self._noise_floor_db = -60.0
        self._noise_samples: deque[float] = deque(maxlen=50)
        # FireRed pVAD: target-speaker-gated barge-in (fallback: AEC+Silero)
        self.pvad = models.get("pvad")
        # Buffer for pvad frame assembly (needs exactly 160 samples)
        self._pvad_buf = np.zeros(0, dtype=np.float32)

    async def emit(self, ev: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(ev))
        except Exception:
            pass

    async def emit_audio(self, samples: np.ndarray, sr: int) -> None:
        i16 = (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        if TTS_BINARY:
            await self.ws.send_text(json.dumps({"type": "audio_header", "sr": sr, "len": len(i16)}))
            await self.ws.send_bytes(i16)
        else:
            await self.ws.send_text(json.dumps({
                "type": "audio_out", "sr": sr,
                "pcm_b64": base64.b64encode(i16).decode("ascii"),
            }))

    async def ingest(self, audio_bytes: bytes) -> None:
        chunk = np.frombuffer(audio_bytes, dtype=np.int16)
        now = time.time()
        if self.barge_in.is_set():
            await self.audio_q.put(chunk.astype(np.float32) / 32768.0)
            return
        if self.tts_playing or now < self.tts_drain_until:
            elapsed_ms = (now - self.tts_started_at) * 1000 if self.tts_started_at else 0
            if elapsed_ms >= AEC_GUARD_MS:
                # ── pVAD (FireRedChat): target-speaker-gated barge-in ─────
                if self.pvad is not None:
                    chunk_f32 = chunk.astype(np.float32) / 32768.0
                    # Accumulate chunks into buffer, drain exactly 160-sample frames
                    self._pvad_buf = np.concatenate([self._pvad_buf, chunk_f32])
                    while self._pvad_buf.size >= 160:
                        frame = self._pvad_buf[:160]
                        self._pvad_buf = self._pvad_buf[160:]
                        p = self.pvad.is_target_speaking(frame)
                        if p > BARGE_IN_THRESHOLD:
                            self.barge_streak += 1
                            if self.barge_streak >= BARGE_IN_FRAMES and not self.barge_in.is_set():
                                logger.info(f"barge-in (pvad p={p:.2f}, streak={self.barge_streak})")
                                self.barge_in.set()
                        else:
                            self.barge_streak = 0
                else:
                    # Fallback: AEC-cleaned Silero VAD (no speaker gate)
                    p = self.fe.vad_prob_clean(chunk)
                    if p > BARGE_IN_THRESHOLD:
                        self.barge_streak += 1
                        if self.barge_streak >= BARGE_IN_FRAMES and not self.barge_in.is_set():
                            logger.info(f"barge-in (cleaned p={p:.2f}, streak={self.barge_streak})")
                            self.barge_in.set()
                    else:
                        self.barge_streak = 0
            return
        await self.audio_q.put(chunk.astype(np.float32) / 32768.0)

    async def dialogue_loop(self) -> None:
        accum = np.zeros(0, dtype=np.float32)
        stt_streaming = hasattr(self.stt, "feed_chunk")
        loop = asyncio.get_running_loop()
        vad_windows_processed = 0
        while True:
            chunk_f32 = await self.audio_q.get()
            accum = np.concatenate([accum, chunk_f32])
            while accum.size >= VAD_WIN:
                window = accum[:VAD_WIN]
                accum = accum[VAD_WIN:]

                # ── NEW: RMS energy gate ─────────────────────────────────────
                rms_db = _rms_dbfs(window)
                # Update noise floor from quiet windows (below VAD threshold)
                if rms_db < VAD_MIN_RMS_DB:
                    self._noise_samples.append(rms_db)
                    if len(self._noise_samples) >= 10:
                        self._noise_floor_db = float(np.percentile(list(self._noise_samples), 50))
                # Require signal to be both above absolute floor AND significantly
                # above the tracked noise floor (6 dB SNR minimum)
                snr = rms_db - self._noise_floor_db
                if rms_db < VAD_MIN_RMS_DB or snr < 6.0:
                    # If we were in a turn, count this as silence
                    if self.in_turn:
                        self.silent_chunks += 1
                        self.utt_buf.append(window)
                        if stt_streaming:
                            partial = await loop.run_in_executor(
                                self.executor, self.stt.feed_chunk, window)
                            if partial:
                                await self.emit({"type": "transcript_partial", "data": partial})
                        if self._exceeded_max_utterance():
                            self._force_finalize = True
                            await self._finalize_turn()
                        elif self.silent_chunks >= self.silence_limit:
                            await self._finalize_turn()
                    continue
                # ── end RMS gate ─────────────────────────────────────────────

                p = float(self.silero(torch.from_numpy(window), SR).item())
                if p > VAD_THRESHOLD:
                    if not self.in_turn:
                        self.in_turn = True
                        self.turn_started_at = time.time()
                        await self.emit({"type": "speech_started"})
                        if stt_streaming:
                            self.stt.begin_utterance()
                    self.silent_chunks = 0
                    self.utt_buf.append(window)
                    if stt_streaming:
                        partial = await loop.run_in_executor(
                            self.executor, self.stt.feed_chunk, window)
                        if partial:
                            await self.emit({"type": "transcript_partial", "data": partial})
                    if self._exceeded_max_utterance():
                        # Speaker hasn't paused but we've hit the hard cap — finalize anyway.
                        self._force_finalize = True
                        await self._finalize_turn()
                elif self.in_turn:
                    self.silent_chunks += 1
                    self.utt_buf.append(window)
                    if stt_streaming:
                        partial = await loop.run_in_executor(
                            self.executor, self.stt.feed_chunk, window)
                        if partial:
                            await self.emit({"type": "transcript_partial", "data": partial})
                    if self._exceeded_max_utterance():
                        self._force_finalize = True
                        await self._finalize_turn()
                    elif self.silent_chunks >= self.silence_limit:
                        await self._finalize_turn()
                if accum.size > SR * 30:
                    accum = accum[-SR * 5:]
                vad_windows_processed += 1
                if vad_windows_processed % 4 == 0:
                    await asyncio.sleep(0)

    def _exceeded_max_utterance(self) -> bool:
        if not self.in_turn or not self.turn_started_at:
            return False
        return (time.time() - self.turn_started_at) >= MAX_UTTERANCE_S

    async def _finalize_turn(self) -> None:
        utterance = np.concatenate(self.utt_buf) if self.utt_buf else np.zeros(0, dtype=np.float32)
        self.utt_buf.clear()
        was_in_turn = self.in_turn
        self.in_turn = False
        self.silent_chunks = 0
        try:
            if hasattr(self.silero, 'reset_states'):
                self.silero.reset_states()
        except Exception:
            pass

        t_vad_end = time.time()
        speech_dur = (t_vad_end - self.turn_started_at) if was_in_turn and self.turn_started_at else 0.0
        metrics: dict[str, float] = {"speech_dur_ms": int(speech_dur * 1000)}
        loop = asyncio.get_running_loop()
        stt_streaming = hasattr(self.stt, "feed_chunk")

        min_samples = int(MIN_SPEECH_MS * SR / 1000)
        if utterance.size < min_samples:
            if stt_streaming:
                self.stt.begin_utterance()
            self._smart_turn_reprieves = 0
            self._force_finalize = False
            return

        if self.smart_turn is not None and not self._force_finalize:
            try:
                t0 = time.time()
                prob = await loop.run_in_executor(
                    self.executor, self.smart_turn.predict, utterance)
                metrics["smart_turn_ms"] = int((time.time() - t0) * 1000)
                if prob < SMART_TURN_THRESHOLD:
                    # "Speaker isn't done" — keep the turn open, keep the STT
                    # encoder cache warm, demand more silence next time.
                    # After SMART_TURN_MAX_REPRIEVES we force-accept regardless
                    # so a chronically uncertain smart-turn can't strand us.
                    if self._smart_turn_reprieves < SMART_TURN_MAX_REPRIEVES:
                        self._smart_turn_reprieves += 1
                        # Restore in-turn state — undo the housekeeping at the
                        # top of this method so accumulation can continue.
                        self.in_turn = was_in_turn
                        self.utt_buf = [utterance]  # re-stash so next chunk grows it
                        # Push silence-end deadline out by REPRIEVE_MS:
                        # subtract (REPRIEVE_MS / VAD_WIN_MS) from silent_chunks
                        # so we demand that many extra ms of silence before
                        # re-checking. Floor at 0 — never make it easier to fire.
                        reprieve_chunks = max(1, int(SMART_TURN_REPRIEVE_MS / (VAD_WIN / SR * 1000)))
                        self.silent_chunks = max(0, self.silence_limit - reprieve_chunks)
                        logger.info(
                            f"smart-turn reprieve {self._smart_turn_reprieves}/{SMART_TURN_MAX_REPRIEVES} "
                            f"(p={prob:.2f}) — keeping turn open"
                        )
                        return
                    logger.info(
                        f"smart-turn still uncertain (p={prob:.2f}) but reprieve budget exhausted — finalizing"
                    )
            except Exception as e:
                logger.warning(f"smart-turn failed: {e}")

        # Committed to finalize — clear reprieve state for next turn.
        self._smart_turn_reprieves = 0
        self._force_finalize = False

        t_stt = time.time()
        if stt_streaming:
            text = await loop.run_in_executor(self.executor, self.stt.finalize)
        else:
            text = await loop.run_in_executor(
                self.executor, self.stt.transcribe, utterance)
        metrics["stt_ms"] = int((time.time() - t_stt) * 1000)
        if not text:
            return
        if LATIN_ONLY and not _is_mostly_latin(text):
            logger.info(f"non-latin transcript dropped: {text!r}")
            return
        if self._looks_like_echo(text, self.last_assistant):
            logger.info(f"echo-loop transcript dropped: {text!r}")
            return
        await self.emit({"type": "transcript", "data": text})
        self.history.append({"role": "user", "content": text})

        self.turn_metrics = metrics
        self.turn_metrics["_vad_end_at"] = t_vad_end

        await self.emit({"type": "generation_start"})
        await self.emit({"type": "assistant_start"})
        full = await self._stream_llm_to_tts(text)
        if full:
            self.history.append({"role": "assistant", "content": full})
        await self.emit({"type": "generation_done"})

        m = {k: v for k, v in self.turn_metrics.items() if not k.startswith("_")}
        m["total_ttfb_ms"] = m.get("smart_turn_ms", 0) + m.get("stt_ms", 0) \
            + m.get("llm_ttft_ms", 0) + m.get("tts_ttfb_ms", 0)
        logger.info("turn_metrics: " + " ".join(f"{k}={int(v)}" for k, v in m.items()))
        await self.emit({"type": "turn_metrics", "data": m})

        if MEMORY_ENABLED and full:
            asyncio.create_task(self._memory_after_turn(text))

    async def _memory_after_turn(self, user_text: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self._update_memory, user_text)
        except Exception as e:
            logger.warning(f"memory update failed: {e}")
        n_turns = sum(1 for m in self.history if m["role"] == "assistant")
        if n_turns and n_turns % MEMORY_CONSOLIDATE_EVERY == 0:
            try:
                await loop.run_in_executor(self.executor, self._consolidate_memory)
            except Exception as e:
                logger.warning(f"memory consolidation failed: {e}")

    def _llm_oneshot(self, prompt: str, max_tokens: int = 80) -> str:
        msgs = [{"role": "user", "content": prompt}]
        out = []
        for tok in self.llm.stream(msgs):
            out.append(tok)
            if len("".join(out)) > max_tokens * 6:
                break
        return "".join(out).strip()

    def _update_memory(self, user_text: str) -> None:
        if not user_text:
            return
        existing = _read_text(MEMORY_MD)
        prompt = (
            f"Current memory:\\n{existing or '(empty)'}\\n\\n"
            f"User just said: {user_text!r}\\n\\n"
            "Did the user state a NEW durable fact about themselves "
            "(name, preference, location, ongoing project, relationship)? "
            "If yes, output ONE short fact per line, each starting with '- '. "
            "If no, output ONLY the single word: NONE. Do not invent facts."
        )
        result = self._llm_oneshot(prompt, max_tokens=80)
        if not result or "NONE" in result.upper():
            return
        new_lines = [ln.strip() for ln in result.splitlines() if ln.strip().startswith("-")]
        if not new_lines:
            return
        MEM_DIR.mkdir(parents=True, exist_ok=True)
        header_needed = not MEMORY_MD.exists()
        with MEMORY_MD.open("a") as f:
            if header_needed:
                f.write("# Memory\\n")
            f.write("\\n".join(new_lines) + "\\n")
        logger.info(f"memory +{len(new_lines)} lines")

    def _consolidate_memory(self) -> None:
        if not MEMORY_MD.exists():
            return
        prompt = (
            f"Here is a memory file about a user:\\n\\n{_read_text(MEMORY_MD)}\\n\\n"
            "Rewrite it: merge duplicates, remove transient/session-specific items "
            "(questions asked, topics discussed), keep only durable facts. Output "
            "the cleaned file starting with '# Memory' followed by lines starting "
            "with '- '. No explanation."
        )
        result = self._llm_oneshot(prompt, max_tokens=300)
        if result and result.startswith("# Memory"):
            MEMORY_MD.write_text(result + "\\n")
            logger.info("memory consolidated")

    def _maybe_tool_roundtrip(self, msgs: list[dict], user_text: str) -> list[dict]:
        if not os.getenv("SERPER_API_KEY") or os.getenv("NOVA_ENABLE_TOOLS", "1") != "1":
            return msgs
        if not (os.getenv("NOVA_ALWAYS_TOOLS") == "1" or _needs_tools(user_text)):
            return msgs
        try:
            reply = self.llm.chat(msgs, tools=SERPER_TOOLS, max_tokens=self.llm.tool_max_tokens)
        except Exception as e:
            logger.warning(f"tool pre-call failed: {e}")
            return msgs

        tool_calls = reply.get("tool_calls") or []
        if not tool_calls and reply.get("content"):
            tool_calls = _extract_tool_calls_from_text(reply["content"])
        if not tool_calls:
            return msgs

        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            key = (fn.get("name", ""), fn.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(tc)
        tool_calls = deduped

        msgs = msgs + [{
            "role": "assistant",
            "content": reply.get("content") or "",
            "tool_calls": tool_calls,
        }]
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            logger.info(f"tool call: {name}({args})")
            result = _exec_tool(name, args)
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": result,
            })
        return msgs

    async def _stream_llm_to_tts(self, user_text: str) -> str:
        loop = asyncio.get_running_loop()
        persona = load_persona()
        sys_prompt = SYSTEM_PROMPT + ("\\n\\n" + persona if persona else "")
        msgs = [{"role": "system", "content": sys_prompt}, *self.history[-10:]]

        msgs = await loop.run_in_executor(self.executor, self._maybe_tool_roundtrip, msgs, user_text)

        sent_q: asyncio.Queue = asyncio.Queue(maxsize=8)
        audio_q: asyncio.Queue = asyncio.Queue(maxsize=4)
        full_text_parts: list[str] = []
        first_sentence_emitted = [False]
        t_llm_start = time.time()

        vad_end_at = self.turn_metrics.get("_vad_end_at", t_llm_start)

        def llm_producer() -> None:
            buf = ""
            first_token_seen = False
            try:
                for delta in self.llm.stream(msgs):
                    if not first_token_seen and delta:
                        first_token_seen = True
                        self.turn_metrics["llm_ttft_ms"] = int(
                            (time.time() - vad_end_at) * 1000)
                    buf += delta
                    while True:
                        m = CLAUSE_BREAK.search(buf)
                        if m:
                            chunk = buf[: m.start() + 1].strip()
                            buf = buf[m.end():]
                            if len(chunk) >= SENT_MIN_CHARS or not first_sentence_emitted[0]:
                                asyncio.run_coroutine_threadsafe(sent_q.put(chunk), loop)
                                full_text_parts.append(chunk)
                                first_sentence_emitted[0] = True
                            continue
                        if len(buf) >= 140 and not first_sentence_emitted[0]:
                            asyncio.run_coroutine_threadsafe(sent_q.put(buf.strip()), loop)
                            full_text_parts.append(buf.strip())
                            first_sentence_emitted[0] = True
                            buf = ""
                        break
                tail = buf.strip()
                if tail:
                    asyncio.run_coroutine_threadsafe(sent_q.put(tail), loop)
                    full_text_parts.append(tail)
            finally:
                asyncio.run_coroutine_threadsafe(sent_q.put(None), loop)

        async def synth_consumer() -> None:
            while True:
                sentence = await sent_q.get()
                if sentence is None or self.barge_in.is_set():
                    await audio_q.put(None)
                    return
                await self.emit({"type": "llm_token", "data": sentence})
                try:
                    async for samples, sr in self.tts.synth_stream(sentence):
                        if self.barge_in.is_set():
                            await audio_q.put(None)
                            return
                        await audio_q.put((sentence, samples, sr))
                except Exception as e:
                    logger.warning(f"Pocket-TTS stream failed, falling back: {e}")
                    samples, sr = await loop.run_in_executor(
                        self.executor, self.tts.synth, sentence)
                    await audio_q.put((sentence, samples, sr))

        total_tts_samples = 0
        first_audio_logged = [False]

        async def play_consumer() -> None:
            nonlocal total_tts_samples
            CHUNK = int(os.getenv("NOVA_TTS_PLAY_CHUNK", "2048"))
            while True:
                item = await audio_q.get()
                if item is None or self.barge_in.is_set():
                    return
                _sentence, samples, sr = item
                self.fe.push_ref(samples, sr)
                total_tts_samples += len(samples)
                for i in range(0, len(samples), CHUNK):
                    if self.barge_in.is_set():
                        return
                    if not first_audio_logged[0]:
                        first_audio_logged[0] = True
                        ttfb_ms = int((time.time() - vad_end_at) * 1000)
                        self.turn_metrics["tts_ttfb_ms"] = ttfb_ms
                        logger.info(f"TTS first-audio TTFB: {ttfb_ms} ms (from vad_end)")
                    await self.emit_audio(samples[i:i + CHUNK], sr)
                    await asyncio.sleep(0)

        prod_task = loop.run_in_executor(self.executor, llm_producer)
        self.barge_in.clear()
        self.barge_streak = 0
        self.fe.reset()
        # Reset pVAD streaming state (mel buffer + GRU) for fresh turn
        if self.pvad is not None:
            self.pvad.reset()
            self._pvad_buf = np.zeros(0, dtype=np.float32)
        self.tts_playing = True
        self.tts_started_at = time.time()
        try:
            await asyncio.gather(synth_consumer(), play_consumer())
        finally:
            barge_happened = self.barge_in.is_set()
            if not barge_happened:
                await prod_task
            self.tts_playing = False
            if barge_happened:
                self.tts_drain_until = 0.0
            else:
                total_dur_s = total_tts_samples / self.tts.sample_rate if total_tts_samples else 0.0
                elapsed_s = time.time() - self.tts_started_at if self.tts_started_at else 0.0
                remaining_s = max(0.0, total_dur_s - elapsed_s)
                self.tts_drain_until = time.time() + remaining_s + 0.7
            self.tts_started_at = 0.0
        full = " ".join(full_text_parts)
        self.last_assistant = full
        return full

    @staticmethod
    def _looks_like_echo(transcript: str, assistant: str) -> bool:
        if not transcript or not assistant:
            return False
        if len(transcript.split()) <= 3 or len(transcript) <= 10:
            return False
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
        t, a = norm(transcript), norm(assistant)
        if not t or not a:
            return False
        if t in a or a in t:
            return True
        ts, as_ = set(t.split()), set(a.split())
        if not ts or not as_:
            return False
        return len(ts & as_) / len(ts | as_) >= 0.6


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from concurrent.futures import ThreadPoolExecutor
    from silero_vad import load_silero_vad  # type: ignore
    logger.info("Loading models …")
    smart_turn = None
    try:
        from pipeline_mp.smart_turn import SmartTurnPredictor, is_enabled as st_enabled  # type: ignore
        if st_enabled():
            smart_turn = SmartTurnPredictor()
            smart_turn.predict(np.zeros(8000, dtype=np.float32))
    except Exception as e:
        logger.warning(f"smart-turn disabled: {e}")
    tts = PocketTTS()
    # ── FireRed pVAD: personalised speaker gate for barge-in ──────────
    pvad = None
    if _PVAD_AVAILABLE:
        try:
            vp_enc = Path(__file__).parent / "nova-l7" / "L-3" / "data" / "voiceprints" / "driver1.enc"
            pvad_model = Path(__file__).parent / "models" / "FireRedChat-pvad"
            vp_ok = vp_enc.is_file()
            model_ok = (pvad_model / "pvad.onnx").is_file()
            if vp_ok and model_ok:
                pvad = FireRedPVAD(voiceprint_enc=str(vp_enc), model_dir=str(pvad_model))
                logger.info("pVAD (FireRedChat): loaded — target-speaker barge-in active")
            else:
                logger.warning(
                    f"pVAD not loaded: voiceprint={vp_ok} model={model_ok}"
                )
        except Exception as e:
            logger.warning(f"pVAD init failed ({e}) — falling back to AEC+Silero barge-in")
    else:
        logger.info("pVAD (FireRedChat): import unavailable — AEC+Silero only")
    # ── end pVAD ───────────────────────────────────────────────────────
    app.state.models = {
        "stt": make_stt(),
        "tts": tts,
        "llm": LLM(),
        "silero": load_silero_vad(onnx=True),
        "smart_turn": smart_turn,
        "pvad": pvad,
    }
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    await tts.warmup_stream()
    logger.info("Ready on :8001")
    yield
    app.state.executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/persona/{name}")
def persona(name: str) -> dict:
    name = name.lower()
    paths = {
        "user": USER_MD,
        "memory": MEMORY_MD,
        "conversation": MEM_DIR / "Conversation.md",
    }
    p = paths.get(name)
    if p is None:
        return {"name": name, "content": "", "exists": False}
    return {"name": name, "content": _read_text(p), "exists": p.exists()}


@app.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    await ws.accept()
    sess = Session(ws, ws.app.state.models, ws.app.state.executor)
    loop_task = asyncio.create_task(sess.dialogue_loop())
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                await sess.ingest(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                try:
                    obj = json.loads(msg["text"])
                except Exception:
                    continue
                if obj.get("type") == "reset":
                    sess.history.clear()
                elif obj.get("type") == "set_voice":
                    voice = obj.get("voice", "alba")
                    sess.tts.set_voice(voice)
                    await sess.emit({"type": "voice_set", "voice": voice})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        loop_task.cancel()


# ── Voice fingerprint enrollment ──────────────────────────────────────────────

# Accumulated fingerprints for dedicated enrollment
_voice_fp_pending: list[np.ndarray] = []
_VOICEPRINT_DIR = Path(__file__).parent / "nova-l7" / "L-3" / "data" / "voiceprints"


@app.post("/enroll/voice-sample")
async def enroll_voice_sample(request: Request):
    """Enroll voice fingerprint from 5 uploaded WAV samples."""
    global _voice_fp_pending

    form = await request.form()
    index = int(form.get("index", "0"))
    total = int(form.get("total", "5"))
    driver_id = form.get("driver_id", "driver1")

    if driver_id not in ("driver1", "driver2"):
        return JSONResponse(content={"status": "error", "error": "Only driver1 and driver2 supported"}, status_code=400)

    file = form.get("file")
    if file is None:
        return JSONResponse(content={"status": "error", "error": "No file provided"}, status_code=400)

    content = await file.read()

    # Convert to 16kHz mono WAV via pydub if available, otherwise write raw
    try:
        from pydub import AudioSegment  # type: ignore
        audio = AudioSegment.from_file(io.BytesIO(content))
    except Exception:
        # Raw WAV fallback — pydub not available or bad format
        audio = None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        if audio is not None:
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp.name, format="wav")
        else:
            tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        sys.path.insert(0, str(Path(__file__).parent / "nova-l7" / "L-3"))
        try:
            from verify import extract_fingerprint_from_array
            # Read WAV as float32 numpy — avoid torchaudio.load (needs FFmpeg/torchcodec)
            import wave
            with wave.open(str(tmp_path), "rb") as wf:
                assert wf.getnchannels() == 1 and wf.getframerate() == 16000, "bad wav format"
                raw = wf.readframes(wf.getnframes())
            pcm_i16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            fp = extract_fingerprint_from_array(pcm_i16)
        finally:
            sys.path.pop(0)
        _voice_fp_pending.append(fp)
        logger.info(f"Voice fingerprint sample {index}/{total} extracted for {driver_id}")
    except Exception as e:
        logger.error(f"Voiceprint extraction failed: {e}", exc_info=True)
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)

    if index >= total:
        master = np.mean(_voice_fp_pending, axis=0)
        master = master / np.linalg.norm(master)

        sys.path.insert(0, str(Path(__file__).parent / "nova-l7" / "L-3"))
        try:
            from crypto_utils import save_array
            _VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
            save_array(master, _VOICEPRINT_DIR / f"{driver_id}.enc")
        finally:
            sys.path.pop(0)

        _voice_fp_pending.clear()
        logger.info(f"Voice fingerprint enrolled for {driver_id}")
        return {"status": "enrolled", "message": f"Voice fingerprint saved for {driver_id}"}

    return {"status": "ok", "collected": index, "total": total}


@app.get("/enroll/check")
async def enrollment_check():
    """Check if KWS and voice fingerprint are enrolled."""
    vp_ok = (_VOICEPRINT_DIR / "driver1.enc").exists()
    return {"kws_enrolled": False, "voice_enrolled": vp_ok, "fully_enrolled": vp_ok}


@app.get("/pVAD/status")
async def pvad_status():
    """Return pVAD loading status."""
    loaded = False
    models = app.state.models if hasattr(app.state, "models") else {}
    pvad = models.get("pvad") if models else None
    loaded = pvad is not None
    return {"loaded": loaded, "model": "FireRedChat-pvad", "speaker": "driver1" if loaded else None}


if __name__ == "__main__":
    import uvicorn
    from llama_launcher import spawn as _llama_spawn, shutdown as _llama_shutdown

    llama_proc = _llama_spawn()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        _llama_shutdown(llama_proc)