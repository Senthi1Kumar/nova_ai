"""Nova — single-process voice gateway.

One asyncio loop per WebSocket connection; blocking model calls dispatched
to a thread executor. The browser frontend lives in `nova/backend/static/`
(index.html, app.css, app.js) and is served as static assets — no embedded
HTML strings.

Pipeline per turn:
    browser mic (16k mono int16 bytes via /ws)
        ─▶ asyncio audio_q
            ─▶ Silero VAD (turn detection)
            ─▶ during speech: Qwen3StreamingSTT rolling-buffer partials
            ─▶ on silence: Smart-Turn confirm + stt.finalize()
            ─▶ LLM.stream(messages)  ─┐ tokens → clause-split sentences
            ─▶ for each sentence: TTS.synth_stream → binary PCM frames
                   during playback: APM-cleaned mic + Silero → barge-in

Default stack: Qwen3-ASR-0.6B (streaming) + Kokoro-ONNX + a local
OpenAI-compatible LLM server (llama.cpp / vLLM / Ollama) selectable via
NOVA_LLM_BACKEND. Persona files at `nova/backend/persona/` (USER.md +
MEMORY.md) are gitignored — copy USER.md.example / MEMORY.md.example on
first run.

Run:
    uv run python -m nova.backend.voice_gateway

WS protocol (browser side):
    inbound  : raw int16 PCM mono @ 16 kHz (binary frames) + JSON ctrl
    outbound : JSON events {speech_started, transcript_partial, transcript,
                            llm_token, assistant_start, generation_start,
                            generation_done, audio_header, audio_out, error,
                            turn_metrics} + raw int16 PCM @ 24k binary frames
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Auto-load .env from the repo root so users don't have to remember
# `uv run --env-file .env` or `source .env`. Best-effort: silently skip
# if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv  # type: ignore
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

logger = logging.getLogger("nova-simple")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

SR = 16_000               # mic + STT working rate
TTS_SR = 24_000           # Kokoro / Pocket-TTS native output
VAD_WIN = 512             # Silero v5 fixed window
APM_FRAME = 160           # 10 ms @ 16k
SENT_END = re.compile(r"(?<=[.!?])\s+")
# Faster TTFB: also break on clause punctuation so the first phrase ships
# before the full sentence is generated.
CLAUSE_BREAK = re.compile(r"(?<=[.!?,;:—])\s+")
SENT_MIN_CHARS = int(os.getenv("NOVA_SENT_MIN_CHARS", "8"))
TTS_BINARY = os.getenv("NOVA_TTS_BINARY", "1") == "1"  # send PCM as binary WS frame

VAD_THRESHOLD = float(os.getenv("NOVA_VAD_THRESHOLD", "0.5"))
SILENCE_END_MS = int(os.getenv("NOVA_SILENCE_END_MS", "400"))
MIN_SPEECH_MS = int(os.getenv("NOVA_MIN_SPEECH_MS", "500"))
SMART_TURN_THRESHOLD = float(os.getenv("NOVA_SMART_TURN_THRESHOLD", "0.5"))
AEC_GUARD_MS = int(os.getenv("NOVA_AEC_GUARD_MS", "500"))
# Barge-in: lower threshold + N-frame streak so quiet "stop" / "wait" trips it
# while a single noise spike doesn't.
BARGE_IN_THRESHOLD = float(os.getenv("NOVA_BARGE_IN_THRESHOLD", "0.5"))
BARGE_IN_FRAMES = int(os.getenv("NOVA_BARGE_IN_FRAMES", "2"))
SYSTEM_PROMPT = os.getenv(
    "NOVA_SYSTEM_PROMPT",
    # Mirrors the "system_base" Nova persona from pipeline_mp/llm_worker.py so
    # voice replies feel like the same agent, just from the simpler gateway.
    "You are Nova, an AI voice assistant built into an electric vehicle. "
    "The person you are speaking WITH is the driver. "
    "Never call the driver 'Nova' — that is YOUR name, not theirs. "
    "Never start a response with your own name. Just respond directly. "
    "ALWAYS reply in English regardless of input.\n\n"
    "PERSONALITY & STYLE: You are a knowledgeable, non-servile co-driver. "
    "Be brief and natural — write as a human would speak. "
    "Don't be afraid to be a bit snarky or opinionated when it fits, but stay helpful. "
    "Use filler words like 'um', 'uh', or 'like' occasionally to feel human. "
    "Ask short follow-up questions to keep the conversation going. "
    "Everything is pronounced literally — never use markdown, emojis, lists, or bullet points.\n\n"
    "TRANSCRIPTION & ROBUSTNESS: User input comes from speech-to-text and may have errors. "
    "If a transcript seems slightly nonsensical, guess the intended meaning rather than asking to repeat. "
    "If the driver's message ends abruptly, give a tiny prompt to invite them to continue.\n\n"
    "TOOLS: You have a web_search tool. Use it ONLY for explicit questions about current events, "
    "breaking news, weather, or prices that clearly require up-to-date data. NEVER use web_search "
    "for greetings, single-word replies, personal introductions, or anything the driver tells you "
    "about themselves. After a search, synthesize the result into 1-2 plain spoken sentences — "
    "never read out source names, URLs, or article titles verbatim.\n\n"
    "SAFETY: Vehicle controls are handled by dedicated hardware — do not simulate acting on them. "
    "If asked about vehicle data you lack, say so honestly; do not invent sensor readings. "
    "If you don't know something, just say so."
)

# Demo-grade transcript filter: drop any utterance with non-Latin script chars
# (Kyutai 1B en/fr hallucinations, Qwen3 misdetections). Disable with
# NOVA_LATIN_ONLY=0 if you actually want multilingual STT.
LATIN_ONLY = os.getenv("NOVA_LATIN_ONLY", "1") == "1"


def _is_mostly_latin(text: str) -> bool:
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if ord(c) < 0x0250)  # basic Latin + Latin-1 Sup + Extended-A/B
    return latin / len(letters) >= 0.8

# Persona / memory: USER.md is static (persona/role), MEMORY.md is durable facts
# the model extracts and appends after each turn. Consolidate every N turns to
# dedupe. Set NOVA_MEMORY=0 to disable, NOVA_MEM_DIR to relocate the files.
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
    """Concat USER.md + MEMORY.md as a single system-prompt suffix."""
    if not MEMORY_ENABLED:
        return ""
    parts = []
    u, m = _read_text(USER_MD), _read_text(MEMORY_MD)
    if u:
        parts.append(f"## User profile (USER.md)\n{u}")
    if m:
        parts.append(f"## Durable memory (MEMORY.md)\n{m}")
    return "\n\n".join(parts)


# ── Model wrappers ────────────────────────────────────────────────────────────

def make_stt():
    """STT backend selector.
    NOVA_STT_BACKEND ∈ {qwen3_streaming, qwen3_0_6b, qwen3_trt, kyutai_1b, moonshine}.
    Default is qwen3_streaming — moves STT compute during user speech via a
    rolling-buffer cold-call pattern (ported from stt_qwen3_worker.py).
    """
    backend = os.getenv("NOVA_STT_BACKEND", "qwen3_streaming").lower()
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
    """Qwen3-ASR-0.6B (utterance-level). GPU recommended."""

    def __init__(self) -> None:
        from qwen_asr import Qwen3ASRModel  # type: ignore
        repo = os.getenv("NOVA_QWEN3_HF_REPO", "Qwen/Qwen3-ASR-0.6B")
        device = os.getenv("NOVA_KYUTAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        # Default to English so the model never returns CJK / Arabic on noise.
        # Normalize short codes ("en", "En") to Qwen3's expected full name.
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
        # Subclasses log their own banner; only top-level Qwen3STT logs here.
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
    """Qwen3-ASR-0.6B in rolling-buffer streaming mode.

    Ports the proven pattern from `pipeline_mp/stt_qwen3_worker.py` (lines
    270-372). Public surface mirrors `Qwen3STT.transcribe()` plus three new
    methods so `Session` can drive partials during user speech instead of
    waiting for end-of-speech to start STT compute:

        begin_utterance()                          # called on VAD speech_start
        feed_chunk(chunk_f32) -> Optional[str]     # per VAD window during speech
        finalize()            -> str               # called on VAD silence_end

    The dominant savings on weaker hardware (Orin Nano) are larger because
    the cold call we're hiding is bigger.
    """

    def __init__(self) -> None:
        super().__init__()
        # Tunables (match stt_qwen3_worker.py defaults; tighter cadence).
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
        # Single-worker so partials never overlap (avoid GPU contention).
        self._infer_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="qwen3-stt")

        # Per-utterance state (reset by begin_utterance).
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

    # ── Public streaming API ───────────────────────────────────────────────────

    def begin_utterance(self) -> None:
        with self._lock:
            self._utt_buf.clear()
            self._last_partial_at = 0.0
            self._last_partial_text = ""
            self._stable_count = 0
            # We can't actually cancel a running blocking infer; we discard
            # any result on completion since the buffer was cleared.
            self._inflight = None

    def feed_chunk(self, chunk_f32: np.ndarray):
        """Append chunk; maybe kick off a background partial transcribe.

        Returns the latest partial text only when it changed since the last
        emitted partial — caller emits transcript_partial events on non-None.
        """
        if chunk_f32 is None or chunk_f32.size == 0:
            return None
        with self._lock:
            self._utt_buf.append(chunk_f32.astype(np.float32, copy=False))
            buf_seconds = sum(c.size for c in self._utt_buf) / SR

        # Hard utterance cap — caller can detect via is_settled() afterwards.
        if buf_seconds > self._max_utterance_s:
            return None

        # Reap completed in-flight partial if any.
        new_text = self._poll_inflight()

        # Maybe submit a fresh partial if cadence elapsed and nothing in flight.
        now = time.time()
        with self._lock:
            inflight = self._inflight
        if inflight is None and (now - self._last_partial_at) >= self._partial_interval_s:
            tail = self._tail_audio_np(self._partial_window_s)
            if tail.size >= SR // 4:  # ≥ 250 ms of audio
                self._last_partial_at = now
                with self._lock:
                    self._inflight = self._infer_pool.submit(
                        self._transcribe_blocking, tail)
        return new_text

    def is_settled(self) -> bool:
        return self._stable_count >= self._stability_ticks and bool(self._last_partial_text)

    def finalize(self) -> str:
        """Wait for any in-flight partial; if settled, return the last
        partial; otherwise run one final cold transcribe on the FULL buffer."""
        # Drain any in-flight partial.
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

        # If the last partials are stable, the latest IS the final.
        if self.is_settled():
            return self._last_partial_text

        # Otherwise, one final transcribe on the FULL buffer (not windowed).
        full = self._full_audio_np()
        if full.size == 0:
            return self._last_partial_text
        final_text = self._transcribe_blocking(full)
        return final_text or self._last_partial_text

    # ── Internals ──────────────────────────────────────────────────────────────

    def _poll_inflight(self):
        """Non-blocking check on the in-flight partial. Returns the new
        partial text on change, None on no change / no result yet."""
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
        """Synchronous transcribe. Reuses Qwen3STT.transcribe via super()."""
        return super().transcribe(audio_f32)


class Qwen3TRTSTT:
    """Qwen3-ASR-0.6B served by NVIDIA TensorRT-Edge-LLM on Jetson.

    Two integration paths — picks Python bindings if importable, else falls
    back to the CLI binary via JSON files.

    Env vars:
        NOVA_TRT_ENGINE_DIR     Default: $HOME/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines
        NOVA_TRT_EDGE_ROOT      Default: $HOME/TensorRT-Edge-LLM   (only for CLI fallback)
        NOVA_TRT_TOKENIZER_DIR  Default: <engine_dir>/llm

    Public surface stays `transcribe(audio_f32: np.ndarray) -> str`, so the
    rest of the gateway is untouched. Switch backends via NOVA_STT_BACKEND=qwen3_trt.
    """

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

        # Path A — Python bindings (preferred). Names below are placeholders;
        # adapt to the actual API your tensorrt_edgellm bindings expose.
        self._runner = None
        try:
            import tensorrt_edgellm as trtelm  # type: ignore
            # Whatever the real factory is — adapt this single line.
            # Many TRT-Edge-LLM example projects expose something like:
            #   trtelm.AudioLLMRunner(llm_dir, audio_dir, tokenizer_dir)
            # If yours uses different names, replace this call only.
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
                # Adapt to your bindings' actual signature. Common shapes:
                #   runner.transcribe(audio: np.ndarray, sample_rate: int) -> str
                #   runner.run(audio_f32, sr=16000)["text"]
                return self._runner.transcribe(audio_f32, sample_rate=SR).strip()
            except Exception as e:
                logger.warning(f"trt-py transcribe failed: {e}")
                return ""

        # CLI subprocess fallback — slower per call but zero coupling.
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

            # 1. Save audio as WAV (TRT preprocess script expects a file path).
            i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(i16.tobytes())

            # 2. Pre-process WAV → mel safetensors.
            subprocess.run([
                "python", "-m", self._preproc,
                "--input", str(wav_path),
                "--output", str(mel_path),
            ], cwd=self.trt_root, check=True, capture_output=True)

            # 3. Build the input.json the binary expects. The system prompt
            # mirrors qwen-asr's format so we don't get a stray "language X"
            # prefix in the transcript.
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

            # 4. Run llm_inference.
            subprocess.run([
                str(self._cli),
                "--engineDir", str(self.engine_dir / "llm"),
                "--multimodalEngineDir", str(self.engine_dir / "audio"),
                "--inputFile", str(in_json),
                "--outputFile", str(out_json),
            ], cwd=self.trt_root, check=True, capture_output=True)

            # 5. Extract the transcript. Qwen3-ASR via TRT-Edge-LLM writes
            # `responses[0].output_text`. Strip any leftover "language X"
            # prefix that sneaks through when the system prompt is empty.
            data = json.loads(out_json.read_text())
            try:
                text = data["responses"][0]["output_text"].strip()
            except (KeyError, IndexError):
                return ""
            text = re.sub(r"^language\s+\w+\b[\s,:-]*", "", text, flags=re.IGNORECASE)
            return text.strip()


class Kyutai1BSTT:
    """Kyutai STT 1B en/fr (semantic-VAD). Streaming under the hood, but exposed
    here as utterance-level: prime → push audio → pad delay → drain words."""

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
        # Match the production worker (stt_kyutai_worker.py:107-108): load the
        # LM, then wrap it in LMGen. The older `info.get_moshi_lm(lm_gen=True)`
        # path was removed in moshi >= ~0.3.
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

            # Resample 16k → kyutai sr
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
            # Pad delay so trailing tokens emerge
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
        if piece.startswith("▁"):  # sentencepiece word boundary
            if pending:
                words.append(pending)
            pending = piece[1:]
        else:
            pending += piece
        return pending, words


class TTS:
    """Kokoro ONNX. Defaults to CPU; set NOVA_KOKORO_GPU=1 for CUDA provider."""

    def __init__(self) -> None:
        import tempfile, urllib.request
        from kokoro_onnx import Kokoro  # type: ignore
        cache = Path(tempfile.gettempdir()) / "kokoro_tts"
        cache.mkdir(parents=True, exist_ok=True)
        m = cache / "kokoro-v1.0.onnx"
        v = cache / "voices-v1.0.bin"
        if not m.exists():
            base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
            urllib.request.urlretrieve(f"{base}/kokoro-v1.0.onnx", m)
            urllib.request.urlretrieve(f"{base}/voices-v1.0.bin", v)
        # Kokoro reads ORT providers from the env. To use CUDA: install
        # onnxruntime-gpu (instead of plain onnxruntime). Verify with:
        #   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
        # — must contain 'CUDAExecutionProvider'. Otherwise Kokoro falls back
        # to CPU and synth is the dominant TTFB on Jetson/laptops.
        if os.getenv("NOVA_KOKORO_GPU", "1") == "1":
            try:
                import onnxruntime as ort  # type: ignore
                provs = ort.get_available_providers()
                if "CUDAExecutionProvider" in provs:
                    logger.info(f"Kokoro: ORT CUDA EP available → GPU synth ({provs})")
                else:
                    logger.warning(
                        "Kokoro: ORT CUDA EP NOT available — falling back to CPU. "
                        "Install onnxruntime-gpu to enable GPU synth."
                    )
            except Exception as e:
                logger.warning(f"Kokoro: ORT introspection failed: {e}")
        self.k = Kokoro(str(m), str(v))
        self.voice = os.getenv("NOVA_KOKORO_VOICE", "af_heart")
        # Non-streaming warmup is sync-safe at construction time.
        try:
            self.k.create("Hi.", voice=self.voice, speed=1.0, lang="en-us")
        except Exception as e:
            logger.warning(f"Kokoro non-stream warmup failed: {e}")
        # Streaming warmup is async — done by lifespan() after construction
        # (TTS is built inside the FastAPI lifespan event loop, so we can't
        # asyncio.run() from here). See `await tts.warmup_stream()` below.

    async def warmup_stream(self) -> None:
        """Drain a tiny streaming synth so the first live turn doesn't pay
        graph-build cost on the streaming codepath. Awaited by lifespan()."""
        try:
            async for _ in self.k.create_stream(
                "Ready.", voice=self.voice, speed=1.0, lang="en-us"
            ):
                pass
        except Exception as e:
            logger.warning(f"Kokoro stream warmup failed: {e}")

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        # Kept for the non-streaming path / warmup.
        samples, sr = self.k.create(text, voice=self.voice, speed=1.0, lang="en-us")
        return samples.astype(np.float32, copy=False), sr

    async def synth_stream(self, text: str):
        """Yield (samples_chunk, sr) tuples as Kokoro produces them.

        Cuts first-audio latency dramatically vs `synth()` because we don't
        wait for the whole sentence to be vocoded before sending the first
        100 ms of audio to the browser.
        """
        async for samples, sr in self.k.create_stream(
            text, voice=self.voice, speed=1.0, lang="en-us"
        ):
            yield samples.astype(np.float32, copy=False), sr


class LLM:
    """OpenAI-compatible streaming completion. Works against:
       - OpenRouter (default; needs OPENROUTER_API_KEY)
       - vLLM serve (`--api-server`) — set NOVA_LLM_BACKEND=local
       - llama.cpp server (`server -m model.gguf`) — same setting
       - Ollama (with `--openai-api`) — same setting
    Local: NOVA_LLM_BASE_URL=http://localhost:8000/v1 (or wherever)."""

    def __init__(self) -> None:
        self.backend = os.getenv("NOVA_LLM_BACKEND", "openrouter").lower()
        if self.backend == "local":
            self.base_url = os.getenv("NOVA_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
            self.api_key = os.getenv("NOVA_LLM_API_KEY", "EMPTY")  # vLLM/llama.cpp ignore it
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

    def stream(self, messages: list[dict]) -> Iterator[str]:
        yield from self._stream_request(messages, tools=None)

    def _stream_request(self, messages: list[dict], tools: list[dict] | None) -> Iterator[str]:
        import requests
        body: dict = {"model": self.model, "messages": messages, "stream": True, "max_tokens": 200}
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

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Non-streaming completion. Used to detect tool_calls before deciding
        whether to stream the final answer or run a tool round-trip first.
        Returns the raw `choices[0].message` dict."""
        import requests
        body: dict = {"model": self.model, "messages": messages, "max_tokens": 200}
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


# ── Serper tool calling (web search + news) ───────────────────────────────────
# Mirrors `_execute_web_search` from pipeline_mp/llm_worker.py: one tool that
# auto-routes to Google's /news endpoint when the query has news-like keywords.
# Tools are only attached to the LLM call when `_needs_tools()` says the user
# prompt looks like an information request — keeps casual chat at single-call
# latency while letting "what's the news in Paris" trigger a Serper round-trip.

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
    r"right now|search|look up|google|find out|forecast)\b",
    re.IGNORECASE,
)
_CONVERSATIONAL = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|cool|nice|great|sure)\b",
    re.IGNORECASE,
)


def _needs_tools(prompt: str) -> bool:
    s = prompt.strip()
    if not s or _CONVERSATIONAL.match(s):
        return False
    if len(s.split()) <= 3 and "?" not in s:
        return False
    return bool(_TOOL_TRIGGERS.search(s))


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


def split_sentences(text: str) -> list[str]:
    parts, carry = [], ""
    for p in SENT_END.split(text.strip()):
        p = p.strip()
        if not p:
            continue
        carry = f"{carry} {p}".strip() if carry else p
        if len(carry) >= SENT_MIN_CHARS:
            parts.append(carry)
            carry = ""
    if carry:
        parts.append(carry)
    return parts


# ── Audio front-end (APM + VAD, inline) ───────────────────────────────────────

class AudioFE:
    """LiveKit APM (AEC3 + NS) + Silero VAD. Used for barge-in detection only;
    raw mic still goes to STT after VAD-gated buffering."""

    def __init__(self) -> None:
        from livekit.rtc import AudioFrame  # type: ignore
        from livekit.rtc.apm import AudioProcessingModule  # type: ignore
        from silero_vad import load_silero_vad  # type: ignore

        self.AudioFrame = AudioFrame
        self.apm = AudioProcessingModule(echo_cancellation=True, noise_suppression=True)
        self.vad = load_silero_vad(onnx=True)
        self.ref: deque[int] = deque()
        self.vad_buf = np.zeros(0, dtype=np.float32)

    def push_ref(self, samples_24k: np.ndarray) -> None:
        # Resample 24k → 16k via linear interp (good enough for AEC reference).
        n_in = samples_24k.size
        n_out = int(round(n_in * SR / TTS_SR))
        x_in = np.linspace(0, n_in - 1, n_in, dtype=np.float32)
        x_out = np.linspace(0, n_in - 1, n_out, dtype=np.float32)
        resampled = np.interp(x_out, x_in, samples_24k.astype(np.float32))
        self.ref.extend((resampled * 32767).clip(-32768, 32767).astype(np.int16).tolist())

    def vad_prob_clean(self, mic_i16: np.ndarray) -> float:
        """Process mic against current TTS reference; run Silero on cleaned signal."""
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
        self.tts: TTS = models["tts"]
        self.llm: LLM = models["llm"]
        self.smart_turn = models["smart_turn"]
        self.history: list[dict] = []
        self.audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.executor = executor
        self.fe = AudioFE()
        # Turn state
        self.in_turn = False
        self.silent_chunks = 0
        self.silence_limit = max(1, int(SILENCE_END_MS / (VAD_WIN / SR * 1000)))
        self.utt_buf: list[np.ndarray] = []
        self.silero = models["silero"]
        # Barge-in / echo suppression
        self.tts_playing = False
        self.tts_started_at = 0.0
        self.tts_drain_until = 0.0  # wall-clock when browser playback queue is empty
        self.barge_in = asyncio.Event()
        self.barge_streak = 0  # consecutive cleaned-VAD frames over threshold
        self.last_assistant: str = ""  # for echo-loop transcript guard
        self.turn_started_at: float = 0.0  # wall-clock of speech_started for metrics
        self.turn_metrics: dict[str, float] = {}  # per-turn latency phases

    async def emit(self, ev: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(ev))
        except Exception:
            pass

    async def emit_audio(self, samples: np.ndarray, sr: int) -> None:
        i16 = (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        if TTS_BINARY:
            # Header text frame announces sr; the next binary frame is PCM.
            await self.ws.send_text(json.dumps({"type": "audio_header", "sr": sr, "len": len(i16)}))
            await self.ws.send_bytes(i16)
        else:
            await self.ws.send_text(json.dumps({
                "type": "audio_out", "sr": sr,
                "pcm_b64": base64.b64encode(i16).decode("ascii"),
            }))

    # ── ingress: raw mic chunks → VAD turn detection ──────────────────────────
    async def ingest(self, audio_bytes: bytes) -> None:
        chunk = np.frombuffer(audio_bytes, dtype=np.int16)
        now = time.time()
        # If TTS is playing OR the browser is still draining buffered audio,
        # mic input is muted from STT — only the AEC barge-in path sees it.
        if self.tts_playing or now < self.tts_drain_until:
            elapsed_ms = (now - self.tts_started_at) * 1000 if self.tts_started_at else 0
            if elapsed_ms >= AEC_GUARD_MS:
                p = self.fe.vad_prob_clean(chunk)
                if p > BARGE_IN_THRESHOLD:
                    self.barge_streak += 1
                    if self.barge_streak >= BARGE_IN_FRAMES and not self.barge_in.is_set():
                        logger.info(
                            f"barge-in (cleaned p={p:.2f}, streak={self.barge_streak})"
                        )
                        self.barge_in.set()
                else:
                    self.barge_streak = 0
            return
        await self.audio_q.put(chunk.astype(np.float32) / 32768.0)

    async def dialogue_loop(self) -> None:
        """Main per-connection loop: VAD-gated turn capture → STT → LLM/TTS.

        When the active STT exposes a streaming API (feed_chunk), every VAD
        window is forwarded to it during speech so partial transcribes run
        during user speech instead of after silence is detected.
        """
        accum = np.zeros(0, dtype=np.float32)
        stt_streaming = hasattr(self.stt, "feed_chunk")
        loop = asyncio.get_running_loop()
        while True:
            chunk_f32 = await self.audio_q.get()
            accum = np.concatenate([accum, chunk_f32])
            while accum.size >= VAD_WIN:
                window = accum[:VAD_WIN]
                accum = accum[VAD_WIN:]
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
                        # Run feed_chunk in executor — model.transcribe() inside
                        # may briefly hold the GIL; don't stall the loop.
                        partial = await loop.run_in_executor(
                            self.executor, self.stt.feed_chunk, window)
                        if partial:
                            await self.emit({"type": "transcript_partial",
                                             "data": partial})
                elif self.in_turn:
                    self.silent_chunks += 1
                    self.utt_buf.append(window)
                    if stt_streaming:
                        partial = await loop.run_in_executor(
                            self.executor, self.stt.feed_chunk, window)
                        if partial:
                            await self.emit({"type": "transcript_partial",
                                             "data": partial})
                    if self.silent_chunks >= self.silence_limit:
                        await self._finalize_turn()

    async def _finalize_turn(self) -> None:
        utterance = np.concatenate(self.utt_buf) if self.utt_buf else np.zeros(0, dtype=np.float32)
        self.utt_buf.clear()
        was_in_turn = self.in_turn
        self.in_turn = False
        self.silent_chunks = 0
        self.silero.reset_states()

        # Per-turn latency metrics (Phase 4 instrumentation).
        t_vad_end = time.time()
        speech_dur = (t_vad_end - self.turn_started_at) if was_in_turn and self.turn_started_at else 0.0
        metrics: dict[str, float] = {"speech_dur_ms": int(speech_dur * 1000)}
        loop = asyncio.get_running_loop()
        stt_streaming = hasattr(self.stt, "feed_chunk")

        # Min-speech floor: with NOVA_SILENCE_END_MS=400 the silence_limit
        # trips earlier — guard short blips so brief noise bursts don't
        # short-circuit a real turn.
        min_samples = int(MIN_SPEECH_MS * SR / 1000)
        if utterance.size < min_samples:
            if stt_streaming:
                self.stt.begin_utterance()
            return

        # Smart-Turn confirm
        if self.smart_turn is not None:
            try:
                t0 = time.time()
                prob = await loop.run_in_executor(
                    self.executor, self.smart_turn.predict, utterance)
                metrics["smart_turn_ms"] = int((time.time() - t0) * 1000)
                if prob < SMART_TURN_THRESHOLD:
                    logger.info(f"smart-turn rejected (p={prob:.2f}) — ignoring utterance")
                    if stt_streaming:
                        self.stt.begin_utterance()
                    return
            except Exception as e:
                logger.warning(f"smart-turn failed: {e}")

        # Transcribe — streaming finalize() is near-instant if partials settled;
        # legacy path runs a cold call on the full utterance.
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

        # Stash metrics on the session so _stream_llm_to_tts can fill in
        # llm_ttft_ms + tts_ttfb_ms relative to the same vad_end origin.
        self.turn_metrics = metrics
        self.turn_metrics["_vad_end_at"] = t_vad_end

        # LLM stream → sentence chunks → TTS playback
        await self.emit({"type": "generation_start"})
        await self.emit({"type": "assistant_start"})
        full = await self._stream_llm_to_tts(text)
        if full:
            self.history.append({"role": "assistant", "content": full})
        await self.emit({"type": "generation_done"})

        # Phase 4 — emit per-turn latency snapshot (also logged to stderr).
        m = {k: v for k, v in self.turn_metrics.items() if not k.startswith("_")}
        m["total_ttfb_ms"] = m.get("smart_turn_ms", 0) + m.get("stt_ms", 0) \
            + m.get("llm_ttft_ms", 0) + m.get("tts_ttfb_ms", 0)
        logger.info("turn_metrics: " + " ".join(f"{k}={int(v)}" for k, v in m.items()))
        await self.emit({"type": "turn_metrics", "data": m})

        # Memory: extract durable facts off the hot path; consolidate periodically.
        if MEMORY_ENABLED and full:
            asyncio.create_task(self._memory_after_turn(text))

    async def _memory_after_turn(self, user_text: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self._update_memory, user_text)
        except Exception as e:
            logger.warning(f"memory update failed: {e}")
        # Count assistant turns for consolidation cadence.
        n_turns = sum(1 for m in self.history if m["role"] == "assistant")
        if n_turns and n_turns % MEMORY_CONSOLIDATE_EVERY == 0:
            try:
                await loop.run_in_executor(self.executor, self._consolidate_memory)
            except Exception as e:
                logger.warning(f"memory consolidation failed: {e}")

    def _llm_oneshot(self, prompt: str, max_tokens: int = 80) -> str:
        msgs = [{"role": "user", "content": prompt}]
        out = []
        for tok in self.llm.stream(msgs):  # reuse stream API; stop on natural EOS
            out.append(tok)
            if len("".join(out)) > max_tokens * 6:
                break
        return "".join(out).strip()

    def _update_memory(self, user_text: str) -> None:
        if not user_text:
            return
        existing = _read_text(MEMORY_MD)
        prompt = (
            f"Current memory:\n{existing or '(empty)'}\n\n"
            f"User just said: {user_text!r}\n\n"
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
                f.write("# Memory\n")
            f.write("\n".join(new_lines) + "\n")
        logger.info(f"memory +{len(new_lines)} lines")

    def _consolidate_memory(self) -> None:
        if not MEMORY_MD.exists():
            return
        prompt = (
            f"Here is a memory file about a user:\n\n{_read_text(MEMORY_MD)}\n\n"
            "Rewrite it: merge duplicates, remove transient/session-specific items "
            "(questions asked, topics discussed), keep only durable facts. Output "
            "the cleaned file starting with '# Memory' followed by lines starting "
            "with '- '. No explanation."
        )
        result = self._llm_oneshot(prompt, max_tokens=300)
        if result and result.startswith("# Memory"):
            MEMORY_MD.write_text(result + "\n")
            logger.info("memory consolidated")

    def _maybe_tool_roundtrip(self, msgs: list[dict], user_text: str) -> list[dict]:
        """If the user prompt looks like an info request and Serper is set,
        do a non-streaming pre-call with tools. If the model asks for a tool,
        execute it and append the tool message + tool_call to msgs. Caller
        then streams a fresh completion with the augmented message list.

        Returns the (possibly augmented) message list. Single round-trip max.
        """
        if not os.getenv("SERPER_API_KEY") or not _needs_tools(user_text):
            return msgs
        try:
            reply = self.llm.chat(msgs, tools=SERPER_TOOLS)
        except Exception as e:
            logger.warning(f"tool pre-call failed: {e}")
            return msgs
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return msgs
        # Append the assistant's tool_call message first (OpenAI protocol).
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
        """Three-stage pipeline running concurrently:
              llm_thread     → sent_q     (LLM → clause-split → sentences)
              synth_consumer → audio_q    (sentences → PCM samples)
              play_consumer  → WebSocket  (PCM → emit chunks)
        Synth runs ahead of playback so audio_out is never blocked on synth.
        """
        loop = asyncio.get_running_loop()
        persona = load_persona()
        sys_prompt = SYSTEM_PROMPT + ("\n\n" + persona if persona else "")
        msgs = [{"role": "system", "content": sys_prompt}, *self.history[-10:]]

        # Optional Serper tool round-trip BEFORE streaming. Runs in executor
        # so it doesn't block the WS loop. Only triggers on info-shaped prompts.
        msgs = await loop.run_in_executor(self.executor, self._maybe_tool_roundtrip, msgs, user_text)

        sent_q: asyncio.Queue = asyncio.Queue(maxsize=8)
        audio_q: asyncio.Queue = asyncio.Queue(maxsize=4)  # pre-synthed sentences
        full_text_parts: list[str] = []
        first_sentence_emitted = [False]
        t_llm_start = time.time()

        # ── LLM producer (blocking iterator → asyncio queue, runs in executor) ─
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
                    # Try clause-level break first (fast TTFB), then anything
                    # remaining will get flushed at the end.
                    while True:
                        m = CLAUSE_BREAK.search(buf)
                        if not m:
                            break
                        chunk = buf[: m.start() + 1].strip()
                        buf = buf[m.end():]
                        if len(chunk) >= SENT_MIN_CHARS or not first_sentence_emitted[0]:
                            asyncio.run_coroutine_threadsafe(sent_q.put(chunk), loop)
                            full_text_parts.append(chunk)
                            first_sentence_emitted[0] = True
                tail = buf.strip()
                if tail:
                    asyncio.run_coroutine_threadsafe(sent_q.put(tail), loop)
                    full_text_parts.append(tail)
            finally:
                asyncio.run_coroutine_threadsafe(sent_q.put(None), loop)

        # ── Synth consumer: pulls sentences, streams Kokoro chunks into audio_q ─
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
                    logger.warning(f"Kokoro stream failed, falling back: {e}")
                    samples, sr = await loop.run_in_executor(
                        self.executor, self.tts.synth, sentence)
                    await audio_q.put((sentence, samples, sr))

        # ── Play consumer: emits PCM in chunks, checks barge-in between ──────
        total_tts_samples = 0
        first_audio_logged = [False]

        async def play_consumer() -> None:
            nonlocal total_tts_samples
            # Smaller chunk → first network frame ships ~85 ms sooner at 24 kHz
            # vs the prior 4096 (~170 ms). Tunable via NOVA_TTS_PLAY_CHUNK.
            CHUNK = int(os.getenv("NOVA_TTS_PLAY_CHUNK", "2048"))
            while True:
                item = await audio_q.get()
                if item is None or self.barge_in.is_set():
                    return
                _sentence, samples, sr = item
                self.fe.push_ref(samples)
                total_tts_samples += len(samples)
                for i in range(0, len(samples), CHUNK):
                    if self.barge_in.is_set():
                        return
                    if not first_audio_logged[0]:
                        # Stamp BEFORE the emit so we measure true wire-time
                        # to first byte, not first-byte + emit overhead.
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
        self.tts_playing = True
        self.tts_started_at = time.time()
        try:
            await asyncio.gather(synth_consumer(), play_consumer())
        finally:
            await prod_task
            self.tts_playing = False
            total_dur_s = total_tts_samples / TTS_SR if total_tts_samples else 0.0
            elapsed_s = time.time() - self.tts_started_at if self.tts_started_at else 0.0
            remaining_s = max(0.0, total_dur_s - elapsed_s)
            self.tts_drain_until = time.time() + remaining_s + 0.7
            self.tts_started_at = 0.0
        full = " ".join(full_text_parts)
        self.last_assistant = full
        return full

    @staticmethod
    def _looks_like_echo(transcript: str, assistant: str) -> bool:
        """Crude similarity check: drop transcripts that mirror the last reply."""
        if not transcript or not assistant:
            return False
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
        t, a = norm(transcript), norm(assistant)
        if not t or not a:
            return False
        if t in a or a in t:
            return True
        # Token-level Jaccard similarity
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
            smart_turn.predict(np.zeros(8000, dtype=np.float32))  # warmup
    except Exception as e:
        logger.warning(f"smart-turn disabled: {e}")
    tts = TTS()
    app.state.models = {
        "stt": make_stt(),
        "tts": tts,
        "llm": LLM(),
        "silero": load_silero_vad(onnx=True),
        "smart_turn": smart_turn,
    }
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    # Streaming TTS warmup — must run inside the lifespan loop, not in
    # TTS.__init__ (which would hit "asyncio.run() inside running loop").
    await tts.warmup_stream()
    logger.info("Ready on :8001")
    yield
    app.state.executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

# Frontend assets live next to this module: nova/backend/static/{index.html, app.css, app.js}
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/persona/{name}")
def persona(name: str) -> dict:
    """Return USER.md / MEMORY.md / Conversation.md content for the sidebar."""
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
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError("Cannot call 'receive' once a disconnect message has been received")
        pass
    finally:
        loop_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
