"""Nova — simple single-process voice gateway (demo skeleton).

Mirrors the architecture of `voice-loop/voice_loop_mac.py` but served over
WebSocket so the existing browser frontend talks to it. One process, one
asyncio loop, blocking model calls dispatched to a thread executor.

Pipeline per turn:
    browser mic (16k mono int16 bytes via /ws)
        ─▶ asyncio audio_q
            ─▶ Silero VAD (turn detection)
            ─▶ on speech_end: Smart-Turn confirm
            ─▶ STT.transcribe(np.float32)
            ─▶ LLM.stream(messages)  ─┐ tokens → split into sentences
            ─▶ for each sentence: TTS.synth → push pcm to /ws + APM ref buffer
                   during playback: APM-cleaned mic + Silero → barge-in cancels

Models default to Moonshine (CPU) + Kokoro (CPU) + OpenRouter (cloud) so the
whole stack loads in <30s with no GPU. Swap the loaders below to use the
production Kyutai / Pocket-TTS / local LLM engines once GPU is available.

Run:
    uv run python -m nova.backend.gateway_simple

WS protocol (browser side):
    inbound  : raw int16 PCM mono @ 16 kHz (binary frames) + JSON ctrl
    outbound : JSON events {speech_started, transcript_partial, transcript,
                            llm_token, assistant_start, generation_start,
                            generation_done, audio_out{b64 pcm @ 24k}, error}
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
from fastapi.responses import HTMLResponse

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
SILENCE_END_MS = int(os.getenv("NOVA_SILENCE_END_MS", "700"))
SMART_TURN_THRESHOLD = float(os.getenv("NOVA_SMART_TURN_THRESHOLD", "0.5"))
AEC_GUARD_MS = int(os.getenv("NOVA_AEC_GUARD_MS", "500"))
SYSTEM_PROMPT = os.getenv(
    "NOVA_SYSTEM_PROMPT",
    "You are Nova, a concise in-car voice assistant. "
    "ALWAYS reply in English regardless of input. "
    "Reply in 1–2 short sentences.",
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
    """STT backend selector. NOVA_STT_BACKEND ∈ {qwen3_0_6b, kyutai_1b, moonshine}."""
    backend = os.getenv("NOVA_STT_BACKEND", "qwen3_0_6b").lower()
    if backend == "kyutai_1b":
        return Kyutai1BSTT()
    if backend == "moonshine":
        return MoonshineSTT()
    return Qwen3STT()


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


class Kyutai1BSTT:
    """Kyutai STT 1B en/fr (semantic-VAD). Streaming under the hood, but exposed
    here as utterance-level: prime → push audio → pad delay → drain words."""

    def __init__(self) -> None:
        import math
        from moshi.models.loaders import CheckpointInfo  # type: ignore
        repo = os.getenv("NOVA_KYUTAI_HF_REPO", "kyutai/stt-1b-en_fr")
        self.device = os.getenv("NOVA_KYUTAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        info = CheckpointInfo.from_hf_repo(repo)
        self.silence_prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        self.delay_s = info.stt_config.get("audio_delay_seconds", 0.5)
        self.mimi = info.get_mimi(device=self.device)
        self.lm_gen = info.get_moshi_lm(device=self.device, lm_gen=True)
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
        # Try GPU when available — cuts per-sentence synth time ~3-5x on
        # mid-range GPUs vs CPU. Falls back gracefully if CUDA EP unavailable.
        sess = None
        if os.getenv("NOVA_KOKORO_GPU", "1") == "1":
            try:
                import onnxruntime as ort  # type: ignore
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                sess = ort.InferenceSession(str(m), providers=providers)
                logger.info(f"Kokoro ONNX providers: {sess.get_providers()}")
            except Exception as e:
                logger.warning(f"Kokoro CUDA provider unavailable: {e}")
        self.k = Kokoro(str(m), str(v), session=sess) if sess is not None else Kokoro(str(m), str(v))
        self.voice = os.getenv("NOVA_KOKORO_VOICE", "af_heart")
        # Warmup so the first sentence doesn't pay graph-build cost.
        try:
            self.k.create("Hi.", voice=self.voice, speed=1.0, lang="en-us")
        except Exception:
            pass

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self.k.create(text, voice=self.voice, speed=1.0, lang="en-us")
        return samples.astype(np.float32, copy=False), sr


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
            self.api_key = os.environ["OPENROUTER_API_KEY"]
            self.model = os.getenv("NOVA_LLM_MODEL", "openai/gpt-4o-mini")
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
        self.last_assistant: str = ""  # for echo-loop transcript guard

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
                if p > 0.8:
                    if not self.barge_in.is_set():
                        logger.info(f"barge-in (cleaned p={p:.2f})")
                        self.barge_in.set()
            return
        await self.audio_q.put(chunk.astype(np.float32) / 32768.0)

    async def dialogue_loop(self) -> None:
        """Main per-connection loop: VAD-gated turn capture → STT → LLM/TTS."""
        accum = np.zeros(0, dtype=np.float32)
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
                        await self.emit({"type": "speech_started"})
                    self.silent_chunks = 0
                    self.utt_buf.append(window)
                elif self.in_turn:
                    self.silent_chunks += 1
                    self.utt_buf.append(window)
                    if self.silent_chunks >= self.silence_limit:
                        await self._finalize_turn()

    async def _finalize_turn(self) -> None:
        utterance = np.concatenate(self.utt_buf) if self.utt_buf else np.zeros(0, dtype=np.float32)
        self.utt_buf.clear()
        self.in_turn = False
        self.silent_chunks = 0
        self.silero.reset_states()
        if utterance.size < SR // 2:
            return  # too short

        # Smart-Turn confirm
        if self.smart_turn is not None:
            try:
                prob = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.smart_turn.predict, utterance)
                if prob < SMART_TURN_THRESHOLD:
                    logger.info(f"smart-turn rejected (p={prob:.2f}) — ignoring utterance")
                    return
            except Exception as e:
                logger.warning(f"smart-turn failed: {e}")

        # Transcribe
        text = await asyncio.get_running_loop().run_in_executor(
            self.executor, self.stt.transcribe, utterance)
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

        # LLM stream → sentence chunks → TTS playback
        await self.emit({"type": "generation_start"})
        await self.emit({"type": "assistant_start"})
        full = await self._stream_llm_to_tts(text)
        if full:
            self.history.append({"role": "assistant", "content": full})
        await self.emit({"type": "generation_done"})

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
        def llm_producer() -> None:
            buf = ""
            try:
                for delta in self.llm.stream(msgs):
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

        # ── Synth consumer: pulls sentences, synthesizes, queues PCM ─────────
        async def synth_consumer() -> None:
            while True:
                sentence = await sent_q.get()
                if sentence is None or self.barge_in.is_set():
                    await audio_q.put(None)
                    return
                await self.emit({"type": "llm_token", "data": sentence})
                samples, sr = await loop.run_in_executor(self.executor, self.tts.synth, sentence)
                if self.barge_in.is_set():
                    await audio_q.put(None)
                    return
                await audio_q.put((sentence, samples, sr))

        # ── Play consumer: emits PCM in chunks, checks barge-in between ──────
        total_tts_samples = 0
        first_audio_logged = [False]

        async def play_consumer() -> None:
            nonlocal total_tts_samples
            CHUNK = 4096
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
                    await self.emit_audio(samples[i:i + CHUNK], sr)
                    if not first_audio_logged[0]:
                        first_audio_logged[0] = True
                        ttfb = (time.time() - t_llm_start) * 1000
                        logger.info(f"TTS first-audio TTFB: {ttfb:.0f} ms")
                    await asyncio.sleep(0)

        prod_task = loop.run_in_executor(self.executor, llm_producer)
        self.barge_in.clear()
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
    app.state.models = {
        "stt": make_stt(),
        "tts": TTS(),
        "llm": LLM(),
        "silero": load_silero_vad(onnx=True),
        "smart_turn": smart_turn,
    }
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Ready on :8001")
    yield
    app.state.executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _DEMO_HTML


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


# ── Minimal browser demo (mic capture + audio playback) ───────────────────────

_DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Nova</title>
  <style>
    *,*::before,*::after { box-sizing: border-box }
    :root {
      --bg-0: #07090d; --bg-1: #0d1218; --line: #1c2230;
      --fg: #e6ecf3; --muted: #7e8a9c; --accent: #4cd2c8; --accent-2: #4f7cff;
      --user: #ffd166; --error: #ff6b6b;
    }
    html,body { height: 100% }
    body {
      margin: 0; background: radial-gradient(1200px 600px at 70% -10%, #14223a 0%, var(--bg-0) 60%);
      color: var(--fg); font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      display: grid; grid-template-rows: auto 1fr auto; min-height: 100vh;
    }
    header {
      padding: 18px 28px; display: flex; align-items: center; gap: 14px;
      border-bottom: 1px solid var(--line);
    }
    .brand { font-weight: 600; letter-spacing: .12em; text-transform: uppercase; font-size: 13px; color: var(--muted) }
    .brand b { color: var(--fg); letter-spacing: .04em }
    .pill {
      margin-left: auto; padding: 4px 10px; border: 1px solid var(--line); border-radius: 999px;
      font-size: 12px; color: var(--muted); display: inline-flex; align-items: center; gap: 8px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); transition: background .2s }
    .dot.idle { background: #4a5568 }
    .dot.listen { background: var(--accent); box-shadow: 0 0 12px var(--accent) }
    .dot.think { background: #a78bfa; animation: pulse 1.2s ease-in-out infinite }
    .dot.speak { background: var(--accent-2); box-shadow: 0 0 12px var(--accent-2) }
    @keyframes pulse { 0%,100% { opacity: .35 } 50% { opacity: 1 } }

    main {
      display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 24px;
      padding: 28px; max-width: 1200px; margin: 0 auto; width: 100%;
    }
    @media (max-width: 900px) { main { grid-template-columns: 1fr } }

    .turns { display: flex; flex-direction: column; gap: 12px; min-height: 60vh }
    .turn {
      padding: 14px 16px; border-radius: 14px; border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
      max-width: 720px;
    }
    .turn.user { align-self: flex-end; border-color: rgba(255, 209, 102, 0.25) }
    .turn .who { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px }
    .turn.user .who { color: var(--user) }
    .turn.assistant .who { color: var(--accent) }

    aside {
      display: flex; flex-direction: column; gap: 14px;
      border: 1px solid var(--line); border-radius: 16px; padding: 18px; background: var(--bg-1);
      align-self: start; position: sticky; top: 20px;
    }
    aside h3 { margin: 0 0 4px; font-size: 12px; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; font-weight: 600 }
    .row { display: flex; gap: 8px; align-items: center; justify-content: space-between }
    .row .v { color: var(--fg); font-variant-numeric: tabular-nums }
    .row .k { color: var(--muted); font-size: 13px }
    .meter { height: 8px; background: #131a25; border-radius: 999px; overflow: hidden }
    .meter > span { display: block; height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); transition: width 60ms linear }

    footer {
      padding: 18px 28px; border-top: 1px solid var(--line);
      display: flex; align-items: center; gap: 16px; justify-content: center;
    }
    .mic-wrap { position: relative; display: inline-flex; align-items: center; justify-content: center }
    .mic-wrap::before, .mic-wrap::after {
      content: ""; position: absolute; inset: 0; border-radius: 50%;
      border: 2px solid var(--accent-2); opacity: 0; pointer-events: none;
    }
    body[data-state="listening"] .mic-wrap::before { animation: ring 1.6s ease-out infinite; border-color: var(--accent) }
    body[data-state="listening"] .mic-wrap::after  { animation: ring 1.6s ease-out infinite .8s; border-color: var(--accent) }
    body[data-state="speaking"]  .mic-wrap::before { animation: ring 1.0s ease-out infinite; border-color: var(--accent-2) }
    body[data-state="speaking"]  .mic-wrap::after  { animation: ring 1.0s ease-out infinite .5s; border-color: var(--accent-2) }
    body[data-state="thinking"]  .mic-wrap::before { animation: ring 0.6s ease-out infinite; border-color: #a78bfa }
    @keyframes ring {
      0% { transform: scale(1); opacity: .8 }
      100% { transform: scale(1.9); opacity: 0 }
    }
    button.mic {
      appearance: none; border: 0; cursor: pointer; position: relative; z-index: 1;
      width: 76px; height: 76px; border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #4f7cff 0%, #2a3a8a 60%, #0a1130 100%);
      color: white; font-size: 26px;
      box-shadow: 0 8px 26px rgba(79, 124, 255, 0.35), inset 0 1px 0 rgba(255,255,255,0.2);
      transition: transform .1s ease;
    }
    button.mic:hover { transform: translateY(-1px) }
    button.mic.on { background: radial-gradient(circle at 30% 30%, #ff6b6b 0%, #8a2a2a 60%, #300a0a 100%); box-shadow: 0 8px 26px rgba(255, 107, 107, 0.35) }
    .tabs { display: flex; gap: 4px; margin-top: 8px }
    .tab { flex: 1; cursor: pointer; padding: 6px 8px; text-align: center; border-radius: 8px; border: 1px solid var(--line); font-size: 11px; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; user-select: none }
    .tab.active { background: var(--bg-1); color: var(--fg); border-color: var(--accent-2) }
    .doc { white-space: pre-wrap; font: 12px/1.55 ui-monospace, SFMono-Regular, monospace; background: #0a0e15; border: 1px solid var(--line); border-radius: 10px; padding: 10px; max-height: 280px; overflow: auto; color: #cfd8e3 }
    .hint { color: var(--muted); font-size: 13px }
    .err { color: var(--error); font-size: 12px; text-align: center }
  </style>
</head>
<body>
  <header>
    <div class="brand"><b>NOVA</b> &nbsp;·&nbsp; in-car voice agent</div>
    <div class="pill"><span id="dot" class="dot idle"></span><span id="status">idle</span></div>
  </header>

  <main>
    <div id="turns" class="turns" aria-live="polite"></div>
    <aside>
      <h3>Session</h3>
      <div class="row"><div class="k">Mic level</div><div class="v" id="lvl">—</div></div>
      <div class="meter"><span id="lvlBar"></span></div>
      <div class="row"><div class="k">Last latency</div><div class="v" id="lat">—</div></div>
      <div class="row"><div class="k">Turns</div><div class="v" id="nTurns">0</div></div>
      <div class="err" id="err"></div>

      <h3 style="margin-top:8px">Persona</h3>
      <div class="tabs">
        <div class="tab active" data-doc="user">USER</div>
        <div class="tab" data-doc="memory">MEMORY</div>
        <div class="tab" data-doc="conversation">CHAT</div>
      </div>
      <pre class="doc" id="doc">(loading…)</pre>
    </aside>
  </main>

  <footer>
    <span class="hint">Click the mic and speak. Nova will reply in voice.</span>
    <span class="mic-wrap"><button id="micBtn" class="mic" title="Start / stop mic">🎙</button></span>
    <span class="hint">Press space to interrupt.</span>
  </footer>

<script>
(() => {
  const $ = id => document.getElementById(id);
  const setStatus = (k, t) => {
    document.body.dataset.state = k === 'listen' ? 'listening' : k === 'think' ? 'thinking' : k === 'speak' ? 'speaking' : 'idle';
    $('dot').className = 'dot ' + k; $('status').textContent = t;
  };
  setStatus('idle', 'idle');
  const turns = $('turns');
  const newTurn = (who, text) => {
    const el = document.createElement('div');
    el.className = 'turn ' + who;
    el.innerHTML = `<div class="who">${who}</div><div class="t"></div>`;
    el.querySelector('.t').textContent = text;
    turns.appendChild(el); el.scrollIntoView({ behavior: 'smooth', block: 'end' });
    $('nTurns').textContent = turns.children.length;
    return el.querySelector('.t');
  };
  const showErr = m => { $('err').textContent = m; setTimeout(() => $('err').textContent = '', 4000); };

  let ws, micCtx, playCtx, playT = 0, on = false, srcNode, workletNode, t0 = 0;
  let assistantSpan = null;
  let pendingAudioSr = 0;
  function playPCM(i16, sr) {
    if (!playCtx) return;
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = playCtx.createBuffer(1, f32.length, sr);
    buf.copyToChannel(f32, 0);
    const s = playCtx.createBufferSource(); s.buffer = buf; s.connect(playCtx.destination);
    const t = Math.max(playT, playCtx.currentTime); s.start(t); playT = t + buf.duration;
  }

  async function start() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch (e) { showErr('mic permission denied'); return; }

    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => setStatus('listen', 'listening');
    ws.onclose = () => setStatus('idle', 'disconnected');
    ws.onerror = () => showErr('connection error');
    ws.onmessage = onMessage;
    // Track expected binary audio frame from preceding header
    pendingAudioSr = 0;

    micCtx = new AudioContext({ sampleRate: 16000 });
    await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([`
      class P extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0]; if (!ch) return true;
          let peak = 0;
          const i16 = new Int16Array(ch.length);
          for (let i = 0; i < ch.length; i++) {
            const v = Math.max(-1, Math.min(1, ch[i]));
            i16[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
            const a = v < 0 ? -v : v; if (a > peak) peak = a;
          }
          this.port.postMessage({ buf: i16.buffer, peak }, [i16.buffer]);
          return true;
        }
      }
      registerProcessor('p', P);`], { type: 'application/javascript' })));
    srcNode = micCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(micCtx, 'p');
    workletNode.port.onmessage = e => {
      if (ws && ws.readyState === 1) ws.send(e.data.buf);
      const pct = Math.min(100, Math.round(e.data.peak * 200));
      $('lvlBar').style.width = pct + '%';
      $('lvl').textContent = `${pct}%`;
    };
    srcNode.connect(workletNode);

    playCtx = new AudioContext({ sampleRate: 24000 });
    playT = playCtx.currentTime;
    on = true; $('micBtn').classList.add('on');
  }

  function stop() {
    on = false; $('micBtn').classList.remove('on');
    if (workletNode) workletNode.disconnect();
    if (srcNode) srcNode.disconnect();
    if (micCtx) micCtx.close().catch(()=>{});
    if (playCtx) playCtx.close().catch(()=>{});
    if (ws && ws.readyState === 1) ws.close();
    setStatus('idle', 'idle');
  }

  function onMessage(e) {
    // Binary frame arrives right after an audio_header text frame.
    if (e.data instanceof ArrayBuffer) {
      const i16 = new Int16Array(e.data);
      playPCM(i16, pendingAudioSr || 24000);
      pendingAudioSr = 0;
      return;
    }
    let msg; try { msg = JSON.parse(e.data) } catch { return; }
    switch (msg.type) {
      case 'speech_started':
        setStatus('listen', 'listening'); break;
      case 'transcript':
        newTurn('user', msg.data || '');
        t0 = performance.now();
        setStatus('think', 'thinking…');
        break;
      case 'assistant_start':
        assistantSpan = newTurn('assistant', '');
        break;
      case 'llm_token':
        if (assistantSpan) assistantSpan.textContent += (assistantSpan.textContent ? ' ' : '') + (msg.data || '');
        if (t0) { $('lat').textContent = `${Math.round(performance.now() - t0)} ms`; t0 = 0; }
        setStatus('speak', 'speaking…');
        break;
      case 'generation_done':
        setStatus('listen', 'listening');
        assistantSpan = null;
        // Refresh sidebar after the turn — memory may have grown.
        loadDoc(currentDoc);
        break;
      case 'audio_header': pendingAudioSr = msg.sr || 24000; break;
      case 'audio_out': {
        // Legacy base64 path (NOVA_TTS_BINARY=0).
        const bin = atob(msg.pcm_b64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        playPCM(new Int16Array(arr.buffer), msg.sr);
        break;
      }
      case 'error': showErr(msg.data || 'error'); break;
    }
  }

  // Sidebar tabs
  let currentDoc = 'user';
  async function loadDoc(name) {
    try {
      const r = await fetch('/persona/' + name); const j = await r.json();
      $('doc').textContent = j.content || '(empty)';
    } catch { $('doc').textContent = '(failed to load)' }
  }
  document.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      currentDoc = t.dataset.doc; loadDoc(currentDoc);
    };
  });
  loadDoc('user');

  $('micBtn').onclick = () => on ? stop() : start();
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && on && playCtx) {
      // local interrupt: drop scheduled playback so user feels barge-in immediately
      try { playCtx.close(); } catch {}
      playCtx = new AudioContext({ sampleRate: 24000 });
      playT = playCtx.currentTime;
      setStatus('listen', 'listening');
      e.preventDefault();
    }
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
