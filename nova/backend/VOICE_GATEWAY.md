# Nova — single-process voice gateway

`nova_loop.py` is the active one-process gateway for Nova. One asyncio loop
per WebSocket connection, blocking model calls dispatched to a thread executor,
browser frontend served from `static/` — no FastRTC, no multiprocessing, no FSM.

The original `voice_gateway.py` (pre-pVAD, Kokoro TTS) is archived at
`nova/archive/voice_gateway.py`.

```
browser mic ──/ws bytes──▶ Silero VAD (+ RMS energy gate, noise-floor tracker)
                                  │
                                  │  (during speech: Qwen3StreamingSTT
                                  │   rolling-buffer partials)
                                  ▼  (silence_end: Smart-Turn confirm)
                                stt.finalize() ─▶ LLM stream ─▶ clause split
                                                                    │
                                                                    ▼
                                                    Pocket-TTS streaming
                                                                    │
                                                                    ▼
                                  binary PCM frames + APM ref ▶ /ws audio_out

                     during TTS playback: APM mic → FireRed pVAD → barge-in
                     (target-speaker-gated — ignores passengers, noise, echo)
```

Per-turn pipeline is fully concurrent: STT runs *during* user speech (rolling
buffer), synth for sentence N+1 starts the moment N enters the audio queue,
playback never blocks on synth.

## What's new vs voice_gateway.py

| Feature | voice_gateway.py (archived) | nova_loop.py (active) |
|---|---|---|
| Barge-in | AEC+Silero only | FireRedChat pVAD (speaker-gated) |
| TTS | Kokoro ONNX | Pocket-TTS |
| VAD | Silero 0.5 threshold | 0.65 + RMS energy gate + noise-floor tracker |
| AEC guard | 500ms | 100ms |
| System prompt | Original | Hardened (5 CRITICAL RULES, numbered) |
| Tool triggers | Loose regex | Requires .?! ending + 4+ words |
| Voice enrollment | None | `/enroll/voice-sample` + UI in sidebar |
| pVAD status | None | `/pVAD/status` + 🔐 badge in header |
| Barge-in fallback | N/A | AEC+Silero when pVAD not loaded |

## Layout

```
nova/backend/
  nova_loop.py              # the gateway server (run this)
  pipeline_mp/
    pvad_firered.py         # FireRedChat pVAD ONNX wrapper
    smart_turn.py           # Smart-Turn v3 endpoint detector
  static/
    index.html              # browser UI
    app.css                 # styles + mic animation
    app.js                  # mic capture, WS protocol, audio playback, voice enrollment
  persona/
    USER.md.example         # template (committed)
    MEMORY.md.example       # template (committed)
    USER.md                 # your persona (gitignored)
    MEMORY.md               # auto-grown durable facts (gitignored)
  models/
    FireRedChat-pvad/       # pvad.onnx + ECAPA-TDNN speaker model
  VOICE_GATEWAY.md          # this doc
nova/archive/
  voice_gateway.py          # pre-pVAD version (archived)
```

## Quick start

### 1. Install deps

### Platform-specific ONNX Runtime GPU

This project uses `onnxruntime-gpu` for TTS and other ONNX models.
Before running `uv sync`, uncomment the correct line in `pyproject.toml`:

- **NVIDIA Jetson (Thor / ARM64):**
  Uncomment the line with `@ https://pypi.jetson-ai-lab.io/...` and comment the plain `"onnxruntime-gpu"` line.
- **AMD64/Intel (Arch Linux, etc.):**
  Uncomment the plain `"onnxruntime-gpu"` line and comment the Jetson URL line.

```bash
uv sync
```

Optional: `apt-get install libssl-dev` / `pacman -S openssl` if you want
HTTPS on the gateway port.

### 2. Set up persona files (first run only)

```bash
cd nova/backend/persona
cp USER.md.example USER.md
cp MEMORY.md.example MEMORY.md
$EDITOR USER.md          # set name, role, language, vehicle, prefs
```

`USER.md` is your manual persona. `MEMORY.md` is auto-grown by Nova: after
every assistant turn it extracts durable facts and appends `- ...` lines;
every 5 turns it consolidates the file to dedupe. You can edit it by hand
at any time.

Disable the whole thing with `NOVA_MEMORY=0` if you want a stateless demo.

### 3. Start a local LLM (optional but recommended)

The gateway speaks OpenAI-compatible chat completions. Either point it at
OpenRouter (default, requires `OPENROUTER_API_KEY`) or run any local server
that exposes `/v1/chat/completions` — vLLM, Ollama, llama.cpp.

**Auto-launched llama.cpp (recommended):** `nova_loop.py` will spawn
`llama-server` for you and shut it down on Ctrl-C. Presets live in
`nova/backend/configs/llama/`.

```bash
# 1. Copy a preset template and point `cwd:` at your local llama.cpp checkout.
cd nova/backend/configs/llama
cp gemma-4-e4b.yaml.example gemma-4-e4b.yaml      # or qwen3.6-35b-mtp.yaml.example
$EDITOR gemma-4-e4b.yaml                          # set cwd: /your/path/to/llama.cpp
                                                  # (the binary defaults to ./build/bin/llama-server)

# 2. Pick the active preset in .env (see #------ Voice Gateway/Nova Loop ------ section).
#    NOVA_LLAMA_PRESET=gemma-4-e4b
#    NOVA_LLAMA_PRESET=qwen3.6-35b-mtp
#    NOVA_DISABLE_LLAMA=1   # skip auto-launch if llama-server is already running
```

Each preset is a YAML file with `cwd`, `binary`, and an `args:` list — add a
new file to the directory to register a new model. The `*.yaml` files are
gitignored (host-specific paths); only the `*.yaml.example` templates are
committed.

If you'd rather run llama.cpp by hand, set `NOVA_DISABLE_LLAMA=1` and start
it yourself:

```bash
./build/bin/llama-server \
  -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
  --alias "unsloth/gemma-4-E4B-it" \
  --host 0.0.0.0 --port 8080 \
  --threads -1 --n-gpu-layers 999 \
  --ctx-size 128000 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --jinja \
  --reasoning off
```

### 4. Enroll your voice (required for pVAD barge-in)

Open `http://localhost:8001`, click the mic to connect. Scroll to "Voice
Enrollment" in the sidebar. Record all 5 phrases. Restart the gateway.

```
[crypto] Loaded existing encryption key.
FireRedPVAD INFO FireRedPVAD loaded (model=pvad.onnx, spk_emb norm=1.0000)
pVAD (FireRedChat): loaded — target-speaker barge-in active
```

The 🔐 badge appears in the header when pVAD is loaded.

### 5. Run the gateway

`.env` (see `.env.example` for the full list):

```bash
NOVA_LLM_BACKEND=local
NOVA_LLM_BASE_URL=http://localhost:8080/v1
NOVA_LLM_MODEL=google/gemma-2-2b-it
SERPER_API_KEY=...                      # optional, enables web_search tool
```

```bash
uv run nova/backend/nova_loop.py
```

Open `http://localhost:8001`, click the mic, talk. The browser handles mic
capture + 16 kHz PCM upload via `AudioWorkletNode` and plays back received
24 kHz PCM frames through `AudioBufferSource`.

## Streaming STT (Nemotron native streaming)

The default `NemotronStreamingSTT` (nvidia/nemotron-speech-streaming-en-0.6b)
uses NeMo's native streaming ASR with per-chunk partials and cache-aware
encoding (`NOVA_NEMOTRON_ATT_CONTEXT` controls the (left, right) context
size). The model loads from Hugging Face on first run (~3 GB, cached in
`HF_HOME`).

Also available: `Qwen3StreamingSTT` — runs partial transcribes on a 3 s
rolling window every 400 ms; identical consecutive partials settle early,
otherwise a final cold transcribe runs on silence-end. Switch via
`NOVA_STT_BACKEND=qwen3_streaming`.

Switch to single-call STT for A/B with `NOVA_STT_BACKEND=qwen3_0_6b`.

## pVAD — speaker-gated barge-in

When enrolled, the FireRedChat pVAD (FireRedTeam/FireRedChat-pvad) runs
alongside the AEC path during TTS playback. It uses ECAPA-TDNN speaker
embeddings to distinguish the enrolled driver from passengers, noise,
and echo. Streaming at 10ms granularity, it feeds exactly 160-sample
(10ms @ 16kHz) frames from the APM mic path.

Without enrollment, barge-in falls back to AEC-cleaned Silero VAD
(no speaker gate — any voice can interrupt).

## Tool calling (Serper)

When `SERPER_API_KEY` is set, the gateway exposes a single `web_search` tool
to the LLM. Before each user-message → assistant streaming pass, the gateway
runs a cheap regex (`_needs_tools`) over the transcript:

- Casual chat ("hi", "thanks", "what's 2+2?") → tools skipped, single
  streaming call.
- Info-shaped ("what's the news in Paris?", "weather tomorrow?", "who won
  the match?") → one non-streaming pre-call with `tools=[web_search]`. If
  the model returns a `tool_call`, the gateway hits Google via Serper,
  appends the tool result to the message list, then streams the final
  answer.

**To prevent hallucinated tool calls on sentence fragments**, the trigger
now requires: (1) at least 4 words, and (2) sentence-ending punctuation
(`.`, `?`, or `!`). Mid-sentence fragments like "designing something that…"
no longer fire a web search.

The Serper endpoint auto-routes to `/news` when the query contains news-y
keywords (`news`, `headline`, `latest`, `breaking`, `today`, ...) and
`/search` otherwise. Top 3 results returned as JSON.

## Latency instrumentation

Every turn emits a `turn_metrics` WS event + single-line stderr log:

```text
turn_metrics: speech_dur_ms=1527 smart_turn_ms=189 stt_ms=18
              llm_ttft_ms=160 tts_ttfb_ms=240 total_ttfb_ms=607
```

`stt_ms` should be small (~10–50 ms) when the streaming partial settled
during speech; large means it ran a final cold call. `total_ttfb_ms` is
end-of-speech → first audio frame on the wire.

## Environment knobs (gateway-specific)

| Var | Default | Notes |
|---|---|---|
| `NOVA_STT_BACKEND` | `nemotron_streaming` | `nemotron_streaming` (default) / `qwen3_streaming` / `qwen3_0_6b` / `kyutai_1b` / `moonshine` |
| `NOVA_QWEN3_LANGUAGE` | `English` | Pinned so noise stays in-language |
| `NOVA_QWEN3_PARTIAL_INTERVAL_MS` | `400` | Streaming partial cadence |
| `NOVA_QWEN3_PARTIAL_WINDOW_S` | `3.0` | Rolling-window length |
| `NOVA_QWEN3_STABILITY_TICKS` | `2` | Consecutive identical partials → settled |
| `NOVA_QWEN3_MAX_UTTERANCE_S` | `12.0` | Hard cap |
| `NOVA_LLM_BACKEND` | `openrouter` | `local` for vLLM/llama.cpp/Ollama |
| `NOVA_LLM_BASE_URL` | OpenRouter | e.g. `http://localhost:8080/v1` |
| `NOVA_LLM_MODEL` | `openai/gpt-4o-mini` | Match local server's model |
| `NOVA_SYSTEM_PROMPT` | concise English | Override per deployment |
| `NOVA_POCKET_VOICE` | `alba` | Pocket-TTS voice name |
| `NOVA_VAD_THRESHOLD` | `0.65` | Silero per-frame voice prob (car-noise tuned) |
| `NOVA_VAD_MIN_RMS_DB` | `-45` | Energy floor for RMS gate |
| `NOVA_SILENCE_END_MS` | `400` | End-of-utterance silence |
| `NOVA_MIN_SPEECH_MS` | `500` | Drop turns shorter than this |
| `NOVA_SMART_TURN_THRESHOLD` | `0.5` | Drops if model unsure speaker is done |
| `NOVA_AEC_GUARD_MS` | `100` | Skip first N ms of TTS for barge-in (was 500) |
| `NOVA_BARGE_IN_THRESHOLD` | `0.35` | pVAD prob for interrupt |
| `NOVA_BARGE_IN_FRAMES` | `2` | Consecutive frames to confirm barge-in |
| `NOVA_TTS_BINARY` | `1` | 0 to fall back to base64 JSON |
| `NOVA_TTS_PLAY_CHUNK` | `2048` | Samples per WS network frame @ 24 kHz |
| `NOVA_POCKET_CHUNK` | `2048` | Synth chunk size for Pocket-TTS stream |
| `NOVA_SENT_MIN_CHARS` | `8` | First clause ships once it hits this |
| `NOVA_LATIN_ONLY` | `1` | Drop transcripts >20% non-Latin |
| `NOVA_MEMORY` | `1` | Persona + auto-memory loop |
| `NOVA_MEM_DIR` | `./persona` | Where USER/MEMORY live |
| `NOVA_MEMORY_CONSOLIDATE_EVERY` | `5` | Turns between memory dedupe passes |
| `SERPER_API_KEY` | *(unset)* | Enables `web_search` tool when set |

## What this is NOT

- Not a replacement for `main.py`. The multiprocess gateway has KWS, the FSM,
  multi-client broadcast, Postgres, per-variant routing.
  This file is the *demo + iteration* surface.
- Not multi-tenant. One WS connection at a time is the design (each opens
  its own APM/VAD; concurrent sessions would oversubscribe the GPU). For
  multi-driver demos, run multiple gateway instances on different ports.

## Troubleshooting

**Browser shows "mic permission denied"**: Chrome blocks `getUserMedia` on
non-HTTPS / non-localhost. Use `localhost` (with SSH `-L 8001:localhost:8001`
if remote), HTTPS, or `--unsafely-treat-insecure-origin-as-secure=...`.

**Audio plays glitchy**: usually a browser-side underrun. Confirm
`NOVA_TTS_BINARY=1` and that the gateway's GPU isn't oversubscribed
(`nvidia-smi` while talking should show steady activity).

**Kyutai/Qwen3 OOM**: smaller STT or move to CPU with
`NOVA_KYUTAI_DEVICE=cpu`. Qwen3-ASR-0.6B at fp16 fits on ~1.2 GB VRAM.

**Echo loop (Nova talks to itself)**: the echo guard catches near-identical
transcripts. If real follow-ups also get dropped, lower the Jaccard
threshold in `Session._looks_like_echo` (currently 0.6) or raise
`NOVA_AEC_GUARD_MS` to `800`.

**`turn_metrics: stt_ms` consistently large**: streaming partials aren't
settling. Try `NOVA_QWEN3_STABILITY_TICKS=1` (accept first match) or widen
`NOVA_QWEN3_PARTIAL_WINDOW_S=5.0` so each partial sees more context.

**pVAD not loading after enrollment**: voiceprint is checked at *startup*.
Enroll voice via the UI sidebar, then restart the gateway. The log should show
`pVAD (FireRedChat): loaded — target-speaker barge-in active`.

**pVAD barge-in fires on wrong speaker**: re-enroll with clearer audio.
The ECAPA-TDNN embedding quality depends on clean 3s+ speech samples.
Record in a quiet environment with the same microphone you'll use in the car.
If false positives persist, raise `NOVA_BARGE_IN_THRESHOLD` to `0.50`.

**LLM hallucinating / calling driver "Nova"**: this is a model-size issue.
The system prompt has CRITICAL RULES at the top, but models under ~1B params
struggle to follow multi-part instructions. Use at least Gemma 2 2B or
Llama 3.2 3B for reliable behavior.
