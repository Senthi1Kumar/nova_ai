# Nova — single-process voice gateway

`gateway_simple.py` is a minimal one-process alternative to the multiprocess
`main.py` pipeline. One asyncio loop per WebSocket connection, blocking model
calls dispatched to a thread executor. Designed for fast demo iteration on a
single workstation or a Jetson — no FastRTC, no multiprocessing, no FSM.

```
browser mic ──/ws bytes──▶ Silero VAD turn detection
                                  │
                                  ▼  (Smart-Turn confirm)
                                STT ─▶ LLM stream ─▶ clause split
                                                          │
                                                          ▼
                                          Kokoro TTS (pipelined synth)
                                                          │
                                                          ▼
                              binary PCM frames + APM ref ▶ /ws audio_out
```

Per-turn pipeline is fully concurrent: synth for sentence N+1 starts the
moment N enters the audio queue, so the player never blocks on synth.

## Quick start

### 1. Install deps

The repo's `pyproject.toml` already pins everything. From the project root:

```bash
uv sync
```

Optional: `pacman -S libssl-dev` (Arch) / `apt-get install libssl-dev` (Debian)
if you also want HTTPS on the gateway.

### 2. Set up the persona files

The gateway prepends two markdown files to the LLM system prompt and grows
one of them automatically. They're personal so they're git-ignored — copy
the templates on first run:

```bash
cd nova/backend/persona
cp USER.md.example USER.md
cp MEMORY.md.example MEMORY.md
$EDITOR USER.md          # set name, role, language, vehicle, prefs
```

`USER.md` is your manual persona (name, role, units, language preference).
`MEMORY.md` is auto-grown by Nova: after every assistant turn it extracts
durable facts and appends `- ...` lines; every 5 turns it consolidates the
file to dedupe. You can edit it by hand at any time.

Disable the whole thing with `NOVA_MEMORY=0` if you just want a stateless
demo.

### 3. Start a local LLM (optional but recommended)

The gateway speaks OpenAI-compatible chat completions. Either point it at
OpenRouter (default, requires `OPENROUTER_API_KEY`) or run any local server
that exposes `/v1/chat/completions` — vLLM, Ollama, llama.cpp, etc.

`llama.cpp` example with Gemma-4-E4B (fits 6 GB VRAM at Q4):

```bash
./build/bin/llama-server \
  -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_M \
  --alias "unsloth/gemma-4-E4B-it" \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 999 --ctx-size 4096 --threads -1 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}'
```

`--jinja` is required for tool-calling support. `enable_thinking:false` keeps
TTFB low.

### 4. Run the gateway

```bash
# .env
NOVA_LLM_BACKEND=local
NOVA_LLM_BASE_URL=http://localhost:8080/v1
NOVA_LLM_MODEL=unsloth/gemma-4-E4B-it
SERPER_API_KEY=...                      # optional, enables web_search tool

uv run python -m nova.backend.gateway_simple
```

Open `http://localhost:8001`, click the mic, talk. The browser handles mic
capture + 16 kHz PCM upload via `AudioWorkletNode` and plays back received
24 kHz PCM frames through `AudioBufferSource`.

## Tool calling (Serper)

When `SERPER_API_KEY` is set, the gateway exposes a single `web_search` tool
to the LLM. Before each user-message → assistant streaming pass, the gateway
runs a cheap regex (`_needs_tools`) over the transcript:

- Casual chat ("hi", "thanks", "what's 2+2?") → tools skipped, single
  streaming call, ~150 ms TTFT.
- Info-shaped ("what's the news in Paris", "weather tomorrow", "who won
  the match", "latest price of NVDA") → one non-streaming pre-call with
  `tools=[web_search]`. If the model returns a `tool_call`, the gateway
  hits Google via Serper, appends the tool result to the message list,
  then streams the final answer.

The Serper endpoint auto-routes to `/news` when the query contains news-y
keywords (`news`, `headline`, `latest`, `breaking`, `today`, ...) and
`/search` otherwise. Returns top 3 results as JSON, capped at 5.

Tools are off when:
- `SERPER_API_KEY` is empty.
- Prompt is conversational or under 4 words without a `?`.

You can extend `SERPER_TOOLS` and `_exec_tool()` with more functions
(weather, calendar, vehicle controls). Same OpenAI tool-calling shape —
add a definition, add a branch in `_exec_tool`, you're done.

## Environment knobs (gateway-specific)

| Var | Default | Notes |
|---|---|---|
| `NOVA_STT_BACKEND` | `qwen3_0_6b` | `qwen3_0_6b` / `kyutai_1b` / `moonshine` |
| `NOVA_QWEN3_LANGUAGE` | `English` | Pinned so noise stays in-language |
| `NOVA_LLM_BACKEND` | `openrouter` | `local` for vLLM/llama.cpp/Ollama |
| `NOVA_LLM_BASE_URL` | OpenRouter | e.g. `http://localhost:8080/v1` |
| `NOVA_LLM_MODEL` | `openai/gpt-4o-mini` | Match local server's model |
| `NOVA_SYSTEM_PROMPT` | concise English | Override per deployment |
| `NOVA_KOKORO_GPU` | `1` | Try CUDA EP for Kokoro ONNX |
| `NOVA_KOKORO_VOICE` | `af_heart` | Any Kokoro voice name |
| `NOVA_VAD_THRESHOLD` | `0.5` | Silero per-frame voice prob |
| `NOVA_SILENCE_END_MS` | `700` | End-of-utterance silence |
| `NOVA_SMART_TURN_THRESHOLD` | `0.5` | Drops if model unsure speaker is done |
| `NOVA_AEC_GUARD_MS` | `500` | Skip first N ms of TTS for barge-in |
| `NOVA_TTS_BINARY` | `1` | 0 to fall back to base64 JSON |
| `NOVA_SENT_MIN_CHARS` | `8` | First clause ships once it hits this |
| `NOVA_LATIN_ONLY` | `1` | Drop transcripts >20% non-Latin |
| `NOVA_MEMORY` | `1` | Persona + auto-memory loop |
| `NOVA_MEM_DIR` | `./persona` | Where USER/MEMORY live |
| `NOVA_MEMORY_CONSOLIDATE_EVERY` | `5` | Turns between memory dedupe passes |
| `SERPER_API_KEY` | _(unset)_ | Enables `web_search` tool when set |

## What this is NOT

- Not a replacement for `main.py`. The multiprocess gateway has KWS, the FSM,
  multi-client broadcast, voice enrollment, Postgres, per-variant routing.
  This file is a focused *demo + iteration* surface.
- Not multi-tenant. One WS connection at a time is the design (each opens
  its own `Session` + APM/VAD; concurrent sessions would oversubscribe the
  GPU). For multi-driver demos, run multiple gateway instances on different
  ports behind a reverse proxy.
- Not battle-tested at scale. The echo guard, drain-window math, and
  similarity filter are heuristics — they pass a one-driver demo but a
  real product will want LiveKit Agents + proper turn detection.

## Troubleshooting

**Browser shows "mic permission denied"**: Chrome blocks `getUserMedia` on
non-HTTPS / non-localhost. Either use `localhost`, run behind HTTPS, or
launch Chrome with `--unsafely-treat-insecure-origin-as-secure=http://thor.lan:8001`.

**Audio plays glitchy**: usually an underrun in the browser playback queue.
Set `NOVA_TTS_BINARY=1` (default) and confirm the WebSocket sends arrive
quickly enough — `nvidia-smi` while talking should show steady GPU activity,
not idle then spike.

**Kyutai/Qwen3 OOM**: smaller STT or move to CPU with
`NOVA_KYUTAI_DEVICE=cpu`. Qwen3-ASR-0.6B at fp16 fits on ~1.2 GB VRAM.

**Echo loop (Nova talks to itself)**: the echo guard catches near-identical
transcripts. If real follow-ups also get dropped, lower the Jaccard
threshold in `Session._looks_like_echo` (currently 0.6) or increase
`NOVA_AEC_GUARD_MS` to `800`.

**Latency feels high**: watch the gateway log for
`TTS first-audio TTFB: NNN ms`. Typical decomposition on a single GPU:
VAD finalize (~700 ms) → STT (~200 ms) → LLM TTFT (~150 ms local) →
Kokoro TFFT (~250 ms) ≈ **1.3 s** end-to-end. Most of that is the
`NOVA_SILENCE_END_MS` window, which you can tighten if your STT is
robust.
