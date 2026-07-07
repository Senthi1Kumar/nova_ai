# LiteRT-LM Native Voice Chat

A local FastAPI chat app powered by **LiteRT-LM** with:

- Gemma-4 native audio understanding (no Whisper)
- Gemma-4 native vision — webcam snap or image upload, attached one-shot to any text or voice turn
- MTP (speculative decoding) for fast streaming LLM output
- Brave Search + Serper (Google SERP / Shopping / Maps) web tools, exposed via the IResearcher FastMCP sidecar (`app/tools_v2.py`) — chat app connects over MCP
- Pocket-TTS streaming PCM over SSE
- Push-to-talk browser UI with gapless audio playback

This app is part of the [Nova AI](../) repository. It ships its own venv and
`requirements.txt` independent of Nova's `pyproject.toml`, so it can be
installed and run without the full Nova stack.

## Hardware requirements

LiteRT-LM picks the heaviest model variant your VRAM can hold. Check yours
with `nvidia-smi` (Linux/Windows) or `system_profiler SPDisplaysDataType` (macOS).

| GPU VRAM | Recommended model | Backend |
| --- | --- | --- |
| **≥ 8 GB** | `gemma-4-E4B-it.litertlm` (~4 GB weights @ INT8) | `GPU` |
| **4–8 GB** | `gemma-4-E2B-it.litertlm` (~2 GB weights @ INT8) | `GPU` |
| **< 4 GB or no GPU** | `gemma-4-E2B-it.litertlm` | `CPU` (slower TTFT) |

If you put E4B on a ≤ 4 GB card you will see a wall of WebGPU `Invalid
Buffer` validation errors as Dawn fails to allocate KV-cache buffers. Drop
to E2B or switch the backend to `CPU`.

Audio backend stays on `CPU` regardless — the encoder is small, GPU
bandwidth is better spent on the decoder.

## System dependencies

- **Python 3.12 or 3.13** (LiteRT-LM ships wheels for both)
- **ffmpeg** — required to transcode browser audio (webm/opus → 16 kHz mono WAV)
  before handing it to Gemma's bundled miniaudio decoder, which only accepts
  WAV/FLAC/MP3/Vorbis. Install:
  - Debian/Ubuntu: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## Setup

### 1) Get the code

This app lives on the `litert-chat-app` branch of the Nova AI repository.
Two ways to grab it:

**Option A — clone just this branch (recommended if you only want the chat app):**

```bash
git clone -b litert-chat-app --single-branch \
  https://github.com/Senthi1Kumar/nova_ai.git
cd nova_ai/litert_lm_chat_app
```

**Option B — clone the full repo and switch:**

```bash
git clone https://github.com/Senthi1Kumar/nova_ai.git
cd nova_ai
git checkout litert-chat-app
cd litert_lm_chat_app
```

### 2) Create and activate the venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Shell PATH gotcha:** on some shells (zsh with certain plugins, conda
> hooks, pyenv shims) `source .venv/bin/activate` shows the `(.venv)` prompt
> but doesn't actually prepend `.venv/bin` to `$PATH`. Verify with
> `which python` — it must point inside `.venv/bin/`. If it doesn't, just
> call the venv's binary directly: `./.venv/bin/python run.py`.

### 3) Install dependencies

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 4) Download a Gemma-4 LiteRT model

From Hugging Face:

- E2B: <https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm>
- E4B: <https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm>

```bash
hf download litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm --local-dir .
```

### 5) Configure `.env`

```env
LITERT_MODEL_PATH=/absolute/path/to/gemma-4-E2B-it.litertlm
LITERT_BACKEND=GPU                # or CPU if VRAM-bound
LITERT_AUDIO_BACKEND=CPU
LITERT_VISION_BACKEND=CPU         # GPU vision on ≤4 GB cards stalls or OOMs
LITERT_ENABLE_SPECULATIVE=true    # MTP — set false on CPU backend
LITERT_MAX_NUM_TOKENS=8192        # 4096 on ≤4 GB GPU, 16384 on ≥8 GB
TTS_ENABLED=true
POCKET_TTS_VOICE=alba
POCKET_TTS_LANGUAGE=english
NOVA_MEM0_DISABLED=1              # default: journal + diary recall only
TAVILY_API_KEY=tvly-...           # optional, enables web_search tool
```

### 6) Run

```bash
python run.py
# or, if your shell PATH doesn't pick up the venv:
./.venv/bin/python run.py
```

Open <http://localhost:8000>. Click "Enable Microphone", hold the mic
button, speak, release.

## API endpoints

- `GET /api/health` — engine + TTS status
- `GET /api/debug/system_prompt` — composed system prompt (verify PRIOR DAYS diary splice)
- `POST /api/debug/tool` — invoke a single MCP tool by name, bypass the model (`{"name":"web_search","args":{"query":"...","search_type":"news","num_results":5}}`)
- `POST /api/chat/stream` — text chat SSE
- `POST /api/voice/chat/stream` — native-audio voice pipeline SSE
- `POST /api/voice/chat` — non-streaming voice (returns one WAV URL)
- `POST /api/tts` — synthesize text to a WAV file
- `GET /api/vehicle/state` — simulated CAN/OBD state (HVAC zones, sunroof/windows,
  battery, fuel, media) + reminders + calendar events, from `app/vehicle_db.py`
- `GET /api/system/stats` — host CPU%, RAM used/total, GPU util%/VRAM (via
  `psutil` + `nvidia-smi`), polled by the top-bar HUD every 3s

## Voice streaming event order

`/api/voice/chat/stream` is a server-sent events (SSE) stream. Events
arrive in this order per turn:

```text
session       Session id
status        Pipeline stage (transcode | encode)
token         Streamed reply token
tool_call     Model invoked tavily_search (or other tool)
tool_result   Tool returned
clause        Clause flushed to TTS
audio_chunk   Base64 int16 LE PCM @ 24 kHz
clause_end    End of clause
turn_metrics  Latency snapshot (TTFT, TTFB, tok/s)
done          Stream complete
warning       Non-fatal TTS issue
error         Fatal error
```

## Architecture notes

Gemma-4 accepts audio directly via its USM-style encoder — the model hears
the user and replies in one forward pass; there is no intermediate
transcript.

## Run the IResearcher MCP sidecar

The web-search tools (`web_search`, `google_search`) live in a separate
FastMCP server (`app/tools_v2.py`) that exposes Brave + Serper APIs over
MCP. Start it in a second terminal:

```bash
cd litert_lm_chat_app
export BRAVE_API_KEY=...      # https://api.search.brave.com
export SERPER_API_KEY=...     # https://serper.dev
./.venv/bin/python -m app.tools_v2
```

It listens on `http://127.0.0.1:8765/mcp`. The chat app connects at
startup; if the sidecar isn't up you'll see a warning in the chat app
log and the search tools will return *"IResearcher MCP sidecar not
connected"* until you start it.

Other MCP clients (Claude Desktop, Cursor, MCP Inspector) can also
connect to the same endpoint — the sidecar exposes the full tool set:

| Tool | Backend |
| --- | --- |
| `web_search` | Brave Search API (keyword snippets, web / news) |
| `google_search` | Serper (Google Shopping or Maps) |
| `geocode` | Nominatim (forward + reverse, OpenStreetMap) |
| `execute_sql_query` | Traccar / telematics DB (see prompts.py) |
| `add_skill` | Returns the telematics skill system prompt |

The chat app registers only `web_search` + `google_search` in
`DEFAULT_TOOLS`; the rest are available for other clients.

## In-vehicle demo layer (vehicle store + tool router)

`app/vehicle_db.py` is a small SQLite-backed simulated CAN/OBD store
(`runtime/vehicle.db`) backing the 12-tool IVA toolbox: HVAC per-zone
on/off + temp with sunroof/window interlocks (both auto-close whenever any
HVAC zone is on), reminders and calendar events resolved against the host
clock. `app/vehicle_tools.py` routes each tool call through it first,
falling back to an acknowledge-only stub for unhandled tools. Poll current
state at `GET /api/vehicle/state`; it's rendered in the right-rail VEHICLE
panel and refreshed every 3s.

`NOVA_TOOL_ROUTER` (see `.env.example`) enables a lightweight per-turn
lexical tool router (SkillWeaver-style): instead of binding all 12 tools on
every prefill, it scores tools by name/param/description token overlap
against the current turn and rebinds only the top-k, rebuilding the
conversation when the routed set changes. It currently only fires on text
turns — Gemma-4 ingests voice natively with no transcript to route on, so
voice turns keep the full toolbox bound.

`NOVA_MAX_REPLY_CHARS` caps a single assistant reply; past that length (or
on detecting leaked template markers / short-tail repetition from an
unterminated tool-call arg) the stream is drained-and-discarded rather than
broken off, since abandoning the LiteRT-LM generator mid-decode corrupts
the engine.

## Persistent memory layer (mem0 + pgvector)

Adapted from Nova's `nova_memory_layer.py`. Three cooperating pieces:

| Component | What | Where |
| --- | --- | --- |
| **SessionJournal** | every turn appended to a per-session JSON | `runtime/session_logs/session_<id>_<ts>.json` |
| **DiaryCompactor** | on session close, Gemma-4 summarises 3-5 bullets, appended to today's diary | `runtime/daily_logs/YYYY-MM-DD.md` |
| **mem0 + pgvector** | semantic recall — facts extracted by an LLM, embedded by sentence-transformers, stored in Postgres; top-K injected into the next session's system prompt | Postgres table |

### How mem0 talks to LiteRT-LM (no llama-server required)

mem0's `openai` LLM provider speaks the OpenAI HTTP wire format. We expose
a tiny shim — `app/llm_shim.py` — that wraps the already-loaded LiteRT-LM
engine as `/v1/chat/completions`. mem0 calls it for fact extraction; no
second model is loaded. Fact-extraction LLM calls share the engine with
chat (serialised by LiteRT-LM's per-conversation lock).

Architecture:

```text
DURING SESSION (per turn):
   user turn ends ──▶ memory.add_turn(user) + memory.add_turn(assistant)
                            ↑
                            └── sync, journal-only (no LLM call, no GPU work)

AT SESSION CLOSE (↻ New chat OR app shutdown):
   1. chat_service.close_all_sessions()    ← frees the LiteRT-LM engine
   2. memory.batch_ingest_session()        ← mem0 processes the journal
         │
         ▼  for each (user, assistant) pair in the journal:
   mem0.add() ──▶ POST /v1/chat/completions ──▶ LiteRT shim (throwaway conv)
                                                       │
                                                       ▼
                          extracts durable facts ──▶ embeds (all-MiniLM-L6-v2, CPU)
                                                       │
                                                       ▼
                                                   pgvector
   3. memory.close_session(llm_oneshot)     ← diary summary appended
   4. memory.start_new_session()            ← fresh journal for the next session
```

**Why ingest is deferred to session close**: LiteRT-LM only supports one
conversation per engine. While the user's chat conversation is open, the
shim can't spin up a second one for mem0 — the engine rejects with
`FAILED_PRECONDITION: A session already exists`. So mem0 work is batched
between conversation-close and engine-teardown, when the engine is free.

### One-time Postgres setup

```bash
sudo apt-get install -y postgresql postgresql-contrib postgresql-16-pgvector
sudo systemctl enable --now postgresql

# DB + user (separate from Nova's nova_db so they don't collide)
sudo -u postgres psql -c "CREATE USER nova WITH PASSWORD 'nova_dev';"
sudo -u postgres psql -c "CREATE DATABASE litert_chat_db OWNER nova;"
sudo -u postgres psql -d litert_chat_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

If your distro doesn't have `postgresql-NN-pgvector` in apt, build from source —
see the same instructions in `nova/backend/NOVA_LOOP.md`.

### Fail-open behaviour

If Postgres is down, pgvector missing, or the shim unreachable, the layer
logs a warning and degrades to **journal-only** mode:

- `runtime/session_logs/*.json` still get written on session close
- `runtime/daily_logs/*.md` diary still gets appended (deterministic
  fallback if the LLM call also fails)
- Semantic recall returns empty; system prompt has no `PRIOR CONTEXT` block

Chat continues to work without interruption.

### Disabling

```env
NOVA_MEMORY=0           # disable everything
NOVA_MEM0_DISABLED=1    # keep journal + diary, skip mem0 / Postgres
```

### Lifecycle

| When | What happens |
| --- | --- |
| App startup | `MemoryLayer` initialised, mem0 connects to Postgres + shim |
| Each user turn ends | `journal.add(user)` + `journal.add(assistant)` (sync, journal-only — no LLM call) |
| `↻ New chat` click (or `DELETE /api/session/<id>`) | Journal flushed → diary appended → fresh journal started |
| App shutdown | Final journal flushed + diary appended |

### Knobs

See `.env.example` under "Persistent memory layer" — `NOVA_MEMORY*`,
`NOVA_MEM0_*`, `NOVA_MEMORY_USER_ID`, `NOVA_MEM_RECALL_K`.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

13 unit tests cover the Tavily tool, clause splitter, and Pocket-TTS
chunk math. No model load required, runs in under a second.

## Troubleshooting

**`Invalid Buffer`/`Invalid BindGroup` validation error spam.** GPU is
out of memory. Switch to a smaller model (E4B → E2B) or set
`LITERT_BACKEND=CPU`.

**`module 'litert_lm' has no attribute 'set_min_log_severity'`.** You have
`litert-lm-api-nightly` installed instead of stable `litert-lm-api`. The
nightly has a reshuffled API surface. Reinstall: `pip uninstall -y
litert-lm-api-nightly && pip install litert-lm-api`.

**`Failed to initialize miniaudio decoder`.** The audio file isn't a
container miniaudio supports. The route auto-transcodes via ffmpeg, so this
means ffmpeg either isn't installed or failed. Verify `which ffmpeg`.

**Mic permission denied.** Browsers block `getUserMedia` on non-HTTPS /
non-localhost. Use `http://localhost:8000` (not `0.0.0.0` or a LAN IP).

**`litert_lm_conversation_send_message_stream failed` with no other
context.** Usually means the engine ran out of memory mid-decode (KV cache
grew past the limit). Switch to a smaller model or shorter conversation.

**Model goes silent after 3-5 tool-using turns** (no token events, no error
event, just `done`). Engine `max_num_tokens` budget was exhausted. Bump
`LITERT_MAX_NUM_TOKENS` in `.env` (default 4096; the model's own default of
~2048 is what causes the silent stop). Or click **↻ New chat** in the UI to
reset the conversation (calls `DELETE /api/session/{id}`).

**`VK_ERROR_OUT_OF_DEVICE_MEMORY` at startup**, followed by a wall of
WebGPU `Invalid Buffer` validation errors. KV cache allocation overflowed
your GPU VRAM. KV cache size ≈ `2 × layers × hidden × max_num_tokens × 2
bytes`. For Gemma-4-E2B (24 layers, 2048 hidden, fp16) on a 4 GB card,
`max_num_tokens` must stay ≤ ~4096. If you set it higher, drop it back in
`.env` and restart:

```env
LITERT_MAX_NUM_TOKENS=4096      # 4 GB GPU
LITERT_MAX_NUM_TOKENS=8192      # ≥ 6 GB GPU
LITERT_MAX_NUM_TOKENS=16384     # ≥ 8 GB GPU
```

**Vision turn fails with `vision_litert_compiled_model_executor` /
`Failed to get num packed bytes`.** GPU OOM during vision activations
(image came in, vision encoder tried to allocate on GPU on top of LLM
weights + KV cache). Set `LITERT_VISION_BACKEND=CPU` in `.env` — the
vision encoder is small (~100 MB), CPU runs it in well under a second,
and it stops competing for VRAM.

**`Fatal glibc error: malloc.c:4241 (_int_malloc): assertion failed` /
`free(): invalid next size` crashes the whole process (not just the
turn).** Heap corruption inside the LiteRT-LM C++ engine, typically
surfacing after a degenerate decode (unterminated tool-call arg → runaway
generation) even though the app-level runaway guard (`NOVA_MAX_REPLY_CHARS`)
contains the visible symptom. Not fixable from Python — this is the
open item in `docs/app-audit-2026-07-06.md` priority queue #4. If it's
frequent on your model/build, worth evaluating an alternative on-device
runtime (e.g. Cactus) for the affected model rather than chasing it
further inside LiteRT-LM.
