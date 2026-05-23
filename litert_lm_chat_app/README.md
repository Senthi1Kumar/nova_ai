# LiteRT-LM Native Voice Chat

A local FastAPI chat app powered by **LiteRT-LM** with:

- Gemma-4 native audio understanding (no Whisper)
- Gemma-4 native vision — webcam snap or image upload, attached one-shot to any text or voice turn
- MTP (speculative decoding) for fast streaming LLM output
- Tavily web search / extract / research tools
- Mapbox MCP — 9 geocoding / routing / isochrone / map-image tools via Anthropic's Model Context Protocol
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
LITERT_VISION_BACKEND=GPU         # drop to CPU if VRAM-bound
LITERT_ENABLE_SPECULATIVE=true    # MTP — set false on CPU backend
TTS_ENABLED=true
POCKET_TTS_VOICE=alba
POCKET_TTS_LANGUAGE=english
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
- `POST /api/chat/stream` — text chat SSE
- `POST /api/voice/chat/stream` — native-audio voice pipeline SSE
- `POST /api/voice/chat` — non-streaming voice (returns one WAV URL)
- `POST /api/tts` — synthesize text to a WAV file

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

## MCP integration (Mapbox)

This app demonstrates **MCP (Model Context Protocol)** by bridging the
hosted Mapbox MCP server into LiteRT-LM's tool-calling. MCP is Anthropic's
open protocol for letting LLM apps talk to external "tool servers" over a
small JSON-RPC contract. Each MCP server exposes a set of tools
(`list_tools` → `call_tool`), and a client wires those into the LLM's
tool surface.

### What's wired up

We connect to `https://mcp.mapbox.com/mcp` over streamable-HTTP at app
startup (a daemon thread holds the async `ClientSession` for the lifetime
of the process) and expose 9 hand-picked Mapbox tools as Python functions
that LiteRT-LM auto-registers from their docstrings:

| Tool | What it does |
| --- | --- |
| `mapbox_search_and_geocode` | place name → address + coordinates |
| `mapbox_reverse_geocode` | coordinates → address |
| `mapbox_directions` | turn-by-turn routing + ETA |
| `mapbox_isochrone` | area reachable within N minutes |
| `mapbox_matrix` | many-to-many travel times |
| `mapbox_category_search` | POIs by category near a place |
| `mapbox_static_map` | render a map PNG with markers |
| `mapbox_optimize_route` | optimal visiting order for 3-12 stops |
| `mapbox_map_match` | snap a GPS trace to roads |

Try voice prompts like *"How long to drive from MG Road Bangalore to the
airport?"* or *"Show me coffee shops near MG Road"* — Gemma will pick the
right tool, call it via the MCP bridge, and answer using the result.

### How the bridge works

```text
LiteRT-LM Engine
  │ tools = [tavily_search, mapbox_directions, ...]
  │
  │ Gemma emits tool_call("mapbox_directions", {...})
  │     │
  │     ▼
  │ Python wrapper in app/tools.py
  │     │ calls bridge.call_tool("directions_tool", {...})
  │     ▼
  │ MCPBridge (background asyncio loop in a daemon thread)
  │     │ run_coroutine_threadsafe(session.call_tool(...), loop)
  │     ▼
  │ mcp.ClientSession over streamable-HTTP
  │     │ POST JSON-RPC to https://mcp.mapbox.com/mcp
  │     ▼
  │ Mapbox MCP server (hosted) → calls real Mapbox APIs
  │     │
  │     ◀── CallToolResult.content (text or image)
  │
  ◀── tool_result fed back into Gemma's decode loop
```

The `app/mcp_bridge.py` module owns one persistent session per server, so
the WebSocket / HTTP connection stays warm across turns. The sync↔async
gap is bridged with `asyncio.run_coroutine_threadsafe` from the bridge's
public `call_tool(name, args, timeout)`.

### Enabling Mapbox

```bash
# 1. Get a Mapbox public token: https://account.mapbox.com/access-tokens/
# 2. Add to .env:
echo 'MAPBOX_ACCESS_TOKEN=pk.eyJ...' >> .env
# 3. Restart the server. Logs should show:
#    INFO litert_app.mcp: mapbox MCP ready (20 tools): distance_tool, ...
```

If `MAPBOX_ACCESS_TOKEN` is empty the bridge stays off, the tool wrappers
return `"Map tool unavailable: Mapbox MCP not connected."`, and Gemma is
told via the system prompt to not invoke them. The rest of the app
(Tavily, vision, voice) continues to work normally.

### Adding more MCP servers

The bridge is reusable. To add a second server (e.g. GitHub MCP), copy
the lifespan block in `app/main.py`, pass a different `url` + `token`,
hold its `MCPBridge` instance under a different name, and write thin
wrapper functions in `app/tools.py` that call `bridge.call_tool("…", {…})`.
LiteRT-LM auto-discovers Python functions from their docstrings.

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
