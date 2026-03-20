# Nova AI

Nova is a high-performance, low-latency voice AI assistant designed for EV dashboards. It features a custom wake-word engine, streaming speech-to-text, real-time LLM reasoning with web search, and neural voice synthesis — all running as isolated multiprocessing workers communicating over lock-free queues.

## Key Features

- **Multiprocessing Pipeline**: STT, LLM, TTS, and KWS run as separate processes communicating via `mp.Queue`, each with its own GPU allocation. No GIL contention.
- **Always Listening (KWS)**: Custom-trained wake-word engine using Google Speech Embeddings + PyTorch MLP classifier. Supports versioned model snapshots with rollback.
- **Hybrid Interaction**: Hands-free wake-word activation ("Nova") and manual Push-to-Talk (PTT) via WebRTC or WebSocket.
- **Streaming STT**: Moonshine Voice (Medium) with built-in VAD — starts transcribing while the user is still speaking.
- **Tool-Calling LLM**: OpenRouter API (Nemotron 49B, Qwen3.5, Gemini Flash) with streaming tool calls. Web search via [Dux Distributed Global Search](https://github.com/deedy5/ddgs) (`ddgs`) returns actual news headlines, not website links. Falls back to local Qwen3.5-0.8B (4-bit) when offline.
- **Neural TTS**: FasterQwen3TTS with CUDA graph acceleration (12 kHz output, resampled to 24 kHz). Falls back to Pocket-TTS with multiple voice options.
- **Layer 7 Dialogue Manager**: Intent classification (Gemma-300M semantic embeddings), payment flow with voice verification (ECAPA-TDNN voiceprint → PIN → Face ID), OTP, and mock commerce.
- **Compound Vehicle Control**: "Switch off the AC and open the sunroof" handled as multiple actions in a single command.
- **Echo Suppression**: Accurate TTS playback tracking (`total_samples / 24kHz - elapsed`) with delayed STT unmute prevents the assistant from hearing its own voice.
- **Conversation Storage**: PostgreSQL session + turn history with intent, entities, latency, FSM state, and routing per turn.
- **Real-time Metrics**: Live E2E, STT, LLM (TTFT + throughput), and TTS (TTFA + RTF) latency tracking.

## Tech Stack

| Component | Technology | Details |
| :--- | :--- | :--- |
| **Gateway** | **FastAPI + FastRTC** | WebRTC (SDP) + WebSocket PCM streaming, FSM state machine, echo suppression |
| **STT** | **Moonshine Voice (Medium)** | Streaming transcription with built-in VAD, ~0.5s latency. PCM fed via WebSocket ScriptProcessor |
| **Storage** | **PostgreSQL + asyncpg** | Session + turn history: intent, entities, latency, FSM state per turn |
| **LLM** | **OpenRouter (Nemotron 49B)** | Streaming tool calls, web search, local Qwen3.5-0.8B fallback |
| **TTS** | **FasterQwen3TTS** | CUDA graphs, voice cloning, 12→24 kHz resampling. Pocket-TTS fallback |
| **KWS** | **Google Speech Embeddings + MLP** | Custom wake-word detection with versioned model snapshots |
| **Intent** | **Gemma-300M Embeddings** | Semantic intent classification with regex entity extraction |
| **Voice Auth** | **ECAPA-TDNN (SpeechBrain)** | Voiceprint verification, PIN fallback, Face ID fallback |
| **Commerce** | **SQLite Mock Backend** | Merchant search, basket, checkout, payment with OTP |

## Architecture

```text
Browser (WebRTC/WS)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI Gateway  (main.py)                         │
│  FSM: IDLE → LISTENING → GENERATING → IDLE          │
│  Echo suppression, delayed STT restart              │
└──┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│ KWS  │  │ STT  │  │ LLM  │  │ TTS  │
│Worker│  │Worker│  │Worker│  │Worker│
└──────┘  └──────┘  └──────┘  └──────┘
   mp.Queue ←──→ mp.Queue ←──→ mp.Queue

KWS → detects "Nova" → starts STT
STT → transcribes → sends to LLM
LLM → intent classify → stream tokens → sentence split → TTS
TTS → generate audio chunks → send to browser
```

## Performance

| Metric | Description | Target |
| :--- | :--- | :--- |
| **STT TTFB** | Voice stop → first transcript | < 0.6s |
| **LLM TTFT** | Transcript → first token | < 1.0s (API) |
| **TTS TTFA** | Token → first audio chunk | < 0.5s |
| **E2E** | Voice stop → Nova speaks | < 1.5s |

## KWS Enrollment + Voice Fingerprint

Nova includes a built-in enrollment UI:

1. **Wake Word**: Record 5 samples of "Nova" + 5 noise samples → trains MLP classifier
2. **Voice Fingerprint**: Record 5 speech phrases → computes ECAPA-TDNN embedding → encrypted voiceprint
3. **KWS Versioning**: Each re-enrollment saves a timestamped snapshot. Switch between versions from the settings menu.

## Payment Flow

```text
"Order a coffee from Starbucks"
  → Intent: payment → merchant search → menu display
  → "Yes" to confirm
  → Voice verification (voiceprint match)
    ✗ fail → PIN fallback → Face ID fallback
    ✓ pass → Voice OTP (4-digit, spoken)
  → OTP match → location check → payment processed
```

## Prerequisites

- **OS**: Linux
- **Hardware**: NVIDIA GPU with >= 4GB VRAM
- **CUDA**: 13.0 (required for PyTorch GPU acceleration)
- **Python**: 3.12+
- **Tools**: [uv](https://docs.astral.sh/uv/)
- **PostgreSQL**: 16+ (for conversation storage)

## Quick Start

1. **Clone the repository**:

   ```bash
   git clone --recursive https://github.com/Senthi1Kumar/nova_ai.git
   cd nova_ai
   git checkout v1.2
   git submodule update --init --recursive
   ```

2. **Set environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY and NOVA_DB_URL
   ```

3. **Set up PostgreSQL** (conversation storage):

   ```bash
   # Create DB and user
   sudo -u postgres psql -c "CREATE USER nova WITH PASSWORD 'nova_dev';"
   sudo -u postgres psql -c "CREATE DATABASE nova_db OWNER nova;"

   # Create schema
   psql postgresql://nova:nova_dev@localhost/nova_db <<'SQL'
   CREATE TABLE sessions (
     session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     driver_id  TEXT NOT NULL DEFAULT 'driver1',
     started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
     ended_at   TIMESTAMPTZ
   );
   CREATE TABLE turns (
     turn_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     session_id     UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
     turn_index     INT NOT NULL,
     ts             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
     user_text      TEXT,
     intent         TEXT,
     entities       JSONB,
     nova_response  TEXT,
     fsm_state      TEXT,
     routing        TEXT,
     stt_latency_ms INT
   );
   SQL
   ```

   Set `NOVA_DB_URL=postgresql://nova:nova_dev@localhost/nova_db` in `.env`. Storage is optional — Nova runs without it if `NOVA_DB_URL` is unset.

4. **Patch pocket-tts submodule** (forces CUDA PyTorch):

   ```bash
   ./patch_pocket-tts_submodule.sh
   ```

5. **Install dependencies**:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

6. **Optional: Record a voice reference** for Faster Qwen3-TTS (see `nova/backend/voices/SETUP.md`):

   ```bash
   arecord -d 8 -f cd -r 24000 -c 1 nova/backend/voices/nova_ref.wav
   ```

7. **Start Nova**:

   ```bash
   # With KWS wake-word detection:
   uv run nova/backend/main.py

   # Without KWS (always-listening mode):
   uv run nova/backend/main.py --no-kws
   ```

8. **Open Dashboard**: Navigate to `http://localhost:8000`

## Directory Structure

```text
nova/
├── backend/
│   ├── main.py                  # FastAPI gateway, FSM, WebRTC/WS
│   ├── pipeline_mp/             # Multiprocessing workers
│   │   ├── stt_moonshine_worker.py   # Moonshine streaming STT
│   │   ├── llm_worker.py             # OpenRouter + local fallback LLM
│   │   ├── tts_worker.py             # FasterQwen3TTS + Pocket-TTS
│   │   └── kws_worker.py             # Wake-word detection worker
│   ├── kws/                     # KWS engine, enrollment, synthetic data
│   │   └── kws_engine_v2.py          # MLP classifier with versioned save/load
│   ├── nova-l7/                 # (Submodule) Layer 7 dialogue system
│   │   ├── L-7/                      # Intent classifier + dialogue manager
│   │   └── L-3/                      # Voice/PIN/Face verification + commerce
│   └── voices/                  # TTS voice reference audio
├── frontend/
│   └── index.html               # Dashboard UI (WebRTC + WebSocket)
└── pocket-tts/                  # (Submodule) Pocket-TTS voice synthesis
```
