# Nova AI

Nova is a high-performance, low-latency voice AI assistant designed for EV dashboards. It features a custom wake-word engine, streaming speech-to-text, real-time LLM reasoning with web search, and neural voice synthesis — all running as isolated multiprocessing workers communicating over lock-free queues.

## Key Features

- **Multiprocessing Pipeline**: STT, LLM, TTS, and KWS run as separate processes communicating via `mp.Queue`, each with its own GPU allocation. No GIL contention.
- **Always Listening (KWS)**: MicroWakeWord TFLite engine (`micro-wake-word` submodule) trained on synthetic Nova utterances. Supports versioned model snapshots with rollback. Legacy Google Speech Embeddings + MLP engine available via `NOVA_KWS_ENGINE=v2`.
- **Hybrid Interaction**: Hands-free wake-word activation ("Nova") and manual Push-to-Talk (PTT) via WebRTC or WebSocket.
- **Streaming STT**: Moonshine with built-in VAD — starts transcribing while the user is still speaking. Default: fine-tuned Indian English base model (`in_en`). Switchable to Tiny/Small/Medium streaming variants from the settings menu.
- **Tool-Calling LLM**: OpenRouter API (Nemotron 49B, Qwen3.5, Gemini Flash) with streaming tool calls. Web search via [Serper API](https://serper.dev/) (Google Search) returns actual news snippets and search results. Falls back to local Liquid AI LFM2.5-350M when offline.
- **Personalized VAD (pVAD)**: ECAPA-TDNN speaker gate runs in a parallel process on a 1s rolling window. FSM transitions are suppressed when a non-primary speaker is detected. Fail-open: no voiceprint = always pass. Hysteresis prevents chattering on room noise.
- **Neural TTS**: Pocket-TTS with multiple voice options (default, low VRAM). Optional: FasterQwen3TTS with CUDA graph acceleration (12 kHz output, resampled to 24 kHz) for higher quality and voice cloning.
- **Layer 7 Dialogue Manager**: Intent classification (Gemma-300M semantic embeddings), payment flow with voice verification (ECAPA-TDNN voiceprint → PIN → Face ID), OTP, and mock commerce.
- **Compound Vehicle Control**: "Switch off the AC and open the sunroof" handled as multiple actions in a single command.
- **Echo Suppression**: Accurate TTS playback tracking (`total_samples / 24kHz - elapsed`) with delayed STT unmute prevents the assistant from hearing its own voice.
- **Conversation Storage**: PostgreSQL session + turn history with intent, entities, latency, FSM state, and routing per turn.
- **Real-time Metrics**: Live E2E, STT, LLM (TTFT + throughput), and TTS (TTFA + RTF) latency tracking.

## Tech Stack

| Component | Technology | Details |
| :--- | :--- | :--- |
| **Gateway** | **FastAPI + FastRTC** | WebRTC (SDP) + WebSocket PCM streaming, FSM state machine, echo suppression |
| **STT** | **Moonshine (fine-tuned Indian EN)** | Streaming + non-streaming variants. Default: `pavandheeraj05/moonshine-nova-indian-english`. Switchable via UI. |
| **Storage** | **PostgreSQL + asyncpg** | Session + turn history: intent, entities, latency, FSM state per turn |
| **LLM** | **OpenRouter (Nemotron 49B)** | Streaming tool calls, web search, local LFM2.5-350M fallback |
| **pVAD** | **ECAPA-TDNN (SpeechBrain)** | Parallel speaker gate — suppresses FSM transitions for non-primary speakers. Env: `NOVA_PVAD_*` |
| **TTS** | **Pocket-TTS** | Default engine (low VRAM). Optional: [FasterQwen3TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (CUDA graphs) |
| **KWS** | **MicroWakeWord (TFLite)** | Custom-trained TFLite wake-word model. Versioned snapshots. Legacy MLP engine via `NOVA_KWS_ENGINE=v2` |
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

1. **Wake Word (MicroKWS)**: The default TFLite model is pre-trained and static — no runtime enrollment needed. To retrain with new voices:

   ```bash
   cd nova/backend/kws
   # With Piper TTS (default):
   python train_micro_nova.py --all
   python train_micro_nova.py --export

   # With Qwen3-TTS voice cloning (more diverse, better accuracy):
   # Record 20-30 people saying any random sentence (~10s each):
   arecord -f S16_LE -r 16000 -c 1 -d 10 reference_voices/voice_01.wav
   python train_micro_nova.py --all --tts-engine qwen3 --voices-dir ./reference_voices/
   python train_micro_nova.py --export
   ```

   Each export saves a versioned snapshot under `kws/models/versions/<timestamp>/`. Switch the active version from the settings menu.

2. **Wake Word (Legacy MLP engine)**: Set `NOVA_KWS_ENGINE=v2`. Record 5 samples of "Nova" + 5 noise samples from the enrollment UI → trains MLP classifier.

3. **Voice Fingerprint**: Record 5 speech phrases → computes ECAPA-TDNN embedding → encrypted voiceprint. Used for payment voice verification (L-3).

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

1. **Clone and Setup Repository**:

   - **Option A: Clone via Git** (Recommended)

     ```bash
     git clone https://github.com/Senthi1Kumar/nova_ai.git
     cd nova_ai
     git checkout v1.2
     git submodule update --init --recursive
     ```

   - **Option B: Direct v1.2 Download**
     If you downloaded the `v1.2` source directly, you still need to initialize submodules:

     ```bash
     cd nova_ai
     git submodule update --init --recursive
     ```

2. **Set environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env and add your keys:
   #   OPENROUTER_API_KEY   — from https://openrouter.ai/keys
   #   SERPER_API_KEY       — from https://serper.dev/
   #   NOVA_DB_URL          — PostgreSQL connection string (optional)
   #
   # Optional KWS tuning (MicroKWS defaults are shown):
   #   NOVA_KWS_ENGINE      — "micro" (default) or "v2" (legacy MLP)
   #   NOVA_KWS_THRESHOLD   — wake-word confidence threshold (default: 0.35)
   #   NOVA_KWS_CONSECUTIVE — consecutive triggers required (default: 1)
   #
   # Optional pVAD tuning (speaker gate, fail-open by default):
   #   NOVA_PVAD_ENABLED    — "1" (default) or "0" to disable
   #   NOVA_PVAD_DRIVER_ID  — voiceprint ID (default: driver1)
   #   NOVA_PVAD_THRESHOLD  — cosine similarity cutoff (default: 0.58)
   #   NOVA_PVAD_HYSTERESIS — windows before gate state changes (default: 2)
   #   NOVA_PVAD_ENERGY     — RMS floor below which gate stays open (default: 0.035)
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

4. **Patch submodules** (must run before `uv sync`):

   ```bash
   # Removes audio-metadata from microwakeword's install_requires (training dep, not needed for inference)
   bash patch_microwakeword_submodule.sh

   # Forces pocket-tts to use CUDA PyTorch
   bash patch_pocket-tts_submodule.sh
   ```

5. **Install dependencies**:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

   On **Jetson / aarch64**, `uv sync` automatically pulls torch from the Jetson AI Lab `sbsa/cu130` index (configured in `pyproject.toml`). No extra pip step needed.

6. **Optional: Record a voice reference** for Faster Qwen3-TTS (see `nova/backend/voices/SETUP.md`):

   ```bash
   arecord -d 8 -f cd -r 24000 -c 1 nova/backend/voices/nova_ref.wav
   ```

7. **Start Nova**:

   ```bash
   # With KWS wake-word detection (MicroKWS, default):
   uv run nova/backend/main.py

   # Explicitly set MicroKWS engine (if not set in .env):
   NOVA_KWS_ENGINE=micro uv run nova/backend/main.py

   # Without KWS (always-listening mode):
   uv run nova/backend/main.py --no-kws

   # Disable pVAD speaker gate (e.g. guest/debug mode, or before enrolling a voiceprint):
   NOVA_PVAD_ENABLED=0 uv run nova/backend/main.py
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
│   ├── kws/                     # KWS engine, training pipeline, models
│   │   ├── micro_kws.py              # MicroWakeWord TFLite inference (default)
│   │   ├── train_micro_nova.py       # Training pipeline (Piper or Qwen3-TTS)
│   │   ├── micro-wake-word/          # (Submodule) OHF-Voice/micro-wake-word
│   │   └── kws_engine_v2.py          # Legacy MLP classifier (NOVA_KWS_ENGINE=v2)
│   ├── nova-l7/                 # (Submodule) Layer 7 dialogue system
│   │   ├── L-7/                      # Intent classifier + dialogue manager
│   │   └── L-3/                      # Voice/PIN/Face verification + commerce
│   └── voices/                  # TTS voice reference audio
├── frontend/
│   └── index.html               # Dashboard UI (WebRTC + WebSocket)
└── pocket-tts/                  # (Submodule) Pocket-TTS voice synthesis
```
