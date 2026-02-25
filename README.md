# Nova AI

Nova is a high-performance, low-latency voice AI assistant designed for EV dashboards. It features a robust "Always Listening" wake-word engine, near-instant streaming transcription, and fluid voice synthesis, all optimized to run on consumer hardware.

## 🚀 Key Features

- **Always Listening (KWS)**: A custom-trained Key Word Spotting engine using Google Speech Embeddings and a PyTorch MLP classifier.
- **Hybrid Interaction**: Supports both hands-free wake-word activation ("Nova") and manual "Push to Talk" (PTT) modes.
- **Streaming Pipeline**: True end-to-end streaming. Nova starts "thinking" and "speaking" while the transcription is still finalizing.
- **VRAM Optimized**: Co-locates STT, LLM, and VAD models within 4GB VRAM using 4-bit quantization and shared model instances.
- **Real-time Metrics**: Live tracking of STT, LLM, TTS, and End-to-End latencies for performance benchmarking.

## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Protocol** | **FastAPI WebSockets** | Real-time binary audio and JSON metadata streaming. |
| **STT** | **Faster-Whisper (Small)** | CTranslate2-based Whisper implementation for high-speed transcription. |
| **LLM** | **SmolLM2-1.7B / Qwen2.5-0.5B** | 4-bit quantized (NF4) Instruct models for instant reasoning. |
| **TTS** | **Pocket-TTS** | High-fidelity voice synthesis with background streaming tasks. |
| **KWS/VAD** | **Silero + Google Embeddings** | Advanced voice activity and wake-word detection. |

## 📊 Performance Metrics Explained

Nova tracks four critical latency stages to ensure a seamless "human-like" interaction:

1.  **STT (TTFB - Time to First Byte)**: Time from voice stop to the first transcript appearing. Measures recognition speed.
2.  **LLM (TTFT - Time to First Token)**: Time from transcript completion to the first generated word. Measures "thinking" delay.
3.  **TTS (TTFA - Time to First Audio)**: Time from token generation to the first audible chunk being synthesized. Measures voice synthesis delay.
4.  **END-TO-END (E2E)**: The total duration from the moment the user stops speaking to the moment Nova's voice is heard. **Target: < 1.5s.**

## 🎙 KWS Enrollment

Nova includes a built-in enrollment UI to calibrate the wake-word engine to your specific voice and environment:
1.  **Wake Word**: Record 5 samples of you saying "Nova".
2.  **Noise**: Record 2 samples of your environment (car engine, AC, road noise).
3.  **Training**: The system automatically trains a PyTorch MLP classifier on top of the Google Speech Embeddings for high-precision rejection of non-wake-word speech.

## 📋 Prerequisites

- **OS**: Linux (Ubuntu recommended)
- **Hardware**: NVIDIA GPU with >= 4GB VRAM (e.g., RTX 3050)
- **Python**: 3.12+
- **Tools**: [uv](https://docs.astral.sh/uv/)

## 🏃 Quick Start

1. **Install Dependencies**:
   ```bash
   uv sync --prerelease=allow
   ```
2. **Start Nova**:
   ```bash
   uv run python nova/backend/main.py
   ```
3. **Access Dashboard**:
   Navigate to `http://localhost:8000` and enable **"ALWAYS LISTENING"**.

## 🏗 Directory Structure

- `nova/backend/`: FastAPI server, session management, and model loading.
- `nova/backend/kws/`: Wake-word engine, MLP classifier, and reference samples.
- `nova/frontend/`: WebSocket-based dashboard and sequential audio playback logic.
- `pocket-tts/`: (Submodule) High-performance TTS synthesis.
