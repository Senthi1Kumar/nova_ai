#!/usr/bin/env python3
"""WebSocket loadtest client for nova_loop.py — measures TTFB percentiles.

Connects N concurrent WS sessions, plays a prerecorded WAV (or synthetic noise)
through each at real-time rate, listens for nova_loop events, and reports
per-stage timing percentiles. Use to measure the impact of code/config changes
objectively instead of eyeballing a single log.

Inspired by unmute's loadtest_client.py but adapted to nova_loop's protocol.

Usage:
    uv run nova/backend/loadtest_client.py --url ws://localhost:8001/ws --workers 4 --turns 3
    uv run nova/backend/loadtest_client.py --wav /path/to/16k_mono.wav --workers 1

Required deps already in pyproject: websockets, numpy. Optional: soundfile (for WAV).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK_MS = 20                       # 20 ms of audio per WS send (320 samples @ 16k)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
DEFAULT_PHRASE_SAMPLES = SAMPLE_RATE * 2  # 2 seconds of synthetic audio


@dataclass
class TurnTiming:
    """Per-turn latency in ms (None if event never arrived)."""
    audio_end_at: float = 0.0       # local time we stopped sending audio
    speech_started_ms: float | None = None
    transcript_ms: float | None = None
    generation_start_ms: float | None = None
    first_audio_ms: float | None = None
    generation_done_ms: float | None = None
    turn_metrics: dict = field(default_factory=dict)


def _gen_synthetic_audio(seconds: float = 2.0) -> np.ndarray:
    """500 ms silence + 1 s amplitude-modulated noise + 500 ms silence (16 kHz mono int16).

    Not real speech — won't transcribe to anything meaningful — but triggers
    the VAD speech_start → speech_end cycle so the pipeline runs end-to-end.
    """
    silence_samples = SAMPLE_RATE // 2
    noise_samples = int(SAMPLE_RATE * (seconds - 1.0))
    noise = (np.random.normal(0, 0.15, noise_samples) * 16000).astype(np.int16)
    # Apply a slow envelope so it looks more speech-like
    env = np.sin(np.linspace(0, 3.14, noise_samples)) ** 2
    noise = (noise.astype(np.float32) * env).astype(np.int16)
    silence = np.zeros(silence_samples, dtype=np.int16)
    return np.concatenate([silence, noise, silence])


def _load_wav(path: Path) -> np.ndarray:
    """Load 16 kHz mono int16 WAV. Resamples crudely if needed."""
    with wave.open(str(path), "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise SystemExit(f"WAV must be 16-bit PCM (got {sw*8}-bit): {path}")
    pcm = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        # Crude linear resample — good enough for loadtest timing measurement
        ratio = SAMPLE_RATE / sr
        new_len = int(len(pcm) * ratio)
        idx = np.linspace(0, len(pcm) - 1, new_len)
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)
    return pcm


async def _send_audio(ws, audio_i16: np.ndarray) -> float:
    """Stream audio at real-time rate. Returns local time when the last chunk left."""
    chunk_period = CHUNK_MS / 1000.0
    next_send = asyncio.get_event_loop().time()
    for i in range(0, len(audio_i16), CHUNK_SAMPLES):
        chunk = audio_i16[i:i + CHUNK_SAMPLES]
        await ws.send(chunk.tobytes())
        next_send += chunk_period
        delay = next_send - asyncio.get_event_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)
    return time.time()


async def _run_session(worker_id: int, url: str, audio: np.ndarray,
                       turns: int) -> list[TurnTiming]:
    """One WS session, multiple turns. Records timing for each turn."""
    results: list[TurnTiming] = []
    try:
        async with websockets.connect(url, max_size=None) as ws:
            for turn_idx in range(turns):
                t = TurnTiming()
                send_task = asyncio.create_task(_send_audio(ws, audio))
                # Listen until generation_done or 30s timeout
                deadline = time.time() + 30.0
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
                    except asyncio.TimeoutError:
                        break
                    if isinstance(msg, bytes):
                        # Binary audio frame from gateway
                        if t.first_audio_ms is None and t.audio_end_at:
                            t.first_audio_ms = (time.time() - t.audio_end_at) * 1000
                        continue
                    try:
                        ev = json.loads(msg)
                    except (ValueError, TypeError):
                        continue
                    if send_task.done() and not t.audio_end_at:
                        t.audio_end_at = send_task.result()
                    et = ev.get("type")
                    now_ms_from_end = (time.time() - t.audio_end_at) * 1000 if t.audio_end_at else 0
                    if et == "speech_started" and t.speech_started_ms is None:
                        t.speech_started_ms = now_ms_from_end
                    elif et == "transcript" and t.transcript_ms is None:
                        t.transcript_ms = now_ms_from_end
                    elif et == "generation_start" and t.generation_start_ms is None:
                        t.generation_start_ms = now_ms_from_end
                    elif et == "audio_header" and t.first_audio_ms is None:
                        t.first_audio_ms = now_ms_from_end
                    elif et == "turn_metrics":
                        t.turn_metrics = ev.get("data") or {}
                    elif et == "generation_done":
                        t.generation_done_ms = now_ms_from_end
                        break
                if not send_task.done():
                    send_task.cancel()
                results.append(t)
                print(f"[worker {worker_id} turn {turn_idx + 1}/{turns}] "
                      f"first_audio={t.first_audio_ms} done={t.generation_done_ms} "
                      f"metrics={t.turn_metrics}")
    except Exception as e:
        print(f"[worker {worker_id}] session failed: {e}")
    return results


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return statistics.quantiles(sorted(values), n=100)[int(p) - 1] if len(values) >= 2 else values[0]


def _summarize(all_turns: list[TurnTiming]) -> None:
    def col(field: str) -> list[float]:
        return [getattr(t, field) for t in all_turns if getattr(t, field) is not None]

    print(f"\n=== Summary across {len(all_turns)} turns ===")
    for field_name in ("speech_started_ms", "transcript_ms", "generation_start_ms",
                       "first_audio_ms", "generation_done_ms"):
        values = col(field_name)
        if not values:
            print(f"  {field_name:24s} no samples")
            continue
        print(f"  {field_name:24s} n={len(values):3d}  "
              f"p50={_pct(values, 50):7.0f} ms  "
              f"p95={_pct(values, 95):7.0f} ms  "
              f"max={max(values):7.0f} ms")
    # Also surface server-side turn_metrics if they came through
    server_metrics: dict[str, list[float]] = {}
    for t in all_turns:
        for k, v in t.turn_metrics.items():
            if isinstance(v, (int, float)):
                server_metrics.setdefault(k, []).append(float(v))
    if server_metrics:
        print("\n  server turn_metrics:")
        for k in sorted(server_metrics):
            v = server_metrics[k]
            print(f"    {k:22s} n={len(v):3d}  "
                  f"p50={_pct(v, 50):7.0f}  p95={_pct(v, 95):7.0f}  max={max(v):7.0f}")


async def _main(args) -> None:
    if args.wav:
        audio = _load_wav(Path(args.wav))
        print(f"Loaded WAV: {args.wav} → {len(audio) / SAMPLE_RATE:.2f} s @ 16 kHz")
    else:
        audio = _gen_synthetic_audio(seconds=2.0)
        print(f"Using synthetic audio: {len(audio) / SAMPLE_RATE:.2f} s @ 16 kHz "
              f"(set --wav for realistic transcription)")

    print(f"Launching {args.workers} worker(s) × {args.turns} turn(s) against {args.url}")
    tasks = [_run_session(i, args.url, audio, args.turns) for i in range(args.workers)]
    results = await asyncio.gather(*tasks)
    all_turns = [t for worker_results in results for t in worker_results]
    _summarize(all_turns)


def main() -> None:
    ap = argparse.ArgumentParser(description="Nova WS loadtest client")
    ap.add_argument("--url", default="ws://localhost:8001/ws")
    ap.add_argument("--workers", type=int, default=1, help="concurrent WS sessions")
    ap.add_argument("--turns", type=int, default=3, help="turns per session")
    ap.add_argument("--wav", help="16-bit PCM WAV file (any sr, mono preferred). "
                                  "Synthetic noise is used if omitted.")
    asyncio.run(_main(ap.parse_args()))


if __name__ == "__main__":
    main()
