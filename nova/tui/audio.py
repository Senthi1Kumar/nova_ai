from __future__ import annotations

import io
import queue
import threading
import wave
from typing import Callable

import numpy as np
import sounddevice as sd

MIC_SR = 16000
MIC_BLOCK = 320  # 20 ms @ 16 kHz, matches the worklet cadence loosely


class MicCapture:
    """Continuous 16 kHz mono i16 mic capture.

    Calls on_pcm(bytes) and on_peak(float 0..1) from a background thread.
    Use Textual's app.call_from_thread() in callbacks if updating UI.
    """

    def __init__(
        self,
        on_pcm: Callable[[bytes], None],
        on_peak: Callable[[float], None],
        sample_rate: int = MIC_SR,
        block: int = MIC_BLOCK,
    ) -> None:
        self.on_pcm = on_pcm
        self.on_peak = on_peak
        self.sample_rate = sample_rate
        self.block = block
        self._stream: sd.InputStream | None = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if not self._running:
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        i16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        try:
            self.on_pcm(i16.tobytes())
            self.on_peak(peak)
        except Exception:
            pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class PCMPlayer:
    """Threaded queue-fed PCM player.

    Accepts (i16_bytes, sample_rate) chunks. Opens/reopens the output stream
    when the sample rate changes. interrupt() drops everything queued and
    closes the active stream, mirroring the spacebar barge-in in app.js.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[bytes, int] | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._sr: int = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_stream()

    def play(self, pcm_i16: bytes, sample_rate: int) -> None:
        self._q.put((pcm_i16, sample_rate))

    def interrupt(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._close_stream()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._sr = 0

    def _ensure_stream(self, sr: int) -> None:
        if self._stream is not None and self._sr == sr:
            return
        self._close_stream()
        self._stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="int16"
        )
        self._stream.start()
        self._sr = sr

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            pcm, sr = item
            try:
                self._ensure_stream(sr)
                arr = np.frombuffer(pcm, dtype=np.int16)
                if self._stream is not None:
                    self._stream.write(arr)
            except Exception:
                self._close_stream()


def pcm_i16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw i16 PCM as a WAV file in memory (for /enroll upload)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def record_blocking(seconds: float, sample_rate: int = MIC_SR) -> bytes:
    """Synchronous one-shot recording. Returns raw i16 PCM bytes."""
    frames = int(seconds * sample_rate)
    rec = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    return rec.tobytes()
