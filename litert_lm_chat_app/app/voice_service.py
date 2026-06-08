from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Iterator, Optional

from app.config import Settings

try:
    from pocket_tts import TTSModel
except Exception:
    TTSModel = None  # type: ignore

try:
    import scipy.io.wavfile
except Exception:
    scipy = None  # type: ignore


class VoiceNotReady(RuntimeError):
    pass


class PocketTTSService:
    """Pocket-TTS service. Streaming PCM via `synthesize_stream`; legacy file
    output via `synthesize` for the text-only /api/tts route."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.voice_state = None
        self.ready_error: Optional[str] = None
        self.output_dir = Path(settings.tts_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.settings.tts_enabled:
            raise VoiceNotReady("TTS is disabled. Set TTS_ENABLED=true in .env")
        if self.model is not None and self.voice_state is not None:
            return
        if TTSModel is None:
            raise VoiceNotReady("Pocket TTS is not installed. Run: pip install pocket-tts scipy")
        if scipy is None:
            raise VoiceNotReady("scipy is not installed. Run: pip install scipy")
        try:
            try:
                self.model = TTSModel.load_model(language=self.settings.pocket_tts_language)
            except TypeError:
                self.model = TTSModel.load_model()
            self.voice_state = self.model.get_state_for_audio_prompt(self.settings.pocket_tts_voice)
            self.ready_error = None
        except Exception as exc:
            self.ready_error = str(exc)
            raise VoiceNotReady(f"Failed to load Pocket TTS: {exc}") from exc

    def warmup(self) -> None:
        """Eager-load the model + voice state so the first user turn doesn't
        pay the ~5-7s cold-start (HF download + safetensors decode + voice
        prompt encode). Safe to call repeatedly; subsequent calls are no-ops."""
        if not self.settings.tts_enabled:
            return
        try:
            self._load()
        except VoiceNotReady:
            # Don't block app startup if TTS happens to be misconfigured;
            # the first synthesize call will re-raise with the same error.
            pass

    def synthesize_stream(self, text: str) -> Iterator[tuple[bytes, int]]:
        """Yield (pcm_s16le_bytes, sample_rate) chunks for `text`."""
        clean = (text or "").strip()
        if not clean:
            return
        self._load()
        sr = int(self.model.sample_rate)
        for tensor in self.model.generate_audio_stream(self.voice_state, clean):
            arr = tensor.detach().cpu().numpy()
            arr = arr.clip(-1.0, 1.0)
            pcm = (arr * 32767.0).astype("<i2")
            yield pcm.tobytes(), sr

    def synthesize(self, text: str) -> str:
        """Synthesize to a WAV file and return the /audio/<name> URL."""
        clean = (text or "").strip()
        if not clean:
            raise VoiceNotReady("Cannot synthesize an empty response")
        self._load()
        try:
            clean = clean[:4000]
            audio = self.model.generate_audio(self.voice_state, clean)
            filename = f"reply_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
            out_path = self.output_dir / filename
            scipy.io.wavfile.write(str(out_path), self.model.sample_rate, audio.detach().cpu().numpy())
            return f"/audio/{filename}"
        except Exception as exc:
            raise VoiceNotReady(f"Pocket TTS synthesis failed: {exc}") from exc

    def health(self) -> dict:
        return {
            "enabled": self.settings.tts_enabled,
            "available": TTSModel is not None,
            "loaded": self.model is not None,
            "voice": self.settings.pocket_tts_voice,
            "language": self.settings.pocket_tts_language,
            "error": self.ready_error,
        }
