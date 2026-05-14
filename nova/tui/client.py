from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx
import websockets


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    scheme_http: str = "http"
    scheme_ws: str = "ws"

    @property
    def http_base(self) -> str:
        return f"{self.scheme_http}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"{self.scheme_ws}://{self.host}:{self.port}/ws"


class NovaClient:
    """WebSocket + REST client against the nova_loop FastAPI server.

    Mirrors what static/app.js does: binary PCM up, JSON events + binary
    audio down, plus the /persona, /enroll, /pVAD REST endpoints.
    """

    def __init__(self, cfg: ServerConfig | None = None) -> None:
        self.cfg = cfg or ServerConfig()
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._http = httpx.AsyncClient(base_url=self.cfg.http_base, timeout=10.0)
        self._pending_audio_sr: int = 0

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self.cfg.ws_url, max_size=None, ping_interval=20
        )

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        await self._http.aclose()

    async def send_pcm(self, pcm_bytes: bytes) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(pcm_bytes)
        except websockets.ConnectionClosed:
            pass

    async def send_json(self, obj: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(obj))

    async def reset(self) -> None:
        await self.send_json({"type": "reset"})

    async def events(self) -> AsyncIterator[dict]:
        """Yield normalized event dicts.

        JSON frames pass through unchanged.
        Binary audio frames are wrapped as {"type": "audio_pcm", "sr": <sr>,
        "pcm": <bytes>} using the most recent audio_header sample rate.
        """
        if self._ws is None:
            raise RuntimeError("not connected")
        async for msg in self._ws:
            if isinstance(msg, (bytes, bytearray)):
                sr = self._pending_audio_sr or 24000
                self._pending_audio_sr = 0
                yield {"type": "audio_pcm", "sr": sr, "pcm": bytes(msg)}
                continue
            try:
                obj = json.loads(msg)
            except Exception:
                continue
            if obj.get("type") == "audio_header":
                self._pending_audio_sr = int(obj.get("sr") or 24000)
            yield obj

    # ── REST ────────────────────────────────────────────────────────────────
    async def get_persona(self, name: str) -> str:
        try:
            r = await self._http.get(f"/persona/{name}")
            return (r.json() or {}).get("content") or "(empty)"
        except Exception as e:
            return f"(failed: {e})"

    async def pvad_status(self) -> dict:
        try:
            r = await self._http.get("/pVAD/status")
            return r.json()
        except Exception:
            return {"loaded": False}

    async def enroll_check(self) -> dict:
        try:
            r = await self._http.get("/enroll/check")
            return r.json()
        except Exception:
            return {"voice_enrolled": False}

    async def enroll_voice_sample(
        self,
        wav_bytes: bytes,
        index: int,
        total: int,
        driver_id: str = "driver1",
    ) -> dict:
        files = {"file": (f"voice_{index}.wav", wav_bytes, "audio/wav")}
        data = {"index": str(index), "total": str(total), "driver_id": driver_id}
        try:
            r = await self._http.post(
                "/enroll/voice-sample", data=data, files=files
            )
            return r.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
