from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, ProgressBar, Static, Tab, Tabs

from nova.tui.audio import (
    MIC_SR,
    MicCapture,
    PCMPlayer,
    pcm_i16_to_wav_bytes,
    record_blocking,
)
from nova.tui.client import NovaClient, ServerConfig
from nova.tui.widgets.enrollment import VP_PROMPTS, EnrollPanel
from nova.tui.widgets.turn import Turn


STATE_LABEL = {
    "idle": "idle",
    "listen": "listening",
    "think": "thinking…",
    "speak": "speaking…",
}


class NovaTUI(App):
    """Textual mirror of nova/backend/static/ web UI."""

    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("space", "interrupt", "Interrupt", show=True),
        Binding("m", "toggle_mic", "Mic on/off", show=True),
        Binding("r", "reset", "Reset session", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    state: reactive[str] = reactive("idle")
    mic_on: reactive[bool] = reactive(False)
    turns_count: reactive[int] = reactive(0)

    def __init__(self, cfg: ServerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.client = NovaClient(cfg)
        self.player = PCMPlayer()
        self.mic: MicCapture | None = None
        self._event_task: asyncio.Task | None = None
        self._assistant_turn: Turn | None = None
        self._t0: float = 0.0
        self._current_doc = "user"
        self._captured_loop: asyncio.AbstractEventLoop | None = None

    # ── Layout ──────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static("NOVA  ·  in-car voice agent", id="brand")
            with Horizontal(id="pill"):
                yield Static("🔐", id="pvad-badge", classes="-hidden")
                yield Static("●", id="status-dot", classes="dot-idle")
                yield Static("idle", id="status-text")

        with Horizontal(id="body"):
            yield VerticalScroll(id="turns")
            with Vertical(id="sidebar"):
                yield Static("Session", classes="h3")
                with Horizontal(classes="row"):
                    yield Static("Mic level", classes="k")
                    yield Static("—", id="lvl", classes="v")
                yield ProgressBar(
                    total=100, show_eta=False, show_percentage=False, id="meter"
                )
                with Horizontal(classes="row"):
                    yield Static("Last latency", classes="k")
                    yield Static("—", id="lat", classes="v")
                with Horizontal(classes="row"):
                    yield Static("Turns", classes="k")
                    yield Static("0", id="nTurns", classes="v")
                yield Static("", id="err")

                yield Static("Persona", classes="h3")
                yield Tabs(
                    Tab("USER", id="tab-user"),
                    Tab("MEMORY", id="tab-memory"),
                    Tab("CHAT", id="tab-conversation"),
                    id="tabs",
                )
                yield Static("(loading…)", id="doc")

                yield EnrollPanel()

        with Horizontal(id="footer"):
            yield Static("Click mic or press m. Space to interrupt.", id="hint-left")
            yield Button("🎙  MIC", id="mic-btn")
            yield Static("q to quit · r to reset", id="hint-right")

    # ── Lifecycle ───────────────────────────────────────────────────────────
    async def on_mount(self) -> None:
        self._captured_loop = asyncio.get_running_loop()
        self.player.start()
        await self._refresh_persona("user")
        pvad = await self.client.pvad_status()
        if pvad.get("loaded"):
            self.query_one("#pvad-badge", Static).remove_class("-hidden")
        enroll = await self.client.enroll_check()
        if enroll.get("voice_enrolled"):
            self.query_one(EnrollPanel).mark_already_enrolled()

    async def on_unmount(self) -> None:
        await self._stop_mic_session()
        self.player.stop()
        await self.client.close()

    # ── State plumbing ──────────────────────────────────────────────────────
    def watch_state(self, value: str) -> None:
        dot = self.query_one("#status-dot", Static)
        dot.set_classes(f"dot-{value}")
        self.query_one("#status-text", Static).update(STATE_LABEL.get(value, value))

    def watch_mic_on(self, value: bool) -> None:
        btn = self.query_one("#mic-btn", Button)
        if value:
            btn.add_class("-on")
            btn.label = "■  STOP"
        else:
            btn.remove_class("-on")
            btn.label = "🎙  MIC"

    def watch_turns_count(self, value: int) -> None:
        self.query_one("#nTurns", Static).update(str(value))

    # ── Mic on/off ──────────────────────────────────────────────────────────
    async def _start_mic_session(self) -> None:
        try:
            await self.client.connect()
        except Exception as e:
            self._show_error(f"connect failed: {e}")
            return
        self._event_task = asyncio.create_task(self._event_loop())
        self.mic = MicCapture(
            on_pcm=self._on_pcm_from_mic,
            on_peak=self._on_peak_from_mic,
        )
        try:
            self.mic.start()
        except Exception as e:
            self._show_error(f"mic error: {e}")
            await self._stop_mic_session()
            return
        self.mic_on = True
        self.state = "listen"

    async def _stop_mic_session(self) -> None:
        if self.mic is not None:
            self.mic.stop()
            self.mic = None
        if self._event_task is not None:
            self._event_task.cancel()
            self._event_task = None
        await self.client.close()
        # re-create http client for subsequent REST calls
        self.client = NovaClient(self.cfg)
        self.mic_on = False
        self.state = "idle"

    def _on_pcm_from_mic(self, pcm: bytes) -> None:
        # Called from sounddevice thread. Hop to the asyncio loop.
        loop = self._captured_loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.client.send_pcm(pcm), loop)

    def _on_peak_from_mic(self, peak: float) -> None:
        pct = min(100, int(peak * 200))
        self.call_from_thread(self._update_meter, pct)

    def _update_meter(self, pct: int) -> None:
        self.query_one("#meter", ProgressBar).update(progress=pct)
        self.query_one("#lvl", Static).update(f"{pct}%")

    # ── Event stream from server ────────────────────────────────────────────
    async def _event_loop(self) -> None:
        try:
            async for evt in self.client.events():
                await self._on_event(evt)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._show_error(f"ws error: {e}")
            self.state = "idle"

    async def _on_event(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "speech_started":
            self.state = "listen"
        elif t == "transcript":
            text = evt.get("data") or ""
            self._add_turn("user", text)
            self._t0 = time.perf_counter()
            self.state = "think"
        elif t == "assistant_start":
            self._assistant_turn = self._add_turn("assistant", "")
        elif t == "llm_token":
            if self._assistant_turn is not None:
                self._assistant_turn.append(evt.get("data") or "")
            if self._t0:
                ms = int((time.perf_counter() - self._t0) * 1000)
                self.query_one("#lat", Static).update(f"{ms} ms")
                self._t0 = 0.0
            self.state = "speak"
        elif t == "generation_done":
            self._assistant_turn = None
            self.state = "listen"
            await self._refresh_persona(self._current_doc)
        elif t == "audio_pcm":
            self.player.play(evt["pcm"], int(evt["sr"]))
        elif t == "audio_out":
            import base64

            pcm = base64.b64decode(evt.get("pcm_b64") or "")
            self.player.play(pcm, int(evt.get("sr") or 24000))
        elif t == "error":
            self._show_error(evt.get("data") or "error")

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _add_turn(self, who: str, text: str) -> Turn:
        turn = Turn(who, text)
        scroller = self.query_one("#turns", VerticalScroll)
        scroller.mount(turn)
        scroller.scroll_end(animate=False)
        self.turns_count = self.turns_count + 1
        return turn

    async def _refresh_persona(self, name: str) -> None:
        content = await self.client.get_persona(name)
        self.query_one("#doc", Static).update(content)

    def _show_error(self, msg: str) -> None:
        err = self.query_one("#err", Static)
        err.update(msg)
        self.set_timer(4.0, lambda: err.update(""))

    # ── Actions ─────────────────────────────────────────────────────────────
    async def action_toggle_mic(self) -> None:
        if self.mic_on:
            await self._stop_mic_session()
        else:
            await self._start_mic_session()

    def action_interrupt(self) -> None:
        if not self.mic_on:
            return
        self.player.interrupt()
        self.state = "listen"

    async def action_reset(self) -> None:
        if self.mic_on:
            await self.client.reset()
        self.query_one("#turns", VerticalScroll).remove_children()
        self.turns_count = 0

    # ── Tabs + buttons ──────────────────────────────────────────────────────
    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        tab_id = event.tab.id or ""
        name = tab_id.removeprefix("tab-") or "user"
        self._current_doc = name
        await self._refresh_persona(name)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mic-btn":
            await self.action_toggle_mic()

    async def on_enroll_panel_record_requested(
        self, msg: EnrollPanel.RecordRequested
    ) -> None:
        panel = self.query_one(EnrollPanel)
        try:
            pcm = await asyncio.to_thread(record_blocking, 3.5, MIC_SR)
            wav = pcm_i16_to_wav_bytes(pcm, MIC_SR)
            resp = await self.client.enroll_voice_sample(
                wav, msg.index, msg.total, driver_id="driver1"
            )
            status = resp.get("status")
            if status == "enrolled":
                panel.mark_done(msg.index, all_done=True)
            elif status == "ok":
                panel.mark_done(msg.index, all_done=False)
            else:
                panel.mark_failed(msg.index, resp.get("error") or "unknown")
        except Exception as e:
            panel.mark_failed(msg.index, str(e))


def main() -> None:
    p = argparse.ArgumentParser(prog="nova-tui")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8001, type=int)
    p.add_argument("--tls", action="store_true", help="use https/wss")
    args = p.parse_args()
    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        scheme_http="https" if args.tls else "http",
        scheme_ws="wss" if args.tls else "ws",
    )
    NovaTUI(cfg).run()


if __name__ == "__main__":
    main()
