from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static

VP_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Nova, navigate me home and play some music.",
    "What is the weather forecast for tomorrow?",
    "Turn on the AC and set temperature to twenty-two degrees.",
    "Order me a coffee from the nearest café.",
]


class EnrollPanel(Vertical):
    """Five voice-enrollment buttons + status line.

    Emits EnrollPanel.RecordRequested(index, total, phrase) when a button
    is pressed. The app records audio, uploads it, then calls mark_done()
    or mark_failed() on this widget.
    """

    class RecordRequested(Message):
        def __init__(self, index: int, total: int, phrase: str) -> None:
            super().__init__()
            self.index = index
            self.total = total
            self.phrase = phrase

    def compose(self) -> ComposeResult:
        yield Static("Voice Enrollment", classes="h3")
        yield Static(
            "Record 5 phrases to enroll your voice.",
            id="vp-hint",
            classes="hint",
        )
        for i, phrase in enumerate(VP_PROMPTS, 1):
            yield Button(f'"{phrase}"', id=f"vp-btn-{i}", classes="vp-btn")
        yield Static("", id="vp-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith("vp-btn-"):
            return
        try:
            idx = int(bid.split("-")[-1])
        except ValueError:
            return
        phrase = VP_PROMPTS[idx - 1]
        event.button.add_class("-recording")
        event.button.label = "recording…"
        self.query_one("#vp-status", Static).update(f"Recording phrase {idx}…")
        self.post_message(self.RecordRequested(idx, len(VP_PROMPTS), phrase))

    def mark_done(self, idx: int, all_done: bool = False) -> None:
        btn = self.query_one(f"#vp-btn-{idx}", Button)
        btn.remove_class("-recording")
        btn.add_class("-done")
        btn.label = f'✓ "{VP_PROMPTS[idx - 1]}"'
        status = self.query_one("#vp-status", Static)
        if all_done:
            status.update("Voiceprint enrolled! Restart Nova to activate pVAD.")
            status.add_class("-enrolled")
            self.query_one("#vp-hint", Static).update(
                "Voice enrolled. Restart server to load pVAD speaker gate."
            )
            for i in range(1, len(VP_PROMPTS) + 1):
                b = self.query_one(f"#vp-btn-{i}", Button)
                b.add_class("-done")
                b.label = f'✓ "{VP_PROMPTS[i - 1]}"'
        else:
            status.update(f"{idx}/{len(VP_PROMPTS)} recorded")

    def mark_failed(self, idx: int, err: str) -> None:
        btn = self.query_one(f"#vp-btn-{idx}", Button)
        btn.remove_class("-recording")
        btn.label = f'"{VP_PROMPTS[idx - 1]}"'
        self.query_one("#vp-status", Static).update(f"Error: {err}")

    def mark_already_enrolled(self) -> None:
        self.query_one("#vp-hint", Static).update(
            "Voice already enrolled. pVAD speaker gate active on restart."
        )
        status = self.query_one("#vp-status", Static)
        status.update("Enrolled ✓")
        status.add_class("-enrolled")
