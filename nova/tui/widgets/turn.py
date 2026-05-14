from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class Turn(Vertical):
    """One chat bubble: who-label + body text. Body is appendable."""

    def __init__(self, who: str, text: str = "") -> None:
        super().__init__()
        self.who = who
        self._text = text
        self.add_class(f"turn-{who}")
        self.add_class("turn")

    def compose(self) -> ComposeResult:
        yield Static(self.who.upper(), classes="turn-who")
        yield Static(self._text, id="turn-body")

    def append(self, chunk: str) -> None:
        body = self.query_one("#turn-body", Static)
        sep = " " if self._text and not self._text.endswith(("\n", " ")) else ""
        self._text = f"{self._text}{sep}{chunk}"
        body.update(self._text)

    def set_text(self, text: str) -> None:
        self._text = text
        self.query_one("#turn-body", Static).update(text)
