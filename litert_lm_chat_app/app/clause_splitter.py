from __future__ import annotations

_TERMINAL_PUNCT = {".", "!", "?"}


def maybe_flush_clause(
    buf: list[str],
    min_chars: int = 8,
    comma_min_chars: int = 32,
) -> tuple[str | None, list[str]]:
    """Decide whether the current token buffer should be flushed as a clause.

    Rules:
      - Terminal punctuation (".", "!", "?") flushes once buffer length >= min_chars.
      - Comma flushes only when buffer length >= comma_min_chars (avoid over-splitting).
      - Otherwise return (None, buf) — caller keeps appending.

    Returns:
      (flushed_clause_or_None, new_buf).
    """
    if not buf:
        return None, buf
    joined = "".join(buf)
    last = joined[-1]
    if last in _TERMINAL_PUNCT and len(joined) >= min_chars:
        return joined, []
    if last == "," and len(joined) >= comma_min_chars:
        return joined, []
    return None, buf
