from app.clause_splitter import maybe_flush_clause


def _drive(tokens, min_chars=8, comma_min_chars=32):
    """Feed tokens one at a time; collect flushed clauses + final remainder."""
    buf: list[str] = []
    flushed: list[str] = []
    for tok in tokens:
        buf.append(tok)
        clause, buf = maybe_flush_clause(buf, min_chars=min_chars, comma_min_chars=comma_min_chars)
        if clause:
            flushed.append(clause)
    tail = "".join(buf).strip()
    return flushed, tail


def test_short_reply_no_flush_until_final_punctuation():
    flushed, tail = _drive(["Hi", "!"])
    assert flushed == []
    assert tail == "Hi!"


def test_single_sentence_flushes_on_period():
    flushed, tail = _drive(list("The capital of France is Paris."))
    assert flushed == ["The capital of France is Paris."]
    assert tail == ""


def test_multi_sentence_streams_clause_by_clause():
    text = "Sure thing. Paris is in France. It is beautiful."
    flushed, tail = _drive(list(text))
    assert flushed == [
        "Sure thing.",
        " Paris is in France.",
        " It is beautiful.",
    ]
    assert tail == ""


def test_comma_below_threshold_does_not_flush():
    flushed, tail = _drive(list("Hi, there."))
    assert flushed == ["Hi, there."]
    assert tail == ""


def test_comma_above_threshold_flushes():
    text = "Now this is a sufficiently long opening clause, and then more text."
    flushed, tail = _drive(list(text))
    assert flushed[0].endswith(",")
    assert "and then more text." in (flushed[-1] if not tail else tail)


def test_question_mark_and_exclamation_flush():
    flushed, _ = _drive(list("Are you sure? Yes I am!"))
    assert flushed == ["Are you sure?", " Yes I am!"]


def test_no_punctuation_keeps_buffering():
    flushed, tail = _drive(list("just words with no end yet"))
    assert flushed == []
    assert tail == "just words with no end yet"
