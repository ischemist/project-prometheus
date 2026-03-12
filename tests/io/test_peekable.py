"""
Unit tests for PeekableIterator.

All tests are pure in-memory and run in <1ms — no file I/O, no external deps.
"""

import pytest

from calcflow.io.peekable import PeekableIterator


def make(lines: list[str]) -> PeekableIterator:
    return PeekableIterator(iter(lines))


# ---------------------------------------------------------------------------
# Basic iteration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_basic_iteration():
    """Plain for-loop iteration works correctly."""
    result = list(make(["a", "b", "c"]))
    assert result == ["a", "b", "c"]


@pytest.mark.unit
def test_empty_iterator():
    """Empty iterator yields nothing and doesn't raise."""
    assert list(make([])) == []


# ---------------------------------------------------------------------------
# push_back — LIFO ordering guarantee
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_push_back_single():
    """A pushed-back line is returned before the remaining source lines."""
    it = make(["b", "c"])
    it.push_back("a")
    assert list(it) == ["a", "b", "c"]


@pytest.mark.unit
def test_push_back_lifo_order():
    """Multiple push_back calls are LIFO: last pushed is returned first."""
    it = make(["c"])
    it.push_back("first_pushed")
    it.push_back("second_pushed")
    assert next(it) == "second_pushed"
    assert next(it) == "first_pushed"
    assert next(it) == "c"


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_peek_does_not_consume():
    """peek() returns the next line without advancing the iterator."""
    it = make(["x", "y"])
    assert it.peek() == "x"
    assert it.peek() == "x"  # still "x" — not consumed
    assert next(it) == "x"  # now consumed
    assert next(it) == "y"


@pytest.mark.unit
def test_peek_at_eof():
    """peek() returns None when the iterator is exhausted."""
    it = make([])
    assert it.peek() is None


@pytest.mark.unit
def test_peek_after_exhaust():
    """peek() returns None after all lines have been consumed."""
    it = make(["only"])
    next(it)
    assert it.peek() is None


# ---------------------------------------------------------------------------
# skip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_skip_one():
    """skip() discards exactly one line by default."""
    it = make(["a", "b", "c"])
    it.skip()
    assert list(it) == ["b", "c"]


@pytest.mark.unit
def test_skip_n():
    """skip(n) discards exactly n lines."""
    it = make(["a", "b", "c", "d"])
    it.skip(2)
    assert list(it) == ["c", "d"]


@pytest.mark.unit
def test_skip_more_than_remaining():
    """skip(n) silently stops at EOF when n exceeds remaining lines."""
    it = make(["a", "b"])
    it.skip(100)  # should not raise
    assert list(it) == []


@pytest.mark.unit
def test_skip_zero():
    """skip(0) discards nothing."""
    it = make(["a", "b"])
    it.skip(0)
    assert list(it) == ["a", "b"]


# ---------------------------------------------------------------------------
# take_while
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_take_while_normal():
    """take_while collects lines while predicate holds and pushes back the first non-match."""
    it = make(["aa", "bb", "c", "dd"])
    result = it.take_while(lambda ln: len(ln) == 2)
    assert result == ["aa", "bb"]
    assert list(it) == ["c", "dd"]  # "c" was pushed back


@pytest.mark.unit
def test_take_while_stop_on_first_line():
    """take_while returns [] and pushes back the first line when it immediately fails."""
    it = make(["stop", "go", "go"])
    result = it.take_while(lambda ln: ln != "stop")
    assert result == []
    assert next(it) == "stop"  # pushed back, still available


@pytest.mark.unit
def test_take_while_all_match():
    """take_while consumes all lines when every line satisfies the predicate."""
    it = make(["a", "b", "c"])
    result = it.take_while(lambda ln: True)
    assert result == ["a", "b", "c"]
    assert it.peek() is None


# ---------------------------------------------------------------------------
# take_until
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_take_until_normal():
    """take_until collects lines until predicate is True and pushes back the matching line."""
    it = make(["a", "b", "STOP", "c"])
    result = it.take_until(lambda ln: ln == "STOP")
    assert result == ["a", "b"]
    assert next(it) == "STOP"  # pushed back


@pytest.mark.unit
def test_take_until_stop_on_first_line():
    """take_until returns [] and pushes back the first line when it immediately matches."""
    it = make(["STOP", "a", "b"])
    result = it.take_until(lambda ln: ln == "STOP")
    assert result == []
    assert next(it) == "STOP"


@pytest.mark.unit
def test_take_until_no_match():
    """take_until consumes all lines when the predicate never becomes True."""
    it = make(["a", "b", "c"])
    result = it.take_until(lambda ln: ln == "NEVER")
    assert result == ["a", "b", "c"]
    assert it.peek() is None
