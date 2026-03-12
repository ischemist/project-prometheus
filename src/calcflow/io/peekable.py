"""
A peekable line iterator for use by block parsers.

Replaces the old `state.buffered_line` hack: instead of storing an over-read
line on the shared ParseState (invisible to the caller, easy to forget), parsers
push it back onto the iterator itself via `push_back()`.  The core loop can then
be a plain `for line in iterator:` with no special buffering logic.
"""

from collections.abc import Callable, Iterator
from typing import Self


class PeekableIterator:
    """
    A line iterator that supports single-line peek and push-back.

    Wraps a plain ``Iterator[str]`` and adds:

    * ``peek()``  — look at the next line without consuming it (returns ``None`` at EOF)
    * ``push_back(line)`` — put a line back so it is returned by the next ``__next__`` call
    * ``take_while(predicate)`` — consume lines while predicate holds; pushes back the
      first non-matching line so the caller never has to call ``push_back`` manually
    * ``take_until(predicate)`` — consume lines *until* predicate holds (pushes back the
      matching line)
    * ``skip(n)`` — discard the next *n* lines (silently stops at EOF)

    The internal push-back stack is a list used as a LIFO queue, so multiple
    consecutive ``push_back`` calls work correctly (last pushed == first returned).
    """

    __slots__ = ("_iterator", "_stack")

    def __init__(self, iterator: Iterator[str]) -> None:
        self._iterator = iterator
        self._stack: list[str] = []

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        if self._stack:
            return self._stack.pop()
        return next(self._iterator)

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def peek(self) -> str | None:
        """Return the next line without consuming it, or ``None`` at EOF."""
        try:
            line = next(self)
        except StopIteration:
            return None
        self._stack.append(line)
        return line

    def push_back(self, line: str) -> None:
        """Return *line* to the front of the iterator (LIFO)."""
        self._stack.append(line)

    def skip(self, n: int = 1) -> None:
        """Silently discard the next *n* lines (stops at EOF without raising)."""
        for _ in range(n):
            try:
                next(self)
            except StopIteration:
                return

    def take_while(self, predicate: Callable[[str], bool]) -> list[str]:
        """
        Consume and return lines for which *predicate* is ``True``.

        The first line for which *predicate* is ``False`` is pushed back so the
        core loop (or the calling parser) will see it next.
        """
        lines: list[str] = []
        for line in self:
            if predicate(line):
                lines.append(line)
            else:
                self.push_back(line)
                break
        return lines

    def take_until(self, predicate: Callable[[str], bool]) -> list[str]:
        """
        Consume and return lines until *predicate* is ``True``.

        The first line for which *predicate* is ``True`` is pushed back.
        Equivalent to ``take_while(lambda line: not predicate(line))``.
        """
        return self.take_while(lambda line: not predicate(line))
