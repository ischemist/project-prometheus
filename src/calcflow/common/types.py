"""common, non-pydantic type aliases used throughout the library."""

from collections.abc import Iterator

# an iterator that yields lines from a file or text block.
# used extensively by all parsers.
LineIterator = Iterator[str]

# a 3d coordinate tuple.
type Coord3d = tuple[float, float, float]
type AtomCoords = tuple[str, Coord3d]
