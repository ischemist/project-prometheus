"""common, non-pydantic type aliases used throughout the library."""

from collections.abc import Iterator

# an iterator that yields lines from a file or text block.
# used extensively by all parsers.
LineIterator = Iterator[str]

# a 3d coordinate tuple.
type Coord3d = tuple[float, float, float]
type AtomCoords = tuple[str, Coord3d]

# a 3x3 matrix stored as a tuple of three row tuples.
# used for two-photon absorption matrices and similar tensor quantities.
type Matrix3x3 = tuple[Coord3d, Coord3d, Coord3d]
