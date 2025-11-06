import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import Atom
from calcflow.io.state import ParseState
from calcflow.utils import logger

GEOMETRY_START_PAT = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)")
GEOMETRY_LINE_PAT = re.compile(r"^\s*([A-Za-z]{1,3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class GeometryParser:
    def matches(self, line: str, state: ParseState) -> bool:
        return not state.parsed_geometry and bool(GEOMETRY_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        logger.debug("Parsing geometry block.")
        next(iterator, None)  # Consume header
        geometry: list[Atom] = []
        for line in iterator:
            line_stripped = line.strip()
            if not line_stripped:
                break
            match = GEOMETRY_LINE_PAT.match(line_stripped)
            if match:
                symbol, x, y, z = match.groups()
                geometry.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))
            else:
                state.buffered_line = line
                break

        if not geometry:
            raise ParsingError("Geometry block found but no atoms parsed.")

        state.input_geometry = tuple(geometry)
        state.parsed_geometry = True
        logger.debug(f"Parsed {len(geometry)} atoms.")
