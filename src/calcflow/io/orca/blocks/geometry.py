import re

from calcflow.common.exceptions import ParsingError
from calcflow.common.models import Atom
from calcflow.geometry.static import Geometry
from calcflow.io.peekable import PeekableIterator
from calcflow.io.state import ParseState
from calcflow.utils import logger

GEOMETRY_START_PAT = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)")
GEOMETRY_LINE_PAT = re.compile(r"^\s*([A-Za-z]{1,3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class GeometryParser:
    def matches(self, line: str, state: ParseState) -> bool:
        return not state.parsed_geometry and bool(GEOMETRY_START_PAT.search(line))

    def parse(self, iterator: PeekableIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing geometry block.")
        iterator.skip()  # Consume blank header line
        geometry: list[Atom] = []
        for line in iterator.take_while(lambda ln: bool(ln.strip())):
            match = GEOMETRY_LINE_PAT.match(line.strip())
            if match:
                symbol, x, y, z = match.groups()
                geometry.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))

        if not geometry:
            raise ParsingError("Geometry block found but no atoms parsed.")

        state.input_geometry = Geometry(comment="", atoms=tuple(geometry))
        state.parsed_geometry = True
        logger.debug(f"Parsed {len(geometry)} atoms.")
