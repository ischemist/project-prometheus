"""
Block parser for QChem geometry sections: the initial $molecule block and the
final standard nuclear orientation.
"""

import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import Atom
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Regex Patterns ---
INPUT_GEOM_START_PAT = re.compile(r"^\s*\$molecule", re.IGNORECASE)
INPUT_GEOM_END_PAT = re.compile(r"^\s*\$end", re.IGNORECASE)
STANDARD_GEOM_START_PAT = re.compile(r"^\s*Standard Nuclear Orientation \(Angstroms\)")
STANDARD_GEOM_END_PAT = re.compile(r"^\s*-{20,}")

# Skips the charge/multiplicity line in the $molecule block
INPUT_GEOM_SKIP_PAT = re.compile(r"^\s*-?\d+\s+\d+\s*$")
# Input format: Symbol X Y Z
INPUT_ATOM_PAT = re.compile(r"^\s*([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
# Standard Orientation format: Index Symbol X Y Z
STANDARD_ATOM_PAT = re.compile(r"^\s*\d+\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class GeometryParser:
    """
    Parses both the input ($molecule) and standard orientation geometry blocks.
    It uses separate flags in the ParseState to handle each block only once.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """Checks if the line starts either the input or standard geometry block."""
        if INPUT_GEOM_START_PAT.search(line) and not state.parsed_geometry:
            # We use a single 'parsed_geometry' flag for the input geometry
            return True
        # The final geometry is part of the main calculation, not a separate block to be parsed once
        # For now, let's assume we only parse the final geometry once.
        return (STANDARD_GEOM_START_PAT.search(line) is not None) and state.final_geometry is None

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """Delegates to the appropriate parsing method based on the start line."""
        if INPUT_GEOM_START_PAT.search(start_line):
            self._parse_input_geometry(iterator, state)
        elif STANDARD_GEOM_START_PAT.search(start_line):
            self._parse_standard_orientation(iterator, state)

    def _parse_input_geometry(self, iterator: Iterator[str], state: ParseState) -> None:
        """Parses the $molecule block."""
        logger.debug("Parsing $molecule geometry block.")
        atoms: list[Atom] = []

        # First line after $molecule is charge/multiplicity, which we skip
        # unless it's "read" (for multi-job files where Job 2+ inherit geometry from Job 1)
        first_line = next(iterator, None)
        if first_line and first_line.strip().lower() == "read":
            logger.debug("Found '$molecule read' directive - skipping geometry parsing (inherited from previous job).")
            state.parsed_geometry = True
            return

        for line in iterator:
            if INPUT_GEOM_END_PAT.search(line):
                break

            match = INPUT_ATOM_PAT.search(line)
            if match:
                symbol, x, y, z = match.groups()
                atoms.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))
            elif line.strip():  # Ignore blank lines but warn on other unexpected content
                logger.warning(f"Skipping unexpected line in $molecule block: {line.strip()}")

        if not atoms:
            raise ParsingError("$molecule block found but no atoms could be parsed.")

        state.input_geometry = tuple(atoms)
        state.parsed_geometry = True  # Use the generic flag for the primary input geometry
        logger.debug(f"Parsed {len(atoms)} atoms from $molecule block.")

    def _parse_standard_orientation(self, iterator: Iterator[str], state: ParseState) -> None:
        """Parses the 'Standard Nuclear Orientation' block."""
        logger.debug("Parsing standard orientation geometry block.")
        atoms: list[Atom] = []

        # Consume header lines ("I Atom X Y Z" and "----")
        _ = next(iterator, None)
        _ = next(iterator, None)

        for line in iterator:
            if STANDARD_GEOM_END_PAT.search(line):
                break

            match = STANDARD_ATOM_PAT.search(line)
            if match:
                symbol, x, y, z = match.groups()
                atoms.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))
            elif line.strip():
                # This could be the start of the next section
                state.buffered_line = line
                logger.debug(f"Non-atom line ended standard orientation block. Buffering: {line.strip()}")
                break

        if not atoms:
            raise ParsingError("Standard orientation block found but no atoms could be parsed.")

        state.final_geometry = tuple(atoms)
        logger.debug(f"Parsed {len(atoms)} atoms from standard orientation.")
