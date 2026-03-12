"""
Parser for QChem Hirshfeld atomic charges.

Handles the "Hirshfeld Atomic Charges" section, which is printed when
HIRSHFELD=True is set in the $rem block. Hirshfeld charges use a stockholder
partitioning of the electron density onto pro-molecular atomic densities.
"""

import re

from calcflow.common.patterns import extract_index
from calcflow.common.results import AtomicCharges
from calcflow.io.peekable import PeekableIterator
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Header pattern — note the trailing spaces in Q-Chem output are trimmed by search
HIRSHFELD_START_PAT = re.compile(r"Hirshfeld Atomic Charges")

# Pattern to match charge lines: "   1 H                     0.128647"
# Captures: (atom_index_1based, element_symbol, charge_value)
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

SUM_LINE_PAT = re.compile(r"Sum of atomic charges")


class HirshfeldParser:
    """
    Parses Hirshfeld atomic charges from QChem output.

    QChem prints a blank line, a column-header line, and a dashes line before
    the actual charge rows. We skip non-matching lines inside the loop rather
    than eagerly consuming iterator entries, which keeps the logic simple.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        return bool(HIRSHFELD_START_PAT.search(line)) and not state.parsed_hirshfeld

    def parse(self, iterator: PeekableIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing QChem Hirshfeld charges block.")

        charges: dict[int, float] = {}

        # sum line is pushed back, not consumed
        for line in iterator.take_until(lambda ln: bool(SUM_LINE_PAT.search(ln))):
            match = CHARGE_LINE_PAT.match(line)
            if match:
                try:
                    atom_idx = extract_index(match.group(1))
                    charge = float(match.group(3))
                    charges[atom_idx] = charge
                except (ValueError, IndexError) as e:
                    state.parsing_warnings.append(f"Could not parse Hirshfeld charge line: {line.strip()} ({e})")

        if not charges:
            state.parsing_warnings.append("Hirshfeld charges block found but no charges were parsed.")
            return

        state.atomic_charges.append(AtomicCharges(method="Hirshfeld", charges=charges))
        state.parsed_hirshfeld = True
        logger.debug(f"Parsed Hirshfeld charges for {len(charges)} atoms")
