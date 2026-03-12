"""
Parser for QChem Mulliken atomic charges.

Handles the "Ground-State Mulliken Net Atomic Charges" section which lists
charges for each atom in the system.
"""

import re

from calcflow.common.patterns import extract_index
from calcflow.common.results import AtomicCharges
from calcflow.io.peekable import PeekableIterator
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Regex pattern for identifying the Mulliken charges block header
MULLIKEN_START_PAT = re.compile(r"Ground-State Mulliken Net Atomic Charges")

# Pattern to match charge lines: "1 H    0.193937"
# Captures: (atom_index_1based, element_symbol, charge_value)
# QChem uses 1-based indexing in output; we convert to 0-based for storage
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

# End markers for the charges block
SUM_LINE_PAT = re.compile(r"Sum of atomic charges")


class MullikenParser:
    """
    Parses Mulliken atomic charges from QChem output.

    QChem reports atomic charges in a tabular format with 1-based atom indices.
    This parser converts to 0-based indices for consistency with the data model.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the beginning of the Mulliken charges block.

        Returns True only if we haven't already parsed Mulliken charges and the line
        contains the Mulliken charges header.
        """
        if "Mulliken" not in line:
            return False
        return not state.parsed_mulliken and bool(MULLIKEN_START_PAT.search(line))

    def parse(self, iterator: PeekableIterator, start_line: str, state: ParseState) -> None:
        """
        Parse the Mulliken atomic charges block.

        Args:
            iterator: Line iterator for the output file
            start_line: The line matching the charges header
            state: Mutable ParseState to store results
        """
        logger.debug("Parsing QChem Mulliken charges block.")

        # Skip the dashes separator line
        iterator.skip()

        charges: dict[int, float] = {}

        # Parse charge lines until we hit the sum line (sum line is pushed back, not consumed)
        for line in iterator.take_until(lambda ln: bool(SUM_LINE_PAT.search(ln))):
            if not line.strip():
                continue

            match = CHARGE_LINE_PAT.match(line.strip())
            if match:
                try:
                    atom_idx = extract_index(match.group(1))
                    charge = float(match.group(3))
                    charges[atom_idx] = charge
                except (ValueError, IndexError) as e:
                    state.parsing_warnings.append(f"Could not parse Mulliken charge line: {line.strip()} ({e})")
                    continue

        # Validate that we parsed charges
        if not charges:
            state.parsing_warnings.append("Mulliken charges block found but no charges were parsed.")
            return

        # Create AtomicCharges model and add to state
        atomic_charges = AtomicCharges(method="Mulliken", charges=charges)
        state.atomic_charges.append(atomic_charges)

        logger.debug(f"Parsed Mulliken charges for {len(charges)} atoms")

        # Set flag to prevent duplicate parsing
        state.parsed_mulliken = True
