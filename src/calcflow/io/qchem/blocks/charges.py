"""
Parser for QChem atomic charges from Mulliken population analysis.

Handles the "Ground-State Mulliken Net Atomic Charges" section which lists
charges for each atom in the system.
"""

import re
from collections.abc import Iterator

from calcflow.common.results import AtomicCharges
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


class ChargesParser:
    """
    Parses Mulliken atomic charges from QChem output.

    QChem reports atomic charges in a tabular format with 1-based atom indices.
    This parser converts to 0-based indices for consistency with the data model.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the beginning of the Mulliken charges block.

        Returns True only if we haven't already parsed charges and the line
        contains the Mulliken charges header.
        """
        return bool(MULLIKEN_START_PAT.search(line)) and not state.parsed_charges

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse the Mulliken atomic charges block.

        Args:
            iterator: Line iterator for the output file
            start_line: The line matching the charges header
            state: Mutable ParseState to store results
        """
        logger.debug("Parsing QChem Mulliken charges block.")

        # Skip the dashes separator line
        try:
            next(iterator)
        except StopIteration:
            logger.warning("Unexpected end of iterator after Mulliken charges header")
            return

        charges: dict[int, float] = {}

        # Parse charge lines until we hit the sum line
        for line in iterator:
            # Check for end marker
            if SUM_LINE_PAT.search(line):
                break

            # Empty line might signal end, but continue to be safe
            if not line.strip():
                continue

            # Try to match a charge line
            match = CHARGE_LINE_PAT.match(line.strip())
            if match:
                try:
                    atom_idx_1based = int(match.group(1))
                    charge = float(match.group(3))
                    # Convert 1-based QChem index to 0-based for storage
                    atom_idx_0based = atom_idx_1based - 1
                    charges[atom_idx_0based] = charge
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
        state.parsed_charges = True
