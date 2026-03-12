"""
Parser for ORCA atomic charges from population analysis methods
(Mulliken and Loewdin).

Handles both MULLIKEN ATOMIC CHARGES and LOEWDIN ATOMIC CHARGES sections.
Both methods share identical formatting and are parsed by the same parser.
"""

import re

from calcflow.common.results import AtomicCharges
from calcflow.io.peekable import PeekableIterator
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Regex patterns for identifying charge blocks
MULLIKEN_START_PAT = re.compile(r"MULLIKEN ATOMIC CHARGES")
LOEWDIN_START_PAT = re.compile(r"LOEWDIN ATOMIC CHARGES")

# Pattern to match charge lines: "   0 H :    0.172827"
# Captures: (atom_index, element_symbol, charge_value)
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

# End markers (Mulliken has explicit sum line, Loewdin stops at next section)
MULLIKEN_SUM_PAT = re.compile(r"Sum of atomic charges")
REDUCED_CHARGES_PAT = re.compile(r"REDUCED ORBITAL CHARGES")


class ChargesParser:
    """
    Parses atomic charges from MULLIKEN and LOEWDIN population analysis blocks.

    Both methods appear in separate sections of the ORCA output but share identical
    formatting for the atomic charges portion. This single parser handles both by
    determining the method from the header line and parsing accordingly.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the beginning of an atomic charges block.

        Returns True for either MULLIKEN or LOEWDIN atomic charges headers.
        Allows the parser to run twice (once per method); stops after both are parsed.
        """
        if state.parsed_mulliken and state.parsed_loewdin:
            return False
        if "MULLIKEN" not in line and "LOEWDIN" not in line:
            return False
        return bool(MULLIKEN_START_PAT.search(line)) or bool(LOEWDIN_START_PAT.search(line))

    def parse(self, iterator: PeekableIterator, start_line: str, state: ParseState) -> None:
        """
        Parse an atomic charges block (either Mulliken or Loewdin).

        Args:
            iterator: Line iterator for the output file
            start_line: The line matching the charges header
            state: Mutable ParseState to store results
        """
        logger.debug("Parsing atomic charges block.")

        # Determine which method this is
        if MULLIKEN_START_PAT.search(start_line):
            method = "Mulliken"
        elif LOEWDIN_START_PAT.search(start_line):
            method = "Loewdin"
        else:
            logger.warning(f"Could not determine charge method from line: {start_line}")
            return

        iterator.skip()  # consume the dashes separator line

        charges: dict[int, float] = {}

        # Parse charge lines until we hit the end marker
        for line in iterator.take_until(lambda ln: bool(MULLIKEN_SUM_PAT.search(ln) or REDUCED_CHARGES_PAT.search(ln))):
            if not line.strip():
                continue

            # Try to match a charge line
            match = CHARGE_LINE_PAT.match(line.strip())
            if match:
                try:
                    atom_idx = int(match.group(1))
                    charge = float(match.group(3))
                    charges[atom_idx] = charge
                except (ValueError, IndexError) as e:
                    state.parsing_warnings.append(f"Could not parse charge line for {method}: {line.strip()} ({e})")
                    continue

        # Validate that we parsed charges
        if not charges:
            state.parsing_warnings.append(f"{method} charges block found but no charges were parsed.")
            return

        # Create AtomicCharges model and add to state
        atomic_charges = AtomicCharges(method=method, charges=charges)
        state.atomic_charges.append(atomic_charges)

        logger.debug(f"Parsed {method} charges for {len(charges)} atoms")

        if method == "Mulliken":
            state.parsed_mulliken = True
        elif method == "Loewdin":
            state.parsed_loewdin = True
