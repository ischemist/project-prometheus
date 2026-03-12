"""
Parser for QChem CM5 (Charge Model 5) atomic charges.

Handles the "Charge Model 5" section, which is printed when CM5=True is set
in the $rem block. CM5 charges apply empirical corrections to Hirshfeld charges
to better reproduce experimental charge distributions; they always follow a
Hirshfeld block since CM5 is derived from the Hirshfeld partitioning.
"""

import re

from calcflow.common.patterns import extract_index
from calcflow.common.results import AtomicCharges
from calcflow.io.peekable import PeekableIterator
from calcflow.io.state import ParseState
from calcflow.utils import logger

CM5_START_PAT = re.compile(r"Charge Model 5")

# Same tabular format as Hirshfeld/Mulliken
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

SUM_LINE_PAT = re.compile(r"Sum of atomic charges")


class Cm5Parser:
    """
    Parses CM5 atomic charges from QChem output.

    CM5 always follows a Hirshfeld block in Q-Chem output (CM5 requires
    HIRSHFELD=True). The table format is identical to Hirshfeld.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        if "Charge Model 5" not in line:
            return False
        return not state.parsed_cm5 and bool(CM5_START_PAT.search(line))

    def parse(self, iterator: PeekableIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing QChem CM5 charges block.")

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
                    state.parsing_warnings.append(f"Could not parse CM5 charge line: {line.strip()} ({e})")

        if not charges:
            state.parsing_warnings.append("CM5 charges block found but no charges were parsed.")
            return

        state.atomic_charges.append(AtomicCharges(method="CM5", charges=charges))
        state.parsed_cm5 = True
        logger.debug(f"Parsed CM5 charges for {len(charges)} atoms")
