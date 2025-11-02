"""
Parser for orbital energies from ORCA output files.

Extracts molecular orbital information (energies, occupations, HOMO/LUMO indices)
from the "ORBITAL ENERGIES" block in ORCA output.
"""

import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import Orbital, OrbitalsSet
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Regex pattern for the start of the orbital energies block
ORBITAL_ENERGIES_START_PAT = re.compile(r"ORBITAL ENERGIES")

# Pattern to match a single orbital data line:
# Format: NO   OCC          E(Eh)            E(eV)
# Example:   0   2.0000     -18.937331      -515.3110
FLOAT_PAT = r"([-+]?\d+\.\d+)"
ORBITAL_LINE_PAT = re.compile(rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}")

# Pattern to detect end of orbital data (truncation message or next section)
ORBITAL_END_PAT = re.compile(r"\*Only the first.*orbitals were printed")


class OrbitalsParser:
    """
    Parser for the ORBITAL ENERGIES block in ORCA output.

    Extracts:
    - Orbital indices, energies (Hartree), energies (eV), and occupations
    - HOMO and LUMO indices based on occupation numbers
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the start of an orbital energies block.

        Args:
            line: The current line from the output file.
            state: The mutable ParseState object.

        Returns:
            True if this is the start of an orbital block and hasn't been parsed yet.
        """
        if state.parsed_orbitals:
            return False
        return bool(ORBITAL_ENERGIES_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse the entire orbital energies block.

        Args:
            iterator: The line iterator for the output file.
            start_line: The line that matched (contains "ORBITAL ENERGIES").
            state: The mutable ParseState object to populate.

        Raises:
            ParsingError: If the block is found but no orbitals are parsed.
        """
        logger.debug("Parsing orbital energies block.")

        orbitals: list[Orbital] = []
        homo_idx: int | None = None
        lumo_idx: int | None = None

        # Skip separator line ("----------------")
        _ = next(iterator, None)

        # Skip empty line
        _ = next(iterator, None)

        # Skip header line ("  NO   OCC          E(Eh)            E(eV)")
        _ = next(iterator, None)

        for line in iterator:
            # Empty line might mark the end of orbital data
            if not line.strip():
                break

            # Check for truncation message
            if ORBITAL_END_PAT.search(line):
                break

            # Try to parse as an orbital data line
            match = ORBITAL_LINE_PAT.match(line.strip())
            if match:
                try:
                    groups = match.groups()
                    orbital_idx = int(groups[0])
                    occupation = float(groups[1])
                    energy_eh = float(groups[2])
                    energy_ev = float(groups[3])

                    orbital = Orbital(
                        index=orbital_idx,
                        energy=energy_eh,
                        occupation=occupation,
                        energy_ev=energy_ev,
                    )
                    orbitals.append(orbital)

                    # Track HOMO and LUMO indices
                    # HOMO = last orbital with occupation > 0
                    # LUMO = first orbital with occupation = 0
                    if occupation > 0:
                        homo_idx = orbital_idx
                    elif lumo_idx is None:
                        lumo_idx = orbital_idx

                except (ValueError, IndexError) as e:
                    state.parsing_warnings.append(f"Could not parse orbital line: {line.strip()}. Error: {e}")
            else:
                # If line doesn't match orbital format and isn't empty/truncation msg,
                # we've likely reached the next section. Buffer it for the core parser.
                state.buffered_line = line
                break

        if not orbitals:
            raise ParsingError("Orbital energies block found but no orbitals were parsed.")

        # Create OrbitalsSet (RHF: only alpha_orbitals, beta_orbitals = None)
        orbitals_set = OrbitalsSet(
            alpha_orbitals=orbitals,
            beta_orbitals=None,
            alpha_homo_index=homo_idx,
            alpha_lumo_index=lumo_idx,
            beta_homo_index=None,
            beta_lumo_index=None,
        )

        state.orbitals = orbitals_set
        state.parsed_orbitals = True
        logger.debug(f"Successfully parsed {len(orbitals)} orbitals. HOMO index: {homo_idx}, LUMO index: {lumo_idx}")
