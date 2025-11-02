"""
QChem orbitals block parser.

Parses the "Orbital Energies (a.u.)" section, extracting molecular orbital energies
for both occupied and virtual orbitals. Handles both restricted (RKS/RHF) and
unrestricted (UKS/UHF) calculations.
"""

import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import Orbital, OrbitalsSet
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Pattern definitions
ORBITAL_BLOCK_START_PAT = re.compile(r"^\s*Orbital Energies \(a\.u\.\)")
SEPARATOR_PAT = re.compile(r"^\s*-{50,}")
ALPHA_MOS_PAT = re.compile(r"^\s*Alpha MOs")
BETA_MOS_PAT = re.compile(r"^\s*Beta MOs")
OCCUPIED_PAT = re.compile(r"^\s*-- Occupied --")
VIRTUAL_PAT = re.compile(r"^\s*-- Virtual --")

# Pattern to extract all floating-point numbers from a line
ENERGY_PAT = re.compile(r"(-?\d+\.\d+)")

# End-of-block markers (next block identifiers)
END_OF_BLOCK_PATS = [
    re.compile(r"^\s*Ground-State Mulliken Net Atomic Charges"),
    re.compile(r"^\s*Cartesian Multipole Moments"),
    re.compile(r"^\s*TDDFT/TDA\s+Excitation"),
]


class OrbitalsParser:
    """Parses the orbital energies block from QChem output."""

    def matches(self, line: str, state: ParseState) -> bool:
        """Check if this line marks the start of the orbital energies block."""
        if state.parsed_orbitals:
            return False
        return bool(ORBITAL_BLOCK_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """Parse the entire orbital energies block."""
        logger.debug("Starting orbitals block parsing.")

        # Storage for parsed data
        alpha_occupied_energies: list[float] = []
        alpha_virtual_energies: list[float] = []
        beta_occupied_energies: list[float] = []
        beta_virtual_energies: list[float] = []

        # State machine variables
        current_spin: str | None = None  # "alpha" or "beta"
        current_section: str | None = None  # "occupied" or "virtual"

        # Skip the initial separator line
        line = next(iterator)
        if not SEPARATOR_PAT.search(line):
            state.parsing_warnings.append(f"Expected separator after orbital block start, got: {line.strip()}")

        # Main parsing loop
        for line in iterator:
            # Check for end-of-block conditions
            if SEPARATOR_PAT.search(line):
                # This is the closing separator - we're done
                logger.debug("Reached end of orbitals block (closing separator).")
                break

            if any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                # Hit the start of next block
                logger.debug(f"Orbitals parser ended on next-block line: {line.strip()}")
                state.buffered_line = line
                break

            # Check for section headers
            if ALPHA_MOS_PAT.search(line):
                current_spin = "alpha"
                current_section = None
                logger.debug("Found Alpha MOs section")
                continue

            if BETA_MOS_PAT.search(line):
                current_spin = "beta"
                current_section = None
                logger.debug("Found Beta MOs section")
                continue

            if OCCUPIED_PAT.search(line):
                current_section = "occupied"
                logger.debug(f"Found occupied section for {current_spin} spin")
                # Extract energies from the same line (after "-- Occupied --")
                energies = self._extract_energies_from_line(line)
                self._store_energies(
                    energies,
                    current_spin,
                    current_section,
                    alpha_occupied_energies,
                    alpha_virtual_energies,
                    beta_occupied_energies,
                    beta_virtual_energies,
                    state,
                )
                continue

            if VIRTUAL_PAT.search(line):
                current_section = "virtual"
                logger.debug(f"Found virtual section for {current_spin} spin")
                # Extract energies from the same line (after "-- Virtual --")
                energies = self._extract_energies_from_line(line)
                self._store_energies(
                    energies,
                    current_spin,
                    current_section,
                    alpha_occupied_energies,
                    alpha_virtual_energies,
                    beta_occupied_energies,
                    beta_virtual_energies,
                    state,
                )
                continue

            # If we're in a section, try to extract energy values
            if current_spin is not None and current_section is not None and line.strip():
                energies = self._extract_energies_from_line(line)
                if energies:
                    self._store_energies(
                        energies,
                        current_spin,
                        current_section,
                        alpha_occupied_energies,
                        alpha_virtual_energies,
                        beta_occupied_energies,
                        beta_virtual_energies,
                        state,
                    )

        # Validate parsed data
        if not alpha_occupied_energies:
            raise ParsingError("Orbital energies block found, but no occupied alpha orbitals parsed.")
        if not alpha_virtual_energies:
            state.parsing_warnings.append("No virtual orbitals found for alpha spin.")

        # Create Orbital objects
        alpha_orbitals = self._create_orbitals(alpha_occupied_energies + alpha_virtual_energies)

        # Determine HOMO/LUMO indices for alpha
        alpha_homo_index = len(alpha_occupied_energies) - 1  # 0-based, last occupied
        alpha_lumo_index = len(alpha_occupied_energies) if alpha_virtual_energies else None

        # Handle beta orbitals if present (unrestricted calculation)
        beta_orbitals = None
        beta_homo_index = None
        beta_lumo_index = None
        if beta_occupied_energies:
            if not beta_virtual_energies:
                state.parsing_warnings.append("No virtual orbitals found for beta spin.")
            beta_orbitals = self._create_orbitals(beta_occupied_energies + beta_virtual_energies)
            beta_homo_index = len(beta_occupied_energies) - 1
            beta_lumo_index = len(beta_occupied_energies) if beta_virtual_energies else None

        # Create OrbitalsSet
        state.orbitals = OrbitalsSet(
            alpha_orbitals=tuple(alpha_orbitals),
            beta_orbitals=tuple(beta_orbitals) if beta_orbitals else None,
            alpha_homo_index=alpha_homo_index,
            alpha_lumo_index=alpha_lumo_index,
            beta_homo_index=beta_homo_index,
            beta_lumo_index=beta_lumo_index,
        )

        state.parsed_orbitals = True
        logger.info(
            f"Parsed orbital energies. Alpha: {len(alpha_orbitals)} orbitals "
            f"(HOMO: {alpha_homo_index}, LUMO: {alpha_lumo_index}). "
            f"Beta: {len(beta_orbitals) if beta_orbitals else 0} orbitals."
        )

    def _extract_energies_from_line(self, line: str) -> list[float]:
        """Extract all floating-point energy values from a line."""
        matches = ENERGY_PAT.findall(line)
        try:
            return [float(match) for match in matches]
        except ValueError as e:
            logger.warning(f"Could not parse energy values from line: {line.strip()} - {e}")
            return []

    def _store_energies(
        self,
        energies: list[float],
        current_spin: str | None,
        current_section: str | None,
        alpha_occupied: list[float],
        alpha_virtual: list[float],
        beta_occupied: list[float],
        beta_virtual: list[float],
        state: ParseState,
    ) -> None:
        """Store parsed energies in the appropriate list."""
        if not energies:
            return

        if current_spin == "alpha" and current_section == "occupied":
            alpha_occupied.extend(energies)
        elif current_spin == "alpha" and current_section == "virtual":
            alpha_virtual.extend(energies)
        elif current_spin == "beta" and current_section == "occupied":
            beta_occupied.extend(energies)
        elif current_spin == "beta" and current_section == "virtual":
            beta_virtual.extend(energies)
        else:
            state.parsing_warnings.append(
                f"Found energy values but no active section: spin={current_spin}, section={current_section}"
            )

    def _create_orbitals(self, energies: list[float]) -> list[Orbital]:
        """Create Orbital objects from a list of energies with sequential 0-based indices."""
        return [Orbital(index=i, energy=energy) for i, energy in enumerate(energies)]
