import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import DipoleMoment, MultipoleResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# Pattern to match the start of the dipole block (e.g., "DIPOLE MOMENT")
DIPOLE_HEADER_PAT = re.compile(r"^\s*DIPOLE MOMENT\s*$")

# Pattern to match the "Total Dipole Moment" line with x, y, z components
TOTAL_DIPOLE_PAT = re.compile(rf"Total Dipole Moment\s*:\s*{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}")

# Pattern to match the magnitude line in Debye
MAGNITUDE_DEBYE_PAT = re.compile(rf"Magnitude \(Debye\)\s*:\s*{FLOAT_PAT}")


class DipoleParser:
    """Parser for ORCA dipole moment block."""

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the start of the dipole moment block.

        The dipole block in ORCA output is identified by a line containing "DIPOLE MOMENT".
        """
        if state.parsed_dipole:
            return False
        return "DIPOLE MOMENT" in line

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse the dipole moment block and populate state.multipole with DipoleMoment.

        The block structure is:
        --------
        DIPOLE MOMENT
        --------
        Method             : SCF
        Type of density    : Electron Density
        Multiplicity       :   1
        ...
        Total Dipole Moment    :     X             Y             Z
                                -----------------------------------------
        Magnitude (a.u.)       :      M_au
        Magnitude (Debye)      :      M_debye
        """
        logger.debug("Parsing dipole moment block.")

        x_component: float | None = None
        y_component: float | None = None
        z_component: float | None = None
        magnitude_debye: float | None = None

        for line in iterator:
            # Skip empty lines
            if not line.strip():
                continue

            # Try to match the "Total Dipole Moment" line
            if total_match := TOTAL_DIPOLE_PAT.search(line):
                try:
                    x_component = float(total_match.group(1))
                    y_component = float(total_match.group(2))
                    z_component = float(total_match.group(3))
                except ValueError as e:
                    raise ParsingError(f"Failed to parse dipole components: {e}") from e

            # Try to match the "Magnitude (Debye)" line
            if mag_match := MAGNITUDE_DEBYE_PAT.search(line):
                try:
                    magnitude_debye = float(mag_match.group(1))
                except ValueError as e:
                    raise ParsingError(f"Failed to parse dipole magnitude: {e}") from e
                # We've found all we need; exit the loop
                break

        # Validate that we found all required values
        if x_component is None or y_component is None or z_component is None:
            raise ParsingError("Dipole block matched but 'Total Dipole Moment' line not found or incomplete.")

        if magnitude_debye is None:
            raise ParsingError("Dipole block matched but 'Magnitude (Debye)' line not found.")

        # Create the DipoleMoment model
        dipole_moment = DipoleMoment(
            x=x_component,
            y=y_component,
            z=z_component,
            magnitude=magnitude_debye,
        )

        # Create or update the MultipoleResults container
        # For ORCA, we expect state.multipole to be None since this is a single-point calculation
        if state.multipole is None:
            state.multipole = MultipoleResults(dipole=dipole_moment)
        else:
            # This should not happen in normal circumstances, but handle gracefully
            logger.warning("MultipoleResults already exists; overwriting dipole moment.")
            state.multipole = MultipoleResults(
                dipole=dipole_moment,
                quadrupole=state.multipole.quadrupole,
            )

        state.parsed_dipole = True
        logger.debug("Finished parsing dipole moment block.")
