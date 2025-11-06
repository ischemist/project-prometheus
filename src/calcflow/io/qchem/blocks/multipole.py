"""
Parser for QChem Cartesian multipole moments.

Handles the "Cartesian Multipole Moments" section which includes charge, dipole,
quadrupole, octopole, and hexadecapole moments.
"""

import re
from collections.abc import Iterator

from calcflow.common.results import (
    DipoleMoment,
    HexadecapoleMoment,
    MultipoleResults,
    OctopoleMoment,
    QuadrupoleMoment,
)
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Regex patterns
MULTIPOLE_START_PAT = re.compile(r"Cartesian Multipole Moments")
MULTIPOLE_END_PAT = re.compile(r"^\s*-+\s*$")
CHARGE_PAT = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
COMPONENT_PAT = re.compile(r"([A-Za-z]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# Expected component keys for each multipole type
MULTIPOLE_KEYS = {
    "quadrupole": {"xx", "xy", "yy", "xz", "yz", "zz"},
    "octopole": {"xxx", "xxy", "xyy", "yyy", "xxz", "xyz", "yyz", "xzz", "yzz", "zzz"},
    "hexadecapole": {
        "xxxx",
        "xxxy",
        "xxyy",
        "xyyy",
        "yyyy",
        "xxxz",
        "xxyz",
        "xyyz",
        "yyyz",
        "xxzz",
        "xyzz",
        "yyzz",
        "xzzz",
        "yzzz",
        "zzzz",
    },
}


class MultipoleParser:
    """
    Parses Cartesian multipole moments from QChem output.

    QChem reports charge, dipole, quadrupole, octopole, and hexadecapole moments
    in a structured format. The parser extracts all available moments and creates
    the corresponding model instances.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """Check if this line marks the beginning of the multipole moments block."""
        return bool(MULTIPOLE_START_PAT.search(line)) and not state.parsed_multipole

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse the Cartesian multipole moments block.

        Args:
            iterator: Line iterator for the output file
            start_line: The line matching the multipole header
            state: Mutable ParseState to store results
        """
        logger.debug("Parsing QChem Cartesian multipole moments block.")

        # Skip the dashes separator line
        try:
            next(iterator)
        except StopIteration:
            logger.warning("Unexpected end of iterator after multipole moments header")
            return

        charge: float | None = None
        dipole: DipoleMoment | None = None
        quadrupole: QuadrupoleMoment | None = None
        octopole: OctopoleMoment | None = None
        hexadecapole: HexadecapoleMoment | None = None
        components: dict[str, dict[str, float]] = {
            "quadrupole": {},
            "octopole": {},
            "hexadecapole": {},
        }
        current_section: str = ""

        for line in iterator:
            if MULTIPOLE_END_PAT.search(line):
                break

            stripped = line.strip()
            if not stripped:
                continue

            # Detect section headers using match
            if "Charge (ESU" in stripped:
                current_section = "charge"
                continue
            elif "Dipole Moment" in stripped:
                current_section = "dipole"
                continue
            elif "Quadrupole Moments" in stripped:
                current_section = "quadrupole"
                continue
            elif "Octopole Moments" in stripped:
                current_section = "octopole"
                components["octopole"].clear()
                continue
            elif "Hexadecapole Moments" in stripped:
                current_section = "hexadecapole"
                components["hexadecapole"].clear()
                continue

            # Parse data based on current section
            match current_section:
                case "charge":
                    if m := CHARGE_PAT.match(stripped):
                        try:
                            charge = float(m.group(1))
                        except ValueError:
                            state.parsing_warnings.append(f"Could not parse charge: {stripped}")
                        finally:
                            current_section = ""

                case "dipole":
                    result = self._parse_dipole(stripped, dipole, state)
                    if result is not None:
                        dipole = result
                        if result.magnitude > 0:  # Only clear section if magnitude was found
                            current_section = ""

                case "quadrupole":
                    self._accumulate_components(stripped, components["quadrupole"])
                    if MULTIPOLE_KEYS["quadrupole"].issubset(components["quadrupole"].keys()):
                        try:
                            quadrupole = QuadrupoleMoment(**components["quadrupole"])
                        except (ValueError, TypeError):
                            state.parsing_warnings.append(f"Could not create quadrupole: {components['quadrupole']}")
                        finally:
                            current_section = ""
                            components["quadrupole"].clear()

                case "octopole":
                    self._accumulate_components(stripped, components["octopole"])
                    if MULTIPOLE_KEYS["octopole"].issubset(components["octopole"].keys()):
                        try:
                            octopole = OctopoleMoment(**components["octopole"])
                        except (ValueError, TypeError):
                            state.parsing_warnings.append(f"Could not create octopole: {components['octopole']}")
                        finally:
                            current_section = ""
                            components["octopole"].clear()

                case "hexadecapole":
                    self._accumulate_components(stripped, components["hexadecapole"])
                    if MULTIPOLE_KEYS["hexadecapole"].issubset(components["hexadecapole"].keys()):
                        try:
                            hexadecapole = HexadecapoleMoment(**components["hexadecapole"])
                        except (ValueError, TypeError):
                            state.parsing_warnings.append(
                                f"Could not create hexadecapole: {components['hexadecapole']}"
                            )
                        finally:
                            current_section = ""
                            components["hexadecapole"].clear()

        if any([charge, dipole, quadrupole, octopole, hexadecapole]):
            state.multipole = MultipoleResults(
                charge=charge,
                dipole=dipole,
                quadrupole=quadrupole,
                octopole=octopole,
                hexadecapole=hexadecapole,
            )
            logger.debug("Parsed Cartesian multipole moments")
        else:
            state.parsing_warnings.append("Multipole moments block found but no moments were parsed.")

        state.parsed_multipole = True

    @staticmethod
    def _parse_dipole(line: str, current_dipole: DipoleMoment | None, state: ParseState) -> DipoleMoment | None:
        """Parse dipole moment from line, handling split across two lines."""
        x_match = re.search(rf"X\s+{FLOAT_PAT}", line)
        y_match = re.search(rf"Y\s+{FLOAT_PAT}", line)
        z_match = re.search(rf"Z\s+{FLOAT_PAT}", line)
        tot_match = re.search(rf"Tot\s+{FLOAT_PAT}", line)

        try:
            if x_match and y_match and z_match:
                x, y, z = float(x_match.group(1)), float(y_match.group(1)), float(z_match.group(1))
                magnitude = float(tot_match.group(1)) if tot_match else 0.0
                return DipoleMoment(x=x, y=y, z=z, magnitude=magnitude)
            elif tot_match and current_dipole:
                magnitude = float(tot_match.group(1))
                return DipoleMoment(x=current_dipole.x, y=current_dipole.y, z=current_dipole.z, magnitude=magnitude)
        except ValueError:
            state.parsing_warnings.append(f"Could not parse dipole: {line}")
        return None

    @staticmethod
    def _accumulate_components(line: str, components: dict[str, float]) -> None:
        """Extract and accumulate multipole components from a line."""
        for m in COMPONENT_PAT.finditer(line):
            try:  # noqa: SIM105
                components[m.group(1).lower()] = float(m.group(2))
            except ValueError:
                pass
