"""
Parser for QChem Cartesian multipole moments.

Handles the "Cartesian Multipole Moments" section which includes charge, dipole,
quadrupole, octopole, and hexadecapole moments.
"""

import re
from collections.abc import Iterator

from calcflow.common.models import (
    DipoleMoment,
    HexadecapoleMoment,
    MultipoleResults,
    OctopoleMoment,
    QuadrupoleMoment,
)
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Regex pattern for identifying the multipole block header
MULTIPOLE_START_PAT = re.compile(r"Cartesian Multipole Moments")

# End marker for the multipole block
MULTIPOLE_END_PAT = re.compile(r"^\s*-+\s*$")

# Component patterns
CHARGE_PAT = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
COMPONENT_PAT = re.compile(r"([A-Za-z]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


class MultipoleParser:
    """
    Parses Cartesian multipole moments from QChem output.

    QChem reports charge, dipole, quadrupole, octopole, and hexadecapole moments
    in a structured format. The parser extracts all available moments and creates
    the corresponding model instances.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the beginning of the multipole moments block.

        Returns True only if we haven't already parsed multipole moments and the line
        contains the multipole block header.
        """
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

        # Accumulate components for multi-line sections
        quadrupole_components: dict[str, float] = {}
        octopole_components: dict[str, float] = {}
        hexadecapole_components: dict[str, float] = {}

        current_section: str = ""

        for line in iterator:
            # Check for end marker
            if MULTIPOLE_END_PAT.search(line):
                break

            stripped = line.strip()
            if not stripped:
                continue

            # Detect section headers
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
                octopole_components.clear()
                continue
            elif "Hexadecapole Moments" in stripped:
                current_section = "hexadecapole"
                hexadecapole_components.clear()
                continue

            # Parse data based on current section
            if current_section == "charge":
                match = CHARGE_PAT.match(stripped)
                if match:
                    try:
                        charge = float(match.group(1))
                        current_section = ""
                    except ValueError:
                        state.parsing_warnings.append(f"Could not parse charge: {stripped}")
                        current_section = ""

            elif current_section == "dipole":
                # Extract X, Y, Z on first line and Tot on second
                x_match = re.search(r"X\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stripped)
                y_match = re.search(r"Y\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stripped)
                z_match = re.search(r"Z\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stripped)
                tot_match = re.search(r"Tot\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stripped)

                try:
                    if x_match and y_match and z_match:
                        x = float(x_match.group(1))
                        y = float(y_match.group(1))
                        z = float(z_match.group(1))
                        if tot_match:
                            magnitude = float(tot_match.group(1))
                            dipole = DipoleMoment(x=x, y=y, z=z, magnitude=magnitude)
                            current_section = ""
                        else:
                            # Magnitude will come on next line
                            dipole = DipoleMoment(x=x, y=y, z=z, magnitude=0.0)
                    elif tot_match and dipole is not None:
                        # This is the Tot line when X/Y/Z were on previous line
                        magnitude = float(tot_match.group(1))
                        dipole = DipoleMoment(x=dipole.x, y=dipole.y, z=dipole.z, magnitude=magnitude)
                        current_section = ""
                except ValueError:
                    state.parsing_warnings.append(f"Could not parse dipole: {stripped}")
                    current_section = ""

            elif current_section == "quadrupole":
                # Accumulate all components from potentially 2 lines
                for match in COMPONENT_PAT.finditer(stripped):
                    component_name = match.group(1).lower()
                    try:
                        value = float(match.group(2))
                        quadrupole_components[component_name] = value
                    except ValueError:
                        pass

                # Check if we have all 6 components
                expected_keys = {"xx", "xy", "yy", "xz", "yz", "zz"}
                if expected_keys.issubset(set(quadrupole_components.keys())):
                    try:
                        quadrupole = QuadrupoleMoment(
                            xx=quadrupole_components["xx"],
                            xy=quadrupole_components["xy"],
                            yy=quadrupole_components["yy"],
                            xz=quadrupole_components["xz"],
                            yz=quadrupole_components["yz"],
                            zz=quadrupole_components["zz"],
                        )
                        current_section = ""
                        quadrupole_components.clear()
                    except (ValueError, KeyError):
                        state.parsing_warnings.append(
                            f"Could not create quadrupole from components: {quadrupole_components}"
                        )
                        current_section = ""

            elif current_section == "octopole":
                # Accumulate all components from this line
                for match in COMPONENT_PAT.finditer(stripped):
                    component_name = match.group(1).lower()
                    try:
                        value = float(match.group(2))
                        octopole_components[component_name] = value
                    except ValueError:
                        pass

                # Check if we have all 10 components
                expected_keys = {"xxx", "xxy", "xyy", "yyy", "xxz", "xyz", "yyz", "xzz", "yzz", "zzz"}
                if expected_keys.issubset(set(octopole_components.keys())):
                    try:
                        octopole = OctopoleMoment(
                            xxx=octopole_components["xxx"],
                            xxy=octopole_components["xxy"],
                            xyy=octopole_components["xyy"],
                            yyy=octopole_components["yyy"],
                            xxz=octopole_components["xxz"],
                            xyz=octopole_components["xyz"],
                            yyz=octopole_components["yyz"],
                            xzz=octopole_components["xzz"],
                            yzz=octopole_components["yzz"],
                            zzz=octopole_components["zzz"],
                        )
                        current_section = ""
                        octopole_components.clear()
                    except (ValueError, KeyError):
                        state.parsing_warnings.append(
                            f"Could not create octopole from components: {octopole_components}"
                        )
                        current_section = ""

            elif current_section == "hexadecapole":
                # Accumulate all components from this line
                for match in COMPONENT_PAT.finditer(stripped):
                    component_name = match.group(1).lower()
                    try:
                        value = float(match.group(2))
                        hexadecapole_components[component_name] = value
                    except ValueError:
                        pass

                # Check if we have all 15 components
                expected_keys = {
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
                }
                if expected_keys.issubset(set(hexadecapole_components.keys())):
                    try:
                        hexadecapole = HexadecapoleMoment(
                            xxxx=hexadecapole_components["xxxx"],
                            xxxy=hexadecapole_components["xxxy"],
                            xxyy=hexadecapole_components["xxyy"],
                            xyyy=hexadecapole_components["xyyy"],
                            yyyy=hexadecapole_components["yyyy"],
                            xxxz=hexadecapole_components["xxxz"],
                            xxyz=hexadecapole_components["xxyz"],
                            xyyz=hexadecapole_components["xyyz"],
                            yyyz=hexadecapole_components["yyyz"],
                            xxzz=hexadecapole_components["xxzz"],
                            xyzz=hexadecapole_components["xyzz"],
                            yyzz=hexadecapole_components["yyzz"],
                            xzzz=hexadecapole_components["xzzz"],
                            yzzz=hexadecapole_components["yzzz"],
                            zzzz=hexadecapole_components["zzzz"],
                        )
                        current_section = ""
                        hexadecapole_components.clear()
                    except (ValueError, KeyError):
                        state.parsing_warnings.append(
                            f"Could not create hexadecapole from components: {hexadecapole_components}"
                        )
                        current_section = ""

        # Create MultipoleResults and add to state
        if (
            charge is not None
            or dipole is not None
            or quadrupole is not None
            or octopole is not None
            or hexadecapole is not None
        ):
            multipole = MultipoleResults(
                charge=charge,
                dipole=dipole,
                quadrupole=quadrupole,
                octopole=octopole,
                hexadecapole=hexadecapole,
            )
            state.multipole = multipole
            logger.debug("Parsed Cartesian multipole moments")
        else:
            state.parsing_warnings.append("Multipole moments block found but no moments were parsed.")

        # Set flag to prevent duplicate parsing
        state.parsed_multipole = True
