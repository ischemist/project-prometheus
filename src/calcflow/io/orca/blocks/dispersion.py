import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import DispersionCorrection
from calcflow.io.state import ParseState
from calcflow.utils import logger

FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# Block markers
START_PAT = re.compile(r"DFT DISPERSION CORRECTION")
METHOD_PAT = re.compile(r"(DFTD[34])\s+V(\d+\.\d+)\s+Rev\s+(\d+)")
DAMPING_PAT = re.compile(r"USING\s+(.+?)(?:\s*$|$)")
FUNCTIONAL_PAT = re.compile(r"The\s+(.+?)\s+functional is recognized")
C6_PAT = re.compile(rf"molecular C6\(AA\)\s+\[au\]\s+=\s+{FLOAT_PAT}")

# Parameter patterns
S_PARAM_PAT = re.compile(rf"^\s*(s\d|rs\d)\s+scaling factor\s+:\s+{FLOAT_PAT}")
DAMPING_PARAM_PAT = re.compile(rf"^\s*Damping factor\s+(alpha\d)\s+:\s+{FLOAT_PAT}")
K_PARAMS_PAT = re.compile(rf"ad hoc parameters k1-k3\s+:\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}")

# Energy patterns
EDISP_PAT = re.compile(rf"Edisp/kcal,au:\s+{FLOAT_PAT}\s+{FLOAT_PAT}")
E6_PAT = re.compile(rf"E6\s+/kcal\s+:\s+{FLOAT_PAT}")
E8_PAT = re.compile(rf"E8\s+/kcal\s+:\s+{FLOAT_PAT}")
E8_PERCENT_PAT = re.compile(rf"%\s+E8\s+:\s+{FLOAT_PAT}")
FINAL_CORRECTION_PAT = re.compile(rf"Dispersion correction\s+{FLOAT_PAT}")


class DispersionParser:
    """Parser for ORCA DFT-D dispersion correction blocks."""

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.method: str | None = None
        self.version: str | None = None
        self.rev: str | None = None
        self.damping: str | None = None
        self.functional: str | None = None
        self.molecular_c6_au: float | None = None
        self.parameters: dict[str, float] = {}
        self.e_disp_kcal: float | None = None
        self.e_disp_au: float | None = None
        self.e6_kcal: float | None = None
        self.e8_kcal: float | None = None
        self.e8_percentage: float | None = None

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line marks the start of a dispersion block.
        """
        if state.parsed_dispersion:
            return False
        return bool(START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse the entire dispersion block.
        """
        self._reset_state()
        logger.debug("Parsing dispersion block.")

        in_parameters_section = False

        for line in iterator:
            # Skip empty lines
            if not line.strip():
                continue

            # Parse method/version line (e.g., "DFTD3 V3.1  Rev 1")
            if method_match := METHOD_PAT.search(line):
                self.method = method_match.group(1)
                self.version = method_match.group(2)
                self.rev = method_match.group(3)
                logger.debug(f"Found method: {self.method} V{self.version} Rev {self.rev}")
                continue

            # Parse damping line
            if "USING" in line:
                if damping_match := DAMPING_PAT.search(line):
                    self.damping = damping_match.group(1).strip()
                    logger.debug(f"Found damping: {self.damping}")
                continue

            # Parse functional line
            if "functional is recognized" in line:
                if func_match := FUNCTIONAL_PAT.search(line):
                    self.functional = func_match.group(1).strip()
                    logger.debug(f"Found functional: {self.functional}")
                continue

            # Parse molecular C6
            if "molecular C6(AA)" in line:
                if c6_match := C6_PAT.search(line):
                    self.molecular_c6_au = float(c6_match.group(1))
                    logger.debug(f"Found molecular C6: {self.molecular_c6_au}")
                continue

            # Detect start of parameters section
            if "DFT-D V3" in line or "DFT-D V4" in line:
                in_parameters_section = True
                continue

            # Parse parameters section
            if in_parameters_section:
                # End parameters section when we hit empty line followed by energies
                if "Edisp/kcal,au:" in line:
                    in_parameters_section = False
                    # Parse energy line here
                    if edisp_match := EDISP_PAT.search(line):
                        self.e_disp_kcal = float(edisp_match.group(1))
                        self.e_disp_au = float(edisp_match.group(2))
                        logger.debug(f"Found Edisp: {self.e_disp_kcal} kcal/mol, {self.e_disp_au} au")
                    continue

                # Parse s-scaling parameters (s6, rs6, s8, rs8)
                if s_match := S_PARAM_PAT.match(line.strip()):
                    param_name = s_match.group(1)
                    param_value = float(s_match.group(2))
                    self.parameters[param_name] = param_value
                    logger.debug(f"Found parameter {param_name}: {param_value}")
                    continue

                # Parse damping factor parameters (alpha6, alpha8)
                if damp_match := DAMPING_PARAM_PAT.match(line.strip()):
                    param_name = damp_match.group(1)
                    param_value = float(damp_match.group(2))
                    self.parameters[param_name] = param_value
                    logger.debug(f"Found parameter {param_name}: {param_value}")
                    continue

                # Parse ad hoc k parameters (k1, k2, k3)
                if k_match := K_PARAMS_PAT.search(line):
                    self.parameters["k1"] = float(k_match.group(1))
                    self.parameters["k2"] = float(k_match.group(2))
                    self.parameters["k3"] = float(k_match.group(3))
                    logger.debug(
                        f"Found k parameters: k1={self.parameters['k1']}, k2={self.parameters['k2']}, k3={self.parameters['k3']}"
                    )
                    continue

            # Parse E6, E8, and percentage lines
            if "E6" in line and "/kcal" in line:
                if e6_match := E6_PAT.search(line):
                    self.e6_kcal = float(e6_match.group(1))
                    logger.debug(f"Found E6: {self.e6_kcal}")
                continue

            if "E8" in line and "/kcal" in line and "%" not in line:
                if e8_match := E8_PAT.search(line):
                    self.e8_kcal = float(e8_match.group(1))
                    logger.debug(f"Found E8: {self.e8_kcal}")
                continue

            if "% E8" in line:
                if e8_pct_match := E8_PERCENT_PAT.search(line):
                    self.e8_percentage = float(e8_pct_match.group(1))
                    logger.debug(f"Found E8%: {self.e8_percentage}")
                continue

            # Parse final correction line
            if "Dispersion correction" in line and "---" not in line:
                if final_match := FINAL_CORRECTION_PAT.search(line):
                    # This is the definitive final energy value with full precision
                    self.e_disp_au = float(final_match.group(1))
                    logger.debug(f"Found final dispersion correction: {self.e_disp_au}")
                continue

            # Check if we've reached the end of the dispersion section
            if line.strip().startswith("---") and self.e_disp_au is not None:
                # We've reached the closing dashes, block is complete
                break

        # Validate that we got the critical energy value
        if self.e_disp_au is None:
            raise ParsingError("Dispersion block matched but no final energy found.")

        # Ensure method is set
        if self.method is None:
            self.method = "DFTD3"  # Default assumption if not explicitly found
            logger.warning("Dispersion method not found; assuming DFTD3")

        # Create the dispersion correction model
        state.dispersion = DispersionCorrection(
            method=self.method,
            functional=self.functional,
            damping=self.damping,
            molecular_c6_au=self.molecular_c6_au,
            parameters=self.parameters if self.parameters else None,
            e_disp_kcal=self.e_disp_kcal,
            e_disp_au=self.e_disp_au,
            e6_kcal=self.e6_kcal,
            e8_kcal=self.e8_kcal,
            e8_percentage=self.e8_percentage,
        )

        state.parsed_dispersion = True
        logger.debug("Finished parsing dispersion block.")
