import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import ScfEnergyComponents, ScfIteration, ScfResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
SCF_CONVERGED_LINE_PAT = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")

# DIIS iteration format: Iteration Energy Delta-E RMSDP MaxDP DIISErr Damp Time(sec)
SCF_DIIS_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)

# SOSCF iteration format: Iteration Energy Delta-E RMSDP MaxDP MaxGrad Time(sec)
SCF_SOSCF_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)

SCF_ENERGY_COMPONENTS_START_PAT = re.compile(r"TOTAL SCF ENERGY")
SCF_TOTAL_ENERGY_PAT = re.compile(r"^\s*Total Energy\s*:\s*(-?\d+\.\d+)")
SCF_FINAL_INTEGRATED_ENERGY_PAT = re.compile(r"Total energy after final integration\s*:\s*(-?\d+\.\d+)")
SCF_NUCLEAR_REP_PAT = re.compile(r"^\s*Nuclear Repulsion\s*:\s*(-?\d+\.\d+)")
SCF_ELECTRONIC_PAT = re.compile(r"^\s*Electronic Energy\s*:\s*(-?\d+\.\d+)")
SCF_ONE_ELECTRON_PAT = re.compile(r"^\s*One Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_TWO_ELECTRON_PAT = re.compile(r"^\s*Two Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_XC_ENERGY_PAT = re.compile(r"^\s*E\(XC\)\s*:\s*(-?\d+\.\d+)")


class ScfParser:
    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.converged: bool = False
        self.n_iterations: int = 0
        self.final_integrated_energy: float | None = None
        self.total_energy: float | None = None
        self.nuclear_rep_eh: float | None = None
        self.electronic_eh: float | None = None
        self.one_electron_eh: float | None = None
        self.two_electron_eh: float | None = None
        self.xc_eh: float | None = None
        self.iterations: list[ScfIteration] = []

    def matches(self, line: str, state: ParseState) -> bool:
        return not state.parsed_scf and "D-I-I-S" in line

    def _is_iteration_line(self, line: str) -> bool:
        """Check if line is a numeric iteration line (starts with whitespace + number)."""
        stripped = line.strip()
        if not stripped:
            return False
        try:
            int(stripped.split()[0])
            return True
        except (ValueError, IndexError):
            return False

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        self._reset_state()
        logger.debug("Parsing SCF block.")

        # Parse DIIS iterations
        next(iterator, None)  # consume dashes line
        in_diis = True

        for line in iterator:
            # Skip empty lines and informational messages
            if not line.strip() or line.strip().startswith("***"):
                continue

            # Detect transition to SOSCF block
            if "S-O-S-C-F" in line:
                in_diis = False
                next(iterator, None)  # consume header line
                next(iterator, None)  # consume dashes
                continue

            # Stop at convergence message or energy sections
            if "SCF CONVERGED" in line or "TOTAL SCF ENERGY" in line:
                if SCF_CONVERGED_LINE_PAT.search(line):
                    self.converged = True
                    self.n_iterations = int(SCF_CONVERGED_LINE_PAT.search(line).group(1))  # type: ignore
                state.buffered_line = line
                break

            # Parse iteration line
            if self._is_iteration_line(line):
                if in_diis:
                    diis_match = SCF_DIIS_ITER_PAT.match(line.strip())
                    if diis_match:
                        vals = diis_match.groups()
                        self.iterations.append(
                            ScfIteration(
                                iteration=int(vals[0]),
                                energy=float(vals[1]),
                                delta_e_eh=float(vals[2]),
                                rmsdp=float(vals[3]),
                                maxdp=float(vals[4]),
                                diis_error=float(vals[5]),
                            )
                        )
                else:
                    # SOSCF iteration - has MaxGrad instead of DIISErr/Damp
                    soscf_match = SCF_SOSCF_ITER_PAT.match(line.strip())
                    if soscf_match:
                        vals = soscf_match.groups()
                        self.iterations.append(
                            ScfIteration(
                                iteration=int(vals[0]),
                                energy=float(vals[1]),
                                delta_e_eh=float(vals[2]),
                                rmsdp=float(vals[3]),
                                maxdp=float(vals[4]),
                                # MaxGrad stored in diis_error field for unified handling
                                diis_error=float(vals[5]),
                            )
                        )

        # Find convergence and energy components
        for line in iterator:
            # Check for "Total energy after final integration"
            if final_int_match := SCF_FINAL_INTEGRATED_ENERGY_PAT.search(line):
                self.final_integrated_energy = float(final_int_match.group(1))

            if SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                # Now we are in the TOTAL SCF ENERGY block
                for comp_line in iterator:
                    if total_match := SCF_TOTAL_ENERGY_PAT.search(comp_line):
                        self.total_energy = float(total_match.group(1))
                    if nr_match := SCF_NUCLEAR_REP_PAT.search(comp_line):
                        self.nuclear_rep_eh = float(nr_match.group(1))
                    if el_match := SCF_ELECTRONIC_PAT.search(comp_line):
                        self.electronic_eh = float(el_match.group(1))
                    if one_el_match := SCF_ONE_ELECTRON_PAT.search(comp_line):
                        self.one_electron_eh = float(one_el_match.group(1))
                    if two_el_match := SCF_TWO_ELECTRON_PAT.search(comp_line):
                        self.two_electron_eh = float(two_el_match.group(1))
                    if xc_match := SCF_XC_ENERGY_PAT.search(comp_line):
                        self.xc_eh = float(xc_match.group(1))

                    if "FINAL SINGLE POINT ENERGY" in comp_line or "SCF CONVERGENCE" in comp_line:
                        state.buffered_line = comp_line
                        break
                break  # Exit components search

        if not self.iterations:
            raise ParsingError("SCF block matched but no iterations found.")

        # Validate iteration count
        if self.converged and len(self.iterations) != self.n_iterations:
            logger.warning(
                f"SCF convergence message reports {self.n_iterations} cycles, "
                f"but parsed {len(self.iterations)} iterations"
            )

        # Use final integrated energy if available, otherwise use total energy
        final_energy = self.final_integrated_energy or self.total_energy
        if final_energy is None:
            final_energy = self.iterations[-1].energy
            logger.warning("Using last iteration energy as SCF final energy")

        assert self.nuclear_rep_eh is not None
        assert self.electronic_eh is not None
        assert self.one_electron_eh is not None
        assert self.two_electron_eh is not None

        components = ScfEnergyComponents(
            nuclear_repulsion=self.nuclear_rep_eh,
            electronic_eh=self.electronic_eh,
            one_electron_eh=self.one_electron_eh,
            two_electron_eh=self.two_electron_eh,
            xc_eh=self.xc_eh,
        )

        state.scf = ScfResults(
            converged=self.converged,
            energy=final_energy,
            n_iterations=self.n_iterations if self.converged else len(self.iterations),
            iterations=self.iterations,
            components=components,
        )
        state.parsed_scf = True
