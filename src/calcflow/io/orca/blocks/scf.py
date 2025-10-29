import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.models import ScfEnergyComponents, ScfIteration, ScfResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
SCF_CONVERGED_LINE_PAT = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
SCF_DIIS_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
SCF_ENERGY_COMPONENTS_START_PAT = re.compile(r"TOTAL SCF ENERGY")
SCF_NUCLEAR_REP_PAT = re.compile(r"^\s*Nuclear Repulsion\s*:\s*(-?\d+\.\d+)")
SCF_ELECTRONIC_PAT = re.compile(r"^\s*Electronic Energy\s*:\s*(-?\d+\.\d+)")
SCF_ONE_ELECTRON_PAT = re.compile(r"^\s*One Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_TWO_ELECTRON_PAT = re.compile(r"^\s*Two Electron Energy\s*:\s*(-?\d+\.\d+)")


class ScfParser:
    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.converged: bool = False
        self.n_iterations: int = 0
        self.nuclear_rep_eh: float | None = None
        self.electronic_eh: float | None = None
        self.one_electron_eh: float | None = None
        self.two_electron_eh: float | None = None
        self.iterations: list[ScfIteration] = []

    def matches(self, line: str, state: ParseState) -> bool:
        return not state.parsed_scf and "D-I-I-S" in line

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        self._reset_state()
        logger.debug("Parsing SCF block.")

        # Simplified parsing loop for demonstration
        # A more robust version would use your state machine logic

        # Parse iterations
        next(iterator, None)  # consume dashes
        for line in iterator:
            if not line.strip():
                break
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

        # Find convergence and components
        for line in iterator:
            if SCF_CONVERGED_LINE_PAT.search(line):
                self.converged = True
                self.n_iterations = int(SCF_CONVERGED_LINE_PAT.search(line).group(1))  # type: ignore

            if SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                # Now we are in the components block
                for comp_line in iterator:
                    if nr_match := SCF_NUCLEAR_REP_PAT.search(comp_line):
                        self.nuclear_rep_eh = float(nr_match.group(1))
                    if el_match := SCF_ELECTRONIC_PAT.search(comp_line):
                        self.electronic_eh = float(el_match.group(1))
                    if one_el_match := SCF_ONE_ELECTRON_PAT.search(comp_line):
                        self.one_electron_eh = float(one_el_match.group(1))
                    if two_el_match := SCF_TWO_ELECTRON_PAT.search(comp_line):
                        self.two_electron_eh = float(two_el_match.group(1))

                    if "FINAL SINGLE POINT ENERGY" in comp_line:
                        state.buffered_line = comp_line
                        break
                break  # Exit components search

        if not self.iterations:
            raise ParsingError("SCF block matched but no iterations found.")

        assert self.nuclear_rep_eh is not None
        assert self.electronic_eh is not None
        assert self.one_electron_eh is not None
        assert self.two_electron_eh is not None

        components = ScfEnergyComponents(
            nuclear_repulsion=self.nuclear_rep_eh,
            electronic_eh=self.electronic_eh,
            one_electron_eh=self.one_electron_eh,
            two_electron_eh=self.two_electron_eh,
        )

        state.scf = ScfResults(
            converged=self.converged,
            energy=self.iterations[-1].energy,
            n_iterations=self.n_iterations if self.converged else len(self.iterations),
            iterations=self.iterations,
            components=components,
        )
        state.parsed_scf = True
