import re
from collections.abc import Iterator

from calcflow.io.state import ParseState

FINAL_ENERGY_PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
NORMAL_TERM_PAT = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")


class FinalEnergyParser:
    def matches(self, line: str, state: ParseState) -> bool:
        return state.final_energy is None and bool(FINAL_ENERGY_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        match = FINAL_ENERGY_PAT.search(start_line)
        if match:
            state.final_energy = float(match.group(1))


class TerminationParser:
    def matches(self, line: str, state: ParseState) -> bool:
        return state.termination_status == "UNKNOWN" and bool(NORMAL_TERM_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        state.termination_status = "NORMAL"
