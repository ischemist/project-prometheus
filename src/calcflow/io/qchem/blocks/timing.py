"""
Block parser for QChem timing information (CPU and wall time).

QChem provides timing in the format:
"Total job time:  0.77s(wall), 0.22s(cpu)"
"""

import re
from collections.abc import Iterator

from calcflow.common.results import TimingResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

# Pattern for total job time: "Total job time:  0.77s(wall), 0.22s(cpu)"
TOTAL_TIME_PAT = re.compile(r"Total job time:\s+(\d+\.\d+)s\(wall\),\s+(\d+\.\d+)s\(cpu\)")


class TimingParser:
    """
    Parses QChem timing information, extracting both wall time and CPU time.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """Matches on the total job time line."""
        if state.parsed_timing:
            return False
        return bool(TOTAL_TIME_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parses the total job time line to extract wall and CPU times.
        """
        match = TOTAL_TIME_PAT.search(start_line)
        if match:
            wall_time = float(match.group(1))
            cpu_time = float(match.group(2))

            state.timing = TimingResults(
                total_wall_time_seconds=wall_time,
                total_cpu_time_seconds=cpu_time,
            )
            state.parsed_timing = True
            logger.debug(f"Parsed timing: wall={wall_time}s, cpu={cpu_time}s")
