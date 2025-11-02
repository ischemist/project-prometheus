"""
Block parser for ORCA timing information (total wall time and module-specific times).

ORCA provides:
1. Total wall time: "TOTAL RUN TIME: X days Y hours Z minutes W seconds V msec"
2. Module-specific times in a "Timings for individual modules:" section
"""

import re
from collections.abc import Iterator

from calcflow.common.results import TimingResults
from calcflow.io.state import ParseState

# Pattern for total wall time: "TOTAL RUN TIME: 0 days 0 hours 1 minutes 31 seconds 640 msec"
TOTAL_TIME_PAT = re.compile(
    r"TOTAL RUN TIME:\s+(\d+)\s+days?\s+(\d+)\s+hours?\s+(\d+)\s+minutes?\s+(\d+)\s+seconds?\s+(\d+)\s+msec"
)

# Pattern for module-specific times: "SCF iterations                   ...       76.454 sec (=   1.274 min)  91.9 %"
MODULE_TIME_PAT = re.compile(r"^(.+?)\s+\.\.\.\s+(\d+\.\d+)\s+sec")


class TimingParser:
    """
    Parses ORCA timing information, including total wall time and module-specific times.

    Strategy:
    - Collect module-specific times as we encounter them (MODULE_TIME_PAT)
    - When we hit the total time line (TOTAL_TIME_PAT), finalize the timing
    """

    def __init__(self):
        self.module_times_buffer: dict[str, float] = {}

    def matches(self, line: str, state: ParseState) -> bool:
        """Matches on total wall time line or module timing lines."""
        if state.parsed_timing:
            return False
        return bool(TOTAL_TIME_PAT.search(line)) or bool(MODULE_TIME_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parses timing information. Accumulates module times and finalizes on total time line.
        """
        # Check if this is the total time line
        total_match = TOTAL_TIME_PAT.search(start_line)
        if total_match:
            days = int(total_match.group(1))
            hours = int(total_match.group(2))
            minutes = int(total_match.group(3))
            seconds = int(total_match.group(4))
            msec = int(total_match.group(5))

            # Convert everything to seconds
            total_wall_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + msec / 1000

            # Create final timing result with accumulated module times
            module_times = self.module_times_buffer if self.module_times_buffer else None
            state.timing = TimingResults(
                total_wall_time_seconds=total_wall_seconds,
                module_times=module_times,
            )
            state.parsed_timing = True
            # Reset buffer for potential future use
            self.module_times_buffer = {}
        else:
            # Try to match module-specific timing line
            module_match = MODULE_TIME_PAT.search(start_line)
            if module_match:
                module_name = module_match.group(1).strip()
                wall_time = float(module_match.group(2))
                self.module_times_buffer[module_name] = wall_time
