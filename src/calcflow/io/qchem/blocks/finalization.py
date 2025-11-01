"""
Block parser for QChem termination status messages.
"""

import re
from collections.abc import Iterator

from calcflow.io.state import ParseState
from calcflow.utils import logger

NORMAL_TERM_PAT = re.compile(r"Thank you very much for using Q-Chem\.\s+Have a nice day\.")
ERROR_TERM_PAT = re.compile(r"(ERROR:|error:|aborting|failed)", re.IGNORECASE)


class TerminationParser:
    """
    Parses lines to determine the final termination status of the job.
    This parser is designed to be checked on every line if the status is still unknown.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Matches if the termination status is not yet decided and the line contains
        a known termination message.
        """
        if state.termination_status != "UNKNOWN":
            return False
        return bool(NORMAL_TERM_PAT.search(line) or ERROR_TERM_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Sets the termination status based on the matched line. This parser does not
        consume any additional lines from the iterator.
        """
        if NORMAL_TERM_PAT.search(start_line):
            state.termination_status = "NORMAL"
            logger.debug("Found normal termination message.")
        elif ERROR_TERM_PAT.search(start_line):
            state.termination_status = "ERROR"
            logger.debug(f"Found error termination signature: {start_line.strip()}")
