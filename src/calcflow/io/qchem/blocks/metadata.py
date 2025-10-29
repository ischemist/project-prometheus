import re
from collections.abc import Iterator

from calcflow.common.patterns import VersionSpec
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Regex Patterns ---
QCHEM_VERSION_PAT = re.compile(r"A\. Remainder of .* Q-Chem (\d+\.\d+(\.\d+)?), Q-Chem, Inc\.")
HOST_PAT = re.compile(r"^\s*Host:\s*(\S+)")
RUN_DATE_PAT = re.compile(r"^\s*Q-Chem begins on\s*(.*)")


class MetadataParser:
    """
    Parses general metadata lines like Q-Chem version, host, and run date.
    This parser is designed to be checked on many lines, but only act once for
    each piece of metadata it finds.
    """

    _patterns: dict[str, re.Pattern] = {
        "program_version": QCHEM_VERSION_PAT,
        "host": HOST_PAT,
        "run_date": RUN_DATE_PAT,
    }

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Checks if the line matches any metadata pattern that hasn't been parsed yet.
        """
        # Check if all metadata is already found to short-circuit
        if all([state.metadata.program_version, state.metadata.run_date]):
            return False

        for key, pattern in self._patterns.items():
            if getattr(state.metadata, key) is None and pattern.search(line):
                return True
        return False

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parses the matched metadata line and updates state. Consumes only the current line.
        """
        for key, pattern in self._patterns.items():
            # Check flag again to be safe
            if getattr(state.metadata, key) is not None:
                continue

            match = pattern.search(start_line)
            if match:
                value = match.group(1).strip()

                # Special handling for version
                if key == "program_version":
                    # Use model_copy to create a mutable copy, update, and reassign
                    new_metadata = state.metadata.model_copy(update={key: VersionSpec.from_str(value).version})
                else:
                    new_metadata = state.metadata.model_copy(update={key: value})

                state.metadata = new_metadata
                logger.debug(f"Parsed metadata - {key}: {getattr(state.metadata, key)}")
                # We only parse one piece of metadata per line, so we can stop.
                return
