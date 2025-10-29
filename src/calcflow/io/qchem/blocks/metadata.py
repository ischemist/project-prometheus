import re
from collections.abc import Iterator

from calcflow.common.patterns import VersionSpec
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Regex Patterns ---
# Matches: " Q-Chem 6.2, Q-Chem, Inc., Pleasanton, CA (2024)"
QCHEM_VERSION_PAT = re.compile(r"Q-Chem (\d+\.\d+(\.\d+)?), Q-Chem, Inc\.")
HOST_PAT = re.compile(r"^\s*Host:\s*(\S+)")
RUN_DATE_PAT = re.compile(r"^\s*Q-Chem begins on\s*(.*)")


class MetadataParser:
    """
    Parses Q-Chem program version from metadata lines.
    The primary goal is to extract the Q-Chem version, which is required by other parsers.
    Once the version is found, parsing is complete.
    """

    _patterns: dict[str, re.Pattern] = {
        "program_version": QCHEM_VERSION_PAT,
        "host": HOST_PAT,
        "run_date": RUN_DATE_PAT,
    }

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Checks if the line matches any metadata pattern and version hasn't been parsed yet.
        Once version is parsed, we're done - return False to stop checking.
        """
        # Early exit: once we have the version, we're done
        if state.metadata.program_version is not None:
            return False

        # Check if any metadata pattern matches
        for pattern in self._patterns.values():
            if pattern.search(line):
                return True
        return False

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parses metadata from the matched line. Consumes only the current line.

        After extracting the Q-Chem version, sets the completion flag.
        Other fields (host, run_date) are extracted as a bonus but are not required.
        """
        for key, pattern in self._patterns.items():
            # Skip if this field is already populated
            if getattr(state.metadata, key) is not None:
                continue

            match = pattern.search(start_line)
            if match:
                value = match.group(1).strip()

                # Special handling for version: normalize using VersionSpec
                if key == "program_version":
                    normalized_version = VersionSpec.from_str(value).version
                    new_metadata = state.metadata.model_copy(update={key: normalized_version})
                    state.metadata = new_metadata
                    # Once we have the version, we're done
                    state.parsed_metadata = True
                    logger.debug(f"Parsed Q-Chem version: {normalized_version}")
                    return
                else:
                    # Extract other fields as bonus if found
                    new_metadata = state.metadata.model_copy(update={key: value})
                    state.metadata = new_metadata
                    logger.debug(f"Parsed metadata - {key}: {value}")
