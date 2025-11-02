import dataclasses
import re
from collections.abc import Iterator

from calcflow.common.patterns import VersionSpec
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Regex Pattern ---
# Matches: " Q-Chem 6.2, Q-Chem, Inc., Pleasanton, CA (2024)"
QCHEM_VERSION_PAT = re.compile(r"Q-Chem (\d+\.\d+(\.\d+)?), Q-Chem, Inc\.")


class MetadataParser:
    """
    Parses Q-Chem software version from metadata lines.
    The version is required by other parsers for version-specific pattern matching.
    """

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Checks if the line contains the Q-Chem version and hasn't been parsed yet.
        """
        if state.metadata.software_version is not None:
            return False
        return QCHEM_VERSION_PAT.search(line) is not None

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Extracts the Q-Chem version, normalizes it, and sets the completion flag.
        """
        match = QCHEM_VERSION_PAT.search(start_line)
        if match:
            version_str = match.group(1).strip()
            normalized_version = VersionSpec.from_str(version_str).version
            new_metadata = dataclasses.replace(
                state.metadata,
                software_name="Q-Chem",
                software_version=normalized_version,
            )
            state.metadata = new_metadata
            state.parsed_metadata = True
            logger.debug(f"Parsed Q-Chem version: {normalized_version}")
