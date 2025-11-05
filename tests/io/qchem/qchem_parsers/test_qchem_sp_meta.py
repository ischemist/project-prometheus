"""
Tests for the QChem metadata block parser.

These tests verify that the metadata parser correctly extracts the Q-Chem software version,
which is critical because other parsers (like ScfParser) depend on it for version-specific
pattern matching.
"""

import pytest

from calcflow.common.patterns import VersionSpec
from calcflow.common.results import CalculationMetadata
from calcflow.io.qchem.blocks.metadata import MetadataParser
from calcflow.io.state import ParseState


@pytest.mark.unit
def test_metadata_parser_matches_version_line():
    """
    Unit test: verify MetadataParser.matches() recognizes version lines.
    """

    parser = MetadataParser()
    state = ParseState(raw_output="")

    # Should match Q-Chem version line (from actual output format)
    version_line = " Q-Chem 6.2, Q-Chem, Inc., Pleasanton, CA (2024)"
    assert parser.matches(version_line, state) is True


@pytest.mark.unit
def test_metadata_parser_does_not_match_non_metadata():
    """
    Unit test: verify MetadataParser.matches() ignores non-metadata lines.
    """

    parser = MetadataParser()
    state = ParseState(raw_output="")

    # Should not match random lines
    random_line = "Some arbitrary output from the calculation"
    assert parser.matches(random_line, state) is False


@pytest.mark.unit
def test_metadata_parser_stops_after_version():
    """
    Unit test: verify MetadataParser.matches() returns False once version is parsed.
    Once we have the version, we're done - no need to check further lines.
    """

    parser = MetadataParser()
    # Create a state with version already populated
    state = ParseState(raw_output="")
    state.metadata = CalculationMetadata(
        software_name="QChem",
        software_version="6.2",
    )

    # Even with valid metadata lines, should not match because version is already set
    version_line = " Q-Chem 6.3, Q-Chem, Inc., Pleasanton, CA (2025)"
    assert parser.matches(version_line, state) is False


@pytest.mark.unit
def test_metadata_parser_sets_software_name_and_version():
    """
    Unit test: verify MetadataParser.parse() sets both software_name and software_version.
    """

    parser = MetadataParser()
    state = ParseState(raw_output="")

    version_line = " Q-Chem 6.2, Q-Chem, Inc., Pleasanton, CA (2024)"
    parser.parse(iter([]), version_line, state)

    assert state.metadata.software_name == "Q-Chem"
    assert state.metadata.software_version == "6.2"
    assert state.parsed_metadata is True


@pytest.mark.unit
def test_version_spec_normalization():
    """
    Unit test: verify that VersionSpec.version property normalizes versions correctly.
    Versions with patch=0 should omit the patch number (e.g., "6.2" not "6.2.0").
    """

    v1 = VersionSpec.from_str("6.2")
    assert v1.version == "6.2"

    v2 = VersionSpec.from_str("6.2.0")
    assert v2.version == "6.2"

    v3 = VersionSpec.from_str("6.2.1")
    assert v3.version == "6.2.1"
