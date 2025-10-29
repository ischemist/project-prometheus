"""
Tests for the QChem metadata block parser.

These tests verify that the metadata parser correctly extracts the Q-Chem program version,
which is the critical requirement. Version parsing is essential because other parsers
(like ScfParser) depend on knowing the Q-Chem version for version-specific pattern matching.

The parser also optionally extracts host and run_date as secondary metadata.
"""

import pytest

from calcflow.io.state import ParseState


@pytest.mark.unit
def test_metadata_parser_matches_version_line():
    """
    Unit test: verify MetadataParser.matches() recognizes version lines.
    """
    from calcflow.io.qchem.blocks.metadata import MetadataParser

    parser = MetadataParser()
    state = ParseState(raw_output="")

    # Should match Q-Chem version line (from actual output format)
    version_line = " Q-Chem 6.2, Q-Chem, Inc., Pleasanton, CA (2024)"
    assert parser.matches(version_line, state) is True


@pytest.mark.unit
def test_metadata_parser_matches_run_date_line():
    """
    Unit test: verify MetadataParser.matches() recognizes run date lines.
    """
    from calcflow.io.qchem.blocks.metadata import MetadataParser

    parser = MetadataParser()
    state = ParseState(raw_output="")

    # Should match "Q-Chem begins" line
    date_line = "Q-Chem begins on Sun May  4 14:52:50 2025"
    assert parser.matches(date_line, state) is True


@pytest.mark.unit
def test_metadata_parser_matches_host_line():
    """
    Unit test: verify MetadataParser.matches() recognizes host lines.
    """
    from calcflow.io.qchem.blocks.metadata import MetadataParser

    parser = MetadataParser()
    state = ParseState(raw_output="")

    # Should match "Host:" line
    host_line = "Host: login30"
    assert parser.matches(host_line, state) is True


@pytest.mark.unit
def test_metadata_parser_does_not_match_non_metadata():
    """
    Unit test: verify MetadataParser.matches() ignores non-metadata lines.
    """
    from calcflow.io.qchem.blocks.metadata import MetadataParser

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
    from calcflow.common.models import CalculationMetadata
    from calcflow.io.qchem.blocks.metadata import MetadataParser

    parser = MetadataParser()
    # Create a state with version already populated
    metadata = CalculationMetadata(
        program_name="QChem",
        program_version="6.2",
    )
    state = ParseState(raw_output="", metadata=metadata)

    # Even with valid metadata lines, should not match because version is already set
    version_line = " Q-Chem 6.3, Q-Chem, Inc., Pleasanton, CA (2025)"
    assert parser.matches(version_line, state) is False


@pytest.mark.unit
def test_version_spec_normalization():
    """
    Unit test: verify that VersionSpec.version property normalizes versions correctly.
    Versions with patch=0 should omit the patch number (e.g., "6.2" not "6.2.0").
    """
    from calcflow.common.patterns import VersionSpec

    v1 = VersionSpec.from_str("6.2")
    assert v1.version == "6.2"

    v2 = VersionSpec.from_str("6.2.0")
    assert v2.version == "6.2"

    v3 = VersionSpec.from_str("6.2.1")
    assert v3.version == "6.2.1"
