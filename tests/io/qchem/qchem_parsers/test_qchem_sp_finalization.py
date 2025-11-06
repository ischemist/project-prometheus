"""Tests for QChem finalization parser (termination status detection)."""

import pytest

from calcflow.common.results import CalculationResult
from calcflow.io.qchem.blocks.finalization import TerminationParser
from calcflow.io.state import ParseState

# =============================================================================
# UNIT TESTS: Fast, isolated tests of TerminationParser logic
# =============================================================================


@pytest.fixture
def parse_state_empty() -> ParseState:
    """Create an empty ParseState for unit tests."""
    return ParseState(raw_output="")


class TestTerminationParserMatches:
    """Tests for TerminationParser.matches() method."""

    @pytest.mark.unit
    def test_matches_normal_termination_pattern(self, parse_state_empty):
        """Should match when line contains the Q-Chem thank you message."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "        *  Thank you very much for using Q-Chem.  Have a nice day.  *"

        assert parser.matches(line, state)

    @pytest.mark.unit
    def test_matches_error_pattern_uppercase(self, parse_state_empty):
        """Should match when line contains ERROR in uppercase."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "ERROR: SCF did not converge"

        assert parser.matches(line, state)

    @pytest.mark.unit
    def test_matches_error_pattern_lowercase(self, parse_state_empty):
        """Should match when line contains error in lowercase."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "error: Geometry optimization failed"

        assert parser.matches(line, state)

    @pytest.mark.unit
    def test_matches_aborting_pattern(self, parse_state_empty):
        """Should match when line contains 'aborting' keyword."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "Aborting SCF due to convergence failure"

        assert parser.matches(line, state)

    @pytest.mark.unit
    def test_matches_failed_pattern(self, parse_state_empty):
        """Should match when line contains 'failed' keyword."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "Job failed during basis set parsing"

        assert parser.matches(line, state)

    @pytest.mark.unit
    def test_no_match_if_status_already_set(self, parse_state_empty):
        """Should not match if termination_status is already set to NORMAL."""
        parser = TerminationParser()
        state = parse_state_empty
        state.termination_status = "NORMAL"
        line = "        *  Thank you very much for using Q-Chem.  Have a nice day.  *"

        assert not parser.matches(line, state)

    @pytest.mark.unit
    def test_no_match_if_status_already_set_error(self, parse_state_empty):
        """Should not match if termination_status is already set to ERROR."""
        parser = TerminationParser()
        state = parse_state_empty
        state.termination_status = "ERROR"
        line = "        *  Thank you very much for using Q-Chem.  Have a nice day.  *"

        assert not parser.matches(line, state)

    @pytest.mark.unit
    def test_no_match_unrelated_line(self, parse_state_empty):
        """Should not match lines without termination patterns."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "      Alpha    Occupation (Alpha)    Beta    Occupation (Beta)"

        assert not parser.matches(line, state)


class TestTerminationParserParse:
    """Tests for TerminationParser.parse() method."""

    @pytest.mark.unit
    def test_parse_sets_normal_termination(self, parse_state_empty):
        """Should set termination_status to NORMAL for thank you message."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "        *  Thank you very much for using Q-Chem.  Have a nice day.  *"

        parser.parse(iter([]), line, state)

        assert state.termination_status == "NORMAL"

    @pytest.mark.unit
    def test_parse_sets_error_termination_error_uppercase(self, parse_state_empty):
        """Should set termination_status to ERROR for ERROR pattern."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "ERROR: SCF failed to converge"

        parser.parse(iter([]), line, state)

        assert state.termination_status == "ERROR"

    @pytest.mark.unit
    def test_parse_sets_error_termination_error_lowercase(self, parse_state_empty):
        """Should set termination_status to ERROR for error pattern."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "error: Geometry optimization failed"

        parser.parse(iter([]), line, state)

        assert state.termination_status == "ERROR"

    @pytest.mark.unit
    def test_parse_sets_error_termination_aborting(self, parse_state_empty):
        """Should set termination_status to ERROR for aborting pattern."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "Aborting calculation due to memory issues"

        parser.parse(iter([]), line, state)

        assert state.termination_status == "ERROR"

    @pytest.mark.unit
    def test_parse_sets_error_termination_failed(self, parse_state_empty):
        """Should set termination_status to ERROR for failed pattern."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "Job failed at basis set allocation"

        parser.parse(iter([]), line, state)

        assert state.termination_status == "ERROR"

    @pytest.mark.unit
    def test_parse_does_not_consume_iterator(self, parse_state_empty):
        """Should not consume any lines from the iterator (single-line parser)."""
        parser = TerminationParser()
        state = parse_state_empty
        line = "        *  Thank you very much for using Q-Chem.  Have a nice day.  *"
        test_lines = ["line1", "line2", "line3"]
        iterator = iter(test_lines)

        parser.parse(iterator, line, state)

        # Iterator should still have all original lines
        remaining = list(iterator)
        assert remaining == test_lines


class TestTerminationParserEdgeCases:
    """Edge case tests for TerminationParser."""

    @pytest.mark.unit
    def test_case_insensitive_error_matching(self, parse_state_empty):
        """ERROR matching should be case-insensitive."""
        parser = TerminationParser()
        state = parse_state_empty
        test_cases = ["ERROR:", "error:", "Error:", "ErRoR:"]

        for line in test_cases:
            state.termination_status = "UNKNOWN"
            assert parser.matches(line, state), f"Failed to match: {line}"

    @pytest.mark.unit
    def test_case_insensitive_aborting_matching(self, parse_state_empty):
        """'aborting' matching should be case-insensitive."""
        parser = TerminationParser()
        state = parse_state_empty
        test_cases = ["aborting", "Aborting", "ABORTING", "AbOrTiNg"]

        for line in test_cases:
            state.termination_status = "UNKNOWN"
            assert parser.matches(line, state), f"Failed to match: {line}"

    @pytest.mark.unit
    def test_case_insensitive_failed_matching(self, parse_state_empty):
        """'failed' matching should be case-insensitive."""
        parser = TerminationParser()
        state = parse_state_empty
        test_cases = ["failed", "Failed", "FAILED", "FaIlEd"]

        for line in test_cases:
            state.termination_status = "UNKNOWN"
            assert parser.matches(line, state), f"Failed to match: {line}"

    @pytest.mark.unit
    def test_error_pattern_in_middle_of_line(self, parse_state_empty):
        """Should match ERROR/error patterns anywhere in the line."""
        parser = TerminationParser()
        state = parse_state_empty
        lines = [
            "Before ERROR: SCF convergence failure",
            "SCF convergence error: Maximum iterations reached",
            "Status: aborting due to memory",
            "The calculation failed at step 5",
        ]

        for line in lines:
            state.termination_status = "UNKNOWN"
            assert parser.matches(line, state), f"Failed to match: {line}"

    @pytest.mark.unit
    def test_normal_termination_requires_full_message(self, parse_state_empty):
        """Normal termination requires the specific thank you message."""
        parser = TerminationParser()
        state = parse_state_empty
        # Partial messages should not match
        partial_messages = [
            "Thank you for using Q-Chem",
            "Have a nice day",
            "Q-Chem is done",
        ]

        for line in partial_messages:
            state.termination_status = "UNKNOWN"
            assert not parser.matches(line, state), f"Should not match: {line}"


# =============================================================================
# CONTRACT TESTS: Multi-component integration with parsed files
# =============================================================================


class TestFinalizationContractNormalTermination:
    """Contract tests verifying finalization parser produces correct structure."""

    @pytest.mark.contract
    def test_specific_54_sp_termination(self, parsed_qchem_54_h2o_sp_data: CalculationResult):
        """QChem 5.4 SP calculation should have NORMAL termination."""
        assert parsed_qchem_54_h2o_sp_data.termination_status == "NORMAL"

    @pytest.mark.contract
    def test_specific_62_sp_termination(self, parsed_qchem_62_h2o_sp_data: CalculationResult):
        """QChem 6.2 SP calculation should have NORMAL termination."""
        assert parsed_qchem_62_h2o_sp_data.termination_status == "NORMAL"

    @pytest.mark.contract
    def test_specific_54_uks_tddft_termination(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
        """QChem 5.4 UKS-TDDFT calculation should have NORMAL termination."""
        assert parsed_qchem_54_h2o_uks_tddft_data.termination_status == "NORMAL"

    @pytest.mark.contract
    def test_specific_62_uks_tddft_termination(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """QChem 6.2 UKS-TDDFT calculation should have NORMAL termination."""
        assert parsed_qchem_62_h2o_uks_tddft_data.termination_status == "NORMAL"

    @pytest.mark.contract
    def test_specific_62_rks_tddft_termination(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
        """QChem 6.2 RKS-TDDFT calculation should have NORMAL termination."""
        assert parsed_qchem_62_h2o_rks_tddft_data.termination_status == "NORMAL"

    @pytest.mark.contract
    def test_mom_sp_job1_termination(self, parsed_qchem_54_h2o_mom_sp_job1: CalculationResult):
        """QChem MOM SP job1 should have NORMAL termination."""
        assert parsed_qchem_54_h2o_mom_sp_job1.termination_status == "NORMAL"

    @pytest.mark.contract
    def test_mom_xas_job1_termination(self, parsed_qchem_54_h2o_mom_xas_job1: CalculationResult):
        """QChem MOM XAS job1 should have NORMAL termination."""
        assert parsed_qchem_54_h2o_mom_xas_job1.termination_status == "NORMAL"
