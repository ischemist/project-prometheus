"""
Tests for the QChem NTO (Natural Transition Orbital) decomposition block parser.

These tests verify that the NTO parser correctly extracts decomposition data
including hole-electron transitions, weights, and omega values from both
restricted (RKS) and unrestricted (UKS) TDDFT calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure
- integration: multiple components working together
- regression: exact numerical values match expected

Format notes:
- RKS: "Singlet N :" headers, all contributions on same spin
- UKS: "Excited State N :" headers, separate "Alpha spin:" and "Beta spin:" sections
- Each contribution: "H- X -> L+ Y: ±0.XXXX ( XX.X%)"
- Total omega: "omega = XXX.X%"
"""

import pytest

from calcflow.common.results import CalculationResult, NTOStateAnalysis
from calcflow.io.qchem.blocks.tddft.nto import NTOParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA (from ex-nto.md)
# =============================================================================

# RKS TDDFT data (parsed_qchem_62_h2o_rks_tddft_data)
EXPECTED_RKS_NUM_NTO_STATES = 10

# State 1: Single dominant contribution
EXPECTED_RKS_STATE_1_NUM_CONTRIBUTIONS = 1
EXPECTED_RKS_STATE_1_CONTRIBUTION_1 = {
    "hole_offset": 0,
    "electron_offset": 0,
    "weight_percent": 99.9,
    "is_alpha_spin": True,  # RKS is treated as single spin
}
EXPECTED_RKS_STATE_1_OMEGA = 100.1

# State 7: Two contributions
EXPECTED_RKS_STATE_7_NUM_CONTRIBUTIONS = 2
EXPECTED_RKS_STATE_7_CONTRIBUTION_1 = {
    "hole_offset": 0,
    "electron_offset": 2,
    "weight_percent": 52.0,
    "is_alpha_spin": True,
}
EXPECTED_RKS_STATE_7_CONTRIBUTION_2 = {
    "hole_offset": 0,
    "electron_offset": 4,
    "weight_percent": 47.5,
    "is_alpha_spin": True,
}
EXPECTED_RKS_STATE_7_OMEGA = 100.2

# State 10: Three contributions
EXPECTED_RKS_STATE_10_NUM_CONTRIBUTIONS = 3
EXPECTED_RKS_STATE_10_CONTRIBUTION_1 = {
    "hole_offset": 1,
    "electron_offset": 2,
    "weight_percent": 51.5,
    "is_alpha_spin": True,
}
EXPECTED_RKS_STATE_10_CONTRIBUTION_2 = {
    "hole_offset": 0,
    "electron_offset": 5,
    "weight_percent": 43.0,
    "is_alpha_spin": True,
}
EXPECTED_RKS_STATE_10_CONTRIBUTION_3 = {
    "hole_offset": 2,
    "electron_offset": 1,
    "weight_percent": 4.8,
    "is_alpha_spin": True,
}
EXPECTED_RKS_STATE_10_OMEGA = 100.2

# UKS TDDFT data (parsed_qchem_62_h2o_uks_tddft_data)
EXPECTED_UKS_NUM_NTO_STATES = 10

# State 1: Alpha and beta contributions
EXPECTED_UKS_STATE_1_NUM_CONTRIBUTIONS = 2
EXPECTED_UKS_STATE_1_ALPHA = {
    "hole_offset": 0,
    "electron_offset": 0,
    "weight_percent": 49.9,
    "is_alpha_spin": True,
}
EXPECTED_UKS_STATE_1_BETA = {
    "hole_offset": 0,
    "electron_offset": 0,
    "weight_percent": 49.9,
    "is_alpha_spin": False,
}
EXPECTED_UKS_STATE_1_OMEGA_ALPHA = 50.1
EXPECTED_UKS_STATE_1_OMEGA_BETA = 50.1

# State 9: Multiple alpha and beta
EXPECTED_UKS_STATE_9_NUM_CONTRIBUTIONS = 4
EXPECTED_UKS_STATE_9_ALPHA_1 = {
    "hole_offset": 2,
    "electron_offset": 0,
    "weight_percent": 48.6,
    "is_alpha_spin": True,
}
EXPECTED_UKS_STATE_9_ALPHA_2 = {
    "hole_offset": 1,
    "electron_offset": 1,
    "weight_percent": 1.0,
    "is_alpha_spin": True,
}
EXPECTED_UKS_STATE_9_BETA_1 = {
    "hole_offset": 2,
    "electron_offset": 0,
    "weight_percent": 48.6,
    "is_alpha_spin": False,
}
EXPECTED_UKS_STATE_9_BETA_2 = {
    "hole_offset": 1,
    "electron_offset": 1,
    "weight_percent": 1.0,
    "is_alpha_spin": False,
}
EXPECTED_UKS_STATE_9_OMEGA = 50.1

# UKS 5.4 TDDFT data (parsed_qchem_54_h2o_uks_tddft_data)
EXPECTED_UKS_54_NUM_NTO_STATES = 10

# State 1: Single contribution per spin with opposite signs
EXPECTED_UKS_54_STATE_1_NUM_CONTRIBUTIONS = 2
EXPECTED_UKS_54_STATE_1_ALPHA = {
    "hole_offset": 0,
    "electron_offset": 0,
    "weight_percent": 49.9,
    "is_alpha_spin": True,
}
EXPECTED_UKS_54_STATE_1_BETA = {
    "hole_offset": 0,
    "electron_offset": 0,
    "weight_percent": 49.9,
    "is_alpha_spin": False,
}
EXPECTED_UKS_54_STATE_1_OMEGA_ALPHA = 50.1
EXPECTED_UKS_54_STATE_1_OMEGA_BETA = 50.1

# State 9: Multiple contributions per spin
EXPECTED_UKS_54_STATE_9_NUM_CONTRIBUTIONS = 4
EXPECTED_UKS_54_STATE_9_ALPHA_1 = {
    "hole_offset": 2,
    "electron_offset": 0,
    "weight_percent": 48.6,
    "is_alpha_spin": True,
}
EXPECTED_UKS_54_STATE_9_ALPHA_2 = {
    "hole_offset": 1,
    "electron_offset": 1,
    "weight_percent": 1.0,
    "is_alpha_spin": True,
}
EXPECTED_UKS_54_STATE_9_BETA_1 = {
    "hole_offset": 2,
    "electron_offset": 0,
    "weight_percent": 48.6,
    "is_alpha_spin": False,
}
EXPECTED_UKS_54_STATE_9_BETA_2 = {
    "hole_offset": 1,
    "electron_offset": 1,
    "weight_percent": 1.0,
    "is_alpha_spin": False,
}
EXPECTED_UKS_54_STATE_9_OMEGA = 50.1

# Numerical tolerance
WEIGHT_TOL = 0.1
OMEGA_TOL = 0.2


# =============================================================================
# UNIT TESTS: NTOParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_nto_parser_matches_sa_nto_header():
    """Unit test: verify NTOParser.matches() recognizes SA-NTO block start."""
    parser = NTOParser()
    state = ParseState(raw_output="")

    assert parser.matches("  SA-NTO Decomposition", state) is True
    assert parser.matches("           SA-NTO Decomposition", state) is True


@pytest.mark.unit
def test_nto_parser_does_not_match_non_nto_lines():
    """Unit test: verify NTOParser.matches() rejects non-NTO lines."""
    parser = NTOParser()
    state = ParseState(raw_output="")

    assert parser.matches("Singlet 1 :", state) is False
    assert parser.matches("H- 0 -> L+ 0: -0.7067 ( 99.9%)", state) is False
    assert parser.matches("omega = 100.1%", state) is False
    assert parser.matches("Random output line", state) is False


@pytest.mark.unit
def test_nto_parser_skips_if_already_parsed():
    """Unit test: verify NTOParser.matches() returns False when already parsed."""
    parser = NTOParser()
    state = ParseState(raw_output="")
    state.parsed_nto = True

    assert parser.matches("  SA-NTO Decomposition", state) is False


@pytest.mark.unit
def test_nto_parser_does_not_mutate_state_in_matches():
    """
    Unit test: verify that matches() is read-only and does not mutate state.
    Critical for parser-spec compliance.
    """
    parser = NTOParser()
    state = ParseState(raw_output="")

    line = "  SA-NTO Decomposition"
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    assert result1 is True
    assert result2 is True
    assert state.parsed_nto is False  # Should NOT be set


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_nto"])
def test_nto_analyses_has_correct_type(fixture_name: str, request):
    """Contract test: verify nto_analyses field is sequence of NTOStateAnalysis."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    assert data.tddft.nto_analyses is not None
    assert isinstance(data.tddft.nto_analyses, (list, tuple))
    assert len(data.tddft.nto_analyses) > 0
    assert all(isinstance(state, NTOStateAnalysis) for state in data.tddft.nto_analyses)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_nto"])
def test_nto_state_analysis_has_required_fields(fixture_name: str, request):
    """Contract test: verify each NTOStateAnalysis has required fields."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    nto_analyses = data.tddft.nto_analyses
    assert nto_analyses is not None

    for nto_state in nto_analyses:
        assert isinstance(nto_state.state_number, int)
        assert nto_state.state_number > 0
        assert isinstance(nto_state.contributions, (list, tuple))
        assert len(nto_state.contributions) > 0
        # Omega field depends on whether it's RKS or UKS
        assert nto_state.omega_percent is not None or nto_state.omega_alpha_percent is not None


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_nto"])
def test_nto_contribution_structure(fixture_name: str, request):
    """Contract test: verify NTOContribution structure in NTO states."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    nto_analyses = data.tddft.nto_analyses
    assert nto_analyses is not None

    for nto_state in nto_analyses:
        for contribution in nto_state.contributions:
            assert isinstance(contribution.hole_offset, int)
            assert isinstance(contribution.electron_offset, int)
            assert isinstance(contribution.weight_percent, float)
            assert isinstance(contribution.is_alpha_spin, bool)
            # Offsets should be non-negative
            assert contribution.hole_offset >= 0
            assert contribution.electron_offset >= 0


@pytest.mark.contract
def test_rks_all_contributions_single_spin(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Contract test: verify RKS contributions all have same spin (alpha=True)."""
    assert parsed_qchem_62_h2o_rks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_rks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None

    for nto_state in nto_analyses:
        for contribution in nto_state.contributions:
            # RKS should have all contributions with is_alpha_spin=True
            assert contribution.is_alpha_spin is True


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_nto_uks"])
def test_uks_contributions_have_alpha_and_beta(fixture_name: str, request):
    """Contract test: verify UKS contributions have both alpha and beta spins."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    nto_analyses = data.tddft.nto_analyses
    assert nto_analyses is not None

    # Check first state
    state_1 = nto_analyses[0]
    has_alpha = any(c.is_alpha_spin is True for c in state_1.contributions)
    has_beta = any(c.is_alpha_spin is False for c in state_1.contributions)
    assert has_alpha and has_beta, "UKS state 1 should have both alpha and beta contributions"


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
def test_nto_parsed_alongside_tddft(fixture_name: str, request):
    """Integration test: verify NTO and TDDFT both parsed together."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    assert data.tddft.tddft_states is not None
    assert data.tddft.nto_analyses is not None


@pytest.mark.integration
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
def test_nto_state_numbers_match_tddft(fixture_name: str, request):
    """Integration test: verify NTO state numbers match TDDFT state numbers."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None and tddft.tddft_states is not None
    nto_analyses = tddft.nto_analyses
    assert nto_analyses is not None

    # Number of states should match
    assert len(nto_analyses) == len(tddft.tddft_states)

    # State numbers should match
    for i, nto_state in enumerate(nto_analyses):
        tddft_state = tddft.tddft_states[i]
        assert nto_state.state_number == tddft_state.state_number


# =============================================================================
# REGRESSION TESTS: Exact numerical values (RKS)
# =============================================================================


@pytest.mark.regression
def test_rks_nto_state_count(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify exact number of NTO states parsed for RKS."""
    assert parsed_qchem_62_h2o_rks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_rks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) == EXPECTED_RKS_NUM_NTO_STATES


@pytest.mark.regression
def test_rks_nto_state_1_single_contribution(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify RKS state 1 has single dominant NTO contribution."""
    assert parsed_qchem_62_h2o_rks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_rks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 0

    state_1 = nto_analyses[0]
    assert state_1.state_number == 1
    assert len(state_1.contributions) == EXPECTED_RKS_STATE_1_NUM_CONTRIBUTIONS

    contrib = state_1.contributions[0]
    assert contrib.hole_offset == EXPECTED_RKS_STATE_1_CONTRIBUTION_1["hole_offset"]
    assert contrib.electron_offset == EXPECTED_RKS_STATE_1_CONTRIBUTION_1["electron_offset"]
    assert contrib.weight_percent == pytest.approx(
        EXPECTED_RKS_STATE_1_CONTRIBUTION_1["weight_percent"], abs=WEIGHT_TOL
    )
    assert state_1.omega_percent == pytest.approx(EXPECTED_RKS_STATE_1_OMEGA, abs=OMEGA_TOL)


@pytest.mark.regression
def test_rks_nto_state_7_two_contributions(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify RKS state 7 has two NTO contributions."""
    assert parsed_qchem_62_h2o_rks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_rks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 6

    state_7 = nto_analyses[6]
    assert state_7.state_number == 7
    assert len(state_7.contributions) == EXPECTED_RKS_STATE_7_NUM_CONTRIBUTIONS

    # First contribution
    contrib_1 = state_7.contributions[0]
    assert contrib_1.hole_offset == EXPECTED_RKS_STATE_7_CONTRIBUTION_1["hole_offset"]
    assert contrib_1.electron_offset == EXPECTED_RKS_STATE_7_CONTRIBUTION_1["electron_offset"]
    assert contrib_1.weight_percent == pytest.approx(
        EXPECTED_RKS_STATE_7_CONTRIBUTION_1["weight_percent"], abs=WEIGHT_TOL
    )

    # Second contribution
    contrib_2 = state_7.contributions[1]
    assert contrib_2.hole_offset == EXPECTED_RKS_STATE_7_CONTRIBUTION_2["hole_offset"]
    assert contrib_2.electron_offset == EXPECTED_RKS_STATE_7_CONTRIBUTION_2["electron_offset"]
    assert contrib_2.weight_percent == pytest.approx(
        EXPECTED_RKS_STATE_7_CONTRIBUTION_2["weight_percent"], abs=WEIGHT_TOL
    )

    assert state_7.omega_percent == pytest.approx(EXPECTED_RKS_STATE_7_OMEGA, abs=OMEGA_TOL)


@pytest.mark.regression
def test_rks_nto_state_10_three_contributions(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify RKS state 10 has three NTO contributions."""
    assert parsed_qchem_62_h2o_rks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_rks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 9

    state_10 = nto_analyses[9]
    assert state_10.state_number == 10
    assert len(state_10.contributions) == EXPECTED_RKS_STATE_10_NUM_CONTRIBUTIONS

    # First contribution
    contrib_1 = state_10.contributions[0]
    assert contrib_1.hole_offset == EXPECTED_RKS_STATE_10_CONTRIBUTION_1["hole_offset"]
    assert contrib_1.electron_offset == EXPECTED_RKS_STATE_10_CONTRIBUTION_1["electron_offset"]
    assert contrib_1.weight_percent == pytest.approx(
        EXPECTED_RKS_STATE_10_CONTRIBUTION_1["weight_percent"], abs=WEIGHT_TOL
    )

    # Second contribution
    contrib_2 = state_10.contributions[1]
    assert contrib_2.hole_offset == EXPECTED_RKS_STATE_10_CONTRIBUTION_2["hole_offset"]
    assert contrib_2.electron_offset == EXPECTED_RKS_STATE_10_CONTRIBUTION_2["electron_offset"]
    assert contrib_2.weight_percent == pytest.approx(
        EXPECTED_RKS_STATE_10_CONTRIBUTION_2["weight_percent"], abs=WEIGHT_TOL
    )

    # Third contribution
    contrib_3 = state_10.contributions[2]
    assert contrib_3.hole_offset == EXPECTED_RKS_STATE_10_CONTRIBUTION_3["hole_offset"]
    assert contrib_3.electron_offset == EXPECTED_RKS_STATE_10_CONTRIBUTION_3["electron_offset"]
    assert contrib_3.weight_percent == pytest.approx(
        EXPECTED_RKS_STATE_10_CONTRIBUTION_3["weight_percent"], abs=WEIGHT_TOL
    )

    assert state_10.omega_percent == pytest.approx(EXPECTED_RKS_STATE_10_OMEGA, abs=OMEGA_TOL)


# =============================================================================
# REGRESSION TESTS: Exact numerical values (UKS)
# =============================================================================


@pytest.mark.regression
def test_uks_nto_state_count(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact number of NTO states parsed for UKS."""
    assert parsed_qchem_62_h2o_uks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_uks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) == EXPECTED_UKS_NUM_NTO_STATES


@pytest.mark.regression
def test_uks_nto_state_1_alpha_beta(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify UKS state 1 has alpha and beta contributions."""
    assert parsed_qchem_62_h2o_uks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_uks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 0

    state_1 = nto_analyses[0]
    assert state_1.state_number == 1
    assert len(state_1.contributions) == EXPECTED_UKS_STATE_1_NUM_CONTRIBUTIONS

    # Find alpha and beta contributions
    alpha_contrib = None
    beta_contrib = None
    for contrib in state_1.contributions:
        if contrib.is_alpha_spin is True:
            alpha_contrib = contrib
        elif contrib.is_alpha_spin is False:
            beta_contrib = contrib

    assert alpha_contrib is not None
    assert beta_contrib is not None

    assert alpha_contrib.hole_offset == EXPECTED_UKS_STATE_1_ALPHA["hole_offset"]
    assert alpha_contrib.electron_offset == EXPECTED_UKS_STATE_1_ALPHA["electron_offset"]
    assert alpha_contrib.weight_percent == pytest.approx(EXPECTED_UKS_STATE_1_ALPHA["weight_percent"], abs=WEIGHT_TOL)

    assert beta_contrib.hole_offset == EXPECTED_UKS_STATE_1_BETA["hole_offset"]
    assert beta_contrib.electron_offset == EXPECTED_UKS_STATE_1_BETA["electron_offset"]
    assert beta_contrib.weight_percent == pytest.approx(EXPECTED_UKS_STATE_1_BETA["weight_percent"], abs=WEIGHT_TOL)

    assert state_1.omega_alpha_percent == pytest.approx(EXPECTED_UKS_STATE_1_OMEGA_ALPHA, abs=OMEGA_TOL)
    assert state_1.omega_beta_percent == pytest.approx(EXPECTED_UKS_STATE_1_OMEGA_BETA, abs=OMEGA_TOL)


@pytest.mark.regression
def test_uks_nto_state_9_multiple_alpha_beta(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify UKS state 9 has multiple alpha and beta contributions."""
    assert parsed_qchem_62_h2o_uks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_62_h2o_uks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 8

    state_9 = nto_analyses[8]
    assert state_9.state_number == 9
    assert len(state_9.contributions) == EXPECTED_UKS_STATE_9_NUM_CONTRIBUTIONS

    # Separate alpha and beta
    alpha_contribs = [c for c in state_9.contributions if c.is_alpha_spin is True]
    beta_contribs = [c for c in state_9.contributions if c.is_alpha_spin is False]

    assert len(alpha_contribs) == 2
    assert len(beta_contribs) == 2

    # Alpha contributions
    assert alpha_contribs[0].hole_offset == EXPECTED_UKS_STATE_9_ALPHA_1["hole_offset"]
    assert alpha_contribs[0].electron_offset == EXPECTED_UKS_STATE_9_ALPHA_1["electron_offset"]
    assert alpha_contribs[0].weight_percent == pytest.approx(
        EXPECTED_UKS_STATE_9_ALPHA_1["weight_percent"], abs=WEIGHT_TOL
    )

    assert alpha_contribs[1].hole_offset == EXPECTED_UKS_STATE_9_ALPHA_2["hole_offset"]
    assert alpha_contribs[1].electron_offset == EXPECTED_UKS_STATE_9_ALPHA_2["electron_offset"]
    assert alpha_contribs[1].weight_percent == pytest.approx(
        EXPECTED_UKS_STATE_9_ALPHA_2["weight_percent"], abs=WEIGHT_TOL
    )

    # Beta contributions
    assert beta_contribs[0].hole_offset == EXPECTED_UKS_STATE_9_BETA_1["hole_offset"]
    assert beta_contribs[0].electron_offset == EXPECTED_UKS_STATE_9_BETA_1["electron_offset"]
    assert beta_contribs[0].weight_percent == pytest.approx(
        EXPECTED_UKS_STATE_9_BETA_1["weight_percent"], abs=WEIGHT_TOL
    )

    assert beta_contribs[1].hole_offset == EXPECTED_UKS_STATE_9_BETA_2["hole_offset"]
    assert beta_contribs[1].electron_offset == EXPECTED_UKS_STATE_9_BETA_2["electron_offset"]
    assert beta_contribs[1].weight_percent == pytest.approx(
        EXPECTED_UKS_STATE_9_BETA_2["weight_percent"], abs=WEIGHT_TOL
    )

    assert state_9.omega_alpha_percent == pytest.approx(EXPECTED_UKS_STATE_9_OMEGA, abs=OMEGA_TOL)
    assert state_9.omega_beta_percent == pytest.approx(EXPECTED_UKS_STATE_9_OMEGA, abs=OMEGA_TOL)


# =============================================================================
# REGRESSION TESTS: Exact numerical values (UKS 5.4)
# =============================================================================


@pytest.mark.regression
@pytest.mark.parametrize(
    "fixture_name,expected_count",
    [
        ("parsed_qchem_62_h2o_rks_tddft_data", EXPECTED_RKS_NUM_NTO_STATES),
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_NUM_NTO_STATES),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_UKS_54_NUM_NTO_STATES),
    ],
)
def test_nto_state_count(fixture_name: str, expected_count: int, request):
    """Regression test: verify exact number of NTO states parsed."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    nto_analyses = data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) == expected_count


@pytest.mark.regression
def test_uks_54_nto_state_1_alpha_beta(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify UKS 5.4 state 1 has alpha and beta contributions with opposite signs."""
    assert parsed_qchem_54_h2o_uks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_54_h2o_uks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 0

    state_1 = nto_analyses[0]
    assert state_1.state_number == 1
    assert len(state_1.contributions) == EXPECTED_UKS_54_STATE_1_NUM_CONTRIBUTIONS

    # Find alpha and beta contributions
    alpha_contrib = None
    beta_contrib = None
    for contrib in state_1.contributions:
        if contrib.is_alpha_spin is True:
            alpha_contrib = contrib
        elif contrib.is_alpha_spin is False:
            beta_contrib = contrib

    assert alpha_contrib is not None
    assert beta_contrib is not None

    assert alpha_contrib.hole_offset == EXPECTED_UKS_54_STATE_1_ALPHA["hole_offset"]
    assert alpha_contrib.electron_offset == EXPECTED_UKS_54_STATE_1_ALPHA["electron_offset"]
    assert alpha_contrib.weight_percent == pytest.approx(
        EXPECTED_UKS_54_STATE_1_ALPHA["weight_percent"], abs=WEIGHT_TOL
    )

    assert beta_contrib.hole_offset == EXPECTED_UKS_54_STATE_1_BETA["hole_offset"]
    assert beta_contrib.electron_offset == EXPECTED_UKS_54_STATE_1_BETA["electron_offset"]
    assert beta_contrib.weight_percent == pytest.approx(EXPECTED_UKS_54_STATE_1_BETA["weight_percent"], abs=WEIGHT_TOL)

    assert state_1.omega_alpha_percent == pytest.approx(EXPECTED_UKS_54_STATE_1_OMEGA_ALPHA, abs=OMEGA_TOL)
    assert state_1.omega_beta_percent == pytest.approx(EXPECTED_UKS_54_STATE_1_OMEGA_BETA, abs=OMEGA_TOL)


@pytest.mark.regression
def test_uks_54_nto_state_9_multiple_alpha_beta(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify UKS 5.4 state 9 has multiple alpha and beta contributions."""
    assert parsed_qchem_54_h2o_uks_tddft_data.tddft is not None
    nto_analyses = parsed_qchem_54_h2o_uks_tddft_data.tddft.nto_analyses
    assert nto_analyses is not None
    assert len(nto_analyses) > 8

    state_9 = nto_analyses[8]
    assert state_9.state_number == 9
    assert len(state_9.contributions) == EXPECTED_UKS_54_STATE_9_NUM_CONTRIBUTIONS

    # Separate alpha and beta
    alpha_contribs = [c for c in state_9.contributions if c.is_alpha_spin is True]
    beta_contribs = [c for c in state_9.contributions if c.is_alpha_spin is False]

    assert len(alpha_contribs) == 2
    assert len(beta_contribs) == 2

    # Alpha contributions
    assert alpha_contribs[0].hole_offset == EXPECTED_UKS_54_STATE_9_ALPHA_1["hole_offset"]
    assert alpha_contribs[0].electron_offset == EXPECTED_UKS_54_STATE_9_ALPHA_1["electron_offset"]
    assert alpha_contribs[0].weight_percent == pytest.approx(
        EXPECTED_UKS_54_STATE_9_ALPHA_1["weight_percent"], abs=WEIGHT_TOL
    )

    assert alpha_contribs[1].hole_offset == EXPECTED_UKS_54_STATE_9_ALPHA_2["hole_offset"]
    assert alpha_contribs[1].electron_offset == EXPECTED_UKS_54_STATE_9_ALPHA_2["electron_offset"]
    assert alpha_contribs[1].weight_percent == pytest.approx(
        EXPECTED_UKS_54_STATE_9_ALPHA_2["weight_percent"], abs=WEIGHT_TOL
    )

    # Beta contributions
    assert beta_contribs[0].hole_offset == EXPECTED_UKS_54_STATE_9_BETA_1["hole_offset"]
    assert beta_contribs[0].electron_offset == EXPECTED_UKS_54_STATE_9_BETA_1["electron_offset"]
    assert beta_contribs[0].weight_percent == pytest.approx(
        EXPECTED_UKS_54_STATE_9_BETA_1["weight_percent"], abs=WEIGHT_TOL
    )

    assert beta_contribs[1].hole_offset == EXPECTED_UKS_54_STATE_9_BETA_2["hole_offset"]
    assert beta_contribs[1].electron_offset == EXPECTED_UKS_54_STATE_9_BETA_2["electron_offset"]
    assert beta_contribs[1].weight_percent == pytest.approx(
        EXPECTED_UKS_54_STATE_9_BETA_2["weight_percent"], abs=WEIGHT_TOL
    )

    assert state_9.omega_alpha_percent == pytest.approx(EXPECTED_UKS_54_STATE_9_OMEGA, abs=OMEGA_TOL)
    assert state_9.omega_beta_percent == pytest.approx(EXPECTED_UKS_54_STATE_9_OMEGA, abs=OMEGA_TOL)
