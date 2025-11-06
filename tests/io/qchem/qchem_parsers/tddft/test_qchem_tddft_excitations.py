"""
Tests for the QChem TDDFT excitations block parser.

These tests verify that the excitations parser correctly extracts excited state
properties including excitation energies, transition moments, oscillator strengths,
and orbital transition amplitudes from both TDA and full TDDFT blocks.

Tests cover both restricted (RKS) and unrestricted (UKS) calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure
- integration: multiple components working together
- regression: exact numerical values match expected

Format notes:
- RKS: Single multiplicity section, no spin labels on transitions
- UKS: <S**2> values or Multiplicity, transitions may have alpha/beta labels
- Both files contain TDA block followed by full TDDFT block
- Convergence iteration table appears between the two blocks
"""

import pytest

from calcflow.common.results import CalculationResult, ExcitedState, OrbitalTransition, TddftResults
from calcflow.io.qchem.blocks.tddft.excitations import ExcitationsParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA (from ex-tddft.md)
# =============================================================================

# RKS TDDFT data (parsed_qchem_62_h2o_rks_tddft_data)
EXPECTED_RKS_TDA_STATE_1_EXCITATION_EV = 7.5954
EXPECTED_RKS_TDA_STATE_1_TOTAL_ENERGY_AU = -76.16212816
EXPECTED_RKS_TDA_STATE_1_STRENGTH = 0.0518861703
EXPECTED_RKS_TDA_STATE_1_TRANSITION = (4, 0, 0.9957)  # (from_idx, to_idx, amplitude)
EXPECTED_RKS_TDA_STATE_1_TRANS_MOM = (-0.0869, -0.5093, 0.1092)  # (X, Y, Z)

EXPECTED_RKS_TDDFT_STATE_1_EXCITATION_EV = 7.5725
EXPECTED_RKS_TDDFT_STATE_1_TOTAL_ENERGY_AU = -76.16297039
EXPECTED_RKS_TDDFT_STATE_1_STRENGTH = 0.0518397380
EXPECTED_RKS_TDDFT_STATE_1_TRANS_MOM = (-0.0870, -0.5098, 0.1093)  # (X, Y, Z)

EXPECTED_RKS_TDDFT_STATE_8_NUM_TRANSITIONS = 2  # Has D(3)->V(2) and D(5)->V(6)
EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_1 = (2, 1, 0.9122)
EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_2 = (4, 5, -0.3808)

EXPECTED_RKS_NUM_TDA_STATES = 10
EXPECTED_RKS_NUM_TDDFT_STATES = 10

# UKS TDDFT data (parsed_qchem_62_h2o_uks_tddft_data)
EXPECTED_UKS_TDA_STATE_1_EXCITATION_EV = 7.1479
EXPECTED_UKS_TDA_STATE_1_TOTAL_ENERGY_AU = -76.17857121
EXPECTED_UKS_TDA_STATE_1_STRENGTH = 0.0

EXPECTED_UKS_TDDFT_STATE_1_EXCITATION_EV = 7.1232
EXPECTED_UKS_TDDFT_STATE_1_TOTAL_ENERGY_AU = -76.17948063
EXPECTED_UKS_TDDFT_STATE_1_NUM_TRANSITIONS = 2  # Has alpha and beta

EXPECTED_UKS_TDDFT_STATE_7_EXCITATION_EV = 10.9639
EXPECTED_UKS_TDDFT_STATE_7_NUM_TRANSITIONS = 4  # Has alpha and beta versions

EXPECTED_UKS_NUM_TDA_STATES = 10
EXPECTED_UKS_NUM_TDDFT_STATES = 10

# UKS TDDFT data from QChem 5.4 (parsed_qchem_54_h2o_uks_tddft_data)
EXPECTED_UKS_54_TDA_STATE_1_EXCITATION_EV = 7.1479
EXPECTED_UKS_54_TDA_STATE_1_TOTAL_ENERGY_AU = -76.17857122
EXPECTED_UKS_54_TDA_STATE_1_STRENGTH = 0.0

EXPECTED_UKS_54_TDA_STATE_2_EXCITATION_EV = 7.5954
EXPECTED_UKS_54_TDA_STATE_2_TOTAL_ENERGY_AU = -76.16212816
EXPECTED_UKS_54_TDA_STATE_2_STRENGTH = 0.0518863673

EXPECTED_UKS_54_TDA_STATE_3_EXCITATION_EV = 9.0388
EXPECTED_UKS_54_TDA_STATE_3_TOTAL_ENERGY_AU = -76.10908350

EXPECTED_UKS_54_TDDFT_STATE_1_EXCITATION_EV = 7.1232
EXPECTED_UKS_54_TDDFT_STATE_1_TOTAL_ENERGY_AU = -76.17948063
EXPECTED_UKS_54_TDDFT_STATE_1_STRENGTH = 0.0

EXPECTED_UKS_54_TDDFT_STATE_2_EXCITATION_EV = 7.5725
EXPECTED_UKS_54_TDDFT_STATE_2_TOTAL_ENERGY_AU = -76.16297039
EXPECTED_UKS_54_TDDFT_STATE_2_STRENGTH = 0.0518397939

EXPECTED_UKS_54_TDDFT_STATE_7_EXCITATION_EV = 10.9639
EXPECTED_UKS_54_TDDFT_STATE_7_NUM_TRANSITIONS = 4  # D(4)->V(2), D(4)->V(3) with alpha/beta

EXPECTED_UKS_54_NUM_TDA_STATES = 10
EXPECTED_UKS_54_NUM_TDDFT_STATES = 10

# Numerical tolerance
ENERGY_TOL = 1e-4
STRENGTH_TOL = 1e-8


# =============================================================================
# UNIT TESTS: ExcitationsParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_excitations_parser_matches_tda_start_line():
    """Unit test: verify ExcitationsParser.matches() recognizes TDA block start."""
    parser = ExcitationsParser()
    state = ParseState(raw_output="")

    assert parser.matches("          TDDFT/TDA Excitation Energies", state) is True
    assert parser.matches("  TDDFT/TDA Excitation Energies", state) is True


@pytest.mark.unit
def test_excitations_parser_matches_tddft_start_line():
    """Unit test: verify ExcitationsParser.matches() recognizes full TDDFT block start."""
    parser = ExcitationsParser()
    state = ParseState(raw_output="")

    assert parser.matches("              TDDFT Excitation Energies", state) is True
    assert parser.matches("   TDDFT Excitation Energies", state) is True


@pytest.mark.unit
def test_excitations_parser_does_not_match_non_excitation_lines():
    """Unit test: verify ExcitationsParser.matches() rejects non-excitation lines."""
    parser = ExcitationsParser()
    state = ParseState(raw_output="")

    assert parser.matches("Excited state   1:", state) is False
    assert parser.matches("Multiplicity: Singlet", state) is False
    assert parser.matches("Strength   :     0.0518861703", state) is False
    assert parser.matches("D(    5) --> V(    1) amplitude =  0.9957", state) is False
    assert parser.matches("Random output line", state) is False


@pytest.mark.unit
def test_excitations_parser_skips_if_already_parsed():
    """Unit test: verify ExcitationsParser.matches() returns False when already parsed."""
    parser = ExcitationsParser()
    state = ParseState(raw_output="")
    state.parsed_tddft_tda = True
    state.parsed_tddft_full = True

    assert parser.matches("          TDDFT/TDA Excitation Energies", state) is False


@pytest.mark.unit
def test_excitations_parser_matches_tda_even_if_full_parsed():
    """Unit test: verify TDA can be parsed after full TDDFT."""
    parser = ExcitationsParser()
    state = ParseState(raw_output="")
    state.parsed_tddft_full = True

    # Should still match TDA
    assert parser.matches("          TDDFT/TDA Excitation Energies", state) is True


@pytest.mark.unit
def test_excitations_parser_matches_tddft_even_if_tda_parsed():
    """Unit test: verify full TDDFT can be parsed after TDA."""
    parser = ExcitationsParser()
    state = ParseState(raw_output="")
    state.parsed_tddft_tda = True

    # Should still match full TDDFT
    assert parser.matches("              TDDFT Excitation Energies", state) is True


@pytest.mark.unit
def test_excitations_parser_does_not_mutate_state_in_matches():
    """
    Unit test: verify that matches() is read-only and does not mutate state.
    Critical for parser-spec compliance.
    """
    parser = ExcitationsParser()
    state = ParseState(raw_output="")

    line = "          TDDFT/TDA Excitation Energies"
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    assert result1 is True
    assert result2 is True
    assert state.parsed_tddft_tda is False  # Should NOT be set
    assert state.parsed_tddft_full is False
    assert state.tddft is None  # Should NOT be populated


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tda_excitations"])
def test_tddft_results_has_correct_type(fixture_name: str, request):
    """Contract test: verify tddft field is TddftResults instance across all fixtures."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    assert isinstance(data.tddft, TddftResults)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tda_excitations"])
def test_tda_states_is_sequence(fixture_name: str, request):
    """Contract test: verify tda_states is a sequence of ExcitedState objects."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert isinstance(tddft.tda_states, (list, tuple))
    assert len(tddft.tda_states) > 0
    assert all(isinstance(state, ExcitedState) for state in tddft.tda_states)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
def test_tddft_states_is_sequence(fixture_name: str, request):
    """Contract test: verify tddft_states is a sequence of ExcitedState objects."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert isinstance(tddft.tddft_states, (list, tuple))
    assert len(tddft.tddft_states) > 0
    assert all(isinstance(state, ExcitedState) for state in tddft.tddft_states)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tda_excitations"])
def test_excited_state_required_fields_present(fixture_name: str, request):
    """Contract test: verify each ExcitedState has required fields with correct types."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    for excited_state in tddft.tda_states:
        assert isinstance(excited_state.state_number, int)
        assert excited_state.state_number > 0
        assert isinstance(excited_state.excitation_energy_ev, float)
        assert excited_state.excitation_energy_ev > 0
        assert isinstance(excited_state.total_energy_au, float)
        assert isinstance(excited_state.multiplicity, str)
        # Optional fields
        assert excited_state.oscillator_strength is None or isinstance(excited_state.oscillator_strength, float)
        assert isinstance(excited_state.transitions, (list, tuple))


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tda_excitations"])
def test_excited_state_transition_moment_fields_present(fixture_name: str, request):
    """Contract test: verify ExcitedState has transition moment fields with correct types."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    for excited_state in tddft.tda_states:
        # Transition moment components should be optional floats
        assert excited_state.trans_mom_x is None or isinstance(excited_state.trans_mom_x, float)
        assert excited_state.trans_mom_y is None or isinstance(excited_state.trans_mom_y, float)
        assert excited_state.trans_mom_z is None or isinstance(excited_state.trans_mom_z, float)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tda_excitations"])
def test_orbital_transition_structure(fixture_name: str, request):
    """Contract test: verify OrbitalTransition structure in excited states."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    # Find a state with transitions
    state_with_transitions = None
    for state in tddft.tda_states:
        if len(state.transitions) > 0:
            state_with_transitions = state
            break

    assert state_with_transitions is not None
    for transition in state_with_transitions.transitions:
        assert isinstance(transition, OrbitalTransition)
        assert isinstance(transition.from_idx, int)
        assert isinstance(transition.to_idx, int)
        assert isinstance(transition.amplitude, float)
        assert transition.is_alpha_spin is None or isinstance(transition.is_alpha_spin, bool)


@pytest.mark.contract
def test_rks_transitions_have_no_spin_labels(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Contract test: verify RKS transitions don't have spin labels (is_alpha_spin=None)."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    for state in tddft.tda_states:
        for transition in state.transitions:
            assert transition.is_alpha_spin is None, f"RKS should have no spin labels, got {transition}"


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["beta_orbitals"])
def test_uks_transitions_may_have_spin_labels(fixture_name: str, request):
    """Contract test: verify UKS transitions may have spin labels (is_alpha_spin=True/False)."""
    data = request.getfixturevalue(fixture_name)
    tddft = data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    # UKS calculations should have at least some transitions with spin labels
    has_spin_label = False
    for state in tddft.tda_states:
        for transition in state.transitions:
            if transition.is_alpha_spin is not None:
                has_spin_label = True
                break

    assert has_spin_label, "UKS should have spin-labeled transitions"


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize(
    "parsed_qchem_data",
    [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    indirect=True,
)
def test_tddft_parsed_alongside_scf_and_orbitals(parsed_qchem_data: CalculationResult):
    """Integration test: verify TDDFT, SCF, and orbitals all parsed together."""
    assert parsed_qchem_data.scf is not None
    assert parsed_qchem_data.orbitals is not None
    assert parsed_qchem_data.tddft is not None


@pytest.mark.integration
@pytest.mark.parametrize(
    "parsed_qchem_data",
    [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    indirect=True,
)
def test_state_numbers_are_sequential(parsed_qchem_data: CalculationResult):
    """Integration test: verify excited state numbers are 1-based and sequential."""
    tddft = parsed_qchem_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    for i, state in enumerate(tddft.tda_states, 1):
        assert state.state_number == i, f"Expected state number {i}, got {state.state_number}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "parsed_qchem_data",
    [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    indirect=True,
)
def test_tda_and_tddft_both_present(parsed_qchem_data: CalculationResult):
    """Integration test: verify both TDA and full TDDFT blocks parsed."""
    tddft = parsed_qchem_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tda_states) == len(tddft.tddft_states)


@pytest.mark.integration
@pytest.mark.parametrize(
    "parsed_qchem_data",
    [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    indirect=True,
)
def test_orbital_transitions_reference_valid_indices(parsed_qchem_data: CalculationResult):
    """
    Integration test: verify orbital transition indices are within valid range.
    Indices should be 0-based and refer to occupied/virtual orbitals.
    """
    tddft = parsed_qchem_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    for state in tddft.tda_states:
        for transition in state.transitions:
            # from_idx should be an occupied orbital (< HOMO+1)
            # to_idx should be a virtual orbital (>= LUMO)
            # For now, just check they're non-negative and reasonable
            assert transition.from_idx >= 0
            assert transition.to_idx >= 0
            assert transition.from_idx < 100  # Sanity check
            assert transition.to_idx < 100


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
@pytest.mark.parametrize(
    "parsed_qchem_data,expected_count",
    [
        ("parsed_qchem_62_h2o_rks_tddft_data", EXPECTED_RKS_NUM_TDA_STATES),
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_NUM_TDA_STATES),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_UKS_54_NUM_TDA_STATES),
    ],
    indirect=["parsed_qchem_data"],
)
def test_tda_state_count(parsed_qchem_data: CalculationResult, expected_count: int):
    """Regression test: verify exact number of TDA states parsed."""
    tddft = parsed_qchem_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) == expected_count


@pytest.mark.regression
@pytest.mark.parametrize(
    "parsed_qchem_data,expected_count",
    [
        ("parsed_qchem_62_h2o_rks_tddft_data", EXPECTED_RKS_NUM_TDDFT_STATES),
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_NUM_TDDFT_STATES),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_UKS_54_NUM_TDDFT_STATES),
    ],
    indirect=["parsed_qchem_data"],
)
def test_tddft_state_count(parsed_qchem_data: CalculationResult, expected_count: int):
    """Regression test: verify exact number of TDDFT states parsed."""
    tddft = parsed_qchem_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) == expected_count


@pytest.mark.regression
def test_rks_tda_state_energies_exact(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify exact TDA excitation and total energies for RKS."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    test_cases = [
        (
            0,
            EXPECTED_RKS_TDA_STATE_1_EXCITATION_EV,
            EXPECTED_RKS_TDA_STATE_1_TOTAL_ENERGY_AU,
            EXPECTED_RKS_TDA_STATE_1_STRENGTH,
        ),
        (1, 9.2974, -76.09958062, 0.0000004744),
        (2, 9.8215, -76.08032096, 0.1112256100),
    ]

    for state_idx, expected_energy, expected_total_energy, expected_strength in test_cases:
        assert len(tddft.tda_states) > state_idx
        state = tddft.tda_states[state_idx]
        assert state.excitation_energy_ev == pytest.approx(expected_energy, abs=ENERGY_TOL)
        assert state.total_energy_au == pytest.approx(expected_total_energy, abs=ENERGY_TOL)
        assert state.oscillator_strength == pytest.approx(expected_strength, abs=STRENGTH_TOL)


@pytest.mark.regression
def test_rks_tddft_state_energies_exact(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify exact full TDDFT excitation and total energies for RKS."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None

    test_cases = [
        (
            0,
            EXPECTED_RKS_TDDFT_STATE_1_EXCITATION_EV,
            EXPECTED_RKS_TDDFT_STATE_1_TOTAL_ENERGY_AU,
            EXPECTED_RKS_TDDFT_STATE_1_STRENGTH,
        ),
        (1, 9.2908, -76.09982099, 0.0000004614),
        (2, 9.7830, -76.08173387, 0.1055651660),
    ]

    for state_idx, expected_energy, expected_total_energy, expected_strength in test_cases:
        assert len(tddft.tddft_states) > state_idx
        state = tddft.tddft_states[state_idx]
        assert state.excitation_energy_ev == pytest.approx(expected_energy, abs=ENERGY_TOL)
        assert state.total_energy_au == pytest.approx(expected_total_energy, abs=ENERGY_TOL)
        assert state.oscillator_strength == pytest.approx(expected_strength, abs=STRENGTH_TOL)


@pytest.mark.regression
def test_rks_tda_state_1_transitions_exact(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify exact orbital transitions for RKS TDA state 1."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) > 0

    state = tddft.tda_states[0]
    assert len(state.transitions) > 0

    # State 1 should have single dominant transition
    transition = state.transitions[0]
    assert transition.from_idx == EXPECTED_RKS_TDA_STATE_1_TRANSITION[0]
    assert transition.to_idx == EXPECTED_RKS_TDA_STATE_1_TRANSITION[1]
    assert transition.amplitude == pytest.approx(EXPECTED_RKS_TDA_STATE_1_TRANSITION[2], abs=1e-4)


@pytest.mark.regression
def test_rks_tda_state_1_transition_moment_exact(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify exact transition moment components for RKS TDA state 1."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) > 0

    state = tddft.tda_states[0]
    assert state.trans_mom_x == pytest.approx(EXPECTED_RKS_TDA_STATE_1_TRANS_MOM[0], abs=1e-4)
    assert state.trans_mom_y == pytest.approx(EXPECTED_RKS_TDA_STATE_1_TRANS_MOM[1], abs=1e-4)
    assert state.trans_mom_z == pytest.approx(EXPECTED_RKS_TDA_STATE_1_TRANS_MOM[2], abs=1e-4)


@pytest.mark.regression
def test_rks_tddft_state_1_transition_moment_exact(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify exact transition moment components for RKS TDDFT state 1."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) > 0

    state = tddft.tddft_states[0]
    assert state.trans_mom_x == pytest.approx(EXPECTED_RKS_TDDFT_STATE_1_TRANS_MOM[0], abs=1e-4)
    assert state.trans_mom_y == pytest.approx(EXPECTED_RKS_TDDFT_STATE_1_TRANS_MOM[1], abs=1e-4)
    assert state.trans_mom_z == pytest.approx(EXPECTED_RKS_TDDFT_STATE_1_TRANS_MOM[2], abs=1e-4)


@pytest.mark.regression
def test_rks_tddft_state_8_multiple_transitions(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Regression test: verify multiple transitions for RKS TDDFT state 8."""
    tddft = parsed_qchem_62_h2o_rks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) > 7

    state = tddft.tddft_states[7]  # State 8 (0-indexed)
    assert len(state.transitions) == EXPECTED_RKS_TDDFT_STATE_8_NUM_TRANSITIONS

    # Check first transition
    trans1 = state.transitions[0]
    assert trans1.from_idx == EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_1[0]
    assert trans1.to_idx == EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_1[1]
    assert trans1.amplitude == pytest.approx(EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_1[2], abs=1e-4)

    # Check second transition
    trans2 = state.transitions[1]
    assert trans2.from_idx == EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_2[0]
    assert trans2.to_idx == EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_2[1]
    assert trans2.amplitude == pytest.approx(EXPECTED_RKS_TDDFT_STATE_8_TRANSITION_2[2], abs=1e-4)


@pytest.mark.regression
def test_uks_tda_state_count(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact number of TDA states parsed for UKS."""
    tddft = parsed_qchem_62_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) == EXPECTED_UKS_NUM_TDA_STATES


@pytest.mark.regression
def test_uks_tddft_state_count(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact number of TDDFT states parsed for UKS."""
    tddft = parsed_qchem_62_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) == EXPECTED_UKS_NUM_TDDFT_STATES


@pytest.mark.regression
def test_uks_tda_state_energies_exact(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact TDA energies for QChem 6.2 UKS."""
    tddft = parsed_qchem_62_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    test_cases = [
        (0, EXPECTED_UKS_TDA_STATE_1_EXCITATION_EV, EXPECTED_UKS_TDA_STATE_1_TOTAL_ENERGY_AU),
        (1, 7.5954, -76.16212816),
        (2, 9.0388, -76.10908349),
    ]

    for state_idx, expected_energy, expected_total_energy in test_cases:
        assert len(tddft.tda_states) > state_idx
        state = tddft.tda_states[state_idx]
        assert state.excitation_energy_ev == pytest.approx(expected_energy, abs=ENERGY_TOL)
        assert state.total_energy_au == pytest.approx(expected_total_energy, abs=ENERGY_TOL)


@pytest.mark.regression
def test_uks_tddft_state_energies_exact(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact full TDDFT energies for QChem 6.2 UKS."""
    tddft = parsed_qchem_62_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None

    test_cases = [
        (0, EXPECTED_UKS_TDDFT_STATE_1_EXCITATION_EV, EXPECTED_UKS_TDDFT_STATE_1_TOTAL_ENERGY_AU),
        (1, 7.5725, -76.16297039),
        (2, 9.0177, -76.10986010),
    ]

    for state_idx, expected_energy, expected_total_energy in test_cases:
        assert len(tddft.tddft_states) > state_idx
        state = tddft.tddft_states[state_idx]
        assert state.excitation_energy_ev == pytest.approx(expected_energy, abs=ENERGY_TOL)
        assert state.total_energy_au == pytest.approx(expected_total_energy, abs=ENERGY_TOL)


@pytest.mark.regression
def test_uks_tda_state_1_has_alpha_and_beta(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify UKS TDA state 1 has alpha and beta transitions."""
    tddft = parsed_qchem_62_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) > 0

    state = tddft.tda_states[0]
    assert len(state.transitions) == EXPECTED_UKS_TDDFT_STATE_1_NUM_TRANSITIONS

    # Should have both alpha and beta
    has_alpha = any(t.is_alpha_spin is True for t in state.transitions)
    has_beta = any(t.is_alpha_spin is False for t in state.transitions)
    assert has_alpha and has_beta


@pytest.mark.regression
def test_uks_tddft_state_7_multiple_transitions(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify UKS TDDFT state 7 has multiple alpha/beta transitions."""
    tddft = parsed_qchem_62_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) > 6

    state = tddft.tddft_states[6]  # State 7 (0-indexed)
    assert len(state.transitions) == EXPECTED_UKS_TDDFT_STATE_7_NUM_TRANSITIONS

    # Should have both alpha and beta transitions
    has_alpha = any(t.is_alpha_spin is True for t in state.transitions)
    has_beta = any(t.is_alpha_spin is False for t in state.transitions)
    assert has_alpha and has_beta


# =============================================================================
# REGRESSION TESTS: QChem 5.4 UKS TDDFT (parsed_qchem_54_h2o_uks_tddft_data)
# =============================================================================


@pytest.mark.regression
def test_uks_54_tda_state_energies_exact(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact TDA energies for QChem 5.4 UKS."""
    tddft = parsed_qchem_54_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None

    test_cases = [
        (
            0,
            EXPECTED_UKS_54_TDA_STATE_1_EXCITATION_EV,
            EXPECTED_UKS_54_TDA_STATE_1_TOTAL_ENERGY_AU,
            EXPECTED_UKS_54_TDA_STATE_1_STRENGTH,
        ),
        (
            1,
            EXPECTED_UKS_54_TDA_STATE_2_EXCITATION_EV,
            EXPECTED_UKS_54_TDA_STATE_2_TOTAL_ENERGY_AU,
            EXPECTED_UKS_54_TDA_STATE_2_STRENGTH,
        ),
        (2, EXPECTED_UKS_54_TDA_STATE_3_EXCITATION_EV, EXPECTED_UKS_54_TDA_STATE_3_TOTAL_ENERGY_AU, 0.0),
    ]

    for state_idx, expected_energy, expected_total_energy, expected_strength in test_cases:
        assert len(tddft.tda_states) > state_idx
        state = tddft.tda_states[state_idx]
        assert state.excitation_energy_ev == pytest.approx(expected_energy, abs=ENERGY_TOL)
        assert state.total_energy_au == pytest.approx(expected_total_energy, abs=ENERGY_TOL)
        assert state.oscillator_strength == pytest.approx(expected_strength, abs=STRENGTH_TOL)


@pytest.mark.regression
def test_uks_54_tddft_state_energies_exact(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact full TDDFT energies for QChem 5.4 UKS."""
    tddft = parsed_qchem_54_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None

    test_cases = [
        (
            0,
            EXPECTED_UKS_54_TDDFT_STATE_1_EXCITATION_EV,
            EXPECTED_UKS_54_TDDFT_STATE_1_TOTAL_ENERGY_AU,
            EXPECTED_UKS_54_TDDFT_STATE_1_STRENGTH,
        ),
        (
            1,
            EXPECTED_UKS_54_TDDFT_STATE_2_EXCITATION_EV,
            EXPECTED_UKS_54_TDDFT_STATE_2_TOTAL_ENERGY_AU,
            EXPECTED_UKS_54_TDDFT_STATE_2_STRENGTH,
        ),
        (6, EXPECTED_UKS_54_TDDFT_STATE_7_EXCITATION_EV, -76.03833772, 0.0),
    ]

    for state_idx, expected_energy, expected_total_energy, expected_strength in test_cases:
        assert len(tddft.tddft_states) > state_idx
        state = tddft.tddft_states[state_idx]
        assert state.excitation_energy_ev == pytest.approx(expected_energy, abs=ENERGY_TOL)
        assert state.total_energy_au == pytest.approx(expected_total_energy, abs=ENERGY_TOL)
        assert state.oscillator_strength == pytest.approx(expected_strength, abs=STRENGTH_TOL)


@pytest.mark.regression
def test_uks_54_tda_state_1_has_alpha_and_beta(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify QChem 5.4 UKS TDA state 1 has alpha and beta transitions."""
    tddft = parsed_qchem_54_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) > 0

    state = tddft.tda_states[0]
    assert len(state.transitions) == 2  # One alpha, one beta

    # Should have both alpha and beta
    has_alpha = any(t.is_alpha_spin is True for t in state.transitions)
    has_beta = any(t.is_alpha_spin is False for t in state.transitions)
    assert has_alpha and has_beta


@pytest.mark.regression
def test_uks_54_tddft_state_7_multiple_transitions(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify QChem 5.4 UKS TDDFT state 7 has multiple transitions."""
    tddft = parsed_qchem_54_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) > 6

    state = tddft.tddft_states[6]  # State 7 (0-indexed)
    # State 7 has D(4)->V(2) and D(4)->V(3) transitions with alpha/beta variants
    assert len(state.transitions) == EXPECTED_UKS_54_TDDFT_STATE_7_NUM_TRANSITIONS

    # Should have both alpha and beta transitions
    has_alpha = any(t.is_alpha_spin is True for t in state.transitions)
    has_beta = any(t.is_alpha_spin is False for t in state.transitions)
    assert has_alpha and has_beta


@pytest.mark.regression
def test_uks_54_tda_state_2_strength_exact(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact oscillator strength for QChem 5.4 UKS TDA state 2."""
    tddft = parsed_qchem_54_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tda_states is not None
    assert len(tddft.tda_states) > 1

    state = tddft.tda_states[1]
    assert state.oscillator_strength == pytest.approx(EXPECTED_UKS_54_TDA_STATE_2_STRENGTH, abs=STRENGTH_TOL)


@pytest.mark.regression
def test_uks_54_tddft_state_2_strength_exact(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact oscillator strength for QChem 5.4 UKS TDDFT state 2."""
    tddft = parsed_qchem_54_h2o_uks_tddft_data.tddft
    assert tddft is not None
    assert tddft.tddft_states is not None
    assert len(tddft.tddft_states) > 1

    state = tddft.tddft_states[1]
    assert state.oscillator_strength == pytest.approx(EXPECTED_UKS_54_TDDFT_STATE_2_STRENGTH, abs=STRENGTH_TOL)
