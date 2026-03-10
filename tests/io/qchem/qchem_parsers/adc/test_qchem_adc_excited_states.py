"""
Tests for the Q-Chem ADC excited-states parser.

These tests validate parser matching behavior, structural contracts for the
50-state ADC summary, integration with the ADC ground state, and regression
stability for representative state-level quantities.

Test hierarchy:
- unit: isolated matches() behavior
- contract: parser produces correct data structure
- integration: ADC ground/excited data are assembled together
- regression: exact numerical values remain stable
"""

import pytest

from calcflow.common.results import AdcExcitedState, CalculationResult
from calcflow.io.qchem.blocks.adc.excited_states import AdcExcitedStatesParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA
# =============================================================================

FLOAT_TOL = 1e-6

EXPECTED_NUM_STATES = 50

EXPECTED_STATE_1_EXCITATION_EV = 7.114530
EXPECTED_STATE_1_OSC = 0.000000
EXPECTED_STATE_1_V1_SQUARED = 0.9576
EXPECTED_STATE_1_CT_OMEGA = 0.9422
EXPECTED_STATE_1_CT_PHE = 0.0302
EXPECTED_STATE_1_EXCITON_TRANS_SEP_ANG = 0.651848
EXPECTED_STATE_1_NTO_HOLE = 0
EXPECTED_STATE_1_NTO_ELECTRON = 0
EXPECTED_STATE_1_NTO_WEIGHT = 47.0
EXPECTED_STATE_1_AMP_OCC_I = 5
EXPECTED_STATE_1_AMP_VIR_A = 6
EXPECTED_STATE_1_AMP_SPIN = "B"
EXPECTED_STATE_1_AMP_VALUE = 0.6841
EXPECTED_STATE_1_TPA_CROSS_SECTION = 0.000000

EXPECTED_STATE_2_EXCITATION_EV = 7.816509
EXPECTED_STATE_2_TPA_CROSS_SECTION = 62.112485
EXPECTED_STATE_2_TPA_01 = -1.378300

EXPECTED_STATE_25_TOTAL_ENERGY_AU = -75.1496794655
EXPECTED_STATE_25_EXCITATION_EV = 29.391354
EXPECTED_STATE_25_OSC = 0.146762
EXPECTED_STATE_25_V1_SQUARED = 0.9389
EXPECTED_STATE_25_V2_SQUARED = 0.0611

EXPECTED_STATE_50_TOTAL_ENERGY_AU = -74.8092798618
EXPECTED_STATE_50_EXCITATION_EV = 38.654099
EXPECTED_STATE_50_OSC = 0.000000

EXPECTED_SMD_STATE_1_EXCITATION_EV = 8.045472
EXPECTED_SMD_STATE_1_OSC = 0.000000


# =============================================================================
# UNIT TESTS: AdcExcitedStatesParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_adc_excited_states_parser_matches_header() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")

    assert parser.matches("                             Excited State Summary", state)


@pytest.mark.unit
def test_adc_excited_states_parser_matches_header_with_leading_whitespace() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")

    assert parser.matches("   Excited State Summary", state)


@pytest.mark.unit
def test_adc_excited_states_parser_does_not_match_non_summary_lines() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")

    assert parser.matches("Excited state  1 (A)", state) is False
    assert parser.matches("Two-photon absorption matrix [a.u.]", state) is False
    assert parser.matches("", state) is False


@pytest.mark.unit
def test_adc_excited_states_parser_skips_if_already_parsed() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")
    state.parsed_adc_excited = True

    assert parser.matches("                             Excited State Summary", state) is False


@pytest.mark.unit
def test_adc_excited_states_parser_does_not_mutate_state_in_matches() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")

    line = "                             Excited State Summary"
    result_1 = parser.matches(line, state)
    result_2 = parser.matches(line, state)

    assert result_1 is True
    assert result_2 is True
    assert state.parsed_adc_excited is False
    assert state.adc is None


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_excited_states"])
def test_adc_excited_states_have_expected_collection_shape(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    assert data.adc.excited_states is not None
    assert len(data.adc.excited_states) == EXPECTED_NUM_STATES
    assert all(isinstance(state, AdcExcitedState) for state in data.adc.excited_states)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_excited_states"])
def test_adc_excited_states_required_fields(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    for state in data.adc.excited_states:
        assert isinstance(state.state_number, int)
        assert isinstance(state.total_energy_au, float)
        assert isinstance(state.excitation_energy_ev, float)
        assert len(state.amplitudes) > 0


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_excited_states"])
def test_adc_excited_states_optional_blocks_present_across_sequence(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    assert any(state.two_photon_absorption is not None for state in data.adc.excited_states)
    assert any(state.nto_alpha is not None for state in data.adc.excited_states)
    assert any(state.nto_beta is not None for state in data.adc.excited_states)
    assert any(state.exciton_trans_total is not None for state in data.adc.excited_states)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_excited_states"])
def test_adc_excited_state_amplitude_structure(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    assert len(data.adc.excited_states[0].amplitudes) > 0
    amp = data.adc.excited_states[0].amplitudes[0]

    assert isinstance(amp.occ_i, int)
    assert isinstance(amp.vir_a, int)
    assert isinstance(amp.amplitude, float)
    assert amp.spin in ("A", "B")


# =============================================================================
# INTEGRATION TESTS: ADC ground/excited assembly
# =============================================================================


@pytest.mark.integration
def test_adc_state_numbers_are_contiguous(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    state_numbers = [state.state_number for state in parsed_qchem_54_h2o_adc_svp_data.adc.excited_states]

    assert state_numbers == list(range(1, EXPECTED_NUM_STATES + 1))


@pytest.mark.integration
def test_adc_excited_states_parsed_alongside_ground_state(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    assert parsed_qchem_54_h2o_adc_svp_data.adc.ground_state is not None
    assert len(parsed_qchem_54_h2o_adc_svp_data.adc.excited_states) == EXPECTED_NUM_STATES


@pytest.mark.integration
def test_adc_excited_states_parsed_alongside_metadata(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.metadata is not None
    assert parsed_qchem_54_h2o_adc_svp_data.metadata.software_version is not None
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    assert len(parsed_qchem_54_h2o_adc_svp_data.adc.excited_states) == EXPECTED_NUM_STATES


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
def test_adc_excited_state_count(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    states = parsed_qchem_54_h2o_adc_svp_data.adc.excited_states
    assert len(states) == EXPECTED_NUM_STATES


@pytest.mark.regression
def test_adc_state_1_regression_values(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    state_1 = parsed_qchem_54_h2o_adc_svp_data.adc.excited_states[0]

    assert state_1.excitation_energy_ev == pytest.approx(EXPECTED_STATE_1_EXCITATION_EV, abs=FLOAT_TOL)
    assert state_1.oscillator_strength == pytest.approx(EXPECTED_STATE_1_OSC, abs=FLOAT_TOL)
    assert state_1.v1_squared == pytest.approx(EXPECTED_STATE_1_V1_SQUARED, abs=FLOAT_TOL)
    assert state_1.ct_omega == pytest.approx(EXPECTED_STATE_1_CT_OMEGA, abs=FLOAT_TOL)
    assert state_1.ct_phe == pytest.approx(EXPECTED_STATE_1_CT_PHE, abs=FLOAT_TOL)
    assert state_1.two_photon_absorption is not None
    assert state_1.two_photon_absorption.cross_section_au == pytest.approx(
        EXPECTED_STATE_1_TPA_CROSS_SECTION, abs=FLOAT_TOL
    )

    assert state_1.exciton_trans_total is not None
    assert state_1.exciton_trans_total.separation_ang == pytest.approx(
        EXPECTED_STATE_1_EXCITON_TRANS_SEP_ANG, abs=FLOAT_TOL
    )

    assert state_1.nto_alpha is not None
    assert len(state_1.nto_alpha) > 0
    assert state_1.nto_alpha[0].hole_offset == EXPECTED_STATE_1_NTO_HOLE
    assert state_1.nto_alpha[0].electron_offset == EXPECTED_STATE_1_NTO_ELECTRON
    assert state_1.nto_alpha[0].weight_percent == pytest.approx(EXPECTED_STATE_1_NTO_WEIGHT, abs=FLOAT_TOL)

    assert len(state_1.amplitudes) >= 1
    amp = state_1.amplitudes[0]
    assert amp.occ_i == EXPECTED_STATE_1_AMP_OCC_I
    assert amp.vir_a == EXPECTED_STATE_1_AMP_VIR_A
    assert amp.spin == EXPECTED_STATE_1_AMP_SPIN
    assert amp.amplitude == pytest.approx(EXPECTED_STATE_1_AMP_VALUE, abs=FLOAT_TOL)


@pytest.mark.regression
def test_adc_state_2_tpa_regression_values(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    state_2 = parsed_qchem_54_h2o_adc_svp_data.adc.excited_states[1]

    assert state_2.excitation_energy_ev == pytest.approx(EXPECTED_STATE_2_EXCITATION_EV, abs=FLOAT_TOL)
    assert state_2.two_photon_absorption is not None
    assert state_2.two_photon_absorption.cross_section_au == pytest.approx(
        EXPECTED_STATE_2_TPA_CROSS_SECTION, abs=FLOAT_TOL
    )
    assert state_2.two_photon_absorption.matrix_au[0][1] == pytest.approx(EXPECTED_STATE_2_TPA_01, abs=FLOAT_TOL)


@pytest.mark.regression
def test_adc_state_25_regression_values(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    state_25 = parsed_qchem_54_h2o_adc_svp_data.adc.excited_states[24]

    assert state_25.total_energy_au == pytest.approx(EXPECTED_STATE_25_TOTAL_ENERGY_AU, abs=FLOAT_TOL)
    assert state_25.excitation_energy_ev == pytest.approx(EXPECTED_STATE_25_EXCITATION_EV, abs=FLOAT_TOL)
    assert state_25.oscillator_strength == pytest.approx(EXPECTED_STATE_25_OSC, abs=FLOAT_TOL)
    assert state_25.v1_squared == pytest.approx(EXPECTED_STATE_25_V1_SQUARED, abs=FLOAT_TOL)
    assert state_25.v2_squared == pytest.approx(EXPECTED_STATE_25_V2_SQUARED, abs=FLOAT_TOL)


@pytest.mark.regression
def test_adc_state_50_regression_values(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    state_50 = parsed_qchem_54_h2o_adc_svp_data.adc.excited_states[49]

    assert state_50.total_energy_au == pytest.approx(EXPECTED_STATE_50_TOTAL_ENERGY_AU, abs=FLOAT_TOL)
    assert state_50.excitation_energy_ev == pytest.approx(EXPECTED_STATE_50_EXCITATION_EV, abs=FLOAT_TOL)
    assert state_50.oscillator_strength == pytest.approx(EXPECTED_STATE_50_OSC, abs=FLOAT_TOL)


@pytest.mark.regression
def test_adc_state_1_smd_regression_values(parsed_qchem_54_h2o_adc_svp_smd_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_smd_data.adc is not None
    state_1 = parsed_qchem_54_h2o_adc_svp_smd_data.adc.excited_states[0]

    assert state_1.excitation_energy_ev == pytest.approx(EXPECTED_SMD_STATE_1_EXCITATION_EV, abs=FLOAT_TOL)
    assert state_1.oscillator_strength == pytest.approx(EXPECTED_SMD_STATE_1_OSC, abs=FLOAT_TOL)
