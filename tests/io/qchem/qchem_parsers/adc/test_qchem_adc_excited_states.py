"""Tests for ADC excited-state parsing from Q-Chem outputs."""

import pytest

from calcflow.common.results import AdcExcitedState, CalculationResult
from calcflow.io.qchem.blocks.adc.excited_states import AdcExcitedStatesParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

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

EXPECTED_STATE_2_EXCITATION_EV = 7.816509
EXPECTED_STATE_2_TPA_CROSS_SECTION = 62.112485
EXPECTED_STATE_2_TPA_01 = -1.378300


@pytest.mark.unit
def test_adc_excited_states_parser_matches_header() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")

    assert parser.matches("                             Excited State Summary", state)


@pytest.mark.unit
def test_adc_excited_states_parser_respects_parsed_flag() -> None:
    parser = AdcExcitedStatesParser()
    state = ParseState(raw_output="")
    state.parsed_adc_excited = True

    assert parser.matches("                             Excited State Summary", state) is False


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_excited_states"])
def test_adc_excited_states_exist_and_have_structure(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    assert data.adc.excited_states is not None
    assert len(data.adc.excited_states) > 0

    for state in data.adc.excited_states:
        assert isinstance(state, AdcExcitedState)
        assert state.state_number > 0
        assert state.total_energy_au is not None
        assert state.excitation_energy_ev is not None
    assert any(state.two_photon_absorption is not None for state in data.adc.excited_states)
    assert any(state.nto_alpha is not None for state in data.adc.excited_states)
    assert any(state.nto_beta is not None for state in data.adc.excited_states)


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


@pytest.mark.integration
def test_adc_state_numbers_are_contiguous(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    state_numbers = [state.state_number for state in parsed_qchem_54_h2o_adc_svp_data.adc.excited_states]

    assert state_numbers == list(range(1, EXPECTED_NUM_STATES + 1))
