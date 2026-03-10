"""Tests for ADC ground-state parsing from Q-Chem outputs."""

import pytest

from calcflow.common.results import AdcGroundState, AdcResults, CalculationResult
from calcflow.io.qchem.blocks.adc.ground_state import AdcGroundStateParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

FLOAT_TOL = 1e-6

EXPECTED_HF_ENERGY_AU = -76.0240201204
EXPECTED_MP2_TOTAL_ENERGY_AU = -76.2297919082
EXPECTED_MP2_TOTAL_ENERGY_SMD_AU = -76.1760260684
EXPECTED_HF_ENERGY_SMD_AU = -75.9718959030
EXPECTED_MULLIKEN_H0 = 0.156052
EXPECTED_NOS_ALPHA_FRONTIER = (0.0122, 0.9836)
EXPECTED_EXCITON_TOTAL_SEPARATION_ANG = 0.165114


@pytest.mark.unit
def test_adc_ground_state_parser_matches_banner() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")

    assert parser.matches("|                                 A D C  M A N                                 |", state)


@pytest.mark.unit
def test_adc_ground_state_parser_respects_parsed_flag() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")
    state.parsed_adc_gs = True

    assert (
        parser.matches("|                                 A D C  M A N                                 |", state)
        is False
    )


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_ground_state"])
def test_adc_ground_state_exists_and_has_structure(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    assert isinstance(data.adc, AdcResults)
    assert data.adc.method == "adc(2)"
    assert data.adc.ground_state is not None
    assert isinstance(data.adc.ground_state, AdcGroundState)

    gs = data.adc.ground_state
    assert gs.nos_alpha is not None
    assert gs.nos_beta is not None
    assert gs.nos_spin_traced is not None
    assert gs.mulliken is not None
    assert gs.exciton_total is not None


@pytest.mark.regression
def test_adc_ground_state_regression_values(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    gs = parsed_qchem_54_h2o_adc_svp_data.adc.ground_state
    assert gs is not None

    assert gs.hf_energy_au == pytest.approx(EXPECTED_HF_ENERGY_AU, abs=FLOAT_TOL)
    assert gs.total_energy_au == pytest.approx(EXPECTED_MP2_TOTAL_ENERGY_AU, abs=FLOAT_TOL)
    assert gs.mulliken is not None
    assert gs.mulliken.charges[0] == pytest.approx(EXPECTED_MULLIKEN_H0, abs=FLOAT_TOL)
    assert gs.nos_alpha is not None
    assert gs.nos_alpha.frontier_occupations[0] == pytest.approx(EXPECTED_NOS_ALPHA_FRONTIER[0], abs=FLOAT_TOL)
    assert gs.nos_alpha.frontier_occupations[1] == pytest.approx(EXPECTED_NOS_ALPHA_FRONTIER[1], abs=FLOAT_TOL)
    assert gs.exciton_total is not None
    assert gs.exciton_total.separation_ang == pytest.approx(EXPECTED_EXCITON_TOTAL_SEPARATION_ANG, abs=FLOAT_TOL)


@pytest.mark.regression
def test_adc_ground_state_smd_regression_values(parsed_qchem_54_h2o_adc_svp_smd_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_smd_data.adc is not None
    gs = parsed_qchem_54_h2o_adc_svp_smd_data.adc.ground_state
    assert gs is not None

    assert gs.hf_energy_au == pytest.approx(EXPECTED_HF_ENERGY_SMD_AU, abs=FLOAT_TOL)
    assert gs.total_energy_au == pytest.approx(EXPECTED_MP2_TOTAL_ENERGY_SMD_AU, abs=FLOAT_TOL)
