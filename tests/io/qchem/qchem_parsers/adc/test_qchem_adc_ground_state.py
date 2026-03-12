"""
Tests for the Q-Chem ADC ground-state parser.

These tests verify that ADC ground-state parsing correctly captures ADC(2)
reference energies and analysis blocks, while preserving parser contract
guarantees.

Test hierarchy:
- unit: isolated matches() behavior
- contract: parser produces correct data structure
- integration: ADC data coexists with other parsed blocks
- regression: exact numerical values remain stable
"""

import pytest

from calcflow.common.results import AdcGroundState, AdcResults, CalculationResult
from calcflow.io.qchem.blocks.adc.ground_state import AdcGroundStateParser
from calcflow.io.state import BLOCK_ADC_GS, ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA
# =============================================================================

ENERGY_TOL = 1e-8
FLOAT_TOL = 1e-6

EXPECTED_HF_ENERGY_AU = -76.0240201204
EXPECTED_MP2_TOTAL_ENERGY_AU = -76.2297919082
EXPECTED_MP2_TOTAL_ENERGY_SMD_AU = -76.1760260684
EXPECTED_HF_ENERGY_SMD_AU = -75.9718959030
EXPECTED_MULLIKEN_H0 = 0.156052
EXPECTED_NOS_ALPHA_FRONTIER = (0.0122, 0.9836)
EXPECTED_EXCITON_TOTAL_SEPARATION_ANG = 0.165114


# =============================================================================
# UNIT TESTS: AdcGroundStateParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_adc_ground_state_parser_matches_banner() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")

    assert parser.matches("|                                 A D C  M A N                                 |", state)


@pytest.mark.unit
def test_adc_ground_state_parser_matches_with_leading_whitespace() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")

    assert parser.matches("   |                                 A D C  M A N                                 |", state)


@pytest.mark.unit
def test_adc_ground_state_parser_does_not_match_non_adc_lines() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")

    assert parser.matches("SCF converged", state) is False
    assert parser.matches("Mulliken Population Analysis", state) is False
    assert parser.matches("", state) is False


@pytest.mark.unit
def test_adc_ground_state_parser_skips_if_already_parsed() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")
    state.parsed_blocks.add(BLOCK_ADC_GS)

    assert (
        parser.matches("|                                 A D C  M A N                                 |", state)
        is False
    )


@pytest.mark.unit
def test_adc_ground_state_parser_does_not_mutate_state_in_matches() -> None:
    parser = AdcGroundStateParser()
    state = ParseState(raw_output="")

    line = "|                                 A D C  M A N                                 |"
    result_1 = parser.matches(line, state)
    result_2 = parser.matches(line, state)

    assert result_1 is True
    assert result_2 is True
    assert BLOCK_ADC_GS not in state.parsed_blocks
    assert state.adc_ground_state is None


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_ground_state"])
def test_adc_ground_state_has_correct_type(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    assert isinstance(data.adc, AdcResults)
    assert data.adc.method == "adc(2)"
    assert data.adc.ground_state is not None
    assert isinstance(data.adc.ground_state, AdcGroundState)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_ground_state"])
def test_adc_ground_state_energy_fields_present(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    gs = data.adc.ground_state
    assert gs is not None
    assert isinstance(gs.hf_energy_au, float)
    assert isinstance(gs.mp2_correlation_energy_au, float)
    assert isinstance(gs.total_energy_au, float)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_ground_state"])
def test_adc_ground_state_nos_structure(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    gs = data.adc.ground_state
    assert gs is not None
    assert gs.nos_alpha is not None
    assert gs.nos_beta is not None
    assert gs.nos_spin_traced is not None
    assert len(gs.nos_spin_traced.frontier_occupations) > 0


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["adc_ground_state"])
def test_adc_ground_state_mulliken_and_exciton_structure(fixture_name: str, request) -> None:
    data = request.getfixturevalue(fixture_name)

    assert data.adc is not None
    gs = data.adc.ground_state
    assert gs is not None
    assert gs.mulliken is not None
    assert len(gs.mulliken.charges) > 0
    assert gs.exciton_total is not None


# =============================================================================
# INTEGRATION TESTS: ADC with other parsed blocks
# =============================================================================


@pytest.mark.integration
def test_adc_ground_state_parsed_alongside_geometry(
    parsed_qchem_54_h2o_adc_svp_data: CalculationResult,
) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.input_geometry is not None
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    assert parsed_qchem_54_h2o_adc_svp_data.adc.ground_state is not None


@pytest.mark.integration
def test_adc_ground_state_parsed_alongside_metadata(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.metadata is not None
    assert parsed_qchem_54_h2o_adc_svp_data.metadata.software_version is not None
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
def test_adc_ground_state_regression_values(parsed_qchem_54_h2o_adc_svp_data: CalculationResult) -> None:
    assert parsed_qchem_54_h2o_adc_svp_data.adc is not None
    gs = parsed_qchem_54_h2o_adc_svp_data.adc.ground_state
    assert gs is not None

    assert gs.hf_energy_au == pytest.approx(EXPECTED_HF_ENERGY_AU, abs=ENERGY_TOL)
    assert gs.total_energy_au == pytest.approx(EXPECTED_MP2_TOTAL_ENERGY_AU, abs=ENERGY_TOL)
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

    assert gs.hf_energy_au == pytest.approx(EXPECTED_HF_ENERGY_SMD_AU, abs=ENERGY_TOL)
    assert gs.total_energy_au == pytest.approx(EXPECTED_MP2_TOTAL_ENERGY_SMD_AU, abs=ENERGY_TOL)
