"""
Tests for the ORCA finalization block parsers (final energy and termination status).

These tests verify that the finalization block is correctly parsed, including:
- FINAL SINGLE POINT ENERGY extraction
- ****ORCA TERMINATED NORMALLY**** detection
- Energy validation (final energy = SCF energy + dispersion correction)
"""

import pytest

from calcflow.common.results import CalculationResult

ENERGY_TOL = 1e-8


@pytest.mark.contract
def test_final_energy_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that final_energy field is populated with correct type.
    """
    assert parsed_orca_h2o_sp_data.final_energy is not None
    assert isinstance(parsed_orca_h2o_sp_data.final_energy, float)


@pytest.mark.contract
def test_termination_status_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify termination_status is populated and is a valid Literal type.
    """
    assert parsed_orca_h2o_sp_data.termination_status is not None
    assert isinstance(parsed_orca_h2o_sp_data.termination_status, str)
    assert parsed_orca_h2o_sp_data.termination_status in ["NORMAL", "ERROR", "UNKNOWN"]


@pytest.mark.regression
def test_final_energy_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify final single point energy has correct value.
    This is parsed directly from "FINAL SINGLE POINT ENERGY" line.
    """
    expected_final_energy = -75.313506060725
    assert parsed_orca_h2o_sp_data.final_energy == pytest.approx(expected_final_energy, abs=ENERGY_TOL)


@pytest.mark.regression
def test_termination_status_normal(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify termination status is "NORMAL" for successful run.
    """
    assert parsed_orca_h2o_sp_data.termination_status == "NORMAL"


@pytest.mark.regression
def test_final_energy_equals_scf_plus_dispersion(
    parsed_orca_h2o_sp_data: CalculationResult,
):
    """
    Regression test: verify that final energy equals SCF energy + dispersion correction.
    This validates the energy consistency across different calculation components.
    """
    scf_energy = parsed_orca_h2o_sp_data.scf.energy
    dispersion_energy = parsed_orca_h2o_sp_data.dispersion.e_disp_au
    expected_final = scf_energy + dispersion_energy

    # The final energy should be very close to SCF + dispersion
    assert parsed_orca_h2o_sp_data.final_energy == pytest.approx(expected_final, abs=ENERGY_TOL)


@pytest.mark.integration
def test_all_finalization_fields_populated(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify all finalization-related fields are properly populated
    in the final CalculationResult.
    """
    assert parsed_orca_h2o_sp_data.final_energy is not None
    assert parsed_orca_h2o_sp_data.termination_status is not None
    assert parsed_orca_h2o_sp_data.termination_status != "UNKNOWN"
