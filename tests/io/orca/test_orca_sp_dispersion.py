"""
Tests for the ORCA dispersion correction parser (DFTD3/DFTD4).

These tests verify that dispersion correction results are correctly parsed
from DFT-D output blocks into DispersionCorrection models.
"""

import pytest

from calcflow.common.results import CalculationResult, DispersionCorrection

ENERGY_TOL = 1e-10
PARAM_TOL = 1e-6


@pytest.mark.contract
def test_dispersion_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that a DispersionCorrection object exists and has correct type.
    """
    assert parsed_orca_h2o_sp_data.dispersion is not None
    assert isinstance(parsed_orca_h2o_sp_data.dispersion, DispersionCorrection)


@pytest.mark.contract
def test_dispersion_method(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify method is "DFTD3".
    """
    assert parsed_orca_h2o_sp_data.dispersion is not None
    assert parsed_orca_h2o_sp_data.dispersion.method == "DFTD3"


@pytest.mark.contract
def test_dispersion_has_energy_au(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify primary energy field exists in atomic units.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.e_disp_au is not None
    assert isinstance(disp.e_disp_au, float)


@pytest.mark.contract
def test_dispersion_parameters_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify parameters dict exists with expected keys.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    expected_keys = {"s6", "rs6", "s8", "rs8", "alpha6", "alpha8", "k1", "k2", "k3"}
    assert set(disp.parameters.keys()) == expected_keys


@pytest.mark.regression
def test_dispersion_energy_au_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify e_disp_au has correct value.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.e_disp_au == pytest.approx(-0.000001638282, abs=ENERGY_TOL)


@pytest.mark.regression
def test_dispersion_energy_kcal_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify e_disp_kcal has correct value.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.e_disp_kcal == pytest.approx(-0.001028037712, abs=ENERGY_TOL)


@pytest.mark.regression
def test_dispersion_c6_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify molecular C6 coefficient.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.molecular_c6_au == pytest.approx(44.553144, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_e6_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify E6 component in kcal/mol.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.e6_kcal == pytest.approx(-0.000157178, abs=ENERGY_TOL)


@pytest.mark.regression
def test_dispersion_e8_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify E8 component in kcal/mol.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.e8_kcal == pytest.approx(-0.000870860, abs=ENERGY_TOL)


@pytest.mark.regression
def test_dispersion_e8_percentage(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify E8 percentage contribution.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.e8_percentage == pytest.approx(84.710890315, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_s6(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify s6 scaling factor.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["s6"] == pytest.approx(1.0000, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_rs6(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify rs6 scaling factor.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["rs6"] == pytest.approx(1.2810, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_s8(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify s8 scaling factor.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["s8"] == pytest.approx(1.0000, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_rs8(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify rs8 scaling factor.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["rs8"] == pytest.approx(1.0940, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_alpha6(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify alpha6 damping factor.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["alpha6"] == pytest.approx(14.0000, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_alpha8(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify alpha8 damping factor.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["alpha8"] == pytest.approx(16.0000, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_k1(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify k1 ad hoc parameter.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["k1"] == pytest.approx(16.0000, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_k2(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify k2 ad hoc parameter.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["k2"] == pytest.approx(1.3333, abs=PARAM_TOL)


@pytest.mark.regression
def test_dispersion_parameters_k3(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify k3 ad hoc parameter.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.parameters is not None
    assert disp.parameters["k3"] == pytest.approx(-4.0000, abs=PARAM_TOL)


@pytest.mark.integration
def test_dispersion_functional_recognition(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify functional is correctly recognized.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.functional == "omegaB97X-D3"


@pytest.mark.integration
def test_dispersion_damping_type(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify damping type is correctly identified.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.damping == "zero damping"


@pytest.mark.integration
def test_dispersion_all_components_present(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify all major fields are populated.
    """
    disp = parsed_orca_h2o_sp_data.dispersion
    assert disp is not None
    assert disp.method is not None
    assert disp.functional is not None
    assert disp.damping is not None
    assert disp.molecular_c6_au is not None
    assert disp.e_disp_au is not None
    assert disp.e_disp_kcal is not None
    assert disp.e6_kcal is not None
    assert disp.e8_kcal is not None
    assert disp.e8_percentage is not None
    assert disp.parameters is not None
