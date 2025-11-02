"""
Contract and regression tests for the ORCA dipole moment parser.

These tests verify that the dipole moment block is correctly parsed
into the proper data structures and that numerical values match the known output.
"""

import pytest

from calcflow.common.results import CalculationResult, DipoleMoment, MultipoleResults

DIPOLE_TOL = 1e-8  # Tolerance for dipole components (a.u.)
MAGNITUDE_TOL = 1e-6  # Tolerance for magnitude (Debye)


@pytest.mark.contract
def test_dipole_structure_exists(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that the multipole container exists and
    contains a dipole moment object.
    """
    assert parsed_orca_h2o_sp_data.multipole is not None
    assert isinstance(parsed_orca_h2o_sp_data.multipole, MultipoleResults)


@pytest.mark.contract
def test_dipole_moment_exists(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that the dipole moment object is present
    within the multipole results.
    """
    multipole = parsed_orca_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.dipole is not None
    assert isinstance(multipole.dipole, DipoleMoment)


@pytest.mark.contract
def test_dipole_moment_has_all_fields(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that the dipole moment has all required fields
    (x, y, z components and magnitude).
    """
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert hasattr(dipole, "x")
    assert hasattr(dipole, "y")
    assert hasattr(dipole, "z")
    assert hasattr(dipole, "magnitude")
    assert dipole.x is not None
    assert dipole.y is not None
    assert dipole.z is not None
    assert dipole.magnitude is not None


@pytest.mark.contract
def test_dipole_components_are_numeric(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that dipole components are numeric values.
    """
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert isinstance(dipole.x, float)
    assert isinstance(dipole.y, float)
    assert isinstance(dipole.z, float)
    assert isinstance(dipole.magnitude, float)


@pytest.mark.contract
def test_dipole_magnitude_is_positive(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that the dipole magnitude is positive (basic sanity check).
    """
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert dipole.magnitude > 0


@pytest.mark.regression
def test_dipole_x_component_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify the exact value of the X component of the dipole moment.
    Expected value from ex-dipole.md: -0.319770911 a.u.
    """
    expected_x = -0.319770911
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert dipole.x == pytest.approx(expected_x, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_y_component_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify the exact value of the Y component of the dipole moment.
    Expected value from ex-dipole.md: -0.065576153 a.u.
    """
    expected_y = -0.065576153
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert dipole.y == pytest.approx(expected_y, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_z_component_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify the exact value of the Z component of the dipole moment.
    Expected value from ex-dipole.md: -0.559981644 a.u.
    """
    expected_z = -0.559981644
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert dipole.z == pytest.approx(expected_z, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_magnitude_debye_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify the exact value of the dipole magnitude in Debye.
    Expected value from ex-dipole.md: 1.647534386 Debye
    """
    expected_magnitude = 1.647534386
    dipole = parsed_orca_h2o_sp_data.multipole.dipole
    assert dipole.magnitude == pytest.approx(expected_magnitude, abs=MAGNITUDE_TOL)


@pytest.mark.regression
def test_dipole_magnitude_matches_components(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify that the magnitude is consistent with the components.
    Converts from a.u. to Debye using the conversion factor (1 a.u. = 2.541746 Debye).
    """
    dipole = parsed_orca_h2o_sp_data.multipole.dipole

    # Calculate magnitude from components in atomic units
    import math

    magnitude_au = math.sqrt(dipole.x**2 + dipole.y**2 + dipole.z**2)

    # Convert to Debye (1 a.u. = 2.541746 Debye)
    AU_TO_DEBYE = 2.541746
    expected_magnitude = magnitude_au * AU_TO_DEBYE

    # The parsed magnitude should match our calculation
    # Using relative tolerance as ORCA may use slightly different precision in the calculation
    assert dipole.magnitude == pytest.approx(expected_magnitude, rel=1e-3)
