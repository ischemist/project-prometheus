"""
Contract and regression tests for the ORCA orbitals parser.

These tests verify that the OrbitalsParser correctly extracts molecular orbital
information (energies, occupations, HOMO/LUMO indices) from ORCA output files.
"""

import pytest

from calcflow.common.results import CalculationResult, Orbital, OrbitalsSet

# Tolerances for numerical comparisons
ENERGY_TOL = 1e-8  # Hartree
ENERGY_EV_TOL = 1e-4  # eV has fewer significant figures


@pytest.mark.contract
def test_orbitals_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: Verify that the orbitals block is parsed into correct pydantic models.
    """
    orbitals = parsed_orca_h2o_sp_data.orbitals
    assert orbitals is not None
    assert isinstance(orbitals, OrbitalsSet)

    # For RHF, only alpha_orbitals should be populated
    assert orbitals.alpha_orbitals is not None
    assert orbitals.beta_orbitals is None


@pytest.mark.contract
def test_orbitals_count(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: Verify that all 7 orbitals are parsed (NO 0-6).
    """
    orbitals = parsed_orca_h2o_sp_data.orbitals
    assert len(orbitals.alpha_orbitals) == 7

    # Verify all are Orbital objects with expected fields
    for orbital in orbitals.alpha_orbitals:
        assert isinstance(orbital, Orbital)
        assert orbital.index is not None
        assert orbital.energy is not None
        assert orbital.occupation is not None
        assert orbital.energy_ev is not None


@pytest.mark.contract
def test_orbitals_homo_lumo_detection(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: Verify that HOMO and LUMO indices are correctly identified.
    HOMO should be index 4 (last occupied orbital with OCC=2.0).
    LUMO should be index 5 (first virtual orbital with OCC=0.0).
    """
    orbitals = parsed_orca_h2o_sp_data.orbitals

    # HOMO and LUMO indices for alpha (RHF calculation)
    assert orbitals.alpha_homo_index == 4
    assert orbitals.alpha_lumo_index == 5

    # Beta indices should be None for RHF
    assert orbitals.beta_homo_index is None
    assert orbitals.beta_lumo_index is None


@pytest.mark.contract
def test_orbitals_occupation_pattern(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: Verify the occupation pattern (first 5 occupied, last 2 virtual).
    """
    orbitals = parsed_orca_h2o_sp_data.orbitals

    # First 5 orbitals should be occupied (OCC = 2.0)
    for i in range(5):
        assert orbitals.alpha_orbitals[i].occupation == pytest.approx(2.0, abs=1e-4)

    # Last 2 orbitals should be virtual (OCC = 0.0)
    for i in range(5, 7):
        assert orbitals.alpha_orbitals[i].occupation == pytest.approx(0.0, abs=1e-4)


@pytest.mark.regression
def test_orbital_0_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Verify orbital 0 (1s on oxygen) values.
    """
    orbital = parsed_orca_h2o_sp_data.orbitals.alpha_orbitals[0]

    assert orbital.index == 0
    assert orbital.occupation == pytest.approx(2.0, abs=ENERGY_TOL)
    assert orbital.energy == pytest.approx(-18.937331, abs=ENERGY_TOL)
    assert orbital.energy_ev == pytest.approx(-515.3110, abs=ENERGY_EV_TOL)


@pytest.mark.regression
def test_orbital_1_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Verify orbital 1 (2s on oxygen) values.
    """
    orbital = parsed_orca_h2o_sp_data.orbitals.alpha_orbitals[1]

    assert orbital.index == 1
    assert orbital.occupation == pytest.approx(2.0, abs=ENERGY_TOL)
    assert orbital.energy == pytest.approx(-1.035235, abs=ENERGY_TOL)
    assert orbital.energy_ev == pytest.approx(-28.1702, abs=ENERGY_EV_TOL)


@pytest.mark.regression
def test_orbital_4_homo_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Verify orbital 4 (HOMO) values.
    """
    orbital = parsed_orca_h2o_sp_data.orbitals.alpha_orbitals[4]

    assert orbital.index == 4
    assert orbital.occupation == pytest.approx(2.0, abs=ENERGY_TOL)
    assert orbital.energy == pytest.approx(-0.243811, abs=ENERGY_TOL)
    assert orbital.energy_ev == pytest.approx(-6.6344, abs=ENERGY_EV_TOL)


@pytest.mark.regression
def test_orbital_5_lumo_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Verify orbital 5 (LUMO) values.
    """
    orbital = parsed_orca_h2o_sp_data.orbitals.alpha_orbitals[5]

    assert orbital.index == 5
    assert orbital.occupation == pytest.approx(0.0, abs=ENERGY_TOL)
    assert orbital.energy == pytest.approx(0.434377, abs=ENERGY_TOL)
    assert orbital.energy_ev == pytest.approx(11.8200, abs=ENERGY_EV_TOL)


@pytest.mark.regression
def test_orbital_6_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Verify orbital 6 values.
    """
    orbital = parsed_orca_h2o_sp_data.orbitals.alpha_orbitals[6]

    assert orbital.index == 6
    assert orbital.occupation == pytest.approx(0.0, abs=ENERGY_TOL)
    assert orbital.energy == pytest.approx(0.549984, abs=ENERGY_TOL)
    assert orbital.energy_ev == pytest.approx(14.9658, abs=ENERGY_EV_TOL)


@pytest.mark.regression
def test_all_orbital_energies(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Parametrized test for all orbital energies.
    """
    expected_values = [
        # (index, occupation, energy_eh, energy_ev)
        (0, 2.0, -18.937331, -515.3110),
        (1, 2.0, -1.035235, -28.1702),
        (2, 2.0, -0.533924, -14.5288),
        (3, 2.0, -0.330485, -8.9930),
        (4, 2.0, -0.243811, -6.6344),
        (5, 0.0, 0.434377, 11.8200),
        (6, 0.0, 0.549984, 14.9658),
    ]

    orbitals = parsed_orca_h2o_sp_data.orbitals

    for idx, occ, energy_eh, energy_ev in expected_values:
        orbital = orbitals.alpha_orbitals[idx]
        assert orbital.index == idx
        assert orbital.occupation == pytest.approx(occ, abs=ENERGY_TOL)
        assert orbital.energy == pytest.approx(energy_eh, abs=ENERGY_TOL)
        assert orbital.energy_ev == pytest.approx(energy_ev, abs=ENERGY_EV_TOL)


@pytest.mark.regression
def test_orbital_energies_monotonically_increasing(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: Verify that orbital energies are monotonically increasing.
    This is always true for ground state calculations.
    """
    orbitals = parsed_orca_h2o_sp_data.orbitals

    energies = [orbital.energy for orbital in orbitals.alpha_orbitals]
    for i in range(len(energies) - 1):
        assert energies[i] < energies[i + 1], "Orbital energies should be monotonically increasing"
