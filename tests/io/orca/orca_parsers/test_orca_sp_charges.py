"""
Tests for the ORCA charges parser (Mulliken and Loewdin population analysis).

These tests verify that atomic charges from both Mulliken and Loewdin population
analysis are correctly parsed into AtomicCharges models.
"""

import pytest

from calcflow.common.results import AtomicCharges, CalculationResult

CHARGE_TOL = 1e-6


@pytest.mark.contract
def test_atomic_charges_list_length(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that we parsed both charge methods.

    Note: The parser is flexible and supports files with only Mulliken,
    only Loewdin, or both methods. This specific test fixture has both.
    """
    assert len(parsed_orca_h2o_sp_data.atomic_charges) == 2


@pytest.mark.contract
def test_mulliken_charges_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify the first AtomicCharges is Mulliken with correct structure.
    """
    mulliken = parsed_orca_h2o_sp_data.atomic_charges[0]
    assert isinstance(mulliken, AtomicCharges)
    assert mulliken.method == "Mulliken"
    assert len(mulliken.charges) == 3


@pytest.mark.contract
def test_loewdin_charges_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify the second AtomicCharges is Loewdin with correct structure.
    """
    loewdin = parsed_orca_h2o_sp_data.atomic_charges[1]
    assert isinstance(loewdin, AtomicCharges)
    assert loewdin.method == "Loewdin"
    assert len(loewdin.charges) == 3


@pytest.mark.contract
def test_mulliken_atom_indices(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify all three atoms are present in Mulliken charges.
    """
    mulliken = parsed_orca_h2o_sp_data.atomic_charges[0]
    assert 0 in mulliken.charges  # First H
    assert 1 in mulliken.charges  # O
    assert 2 in mulliken.charges  # Second H


@pytest.mark.contract
def test_loewdin_atom_indices(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify all three atoms are present in Loewdin charges.
    """
    loewdin = parsed_orca_h2o_sp_data.atomic_charges[1]
    assert 0 in loewdin.charges  # First H
    assert 1 in loewdin.charges  # O
    assert 2 in loewdin.charges  # Second H


@pytest.mark.regression
def test_mulliken_charges_h1(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify first hydrogen Mulliken charge.
    """
    mulliken = parsed_orca_h2o_sp_data.atomic_charges[0]
    assert mulliken.charges[0] == pytest.approx(0.172827, abs=CHARGE_TOL)


@pytest.mark.regression
def test_mulliken_charges_oxygen(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify oxygen Mulliken charge.
    """
    mulliken = parsed_orca_h2o_sp_data.atomic_charges[0]
    assert mulliken.charges[1] == pytest.approx(-0.346096, abs=CHARGE_TOL)


@pytest.mark.regression
def test_mulliken_charges_h2(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify second hydrogen Mulliken charge.
    """
    mulliken = parsed_orca_h2o_sp_data.atomic_charges[0]
    assert mulliken.charges[2] == pytest.approx(0.173269, abs=CHARGE_TOL)


@pytest.mark.regression
def test_loewdin_charges_h1(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify first hydrogen Loewdin charge.
    """
    loewdin = parsed_orca_h2o_sp_data.atomic_charges[1]
    assert loewdin.charges[0] == pytest.approx(0.120674, abs=CHARGE_TOL)


@pytest.mark.regression
def test_loewdin_charges_oxygen(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify oxygen Loewdin charge.
    """
    loewdin = parsed_orca_h2o_sp_data.atomic_charges[1]
    assert loewdin.charges[1] == pytest.approx(-0.241589, abs=CHARGE_TOL)


@pytest.mark.regression
def test_loewdin_charges_h2(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify second hydrogen Loewdin charge.
    """
    loewdin = parsed_orca_h2o_sp_data.atomic_charges[1]
    assert loewdin.charges[2] == pytest.approx(0.120916, abs=CHARGE_TOL)


@pytest.mark.integration
def test_mulliken_all_charges(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify all Mulliken charges at once.
    """
    mulliken = parsed_orca_h2o_sp_data.atomic_charges[0]
    assert mulliken.charges[0] == pytest.approx(0.172827, abs=CHARGE_TOL)
    assert mulliken.charges[1] == pytest.approx(-0.346096, abs=CHARGE_TOL)
    assert mulliken.charges[2] == pytest.approx(0.173269, abs=CHARGE_TOL)


@pytest.mark.integration
def test_loewdin_all_charges(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify all Loewdin charges at once.
    """
    loewdin = parsed_orca_h2o_sp_data.atomic_charges[1]
    assert loewdin.charges[0] == pytest.approx(0.120674, abs=CHARGE_TOL)
    assert loewdin.charges[1] == pytest.approx(-0.241589, abs=CHARGE_TOL)
    assert loewdin.charges[2] == pytest.approx(0.120916, abs=CHARGE_TOL)


@pytest.mark.integration
def test_charges_methods_order(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify Mulliken appears before Loewdin in the list.
    """
    methods = [c.method for c in parsed_orca_h2o_sp_data.atomic_charges]
    assert methods == ["Mulliken", "Loewdin"]
