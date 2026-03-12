"""
Tests for the QChem Mulliken charges parser.

These tests verify that the Mulliken charges parser correctly extracts atomic
charges from Q-Chem output files for single-point calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure (non-None values)
- integration: multiple components working together
- regression: exact numerical values match expected
"""

import pytest

from calcflow.common.results import AtomicCharges, CalculationResult
from calcflow.io.qchem.blocks.charges import MullikenParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA
# =============================================================================

# Expected Mulliken charges for H2O (identical in both Q-Chem 5.4 and 6.2)
EXPECTED_MULLIKEN_CHARGES = {
    0: 0.193937,  # H
    1: -0.388200,  # O
    2: 0.194263,  # H
}
EXPECTED_CHARGE_SUM = sum(EXPECTED_MULLIKEN_CHARGES.values())
CHARGE_TOL = 1e-5


# =============================================================================
# UNIT TESTS: MullikenParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_mulliken_parser_matches_start_line():
    """Unit test: verify MullikenParser.matches() recognizes charges block start."""
    parser = MullikenParser()
    state = ParseState(raw_output="")

    start_line = "Ground-State Mulliken Net Atomic Charges"
    assert parser.matches(start_line, state) is True


@pytest.mark.unit
def test_mulliken_parser_matches_with_leading_whitespace():
    """Unit test: verify MullikenParser.matches() handles leading whitespace."""
    parser = MullikenParser()
    state = ParseState(raw_output="")

    start_line = "    Ground-State Mulliken Net Atomic Charges"
    assert parser.matches(start_line, state) is True


@pytest.mark.unit
def test_mulliken_parser_does_not_match_non_charges_lines():
    """Unit test: verify MullikenParser.matches() rejects non-charges lines."""
    parser = MullikenParser()
    state = ParseState(raw_output="")

    assert parser.matches("SCF time:   CPU 0.32s  wall 0.00s", state) is False
    assert parser.matches("Atom          Charge", state) is False
    assert parser.matches("Hirshfeld Atomic Charges", state) is False
    assert parser.matches("Charge Model 5", state) is False
    assert parser.matches("Random calculation output", state) is False


@pytest.mark.unit
def test_mulliken_parser_skips_if_already_parsed():
    """Unit test: verify MullikenParser.matches() returns False when already parsed."""
    parser = MullikenParser()
    state = ParseState(raw_output="")
    state.parsed_mulliken = True

    start_line = "Ground-State Mulliken Net Atomic Charges"
    assert parser.matches(start_line, state) is False


@pytest.mark.unit
def test_mulliken_parser_does_not_mutate_state_in_matches():
    """
    Unit test: verify that matches() is read-only and does not mutate state.
    Critical for parser-spec compliance.
    """
    parser = MullikenParser()
    state = ParseState(raw_output="")

    line = "Ground-State Mulliken Net Atomic Charges"
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    assert result1 is True
    assert result2 is True
    assert state.parsed_mulliken is False  # Should NOT be set
    assert state.atomic_charges == []  # Should NOT be populated


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["charges"],
    indirect=True,
)
def test_charges_list_is_sequence(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify atomic_charges field is a Sequence of AtomicCharges."""
    from collections.abc import Sequence

    assert isinstance(parsed_qchem_data.atomic_charges, Sequence)
    assert len(parsed_qchem_data.atomic_charges) > 0
    assert all(isinstance(charge, AtomicCharges) for charge in parsed_qchem_data.atomic_charges)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["charges"],
    indirect=True,
)
def test_mulliken_charges_present(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Mulliken charges are present in parsed data."""
    mulliken_charges = None
    for charges in parsed_qchem_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break

    assert mulliken_charges is not None, "Mulliken charges not found in parsed data"
    assert mulliken_charges.method == "Mulliken"


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["charges"],
    indirect=True,
)
def test_mulliken_charges_has_values(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Mulliken charges dict is populated with non-None values."""
    mulliken_charges = None
    for charges in parsed_qchem_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break

    assert mulliken_charges is not None
    assert isinstance(mulliken_charges.charges, dict)
    assert len(mulliken_charges.charges) > 0

    for idx, charge_value in mulliken_charges.charges.items():
        assert isinstance(idx, int), f"Key should be int, got {type(idx)}"
        assert isinstance(charge_value, float), f"Value should be float, got {type(charge_value)}"


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["charges"],
    indirect=True,
)
def test_charges_structure_has_three_atoms(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Mulliken charges contains expected number of atoms (H2O = 3)."""
    mulliken_charges = None
    for charges in parsed_qchem_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break

    assert mulliken_charges is not None
    assert len(mulliken_charges.charges) == 3, f"Expected 3 atoms, got {len(mulliken_charges.charges)}"


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
def test_charges_parsed_alongside_geometry(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Integration test: verify charges parser works alongside geometry parser."""
    assert parsed_qchem_62_h2o_sp_data.input_geometry is not None
    assert parsed_qchem_62_h2o_sp_data.input_geometry.num_atoms == 3  # H2O

    mulliken_charges = None
    for charges in parsed_qchem_62_h2o_sp_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break
    assert mulliken_charges is not None


@pytest.mark.integration
def test_mulliken_completion_flag_set(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Integration test: verify that Mulliken charges appear in the result (flag was set)."""
    mulliken_charges = None
    for charges in parsed_qchem_62_h2o_sp_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break
    assert mulliken_charges is not None


@pytest.mark.integration
def test_charges_parsed_alongside_scf(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Integration test: verify both Mulliken charges and SCF results are present."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None

    mulliken_charges = None
    for charges in parsed_qchem_62_h2o_sp_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break
    assert mulliken_charges is not None


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
@pytest.mark.parametrize(
    "atom_idx",
    list(EXPECTED_MULLIKEN_CHARGES),
    ids=[f"atom-{k}" for k in EXPECTED_MULLIKEN_CHARGES],
)
def test_mulliken_charge_values(
    parsed_qchem_62_h2o_sp_data: CalculationResult,
    atom_idx: int,
) -> None:
    """Regression test: verify exact charge values for all atoms."""
    mulliken_charges = None
    for charges in parsed_qchem_62_h2o_sp_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break

    assert mulliken_charges is not None
    assert mulliken_charges.charges[atom_idx] == pytest.approx(EXPECTED_MULLIKEN_CHARGES[atom_idx], abs=CHARGE_TOL)


@pytest.mark.regression
def test_mulliken_charges_sum_approximately_zero(parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
    """Regression test: verify sum of charges is approximately zero."""
    mulliken_charges = None
    for charges in parsed_qchem_62_h2o_sp_data.atomic_charges:
        if charges.method == "Mulliken":
            mulliken_charges = charges
            break

    assert mulliken_charges is not None
    total_charge = sum(mulliken_charges.charges.values())
    assert total_charge == pytest.approx(EXPECTED_CHARGE_SUM, abs=CHARGE_TOL)
