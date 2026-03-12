"""
Tests for the QChem Hirshfeld charges parser.

These tests verify that the Hirshfeld charges parser correctly extracts atomic
charges from Q-Chem output files when HIRSHFELD=True is set in $rem.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure (non-None values)
- integration: multiple components working together
- regression: exact numerical values match expected
"""

import pytest

from calcflow.common.results import AtomicCharges, CalculationResult
from calcflow.io.peekable import PeekableIterator
from calcflow.io.qchem.blocks.hirshfeld import HirshfeldParser
from calcflow.io.state import BLOCK_HIRSHFELD, ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA
# =============================================================================

# Expected Hirshfeld charges for H2O from 6.3-sp-cm5-smd.out
EXPECTED_HIRSHFELD_CHARGES = {
    0: 0.128647,  # H
    1: -0.257050,  # O
    2: 0.128652,  # H
}
CHARGE_TOL = 1e-5


# =============================================================================
# UNIT TESTS: HirshfeldParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_hirshfeld_parser_matches_start_line():
    """Unit test: verify HirshfeldParser.matches() recognizes the Hirshfeld block header."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    assert parser.matches("          Hirshfeld Atomic Charges         ", state) is True


@pytest.mark.unit
def test_hirshfeld_parser_matches_with_leading_whitespace():
    """Unit test: verify HirshfeldParser.matches() handles varying whitespace."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    assert parser.matches("Hirshfeld Atomic Charges", state) is True
    assert parser.matches("   Hirshfeld Atomic Charges   ", state) is True


@pytest.mark.unit
def test_hirshfeld_parser_does_not_match_other_charge_headers():
    """Unit test: verify HirshfeldParser.matches() rejects unrelated lines."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    assert parser.matches("Ground-State Mulliken Net Atomic Charges", state) is False
    assert parser.matches("Charge Model 5", state) is False
    assert parser.matches("SCF time:   CPU 0.32s  wall 0.00s", state) is False
    assert parser.matches("Performing Hirshfeld population analysis.", state) is False


@pytest.mark.unit
def test_hirshfeld_parser_skips_if_already_parsed():
    """Unit test: verify HirshfeldParser.matches() returns False when already parsed."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")
    state.parsed_blocks.add(BLOCK_HIRSHFELD)

    assert parser.matches("          Hirshfeld Atomic Charges         ", state) is False


@pytest.mark.unit
def test_hirshfeld_parser_does_not_mutate_state_in_matches():
    """Unit test: verify matches() is read-only — no state mutation allowed."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    line = "          Hirshfeld Atomic Charges         "
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    assert result1 is True
    assert result2 is True
    assert BLOCK_HIRSHFELD not in state.parsed_blocks
    assert state.atomic_charges == []


@pytest.mark.unit
def test_hirshfeld_parser_parse_basic():
    """Unit test: verify HirshfeldParser.parse() extracts charges from minimal input."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    lines = [
        "",
        "     Atom                 Charge (a.u.)    ",
        "  ---------------------------------------- ",
        "      1 H                     0.128647",
        "      2 O                    -0.257050",
        "      3 H                     0.128652",
        "  ---------------------------------------- ",
        "  Sum of atomic charges =     0.000248",
    ]
    parser.parse(PeekableIterator(iter(lines)), "          Hirshfeld Atomic Charges         ", state)

    assert BLOCK_HIRSHFELD in state.parsed_blocks
    assert len(state.atomic_charges) == 1
    hirshfeld = state.atomic_charges[0]
    assert hirshfeld.method == "Hirshfeld"
    assert hirshfeld.charges[0] == pytest.approx(0.128647, abs=CHARGE_TOL)
    assert hirshfeld.charges[1] == pytest.approx(-0.257050, abs=CHARGE_TOL)
    assert hirshfeld.charges[2] == pytest.approx(0.128652, abs=CHARGE_TOL)


@pytest.mark.unit
def test_hirshfeld_parser_uses_0based_indices():
    """Unit test: verify 1-based Q-Chem indices are converted to 0-based."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    lines = [
        "      1 H                     0.100000",
        "      2 O                    -0.200000",
        "  Sum of atomic charges =    -0.100000",
    ]
    parser.parse(PeekableIterator(iter(lines)), "Hirshfeld Atomic Charges", state)

    assert 0 in state.atomic_charges[0].charges
    assert 1 in state.atomic_charges[0].charges
    assert 2 not in state.atomic_charges[0].charges


@pytest.mark.unit
def test_hirshfeld_parser_empty_block_adds_warning():
    """Unit test: verify empty block produces a warning instead of crashing."""
    parser = HirshfeldParser()
    state = ParseState(raw_output="")

    parser.parse(PeekableIterator(iter(["  Sum of atomic charges =     0.000000"])), "Hirshfeld Atomic Charges", state)

    assert BLOCK_HIRSHFELD not in state.parsed_blocks
    assert len(state.atomic_charges) == 0
    assert len(state.parsing_warnings) == 1


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["hirshfeld"],
    indirect=True,
)
def test_hirshfeld_charges_present(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Hirshfeld charges entry exists in atomic_charges list."""
    hirshfeld = next((c for c in parsed_qchem_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None, "Hirshfeld charges not found in parsed data"
    assert hirshfeld.method == "Hirshfeld"


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["hirshfeld"],
    indirect=True,
)
def test_hirshfeld_charges_is_atomiccharges(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Hirshfeld entry is an AtomicCharges instance."""
    hirshfeld = next((c for c in parsed_qchem_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None
    assert isinstance(hirshfeld, AtomicCharges)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["hirshfeld"],
    indirect=True,
)
def test_hirshfeld_charges_has_values(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Hirshfeld charges dict is populated with float values."""
    hirshfeld = next((c for c in parsed_qchem_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None
    assert len(hirshfeld.charges) > 0
    for idx, val in hirshfeld.charges.items():
        assert isinstance(idx, int)
        assert isinstance(val, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["hirshfeld"],
    indirect=True,
)
def test_hirshfeld_charges_has_three_atoms(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify Hirshfeld charges has 3 entries for H2O."""
    hirshfeld = next((c for c in parsed_qchem_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None
    assert len(hirshfeld.charges) == 3


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
def test_hirshfeld_parsed_alongside_mulliken(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: Hirshfeld and Mulliken charges coexist in atomic_charges list."""
    methods = {c.method for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges}
    assert "Mulliken" in methods
    assert "Hirshfeld" in methods


@pytest.mark.integration
def test_hirshfeld_parsed_alongside_cm5(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: Hirshfeld and CM5 charges both present when CM5=True."""
    methods = {c.method for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges}
    assert "Hirshfeld" in methods
    assert "CM5" in methods


@pytest.mark.integration
def test_hirshfeld_parsed_alongside_scf(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: SCF results and Hirshfeld charges are both present."""
    assert parsed_qchem_63_h2o_cm5_sp_data.scf is not None
    hirshfeld = next((c for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
@pytest.mark.parametrize(
    "atom_idx",
    list(EXPECTED_HIRSHFELD_CHARGES),
    ids=[f"atom-{k}" for k in EXPECTED_HIRSHFELD_CHARGES],
)
def test_hirshfeld_charge_values(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult, atom_idx: int) -> None:
    """Regression test: verify exact Hirshfeld charge values for each atom."""
    hirshfeld = next((c for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None
    assert hirshfeld.charges[atom_idx] == pytest.approx(EXPECTED_HIRSHFELD_CHARGES[atom_idx], abs=CHARGE_TOL)


@pytest.mark.regression
def test_hirshfeld_charges_sum(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult) -> None:
    """Regression test: verify sum of Hirshfeld charges matches expected (non-zero due to integration)."""
    hirshfeld = next((c for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges if c.method == "Hirshfeld"), None)
    assert hirshfeld is not None
    total = sum(hirshfeld.charges.values())
    # Q-Chem reports sum = 0.000248 due to numerical integration error
    assert total == pytest.approx(sum(EXPECTED_HIRSHFELD_CHARGES.values()), abs=CHARGE_TOL)
