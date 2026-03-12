"""
Tests for the QChem CM5 (Charge Model 5) charges parser.

These tests verify that the CM5 charges parser correctly extracts atomic
charges from Q-Chem output files when CM5=True is set in $rem.
CM5 always implies HIRSHFELD=True, so Hirshfeld charges will also be present.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure (non-None values)
- integration: multiple components working together
- regression: exact numerical values match expected
"""

import pytest

from calcflow.common.results import AtomicCharges, CalculationResult
from calcflow.io.peekable import PeekableIterator
from calcflow.io.qchem.blocks.cm5 import Cm5Parser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA
# =============================================================================

# Expected CM5 charges for H2O from 6.3-sp-cm5-smd.out
EXPECTED_CM5_CHARGES = {
    0: 0.285713,  # H
    1: -0.571558,  # O
    2: 0.286093,  # H
}
CHARGE_TOL = 1e-5


# =============================================================================
# UNIT TESTS: Cm5Parser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_cm5_parser_matches_start_line():
    """Unit test: verify Cm5Parser.matches() recognizes the CM5 block header."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    assert parser.matches("          Charge Model 5         ", state) is True


@pytest.mark.unit
def test_cm5_parser_matches_with_varying_whitespace():
    """Unit test: verify Cm5Parser.matches() handles varying whitespace."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    assert parser.matches("Charge Model 5", state) is True
    assert parser.matches("   Charge Model 5   ", state) is True


@pytest.mark.unit
def test_cm5_parser_does_not_match_other_charge_headers():
    """Unit test: verify Cm5Parser.matches() rejects unrelated lines."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    assert parser.matches("Ground-State Mulliken Net Atomic Charges", state) is False
    assert parser.matches("Hirshfeld Atomic Charges", state) is False
    assert parser.matches("SCF time:   CPU 0.32s  wall 0.00s", state) is False
    assert parser.matches("Performing Hirshfeld population analysis.", state) is False


@pytest.mark.unit
def test_cm5_parser_skips_if_already_parsed():
    """Unit test: verify Cm5Parser.matches() returns False when already parsed."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")
    state.parsed_cm5 = True

    assert parser.matches("          Charge Model 5         ", state) is False


@pytest.mark.unit
def test_cm5_parser_does_not_mutate_state_in_matches():
    """Unit test: verify matches() is read-only — no state mutation allowed."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    line = "          Charge Model 5         "
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    assert result1 is True
    assert result2 is True
    assert state.parsed_cm5 is False
    assert state.atomic_charges == []


@pytest.mark.unit
def test_cm5_parser_parse_basic():
    """Unit test: verify Cm5Parser.parse() extracts charges from minimal input."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    lines = [
        "",
        "     Atom                 Charge (a.u.)    ",
        "  ---------------------------------------- ",
        "      1 H                     0.285713",
        "      2 O                    -0.571558",
        "      3 H                     0.286093",
        "  ---------------------------------------- ",
        "  Sum of atomic charges =     0.000248",
    ]
    parser.parse(PeekableIterator(iter(lines)), "          Charge Model 5         ", state)

    assert state.parsed_cm5 is True
    assert len(state.atomic_charges) == 1
    cm5 = state.atomic_charges[0]
    assert cm5.method == "CM5"
    assert cm5.charges[0] == pytest.approx(0.285713, abs=CHARGE_TOL)
    assert cm5.charges[1] == pytest.approx(-0.571558, abs=CHARGE_TOL)
    assert cm5.charges[2] == pytest.approx(0.286093, abs=CHARGE_TOL)


@pytest.mark.unit
def test_cm5_parser_uses_0based_indices():
    """Unit test: verify 1-based Q-Chem indices are converted to 0-based."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    lines = [
        "      1 H                     0.285713",
        "      2 O                    -0.571558",
        "  Sum of atomic charges =    -0.285845",
    ]
    parser.parse(PeekableIterator(iter(lines)), "Charge Model 5", state)

    assert 0 in state.atomic_charges[0].charges
    assert 1 in state.atomic_charges[0].charges
    assert 2 not in state.atomic_charges[0].charges


@pytest.mark.unit
def test_cm5_parser_empty_block_adds_warning():
    """Unit test: verify empty block produces a warning instead of crashing."""
    parser = Cm5Parser()
    state = ParseState(raw_output="")

    parser.parse(PeekableIterator(iter(["  Sum of atomic charges =     0.000000"])), "Charge Model 5", state)

    assert state.parsed_cm5 is False
    assert len(state.atomic_charges) == 0
    assert len(state.parsing_warnings) == 1


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["cm5"],
    indirect=True,
)
def test_cm5_charges_present(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify CM5 charges entry exists in atomic_charges list."""
    cm5 = next((c for c in parsed_qchem_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None, "CM5 charges not found in parsed data"
    assert cm5.method == "CM5"


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["cm5"],
    indirect=True,
)
def test_cm5_charges_is_atomiccharges(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify CM5 entry is an AtomicCharges instance."""
    cm5 = next((c for c in parsed_qchem_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None
    assert isinstance(cm5, AtomicCharges)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["cm5"],
    indirect=True,
)
def test_cm5_charges_has_values(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify CM5 charges dict is populated with float values."""
    cm5 = next((c for c in parsed_qchem_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None
    assert len(cm5.charges) > 0
    for idx, val in cm5.charges.items():
        assert isinstance(idx, int)
        assert isinstance(val, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["cm5"],
    indirect=True,
)
def test_cm5_charges_has_three_atoms(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify CM5 charges has 3 entries for H2O."""
    cm5 = next((c for c in parsed_qchem_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None
    assert len(cm5.charges) == 3


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
def test_cm5_parsed_alongside_hirshfeld(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: CM5 and Hirshfeld charges coexist — CM5 requires Hirshfeld."""
    methods = {c.method for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges}
    assert "Hirshfeld" in methods
    assert "CM5" in methods


@pytest.mark.integration
def test_cm5_parsed_alongside_mulliken(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: all three charge methods present when CM5=True."""
    methods = {c.method for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges}
    assert "Mulliken" in methods
    assert "Hirshfeld" in methods
    assert "CM5" in methods


@pytest.mark.integration
def test_cm5_charges_ordering(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: charges appear in output order — Mulliken, Hirshfeld, CM5."""
    methods = [c.method for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges]
    assert methods.index("Mulliken") < methods.index("Hirshfeld") < methods.index("CM5")


@pytest.mark.integration
def test_cm5_parsed_alongside_scf(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult):
    """Integration test: SCF results and CM5 charges are both present."""
    assert parsed_qchem_63_h2o_cm5_sp_data.scf is not None
    cm5 = next((c for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
@pytest.mark.parametrize(
    "atom_idx",
    list(EXPECTED_CM5_CHARGES),
    ids=[f"atom-{k}" for k in EXPECTED_CM5_CHARGES],
)
def test_cm5_charge_values(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult, atom_idx: int) -> None:
    """Regression test: verify exact CM5 charge values for each atom."""
    cm5 = next((c for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None
    assert cm5.charges[atom_idx] == pytest.approx(EXPECTED_CM5_CHARGES[atom_idx], abs=CHARGE_TOL)


@pytest.mark.regression
def test_cm5_charges_sum(parsed_qchem_63_h2o_cm5_sp_data: CalculationResult) -> None:
    """Regression test: verify sum of CM5 charges matches expected."""
    cm5 = next((c for c in parsed_qchem_63_h2o_cm5_sp_data.atomic_charges if c.method == "CM5"), None)
    assert cm5 is not None
    total = sum(cm5.charges.values())
    # Q-Chem reports sum = 0.000248 (same integration error as Hirshfeld)
    assert total == pytest.approx(sum(EXPECTED_CM5_CHARGES.values()), abs=CHARGE_TOL)
