"""
Tests for the QChem orbitals block parser.

These tests verify that the orbitals parser correctly extracts molecular orbital
energies, occupation information, and HOMO/LUMO indices from Q-Chem output files.
Tests cover both restricted (RKS/RHF) and unrestricted (UKS/UHF) calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure
- integration: multiple components working together
- regression: exact numerical values match expected

Format notes:
- RKS/RHF: Only "Alpha MOs" section (restricted calculation)
- UKS/UHF: Both "Alpha MOs" and "Beta MOs" sections (unrestricted calculation)
- Energies are in atomic units (Hartree)
"""

import pytest

from calcflow.common.results import CalculationResult, Orbital, OrbitalsSet
from calcflow.io.qchem.blocks.orbitals import OrbitalsParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA
# =============================================================================

# RKS calculation (6.2-sp-smd.out) - H2O with 5 occupied, 2 virtual
EXPECTED_RKS_ALPHA_OCCUPIED = [-18.9200, -1.0263, -0.5245, -0.3311, -0.2422]
EXPECTED_RKS_ALPHA_VIRTUAL = [0.4520, 0.5636]
EXPECTED_RKS_HOMO_INDEX = 4  # 0-based: last occupied
EXPECTED_RKS_LUMO_INDEX = 5  # 0-based: first virtual
EXPECTED_RKS_TOTAL_ORBITALS = 7

# UKS calculation (6.2-uks-tddft.out) - H2O with 5 occupied, 53 virtual
EXPECTED_UKS_ALPHA_OCCUPIED = [-19.2346, -1.1182, -0.6261, -0.4888, -0.4147]
EXPECTED_UKS_ALPHA_VIRTUAL_FIRST_10 = [0.0878, 0.1389, 0.3614, 0.3715, 0.4265, 0.4394, 0.5388, 0.7688, 0.8532, 0.8877]
EXPECTED_UKS_ALPHA_VIRTUAL_LAST = 14.8185
EXPECTED_UKS_HOMO_INDEX = 4
EXPECTED_UKS_LUMO_INDEX = 5
EXPECTED_UKS_TOTAL_ORBITALS = 58  # 5 occupied + 53 virtual

# Numerical tolerance
ENERGY_TOL = 1e-4


# =============================================================================
# UNIT TESTS: OrbitalsParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_orbitals_parser_matches_start_line():
    """Unit test: verify OrbitalsParser.matches() recognizes orbital block start."""
    parser = OrbitalsParser()
    state = ParseState(raw_output="")

    assert parser.matches("                    Orbital Energies (a.u.)", state) is True
    assert parser.matches("   Orbital Energies (a.u.)", state) is True


@pytest.mark.unit
def test_orbitals_parser_does_not_match_non_orbital_lines():
    """Unit test: verify OrbitalsParser.matches() rejects non-orbital lines."""
    parser = OrbitalsParser()
    state = ParseState(raw_output="")

    assert parser.matches("Alpha MOs", state) is False
    assert parser.matches("-- Occupied --", state) is False
    assert parser.matches("-- Virtual --", state) is False
    assert parser.matches("-18.9200  -1.0263", state) is False
    assert parser.matches("Random output line", state) is False


@pytest.mark.unit
def test_orbitals_parser_skips_if_already_parsed():
    """Unit test: verify OrbitalsParser.matches() returns False when already parsed."""
    parser = OrbitalsParser()
    state = ParseState(raw_output="")
    state.parsed_orbitals = True

    assert parser.matches("                    Orbital Energies (a.u.)", state) is False


@pytest.mark.unit
def test_orbitals_parser_does_not_mutate_state_in_matches():
    """
    Unit test: verify that matches() is read-only and does not mutate state.
    Critical for parser-spec compliance.
    """
    parser = OrbitalsParser()
    state = ParseState(raw_output="")

    line = "                    Orbital Energies (a.u.)"
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    assert result1 is True
    assert result2 is True
    assert state.parsed_orbitals is False  # Should NOT be set
    assert state.orbitals is None  # Should NOT be populated


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["orbitals"],
    indirect=True,
)
def test_orbitals_set_has_correct_type(parsed_qchem_data: CalculationResult):
    """Contract test: verify orbitals field is OrbitalsSet instance."""
    assert parsed_qchem_data.orbitals is not None
    assert isinstance(parsed_qchem_data.orbitals, OrbitalsSet)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["orbitals"],
    indirect=True,
)
def test_orbitals_set_alpha_orbitals_is_sequence(parsed_qchem_data: CalculationResult):
    """Contract test: verify alpha_orbitals is a sequence of Orbital objects."""
    orbitals = parsed_qchem_data.orbitals
    assert orbitals is not None
    assert isinstance(orbitals.alpha_orbitals, (list, tuple))
    assert len(orbitals.alpha_orbitals) > 0
    assert all(isinstance(orb, Orbital) for orb in orbitals.alpha_orbitals)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["orbitals"],
    indirect=True,
)
def test_orbital_required_fields_present(parsed_qchem_data: CalculationResult):
    """Contract test: verify each Orbital has required fields with correct types."""
    orbitals = parsed_qchem_data.orbitals
    assert orbitals is not None

    for orbital in orbitals.alpha_orbitals:
        assert isinstance(orbital.index, int)
        assert orbital.index >= 0  # 0-based indexing
        assert isinstance(orbital.energy, float)
        # occupation and energy_ev are optional, can be None
        assert orbital.occupation is None or isinstance(orbital.occupation, float)
        assert orbital.energy_ev is None or isinstance(orbital.energy_ev, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["orbitals"],
    indirect=True,
)
def test_orbitals_homo_lumo_indices_set(parsed_qchem_data: CalculationResult):
    """Contract test: verify HOMO/LUMO indices are set and valid."""
    orbitals = parsed_qchem_data.orbitals
    assert orbitals is not None

    # Alpha HOMO and LUMO should always be present
    assert orbitals.alpha_homo_index is not None
    assert orbitals.alpha_lumo_index is not None
    assert isinstance(orbitals.alpha_homo_index, int)
    assert isinstance(orbitals.alpha_lumo_index, int)
    # LUMO should be one more than HOMO for closed-shell systems
    assert orbitals.alpha_lumo_index == orbitals.alpha_homo_index + 1


@pytest.mark.contract
def test_rks_calculation_has_no_beta_orbitals(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Contract test: verify beta_orbitals is None for restricted (RKS) calculation."""
    orbitals = parsed_qchem_62_h2o_sp_data.orbitals
    assert orbitals is not None
    assert orbitals.beta_orbitals is None
    assert orbitals.beta_homo_index is None
    assert orbitals.beta_lumo_index is None


@pytest.mark.contract
def test_uks_calculation_has_beta_orbitals(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Contract test: verify beta_orbitals present for unrestricted (UKS) calculation."""
    orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
    assert orbitals is not None
    assert orbitals.beta_orbitals is not None
    assert isinstance(orbitals.beta_orbitals, (list, tuple))
    assert len(orbitals.beta_orbitals) > 0
    assert all(isinstance(orb, Orbital) for orb in orbitals.beta_orbitals)
    # Beta HOMO/LUMO indices should also be set
    assert orbitals.beta_homo_index is not None
    assert orbitals.beta_lumo_index is not None


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
def test_orbitals_parsed_alongside_scf(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Integration test: verify orbitals parser works with SCF parser."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None
    assert parsed_qchem_62_h2o_sp_data.orbitals is not None


@pytest.mark.integration
def test_orbitals_indices_are_sequential(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Integration test: verify orbital indices are 0-based and sequential."""
    orbitals = parsed_qchem_62_h2o_sp_data.orbitals
    assert orbitals is not None

    # Check alpha orbitals
    for i, orbital in enumerate(orbitals.alpha_orbitals):
        assert orbital.index == i, f"Expected index {i}, got {orbital.index}"

    # Check beta orbitals if present
    if orbitals.beta_orbitals is not None:
        for i, orbital in enumerate(orbitals.beta_orbitals):
            assert orbital.index == i, f"Expected beta index {i}, got {orbital.index}"


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
def test_rks_orbital_count(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact total orbital count for RKS calculation."""
    orbitals = parsed_qchem_62_h2o_sp_data.orbitals
    assert orbitals is not None
    assert len(orbitals.alpha_orbitals) == EXPECTED_RKS_TOTAL_ORBITALS


@pytest.mark.regression
def test_rks_homo_lumo_indices(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact HOMO/LUMO indices for RKS calculation."""
    orbitals = parsed_qchem_62_h2o_sp_data.orbitals
    assert orbitals is not None
    assert orbitals.alpha_homo_index == EXPECTED_RKS_HOMO_INDEX
    assert orbitals.alpha_lumo_index == EXPECTED_RKS_LUMO_INDEX


@pytest.mark.regression
@pytest.mark.parametrize(
    "orbital_idx,expected_energy",
    [
        # Occupied orbitals (0-4)
        (0, -18.9200),
        (1, -1.0263),
        (2, -0.5245),
        (3, -0.3311),
        (4, -0.2422),
        # Virtual orbitals (5-6)
        (5, 0.4520),
        (6, 0.5636),
    ],
    ids=["occ-0", "occ-1", "occ-2", "occ-3", "occ-4(HOMO)", "virt-5(LUMO)", "virt-6"],
)
def test_rks_orbital_energies_exact(
    parsed_qchem_62_h2o_sp_data: CalculationResult,
    orbital_idx: int,
    expected_energy: float,
):
    """Regression test: verify exact orbital energies for RKS calculation."""
    orbitals = parsed_qchem_62_h2o_sp_data.orbitals
    assert orbitals is not None
    assert len(orbitals.alpha_orbitals) > orbital_idx

    orbital = orbitals.alpha_orbitals[orbital_idx]
    assert orbital.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)


@pytest.mark.regression
def test_uks_orbital_count(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact total orbital count for UKS calculation."""
    orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
    assert orbitals is not None
    assert len(orbitals.alpha_orbitals) == EXPECTED_UKS_TOTAL_ORBITALS
    assert orbitals.beta_orbitals is not None
    assert len(orbitals.beta_orbitals) == EXPECTED_UKS_TOTAL_ORBITALS  # Should match


@pytest.mark.regression
def test_uks_homo_lumo_indices(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
    """Regression test: verify exact HOMO/LUMO indices for UKS calculation."""
    orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
    assert orbitals is not None
    assert orbitals.alpha_homo_index == EXPECTED_UKS_HOMO_INDEX
    assert orbitals.alpha_lumo_index == EXPECTED_UKS_LUMO_INDEX
    assert orbitals.beta_homo_index == EXPECTED_UKS_HOMO_INDEX
    assert orbitals.beta_lumo_index == EXPECTED_UKS_LUMO_INDEX


@pytest.mark.regression
@pytest.mark.parametrize(
    "orbital_idx,expected_energy",
    [
        # Alpha occupied orbitals (0-4)
        (0, -19.2346),
        (1, -1.1182),
        (2, -0.6261),
        (3, -0.4888),
        (4, -0.4147),
        # Alpha virtual orbitals - first 10
        (5, 0.0878),  # LUMO
        (6, 0.1389),
        (7, 0.3614),
        (8, 0.3715),
        (9, 0.4265),
        (10, 0.4394),
        (11, 0.5388),
        (12, 0.7688),
        (13, 0.8532),
        (14, 0.8877),
        # Last virtual orbital
        (57, 14.8185),
    ],
    ids=[
        "alpha-occ-0",
        "alpha-occ-1",
        "alpha-occ-2",
        "alpha-occ-3",
        "alpha-occ-4(HOMO)",
        "alpha-virt-5(LUMO)",
        "alpha-virt-6",
        "alpha-virt-7",
        "alpha-virt-8",
        "alpha-virt-9",
        "alpha-virt-10",
        "alpha-virt-11",
        "alpha-virt-12",
        "alpha-virt-13",
        "alpha-virt-14",
        "alpha-virt-57(last)",
    ],
)
def test_uks_alpha_orbital_energies_sample(
    parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
    orbital_idx: int,
    expected_energy: float,
):
    """Regression test: verify sample of exact alpha orbital energies for UKS calculation."""
    orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
    assert orbitals is not None
    assert len(orbitals.alpha_orbitals) > orbital_idx

    orbital = orbitals.alpha_orbitals[orbital_idx]
    assert orbital.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "orbital_idx,expected_energy",
    [
        # Beta occupied orbitals (0-4) - should match alpha for closed-shell
        (0, -19.2346),
        (4, -0.4147),
        # Beta LUMO
        (5, 0.0878),
        # Beta last virtual
        (57, 14.8185),
    ],
    ids=["beta-occ-0", "beta-occ-4(HOMO)", "beta-virt-5(LUMO)", "beta-virt-57(last)"],
)
def test_uks_beta_orbital_energies_sample(
    parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
    orbital_idx: int,
    expected_energy: float,
):
    """Regression test: verify sample of exact beta orbital energies for UKS calculation."""
    orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
    assert orbitals is not None
    assert orbitals.beta_orbitals is not None
    assert len(orbitals.beta_orbitals) > orbital_idx

    orbital = orbitals.beta_orbitals[orbital_idx]
    assert orbital.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)
