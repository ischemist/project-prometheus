"""
Tests for the QChem ground state reference parser (excited state analysis).

These tests verify that the gs_ref parser correctly extracts ground state reference
data from the TDDFT Excited State Analysis block, including frontier NOs, electron counts,
Mulliken charges, and dipole moments.

Tests cover both restricted (RKS) and unrestricted (UKS) calculations.

Test hierarchy:
- contract: parser produces correct data structure with accurate values
- integration: ground state ref integrates with full TDDFT results

Format notes:
- RKS: Single "NOs" section, Mulliken charges only (no spin)
- UKS: Three "NOs" sections (alpha, beta, spin-traced), Mulliken charges + spin column
- Both files contain ground state reference at start of "Excited State Analysis" block
"""

import pytest

from calcflow.common.results import CalculationResult, GroundStateReference, TddftResults
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA (from gs-ref.md and actual QChem output)
# =============================================================================

# RKS TDDFT ground state reference (parsed_qchem_62_h2o_rks_tddft_data)
EXPECTED_RKS_FRONTIER_NOS = [0.0000, 2.0000]
EXPECTED_RKS_NUM_ELECTRONS = 10.0
EXPECTED_RKS_NUM_UNPAIRED = 0.0
EXPECTED_RKS_MULLIKEN_CHARGES = {0: 0.230554, 1: -0.460460, 2: 0.229906}
EXPECTED_RKS_MULLIKEN_SPINS = None
EXPECTED_RKS_DIPOLE_MOMENT = 2.015379
EXPECTED_RKS_DIPOLE_COMPONENTS = (-0.995831, -0.203503, -1.740304)

# UKS TDDFT ground state reference (parsed_qchem_62_h2o_uks_tddft_data)
EXPECTED_UKS_FRONTIER_NOS = [0.0000, 2.0000]  # spin-traced
EXPECTED_UKS_NUM_ELECTRONS = 10.0
EXPECTED_UKS_NUM_UNPAIRED = 0.0
EXPECTED_UKS_MULLIKEN_CHARGES = {0: 0.230554, 1: -0.460460, 2: 0.229906}
EXPECTED_UKS_MULLIKEN_SPINS = {0: 0.000000, 1: 0.000000, 2: 0.000000}
EXPECTED_UKS_DIPOLE_MOMENT = 2.015379
EXPECTED_UKS_DIPOLE_COMPONENTS = (-0.995831, -0.203503, -1.740304)

# 5.4 UKS TDDFT ground state reference (parsed_qchem_54_h2o_uks_tddft_data)
# Note: 5.4 uses "Mulliken Population Analysis" without "(State DM)" suffix
EXPECTED_54_UKS_FRONTIER_NOS = [0.0000, 2.0000]  # spin-traced
EXPECTED_54_UKS_NUM_ELECTRONS = 10.0
EXPECTED_54_UKS_NUM_UNPAIRED = 0.0
EXPECTED_54_UKS_MULLIKEN_CHARGES = {0: 0.230554, 1: -0.460460, 2: 0.229906}
EXPECTED_54_UKS_MULLIKEN_SPINS = {0: 0.000000, 1: 0.000000, 2: 0.000000}
EXPECTED_54_UKS_DIPOLE_MOMENT = 2.015378  # Minor difference from 6.2's 2.015379
EXPECTED_54_UKS_DIPOLE_COMPONENTS = (-0.995831, -0.203503, -1.740304)


# =============================================================================
# CONTRACT TESTS
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_gs_ref"])
def test_gs_ref_exists(fixture_name: str, request):
    """Ground state reference should exist in TDDFT results."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    assert isinstance(data.tddft, TddftResults)
    assert data.tddft.ground_state_ref is not None
    assert isinstance(data.tddft.ground_state_ref, GroundStateReference)


@pytest.mark.contract
@pytest.mark.parametrize(
    "fixture_name,expected_nos",
    [
        ("parsed_qchem_62_h2o_rks_tddft_data", EXPECTED_RKS_FRONTIER_NOS),
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_FRONTIER_NOS),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_54_UKS_FRONTIER_NOS),
    ],
)
def test_frontier_nos(fixture_name: str, expected_nos, request):
    """Frontier NO occupations should be parsed correctly."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    assert list(gs_ref.frontier_nos) == expected_nos


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_gs_ref"])
def test_num_electrons(fixture_name: str, request):
    """Total electron count should be parsed correctly."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    assert gs_ref.num_electrons == 10.0


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_gs_ref"])
def test_num_unpaired_electrons(fixture_name: str, request):
    """Number of unpaired electrons should be parsed correctly."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    assert gs_ref.num_unpaired_electrons == 0.0


@pytest.mark.contract
def test_mulliken_charges_rks(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """Mulliken charges should be parsed correctly for RKS."""
    gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
    assert len(gs_ref.mulliken.charges) == 3
    for atom_idx, expected_charge in EXPECTED_RKS_MULLIKEN_CHARGES.items():
        actual_charge = gs_ref.mulliken.charges[atom_idx]
        assert abs(actual_charge - expected_charge) < 1e-5


@pytest.mark.contract
@pytest.mark.parametrize(
    "fixture_name,expected_charges",
    [
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_MULLIKEN_CHARGES),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_54_UKS_MULLIKEN_CHARGES),
    ],
)
def test_mulliken_charges_uks(fixture_name: str, expected_charges, request) -> None:
    """Mulliken charges should be parsed correctly for UKS."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    assert len(gs_ref.mulliken.charges) == 3
    for atom_idx, expected_charge in expected_charges.items():
        actual_charge = gs_ref.mulliken.charges[atom_idx]
        assert abs(actual_charge - expected_charge) < 1e-5


@pytest.mark.contract
def test_mulliken_spins_none_for_rks(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """Mulliken spins should be None for RKS."""
    gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
    assert gs_ref.mulliken.spins is None


@pytest.mark.contract
@pytest.mark.parametrize(
    "fixture_name,expected_spins",
    [
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_MULLIKEN_SPINS),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_54_UKS_MULLIKEN_SPINS),
    ],
)
def test_mulliken_spins_uks(fixture_name: str, expected_spins, request) -> None:
    """Mulliken spins should be parsed correctly for UKS."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    assert gs_ref.mulliken.spins is not None
    assert len(gs_ref.mulliken.spins) == 3
    for atom_idx, expected_spin in expected_spins.items():
        actual_spin = gs_ref.mulliken.spins[atom_idx]
        assert abs(actual_spin - expected_spin) < 1e-5


@pytest.mark.contract
def test_dipole_moment_rks(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """Total dipole moment should be parsed correctly for RKS."""
    gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
    assert abs(gs_ref.dipole_moment_debye - EXPECTED_RKS_DIPOLE_MOMENT) < 1e-5


@pytest.mark.contract
@pytest.mark.parametrize(
    "fixture_name,expected_moment",
    [
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_DIPOLE_MOMENT),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_54_UKS_DIPOLE_MOMENT),
    ],
)
def test_dipole_moment_uks(fixture_name: str, expected_moment: float, request) -> None:
    """Total dipole moment should be parsed correctly for UKS."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    assert abs(gs_ref.dipole_moment_debye - expected_moment) < 1e-5


@pytest.mark.contract
def test_dipole_components_rks(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """Dipole moment Cartesian components should be parsed correctly for RKS."""
    gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
    for actual, expected in zip(gs_ref.dipole_components_debye, EXPECTED_RKS_DIPOLE_COMPONENTS, strict=True):
        assert abs(actual - expected) < 1e-5


@pytest.mark.contract
@pytest.mark.parametrize(
    "fixture_name,expected_components",
    [
        ("parsed_qchem_62_h2o_uks_tddft_data", EXPECTED_UKS_DIPOLE_COMPONENTS),
        ("parsed_qchem_54_h2o_uks_tddft_data", EXPECTED_54_UKS_DIPOLE_COMPONENTS),
    ],
)
def test_dipole_components_uks(fixture_name: str, expected_components, request) -> None:
    """Dipole moment Cartesian components should be parsed correctly for UKS."""
    data = request.getfixturevalue(fixture_name)
    gs_ref = data.tddft.ground_state_ref
    for actual, expected in zip(gs_ref.dipole_components_debye, expected_components, strict=True):
        assert abs(actual - expected) < 1e-5


# =============================================================================
# REGRESSION TESTS FOR 5.4 VERSION DIFFERENCES
# =============================================================================


@pytest.mark.regression
def test_54_uks_dipole_moment_minor_difference(parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
    """Regression test: 5.4 dipole moment differs slightly from 6.2 (2.015378 vs 2.015379).

    This verifies we handle version-specific numerical differences correctly.
    """
    gs_ref = parsed_qchem_54_h2o_uks_tddft_data.tddft.ground_state_ref
    assert abs(gs_ref.dipole_moment_debye - EXPECTED_54_UKS_DIPOLE_MOMENT) < 1e-5
