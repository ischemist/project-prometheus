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

from calcflow.common.models import CalculationResult, GroundStateReference, TddftResults

# =============================================================================
# HARDCODED TEST DATA (from gs-ref.md and actual QChem output)
# =============================================================================

# RKS TDDFT ground state reference (parsed_qchem_62_h2o_rks_tddft_data)
EXPECTED_RKS_FRONTIER_NOS = [0.0000, 2.0000]
EXPECTED_RKS_NUM_ELECTRONS = 10.0
EXPECTED_RKS_NUM_UNPAIRED = 0.0
EXPECTED_RKS_MULLIKEN_CHARGES = [0.230554, -0.460460, 0.229906]
EXPECTED_RKS_MULLIKEN_SPINS = None
EXPECTED_RKS_DIPOLE_MOMENT = 2.015379
EXPECTED_RKS_DIPOLE_COMPONENTS = (-0.995831, -0.203503, -1.740304)

# UKS TDDFT ground state reference (parsed_qchem_62_h2o_uks_tddft_data)
EXPECTED_UKS_FRONTIER_NOS = [0.0000, 2.0000]  # spin-traced
EXPECTED_UKS_NUM_ELECTRONS = 10.0
EXPECTED_UKS_NUM_UNPAIRED = 0.0
EXPECTED_UKS_MULLIKEN_CHARGES = [0.230554, -0.460460, 0.229906]
EXPECTED_UKS_MULLIKEN_SPINS = [0.000000, 0.000000, 0.000000]
EXPECTED_UKS_DIPOLE_MOMENT = 2.015379
EXPECTED_UKS_DIPOLE_COMPONENTS = (-0.995831, -0.203503, -1.740304)


# =============================================================================
# CONTRACT TESTS
# =============================================================================


@pytest.mark.contract
class TestRksGroundStateReference:
    """Contract tests for RKS ground state reference parsing."""

    def test_gs_ref_exists(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Ground state reference should exist in TDDFT results."""
        assert parsed_qchem_62_h2o_rks_tddft_data.tddft is not None
        assert isinstance(parsed_qchem_62_h2o_rks_tddft_data.tddft, TddftResults)
        assert parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref is not None
        assert isinstance(parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref, GroundStateReference)

    def test_frontier_nos(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Frontier NO occupations should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        assert list(gs_ref.frontier_nos) == EXPECTED_RKS_FRONTIER_NOS

    def test_num_electrons(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Total electron count should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        assert gs_ref.num_electrons == EXPECTED_RKS_NUM_ELECTRONS

    def test_num_unpaired_electrons(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Number of unpaired electrons should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        assert gs_ref.num_unpaired_electrons == EXPECTED_RKS_NUM_UNPAIRED

    def test_mulliken_charges(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Mulliken charges should be parsed correctly for all atoms."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        assert len(gs_ref.mulliken_charges) == 3
        for actual, expected in zip(gs_ref.mulliken_charges, EXPECTED_RKS_MULLIKEN_CHARGES, strict=True):
            assert abs(actual - expected) < 1e-5

    def test_mulliken_spins_none_for_rks(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Mulliken spins should be None for RKS."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        assert gs_ref.mulliken_spins is None

    def test_dipole_moment(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Total dipole moment should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        assert abs(gs_ref.dipole_moment_debye - EXPECTED_RKS_DIPOLE_MOMENT) < 1e-5

    def test_dipole_components(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Dipole moment Cartesian components should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_rks_tddft_data.tddft.ground_state_ref
        for actual, expected in zip(gs_ref.dipole_components_debye, EXPECTED_RKS_DIPOLE_COMPONENTS, strict=True):
            assert abs(actual - expected) < 1e-5


@pytest.mark.contract
class TestUksGroundStateReference:
    """Contract tests for UKS ground state reference parsing."""

    def test_gs_ref_exists(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Ground state reference should exist in TDDFT results."""
        assert parsed_qchem_62_h2o_uks_tddft_data.tddft is not None
        assert isinstance(parsed_qchem_62_h2o_uks_tddft_data.tddft, TddftResults)
        assert parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref is not None
        assert isinstance(parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref, GroundStateReference)

    def test_frontier_nos(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Frontier NO occupations (spin-traced) should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        assert list(gs_ref.frontier_nos) == EXPECTED_UKS_FRONTIER_NOS

    def test_num_electrons(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Total electron count should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        assert gs_ref.num_electrons == EXPECTED_UKS_NUM_ELECTRONS

    def test_num_unpaired_electrons(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Number of unpaired electrons should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        assert gs_ref.num_unpaired_electrons == EXPECTED_UKS_NUM_UNPAIRED

    def test_mulliken_charges(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Mulliken charges should be parsed correctly for all atoms."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        assert len(gs_ref.mulliken_charges) == 3
        for actual, expected in zip(gs_ref.mulliken_charges, EXPECTED_UKS_MULLIKEN_CHARGES, strict=True):
            assert abs(actual - expected) < 1e-5

    def test_mulliken_spins(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Mulliken spins should be parsed correctly for UKS."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        assert gs_ref.mulliken_spins is not None
        assert len(gs_ref.mulliken_spins) == 3
        for actual, expected in zip(gs_ref.mulliken_spins, EXPECTED_UKS_MULLIKEN_SPINS, strict=True):
            assert abs(actual - expected) < 1e-5

    def test_dipole_moment(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Total dipole moment should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        assert abs(gs_ref.dipole_moment_debye - EXPECTED_UKS_DIPOLE_MOMENT) < 1e-5

    def test_dipole_components(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Dipole moment Cartesian components should be parsed correctly."""
        gs_ref = parsed_qchem_62_h2o_uks_tddft_data.tddft.ground_state_ref
        for actual, expected in zip(gs_ref.dipole_components_debye, EXPECTED_UKS_DIPOLE_COMPONENTS, strict=True):
            assert abs(actual - expected) < 1e-5
