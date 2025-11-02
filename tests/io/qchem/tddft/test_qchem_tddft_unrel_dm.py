"""
Tests for the QChem unrelaxed density matrix parser.

These tests verify that the unrelaxed density matrix parser correctly extracts
NOs, Mulliken population, multipole moments, and exciton analysis from both
RKS and UKS TDDFT calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure
- regression: exact numerical values match expected

Format notes:
- RKS: "Singlet N :", single NOs section, 4-column Mulliken, single exciton analysis
- UKS: "Excited State N :", three NOs sections (alpha/beta/spin-traced), 6-column Mulliken, three exciton analyses
"""

import pytest

from calcflow.common.results import (
    CalculationResult,
    ExcitonAnalysis,
    NaturalOrbitals,
    TddftResults,
    UnrelaxedDensityMatrix,
)

# =============================================================================
# HARDCODED TEST DATA (from unrelaxed-den.md)
# =============================================================================

# RKS data - Singlet 1 (parsed_qchem_62_h2o_rks_tddft_data)
EXPECTED_RKS_STATE_1_NUM = 1
EXPECTED_RKS_STATE_1_MULTIPLICITY = "Singlet 1"
EXPECTED_RKS_STATE_1_FRONTIER_NOS = [0.9992, 1.0006]
EXPECTED_RKS_STATE_1_NUM_ELECTRONS = 10.000000
EXPECTED_RKS_STATE_1_NUM_UNPAIRED = 1.99930
EXPECTED_RKS_STATE_1_NUM_UNPAIRED_NL = 2.00000
EXPECTED_RKS_STATE_1_PR_NO = 2.001426
EXPECTED_RKS_STATE_1_MULLIKEN_CHARGES = {0: -0.283169, 1: 0.562822, 2: -0.279653}
EXPECTED_RKS_STATE_1_MOLECULAR_CHARGE = -0.000000
EXPECTED_RKS_STATE_1_DIPOLE_MOMENT = 1.409939
EXPECTED_RKS_STATE_1_DIPOLE_COMPONENTS = (0.710066, 0.138452, 1.210193)
EXPECTED_RKS_STATE_1_R_H = (2.311785, 1.559453, -0.071430)
EXPECTED_RKS_STATE_1_R_E = (1.957004, 1.488335, -0.685054)
EXPECTED_RKS_STATE_1_SEPARATION = 0.712363
EXPECTED_RKS_STATE_1_HOLE_SIZE = 0.827946
EXPECTED_RKS_STATE_1_HOLE_SIZE_COMPONENTS = (0.389955, 0.616973, 0.390864)
EXPECTED_RKS_STATE_1_ELECTRON_SIZE = 1.932807
EXPECTED_RKS_STATE_1_ELECTRON_SIZE_COMPONENTS = (1.229170, 1.031590, 1.077361)

# RKS data - Singlet 10
EXPECTED_RKS_STATE_10_NUM = 10
EXPECTED_RKS_STATE_10_FRONTIER_NOS = [0.5164, 1.4833]
EXPECTED_RKS_STATE_10_NUM_UNPAIRED = 2.00436
EXPECTED_RKS_STATE_10_NUM_UNPAIRED_NL = 2.10719
EXPECTED_RKS_STATE_10_PR_NO = 4.419449
EXPECTED_RKS_STATE_10_MULLIKEN_CHARGES = {0: 0.247645, 1: -0.495641, 2: 0.247997}
EXPECTED_RKS_STATE_10_DIPOLE_MOMENT = 2.601379
EXPECTED_RKS_STATE_10_DIPOLE_COMPONENTS = (-1.287137, -0.261915, -2.245406)
EXPECTED_RKS_STATE_10_R_H = (2.333178, 1.563873, -0.033870)
EXPECTED_RKS_STATE_10_R_E = (2.393695, 1.576007, 0.071060)
EXPECTED_RKS_STATE_10_SEPARATION = 0.121737

# UKS data - Excited State 1 (parsed_qchem_62_h2o_uks_tddft_data)
EXPECTED_UKS_STATE_1_NUM = 1
EXPECTED_UKS_STATE_1_MULTIPLICITY = "Excited State 1"
EXPECTED_UKS_STATE_1_FRONTIER_NOS_ALPHA = [0.4996, 0.5003]
EXPECTED_UKS_STATE_1_NUM_ELECTRONS_ALPHA = 5.000000
EXPECTED_UKS_STATE_1_FRONTIER_NOS_BETA = [0.4996, 0.5003]
EXPECTED_UKS_STATE_1_NUM_ELECTRONS_BETA = 5.000000
EXPECTED_UKS_STATE_1_FRONTIER_NOS_SPIN_TRACED = [0.9991, 1.0007]
EXPECTED_UKS_STATE_1_NUM_ELECTRONS_SPIN_TRACED = 10.000000
EXPECTED_UKS_STATE_1_NUM_UNPAIRED = 2.00012
EXPECTED_UKS_STATE_1_NUM_UNPAIRED_NL = 2.00000
EXPECTED_UKS_STATE_1_PR_NO = 2.003363
EXPECTED_UKS_STATE_1_MULLIKEN_CHARGES = {0: -0.272264, 1: 0.541008, 2: -0.268744}
EXPECTED_UKS_STATE_1_MULLIKEN_SPINS = {0: -0.000000, 1: 0.000000, 2: -0.000000}
EXPECTED_UKS_STATE_1_DIPOLE_MOMENT = 1.231477
EXPECTED_UKS_STATE_1_DIPOLE_COMPONENTS = (0.621980, 0.120499, 1.056010)
# Exciton Total
EXPECTED_UKS_STATE_1_R_H_TOTAL = (2.311701, 1.559437, -0.071560)
EXPECTED_UKS_STATE_1_R_E_TOTAL = (1.975431, 1.492090, -0.652796)
EXPECTED_UKS_STATE_1_SEPARATION_TOTAL = 0.674871
EXPECTED_UKS_STATE_1_HOLE_SIZE_TOTAL = 0.827901
EXPECTED_UKS_STATE_1_ELECTRON_SIZE_TOTAL = 1.861781
# Exciton Alpha
EXPECTED_UKS_STATE_1_R_H_ALPHA = (2.311702, 1.559437, -0.071560)
EXPECTED_UKS_STATE_1_SEPARATION_ALPHA = 0.674872
# Exciton Beta
EXPECTED_UKS_STATE_1_SEPARATION_BETA = 0.674873

# UKS data - Excited State 10
EXPECTED_UKS_STATE_10_NUM = 10
EXPECTED_UKS_STATE_10_FRONTIER_NOS_SPIN_TRACED = [0.9937, 1.0061]
EXPECTED_UKS_STATE_10_NUM_UNPAIRED = 2.00206
EXPECTED_UKS_STATE_10_PR_NO = 2.029150
EXPECTED_UKS_STATE_10_MULLIKEN_CHARGES = {0: -0.107441, 1: 0.212358, 2: -0.104917}
EXPECTED_UKS_STATE_10_DIPOLE_MOMENT = 0.634124
EXPECTED_UKS_STATE_10_R_H_TOTAL = (2.221157, 1.541133, -0.228933)
EXPECTED_UKS_STATE_10_SEPARATION_TOTAL = 0.551031

# UKS 5.4 data - Excited State 1
EXPECTED_UKS_54_STATE_1_FRONTIER_NOS_ALPHA = [0.4996, 0.5003]
EXPECTED_UKS_54_STATE_1_NUM_ELECTRONS_ALPHA = 5.000000
EXPECTED_UKS_54_STATE_1_FRONTIER_NOS_BETA = [0.4996, 0.5003]
EXPECTED_UKS_54_STATE_1_NUM_ELECTRONS_BETA = 5.000000
EXPECTED_UKS_54_STATE_1_FRONTIER_NOS_SPIN_TRACED = [0.9991, 1.0007]
EXPECTED_UKS_54_STATE_1_NUM_ELECTRONS_SPIN_TRACED = 10.000000
EXPECTED_UKS_54_STATE_1_NUM_UNPAIRED = 2.00012
EXPECTED_UKS_54_STATE_1_NUM_UNPAIRED_NL = 2.00000
EXPECTED_UKS_54_STATE_1_PR_NO = 2.003363
EXPECTED_UKS_54_STATE_1_MULLIKEN_CHARGES = {0: -0.272264, 1: 0.541008, 2: -0.268744}
EXPECTED_UKS_54_STATE_1_MULLIKEN_SPINS = {0: 0.000000, 1: 0.000000, 2: -0.000000}
EXPECTED_UKS_54_STATE_1_DIPOLE_MOMENT = 1.231477
EXPECTED_UKS_54_STATE_1_DIPOLE_COMPONENTS = (0.621980, 0.120499, 1.056010)
EXPECTED_UKS_54_STATE_1_R_H_TOTAL = (2.311708, 1.559437, -0.071560)
EXPECTED_UKS_54_STATE_1_R_E_TOTAL = (1.975434, 1.492090, -0.652796)
EXPECTED_UKS_54_STATE_1_SEPARATION_TOTAL = 0.674870
EXPECTED_UKS_54_STATE_1_HOLE_SIZE_TOTAL = 0.827901
EXPECTED_UKS_54_STATE_1_ELECTRON_SIZE_TOTAL = 1.861783

# UKS 5.4 data - Excited State 2
EXPECTED_UKS_54_STATE_2_FRONTIER_NOS_ALPHA = [0.4996, 0.5003]
EXPECTED_UKS_54_STATE_2_NUM_ELECTRONS_ALPHA = 5.000000
EXPECTED_UKS_54_STATE_2_FRONTIER_NOS_BETA = [0.4996, 0.5003]
EXPECTED_UKS_54_STATE_2_NUM_ELECTRONS_BETA = 5.000000
EXPECTED_UKS_54_STATE_2_FRONTIER_NOS_SPIN_TRACED = [0.9992, 1.0006]
EXPECTED_UKS_54_STATE_2_NUM_ELECTRONS_SPIN_TRACED = 10.000000
EXPECTED_UKS_54_STATE_2_NUM_UNPAIRED = 1.99930
EXPECTED_UKS_54_STATE_2_NUM_UNPAIRED_NL = 2.00000
EXPECTED_UKS_54_STATE_2_PR_NO = 2.001426
EXPECTED_UKS_54_STATE_2_MULLIKEN_CHARGES = {0: -0.283169, 1: 0.562822, 2: -0.279653}
EXPECTED_UKS_54_STATE_2_MULLIKEN_SPINS = {0: 0.000000, 1: -0.000000, 2: 0.000000}
EXPECTED_UKS_54_STATE_2_DIPOLE_MOMENT = 1.409928
EXPECTED_UKS_54_STATE_2_DIPOLE_COMPONENTS = (0.710060, 0.138451, 1.210183)
EXPECTED_UKS_54_STATE_2_R_H_TOTAL = (2.311785, 1.559453, -0.071430)
EXPECTED_UKS_54_STATE_2_R_E_TOTAL = (1.957005, 1.488336, -0.685052)
EXPECTED_UKS_54_STATE_2_SEPARATION_TOTAL = 0.712361
EXPECTED_UKS_54_STATE_2_HOLE_SIZE_TOTAL = 0.827946
EXPECTED_UKS_54_STATE_2_ELECTRON_SIZE_TOTAL = 1.932808

# UKS 5.4 data - Excited State 7
EXPECTED_UKS_54_STATE_7_FRONTIER_NOS_ALPHA = [0.4902, 0.5097]
EXPECTED_UKS_54_STATE_7_NUM_ELECTRONS_ALPHA = 5.000000
EXPECTED_UKS_54_STATE_7_FRONTIER_NOS_BETA = [0.4902, 0.5097]
EXPECTED_UKS_54_STATE_7_NUM_ELECTRONS_BETA = 5.000000
EXPECTED_UKS_54_STATE_7_FRONTIER_NOS_SPIN_TRACED = [0.9804, 1.0193]
EXPECTED_UKS_54_STATE_7_NUM_ELECTRONS_SPIN_TRACED = 10.000000
EXPECTED_UKS_54_STATE_7_NUM_UNPAIRED = 2.00541
EXPECTED_UKS_54_STATE_7_NUM_UNPAIRED_NL = 2.00217
EXPECTED_UKS_54_STATE_7_PR_NO = 2.090472
EXPECTED_UKS_54_STATE_7_MULLIKEN_CHARGES = {0: -0.229283, 1: 0.466761, 2: -0.237478}
EXPECTED_UKS_54_STATE_7_MULLIKEN_SPINS = {0: -0.000000, 1: 0.000000, 2: -0.000000}
EXPECTED_UKS_54_STATE_7_DIPOLE_MOMENT = 1.916470
EXPECTED_UKS_54_STATE_7_DIPOLE_COMPONENTS = (0.921424, 0.200661, 1.668403)
EXPECTED_UKS_54_STATE_7_R_H_TOTAL = (2.349369, 1.567221, -0.005281)
EXPECTED_UKS_54_STATE_7_R_E_TOTAL = (1.951284, 1.483303, -0.713040)
EXPECTED_UKS_54_STATE_7_SEPARATION_TOTAL = 0.816355
EXPECTED_UKS_54_STATE_7_HOLE_SIZE_TOTAL = 0.860981
EXPECTED_UKS_54_STATE_7_ELECTRON_SIZE_TOTAL = 2.078160

# Numerical tolerance
FLOAT_TOL = 1e-6


# =============================================================================
# UNIT TESTS: Matches behavior
# =============================================================================


@pytest.mark.unit
def test_unrelaxed_dm_parser_exists():
    """Unit test: verify parser can be imported."""
    from calcflow.io.qchem.blocks.tddft.unrel_dm import UnrelaxedDensityMatrixParser

    parser = UnrelaxedDensityMatrixParser()
    assert parser is not None


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", ["parsed_qchem_62_h2o_rks_tddft_data", "parsed_qchem_62_h2o_uks_tddft_data"])
def test_unrelaxed_dm_exists(fixture_name: str, request) -> None:
    """Unrelaxed density matrices should exist in TDDFT results."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    assert isinstance(data.tddft, TddftResults)
    assert data.tddft.unrelaxed_density_matrices is not None
    assert len(data.tddft.unrelaxed_density_matrices) > 0


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", ["parsed_qchem_62_h2o_rks_tddft_data", "parsed_qchem_62_h2o_uks_tddft_data"])
def test_unrelaxed_dm_structure(fixture_name: str, request) -> None:
    """Each unrelaxed DM should have correct structure."""
    data = request.getfixturevalue(fixture_name)
    unrel_dms = data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None

    for dm in unrel_dms:
        assert isinstance(dm, UnrelaxedDensityMatrix)
        assert isinstance(dm.state_number, int)
        assert dm.state_number > 0
        assert isinstance(dm.multiplicity, str)
        assert isinstance(dm.nos_spin_traced, NaturalOrbitals)
        assert isinstance(dm.exciton_total, ExcitonAnalysis)


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", ["parsed_qchem_62_h2o_rks_tddft_data", "parsed_qchem_62_h2o_uks_tddft_data"])
def test_nos_structure(fixture_name: str, request) -> None:
    """NaturalOrbitals should have correct structure."""
    data = request.getfixturevalue(fixture_name)
    unrel_dms = data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None
    dm = unrel_dms[0]
    nos = dm.nos_spin_traced

    assert isinstance(nos.frontier_occupations, (list, tuple))
    assert len(nos.frontier_occupations) == 2
    assert isinstance(nos.num_electrons, float)
    assert isinstance(nos.num_unpaired, (float, type(None)))
    assert isinstance(nos.pr_no, (float, type(None)))


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", ["parsed_qchem_62_h2o_rks_tddft_data", "parsed_qchem_62_h2o_uks_tddft_data"])
def test_exciton_structure(fixture_name: str, request) -> None:
    """ExcitonAnalysis should have correct structure."""
    data = request.getfixturevalue(fixture_name)
    unrel_dms = data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None
    dm = unrel_dms[0]
    exciton = dm.exciton_total

    assert isinstance(exciton.r_h_ang, tuple)
    assert len(exciton.r_h_ang) == 3
    assert isinstance(exciton.r_e_ang, tuple)
    assert len(exciton.r_e_ang) == 3
    assert isinstance(exciton.separation_ang, float)
    assert isinstance(exciton.hole_size_ang, float)
    assert isinstance(exciton.electron_size_ang, float)


@pytest.mark.contract
def test_rks_has_no_alpha_beta_nos(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """RKS should not have alpha/beta NOs."""
    unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None

    for dm in unrel_dms:
        assert dm.nos_alpha is None, "RKS should not have alpha NOs"
        assert dm.nos_beta is None, "RKS should not have beta NOs"


@pytest.mark.contract
def test_rks_has_no_alpha_beta_exciton(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """RKS should not have alpha/beta exciton analysis."""
    unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None

    for dm in unrel_dms:
        assert dm.exciton_alpha is None, "RKS should not have alpha exciton"
        assert dm.exciton_beta is None, "RKS should not have beta exciton"


@pytest.mark.contract
def test_uks_has_alpha_beta_nos(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
    """UKS should have alpha/beta NOs."""
    unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None

    for dm in unrel_dms:
        assert dm.nos_alpha is not None, "UKS should have alpha NOs"
        assert dm.nos_beta is not None, "UKS should have beta NOs"
        assert isinstance(dm.nos_alpha, NaturalOrbitals)
        assert isinstance(dm.nos_beta, NaturalOrbitals)


@pytest.mark.contract
def test_uks_has_alpha_beta_exciton(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
    """UKS should have alpha/beta exciton analysis."""
    unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None

    for dm in unrel_dms:
        assert dm.exciton_alpha is not None, "UKS should have alpha exciton"
        assert dm.exciton_beta is not None, "UKS should have beta exciton"
        assert isinstance(dm.exciton_alpha, ExcitonAnalysis)
        assert isinstance(dm.exciton_beta, ExcitonAnalysis)


@pytest.mark.contract
def test_mulliken_has_spins(parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
    """UKS Mulliken should have spin densities."""
    unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
    assert unrel_dms is not None

    for dm in unrel_dms:
        assert dm.mulliken.spins is not None, "UKS should have spin densities"
        assert len(dm.mulliken.spins) > 0


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
class TestRksUnrelaxedDensityMatrixRegression:
    """Regression tests for RKS unrelaxed density matrix values."""

    def test_state_1_nos(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact NOs values for RKS Singlet 1."""
        unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        assert dm.state_number == EXPECTED_RKS_STATE_1_NUM
        assert dm.multiplicity == EXPECTED_RKS_STATE_1_MULTIPLICITY

        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(EXPECTED_RKS_STATE_1_FRONTIER_NOS, abs=FLOAT_TOL)
        assert nos.num_electrons == pytest.approx(EXPECTED_RKS_STATE_1_NUM_ELECTRONS, abs=FLOAT_TOL)
        assert nos.num_unpaired == pytest.approx(EXPECTED_RKS_STATE_1_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.num_unpaired_nl == pytest.approx(EXPECTED_RKS_STATE_1_NUM_UNPAIRED_NL, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_RKS_STATE_1_PR_NO, abs=FLOAT_TOL)

    def test_state_1_mulliken(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken charges for RKS Singlet 1."""
        unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        for atom_idx, expected_charge in EXPECTED_RKS_STATE_1_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.mulliken.spins is None, "RKS should not have spins"

    def test_state_1_multipole(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact multipole moments for RKS Singlet 1."""
        unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        assert dm.molecular_charge == pytest.approx(EXPECTED_RKS_STATE_1_MOLECULAR_CHARGE, abs=FLOAT_TOL)
        assert dm.num_electrons == pytest.approx(EXPECTED_RKS_STATE_1_NUM_ELECTRONS, abs=FLOAT_TOL)
        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_RKS_STATE_1_DIPOLE_MOMENT, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_RKS_STATE_1_DIPOLE_COMPONENTS):
            assert dm.dipole_components_debye[i] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_state_1_exciton(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis for RKS Singlet 1."""
        unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]
        exciton = dm.exciton_total

        for i, expected in enumerate(EXPECTED_RKS_STATE_1_R_H):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_RKS_STATE_1_R_E):
            assert exciton.r_e_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_RKS_STATE_1_SEPARATION, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_RKS_STATE_1_HOLE_SIZE, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_RKS_STATE_1_ELECTRON_SIZE, abs=FLOAT_TOL)

        assert exciton.hole_size_components_ang is not None
        for i, expected in enumerate(EXPECTED_RKS_STATE_1_HOLE_SIZE_COMPONENTS):
            assert exciton.hole_size_components_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.electron_size_components_ang is not None
        for i, expected in enumerate(EXPECTED_RKS_STATE_1_ELECTRON_SIZE_COMPONENTS):
            assert exciton.electron_size_components_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_state_10_values(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact values for RKS Singlet 10."""
        unrel_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        assert len(unrel_dms) >= 10

        dm = unrel_dms[9]  # 0-indexed
        assert dm.state_number == EXPECTED_RKS_STATE_10_NUM

        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(EXPECTED_RKS_STATE_10_FRONTIER_NOS, abs=FLOAT_TOL)
        assert nos.num_unpaired == pytest.approx(EXPECTED_RKS_STATE_10_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.num_unpaired_nl == pytest.approx(EXPECTED_RKS_STATE_10_NUM_UNPAIRED_NL, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_RKS_STATE_10_PR_NO, abs=FLOAT_TOL)

        for atom_idx, expected_charge in EXPECTED_RKS_STATE_10_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_RKS_STATE_10_DIPOLE_MOMENT, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_RKS_STATE_10_DIPOLE_COMPONENTS):
            assert dm.dipole_components_debye[i] == pytest.approx(expected, abs=FLOAT_TOL)

        exciton = dm.exciton_total
        for i, expected in enumerate(EXPECTED_RKS_STATE_10_R_H):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_RKS_STATE_10_R_E):
            assert exciton.r_e_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_RKS_STATE_10_SEPARATION, abs=FLOAT_TOL)


@pytest.mark.regression
class TestUksUnrelaxedDensityMatrixRegression:
    """Regression tests for UKS unrelaxed density matrix values."""

    def test_state_1_nos(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact NOs values for UKS Excited State 1."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        assert dm.state_number == EXPECTED_UKS_STATE_1_NUM
        assert dm.multiplicity == EXPECTED_UKS_STATE_1_MULTIPLICITY

        # Alpha NOs
        assert dm.nos_alpha is not None
        assert list(dm.nos_alpha.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_STATE_1_FRONTIER_NOS_ALPHA, abs=FLOAT_TOL
        )
        assert dm.nos_alpha.num_electrons == pytest.approx(EXPECTED_UKS_STATE_1_NUM_ELECTRONS_ALPHA, abs=FLOAT_TOL)

        # Beta NOs
        assert dm.nos_beta is not None
        assert list(dm.nos_beta.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_STATE_1_FRONTIER_NOS_BETA, abs=FLOAT_TOL
        )
        assert dm.nos_beta.num_electrons == pytest.approx(EXPECTED_UKS_STATE_1_NUM_ELECTRONS_BETA, abs=FLOAT_TOL)

        # Spin-traced NOs
        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_STATE_1_FRONTIER_NOS_SPIN_TRACED, abs=FLOAT_TOL
        )
        assert nos.num_electrons == pytest.approx(EXPECTED_UKS_STATE_1_NUM_ELECTRONS_SPIN_TRACED, abs=FLOAT_TOL)
        assert nos.num_unpaired == pytest.approx(EXPECTED_UKS_STATE_1_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.num_unpaired_nl == pytest.approx(EXPECTED_UKS_STATE_1_NUM_UNPAIRED_NL, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_UKS_STATE_1_PR_NO, abs=FLOAT_TOL)

    def test_state_1_mulliken(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken charges and spins for UKS Excited State 1."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        for atom_idx, expected_charge in EXPECTED_UKS_STATE_1_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.mulliken.spins is not None
        for atom_idx, expected_spin in EXPECTED_UKS_STATE_1_MULLIKEN_SPINS.items():
            actual_spin = dm.mulliken.spins[atom_idx]
            assert actual_spin == pytest.approx(expected_spin, abs=FLOAT_TOL)

    def test_state_1_multipole(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact multipole moments for UKS Excited State 1."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_UKS_STATE_1_DIPOLE_MOMENT, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_STATE_1_DIPOLE_COMPONENTS):
            assert dm.dipole_components_debye[i] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_state_1_exciton_total(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis (Total) for UKS Excited State 1."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]
        exciton = dm.exciton_total

        for i, expected in enumerate(EXPECTED_UKS_STATE_1_R_H_TOTAL):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_STATE_1_R_E_TOTAL):
            assert exciton.r_e_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_UKS_STATE_1_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_UKS_STATE_1_ELECTRON_SIZE_TOTAL, abs=FLOAT_TOL)

    def test_state_1_exciton_alpha(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis (Alpha) for UKS Excited State 1."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        assert dm.exciton_alpha is not None
        exciton = dm.exciton_alpha

        for i, expected in enumerate(EXPECTED_UKS_STATE_1_R_H_ALPHA):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_SEPARATION_ALPHA, abs=FLOAT_TOL)

    def test_state_1_exciton_beta(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis (Beta) for UKS Excited State 1."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = unrel_dms[0]

        assert dm.exciton_beta is not None
        exciton = dm.exciton_beta

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_SEPARATION_BETA, abs=FLOAT_TOL)

    def test_state_10_values(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact values for UKS Excited State 10."""
        unrel_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        assert len(unrel_dms) >= 10

        dm = unrel_dms[9]  # 0-indexed
        assert dm.state_number == EXPECTED_UKS_STATE_10_NUM

        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_STATE_10_FRONTIER_NOS_SPIN_TRACED, abs=FLOAT_TOL
        )
        assert nos.num_unpaired == pytest.approx(EXPECTED_UKS_STATE_10_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_UKS_STATE_10_PR_NO, abs=FLOAT_TOL)

        for atom_idx, expected_charge in EXPECTED_UKS_STATE_10_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_UKS_STATE_10_DIPOLE_MOMENT, abs=FLOAT_TOL)

        exciton = dm.exciton_total
        for i, expected in enumerate(EXPECTED_UKS_STATE_10_R_H_TOTAL):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_10_SEPARATION_TOTAL, abs=FLOAT_TOL)

    def test_54_state_1_nos(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact NOs values for UKS 5.4 Excited State 1."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 1)

        assert dm.state_number == 1
        assert dm.multiplicity == "Excited State 1"

        # Alpha NOs
        assert dm.nos_alpha is not None
        assert list(dm.nos_alpha.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_1_FRONTIER_NOS_ALPHA, abs=FLOAT_TOL
        )
        assert dm.nos_alpha.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_1_NUM_ELECTRONS_ALPHA, abs=FLOAT_TOL)

        # Beta NOs
        assert dm.nos_beta is not None
        assert list(dm.nos_beta.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_1_FRONTIER_NOS_BETA, abs=FLOAT_TOL
        )
        assert dm.nos_beta.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_1_NUM_ELECTRONS_BETA, abs=FLOAT_TOL)

        # Spin-traced NOs
        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_1_FRONTIER_NOS_SPIN_TRACED, abs=FLOAT_TOL
        )
        assert nos.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_1_NUM_ELECTRONS_SPIN_TRACED, abs=FLOAT_TOL)
        assert nos.num_unpaired == pytest.approx(EXPECTED_UKS_54_STATE_1_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.num_unpaired_nl == pytest.approx(EXPECTED_UKS_54_STATE_1_NUM_UNPAIRED_NL, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_UKS_54_STATE_1_PR_NO, abs=FLOAT_TOL)

    def test_54_state_1_mulliken(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken charges and spins for UKS 5.4 Excited State 1."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 1)

        for atom_idx, expected_charge in EXPECTED_UKS_54_STATE_1_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.mulliken.spins is not None
        for atom_idx, expected_spin in EXPECTED_UKS_54_STATE_1_MULLIKEN_SPINS.items():
            actual_spin = dm.mulliken.spins[atom_idx]
            assert actual_spin == pytest.approx(expected_spin, abs=FLOAT_TOL)

    def test_54_state_1_multipole(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact multipole moments for UKS 5.4 Excited State 1."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 1)

        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_UKS_54_STATE_1_DIPOLE_MOMENT, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_1_DIPOLE_COMPONENTS):
            assert dm.dipole_components_debye[i] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_54_state_1_exciton_total(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis (Total) for UKS 5.4 Excited State 1."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 1)
        exciton = dm.exciton_total

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_1_R_H_TOTAL):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_1_R_E_TOTAL):
            assert exciton.r_e_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_54_STATE_1_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_UKS_54_STATE_1_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_UKS_54_STATE_1_ELECTRON_SIZE_TOTAL, abs=FLOAT_TOL)

    def test_54_state_2_nos(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact NOs values for UKS 5.4 Excited State 2."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 2)

        assert dm.state_number == 2
        assert dm.multiplicity == "Excited State 2"

        # Alpha NOs
        assert dm.nos_alpha is not None
        assert list(dm.nos_alpha.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_2_FRONTIER_NOS_ALPHA, abs=FLOAT_TOL
        )
        assert dm.nos_alpha.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_2_NUM_ELECTRONS_ALPHA, abs=FLOAT_TOL)

        # Beta NOs
        assert dm.nos_beta is not None
        assert list(dm.nos_beta.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_2_FRONTIER_NOS_BETA, abs=FLOAT_TOL
        )
        assert dm.nos_beta.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_2_NUM_ELECTRONS_BETA, abs=FLOAT_TOL)

        # Spin-traced NOs
        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_2_FRONTIER_NOS_SPIN_TRACED, abs=FLOAT_TOL
        )
        assert nos.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_2_NUM_ELECTRONS_SPIN_TRACED, abs=FLOAT_TOL)
        assert nos.num_unpaired == pytest.approx(EXPECTED_UKS_54_STATE_2_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.num_unpaired_nl == pytest.approx(EXPECTED_UKS_54_STATE_2_NUM_UNPAIRED_NL, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_UKS_54_STATE_2_PR_NO, abs=FLOAT_TOL)

    def test_54_state_2_mulliken(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken charges and spins for UKS 5.4 Excited State 2."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 2)

        for atom_idx, expected_charge in EXPECTED_UKS_54_STATE_2_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.mulliken.spins is not None
        for atom_idx, expected_spin in EXPECTED_UKS_54_STATE_2_MULLIKEN_SPINS.items():
            actual_spin = dm.mulliken.spins[atom_idx]
            assert actual_spin == pytest.approx(expected_spin, abs=FLOAT_TOL)

    def test_54_state_2_multipole(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact multipole moments for UKS 5.4 Excited State 2."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 2)

        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_UKS_54_STATE_2_DIPOLE_MOMENT, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_2_DIPOLE_COMPONENTS):
            assert dm.dipole_components_debye[i] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_54_state_2_exciton_total(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis (Total) for UKS 5.4 Excited State 2."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 2)
        exciton = dm.exciton_total

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_2_R_H_TOTAL):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_2_R_E_TOTAL):
            assert exciton.r_e_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_54_STATE_2_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_UKS_54_STATE_2_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_UKS_54_STATE_2_ELECTRON_SIZE_TOTAL, abs=FLOAT_TOL)

    def test_54_state_7_nos(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact NOs values for UKS 5.4 Excited State 7."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 7)

        assert dm.state_number == 7
        assert dm.multiplicity == "Excited State 7"

        # Alpha NOs
        assert dm.nos_alpha is not None
        assert list(dm.nos_alpha.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_7_FRONTIER_NOS_ALPHA, abs=FLOAT_TOL
        )
        assert dm.nos_alpha.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_7_NUM_ELECTRONS_ALPHA, abs=FLOAT_TOL)

        # Beta NOs
        assert dm.nos_beta is not None
        assert list(dm.nos_beta.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_7_FRONTIER_NOS_BETA, abs=FLOAT_TOL
        )
        assert dm.nos_beta.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_7_NUM_ELECTRONS_BETA, abs=FLOAT_TOL)

        # Spin-traced NOs
        nos = dm.nos_spin_traced
        assert list(nos.frontier_occupations) == pytest.approx(
            EXPECTED_UKS_54_STATE_7_FRONTIER_NOS_SPIN_TRACED, abs=FLOAT_TOL
        )
        assert nos.num_electrons == pytest.approx(EXPECTED_UKS_54_STATE_7_NUM_ELECTRONS_SPIN_TRACED, abs=FLOAT_TOL)
        assert nos.num_unpaired == pytest.approx(EXPECTED_UKS_54_STATE_7_NUM_UNPAIRED, abs=FLOAT_TOL)
        assert nos.num_unpaired_nl == pytest.approx(EXPECTED_UKS_54_STATE_7_NUM_UNPAIRED_NL, abs=FLOAT_TOL)
        assert nos.pr_no == pytest.approx(EXPECTED_UKS_54_STATE_7_PR_NO, abs=FLOAT_TOL)

    def test_54_state_7_mulliken(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken charges and spins for UKS 5.4 Excited State 7."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 7)

        for atom_idx, expected_charge in EXPECTED_UKS_54_STATE_7_MULLIKEN_CHARGES.items():
            actual_charge = dm.mulliken.charges[atom_idx]
            assert actual_charge == pytest.approx(expected_charge, abs=FLOAT_TOL)

        assert dm.mulliken.spins is not None
        for atom_idx, expected_spin in EXPECTED_UKS_54_STATE_7_MULLIKEN_SPINS.items():
            actual_spin = dm.mulliken.spins[atom_idx]
            assert actual_spin == pytest.approx(expected_spin, abs=FLOAT_TOL)

    def test_54_state_7_multipole(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact multipole moments for UKS 5.4 Excited State 7."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 7)

        assert dm.dipole_moment_debye == pytest.approx(EXPECTED_UKS_54_STATE_7_DIPOLE_MOMENT, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_7_DIPOLE_COMPONENTS):
            assert dm.dipole_components_debye[i] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_54_state_7_exciton_total(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton analysis (Total) for UKS 5.4 Excited State 7."""
        unrel_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.unrelaxed_density_matrices
        assert unrel_dms is not None
        dm = next(dm for dm in unrel_dms if dm.state_number == 7)
        exciton = dm.exciton_total

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_7_R_H_TOTAL):
            assert exciton.r_h_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        for i, expected in enumerate(EXPECTED_UKS_54_STATE_7_R_E_TOTAL):
            assert exciton.r_e_ang[i] == pytest.approx(expected, abs=FLOAT_TOL)

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_54_STATE_7_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_UKS_54_STATE_7_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_UKS_54_STATE_7_ELECTRON_SIZE_TOTAL, abs=FLOAT_TOL)
