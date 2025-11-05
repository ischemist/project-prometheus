"""
Tests for the QChem transition density matrix parser.

These tests verify that the transition density matrix parser correctly extracts
Mulliken population, CT numbers, and exciton analysis from both RKS and UKS
TDDFT calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure
- regression: exact numerical values match expected

Format notes:
- RKS: "Singlet N :", 4-column Mulliken, single exciton analysis
- UKS: "Excited State N :", 5-column Mulliken, three exciton analyses (Total/Alpha/Beta)
"""

import pytest

from calcflow.common.results import (
    CalculationResult,
    ExcitonAnalysis,
    TddftResults,
    TransitionDensityMatrix,
)
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA (from ex-tran-dm.md)
# =============================================================================

# RKS data - Singlet 1 (parsed_qchem_62_h2o_rks_tddft_data)
EXPECTED_RKS_STATE_1_NUM = 1
EXPECTED_RKS_STATE_1_MULTIPLICITY = "Singlet 1"
EXPECTED_RKS_STATE_1_TRANS_CHARGES = {0: -0.000020, 1: 0.000026, 2: -0.000007}
EXPECTED_RKS_STATE_1_HOLE_POP = {0: 0.016997, 1: 0.967032, 2: 0.017035}
EXPECTED_RKS_STATE_1_ELEC_POP = {0: -0.530519, 1: 0.055849, 2: -0.526394}
EXPECTED_RKS_STATE_1_DEL_Q = {0: -0.513522, 1: 1.022881, 2: -0.509359}
EXPECTED_RKS_STATE_1_QTA = 0.000053
EXPECTED_RKS_STATE_1_QT2 = 0.000000
EXPECTED_RKS_STATE_1_OMEGA = 1.0011
EXPECTED_RKS_STATE_1_TWO_ALPHA_BETA = 1.0011
EXPECTED_RKS_STATE_1_LOC = 0.0010
EXPECTED_RKS_STATE_1_LOCA = 0.1492
EXPECTED_RKS_STATE_1_PHE = -0.0230
# Exciton analysis
EXPECTED_RKS_STATE_1_TRANS_DIPOLE = 1.343588
EXPECTED_RKS_STATE_1_TRANS_DIPOLE_COMPONENTS = (0.221046, 1.295818, -0.277890)
EXPECTED_RKS_STATE_1_TRANS_R2 = 3.050645
EXPECTED_RKS_STATE_1_TRANS_R2_COMPONENTS = (-0.723448, -2.961617, -0.109020)
EXPECTED_RKS_STATE_1_R_H = (2.311727, 1.559441, -0.071529)
EXPECTED_RKS_STATE_1_R_E = (1.957061, 1.488347, -0.684955)
EXPECTED_RKS_STATE_1_SEPARATION = 0.712134
EXPECTED_RKS_STATE_1_HOLE_SIZE = 0.828081
EXPECTED_RKS_STATE_1_HOLE_SIZE_COMPONENTS = (0.390126, 0.616966, 0.390990)
EXPECTED_RKS_STATE_1_ELEC_SIZE = 1.932792
EXPECTED_RKS_STATE_1_ELEC_SIZE_COMPONENTS = (1.229132, 1.031595, 1.077371)
EXPECTED_RKS_STATE_1_RMS_SEPARATION = 2.219123
EXPECTED_RKS_STATE_1_RMS_SEPARATION_COMPONENTS = (1.336700, 1.203917, 1.299354)
EXPECTED_RKS_STATE_1_COVARIANCE = 0.002016
EXPECTED_RKS_STATE_1_CORRELATION = 0.001259
EXPECTED_RKS_STATE_1_COM_SIZE = 1.051836
EXPECTED_RKS_STATE_1_COM_COMPONENTS = (0.645165, 0.601105, 0.573405)

# RKS data - Singlet 5
EXPECTED_RKS_STATE_5_NUM = 5
EXPECTED_RKS_STATE_5_MULTIPLICITY = "Singlet 5"
EXPECTED_RKS_STATE_5_TRANS_CHARGES = {0: -0.126826, 1: 0.000446, 2: 0.126381}
EXPECTED_RKS_STATE_5_HOLE_POP = {0: 0.191894, 1: 0.617362, 2: 0.191775}
EXPECTED_RKS_STATE_5_ELEC_POP = {0: -0.529774, 1: 0.055230, 2: -0.526487}
EXPECTED_RKS_STATE_5_DEL_Q = {0: -0.337880, 1: 0.672591, 2: -0.334712}
EXPECTED_RKS_STATE_5_QTA = 0.253653
EXPECTED_RKS_STATE_5_QT2 = 0.032057
EXPECTED_RKS_STATE_5_OMEGA = 1.0010
EXPECTED_RKS_STATE_5_LOCA = 0.4627
EXPECTED_RKS_STATE_5_TRANS_DIPOLE = 1.932810
EXPECTED_RKS_STATE_5_SEPARATION = 0.550872

# UKS data - Excited State 1 (parsed_qchem_62_h2o_uks_tddft_data)
EXPECTED_UKS_STATE_1_NUM = 1
EXPECTED_UKS_STATE_1_MULTIPLICITY = "Excited State 1"
EXPECTED_UKS_STATE_1_TRANS_CHARGES = {0: 0.000000, 1: -0.000000, 2: 0.000000}
EXPECTED_UKS_STATE_1_HOLE_POP_ALPHA = {0: 0.008520, 1: 0.483752, 2: 0.008538}
EXPECTED_UKS_STATE_1_HOLE_POP_BETA = {0: 0.008520, 1: 0.483752, 2: 0.008538}
EXPECTED_UKS_STATE_1_ELEC_POP_ALPHA = {0: -0.259858, 1: 0.016841, 2: -0.257793}
EXPECTED_UKS_STATE_1_ELEC_POP_BETA = {0: -0.259858, 1: 0.016841, 2: -0.257793}
EXPECTED_UKS_STATE_1_QTA = 0.000000
EXPECTED_UKS_STATE_1_QT2 = 0.000000
EXPECTED_UKS_STATE_1_OMEGA = 1.0016
EXPECTED_UKS_STATE_1_OMEGA_ALPHA = 0.5008
EXPECTED_UKS_STATE_1_OMEGA_BETA = 0.5008
EXPECTED_UKS_STATE_1_TWO_ALPHA_BETA = -1.0016
EXPECTED_UKS_STATE_1_LOC = 0.0014
EXPECTED_UKS_STATE_1_LOC_ALPHA = 0.0007
EXPECTED_UKS_STATE_1_LOC_BETA = 0.0007
EXPECTED_UKS_STATE_1_LOCA = 0.1682
EXPECTED_UKS_STATE_1_LOCA_ALPHA = 0.0841
EXPECTED_UKS_STATE_1_LOCA_BETA = 0.0841
EXPECTED_UKS_STATE_1_PHE = 0.0381
EXPECTED_UKS_STATE_1_PHE_ALPHA = 0.0381
EXPECTED_UKS_STATE_1_PHE_BETA = 0.0381
# Exciton Total
EXPECTED_UKS_STATE_1_TRANS_DIPOLE_TOTAL = 0.000000
EXPECTED_UKS_STATE_1_TRANS_DIPOLE_COMPONENTS_TOTAL = (-0.000000, -0.000000, 0.000000)
EXPECTED_UKS_STATE_1_R_H_TOTAL = (2.311634, 1.559422, -0.071689)
EXPECTED_UKS_STATE_1_R_E_TOTAL = (1.975509, 1.492105, -0.652667)
EXPECTED_UKS_STATE_1_SEPARATION_TOTAL = 0.674572
EXPECTED_UKS_STATE_1_HOLE_SIZE_TOTAL = 0.828288
EXPECTED_UKS_STATE_1_ELEC_SIZE_TOTAL = 1.861664
EXPECTED_UKS_STATE_1_RMS_SEPARATION_TOTAL = 2.141228
EXPECTED_UKS_STATE_1_COVARIANCE_TOTAL = 0.011024
EXPECTED_UKS_STATE_1_CORRELATION_TOTAL = 0.007149
EXPECTED_UKS_STATE_1_COM_SIZE_TOTAL = 1.021507
# Exciton Alpha
EXPECTED_UKS_STATE_1_TRANS_DIPOLE_ALPHA = 0.643948
EXPECTED_UKS_STATE_1_TRANS_DIPOLE_COMPONENTS_ALPHA = (-0.106069, -0.621037, 0.133160)
EXPECTED_UKS_STATE_1_SEPARATION_ALPHA = 0.674572
EXPECTED_UKS_STATE_1_RMS_SEPARATION_ALPHA = 2.141228
EXPECTED_UKS_STATE_1_COVARIANCE_ALPHA = 0.011024
# Exciton Beta
EXPECTED_UKS_STATE_1_TRANS_DIPOLE_BETA = 0.643948
EXPECTED_UKS_STATE_1_TRANS_DIPOLE_COMPONENTS_BETA = (0.106069, 0.621037, -0.133160)
EXPECTED_UKS_STATE_1_SEPARATION_BETA = 0.674572

# UKS data - Excited State 5
EXPECTED_UKS_STATE_5_NUM = 5
EXPECTED_UKS_STATE_5_MULTIPLICITY = "Excited State 5"
EXPECTED_UKS_STATE_5_TRANS_CHARGES = {0: -0.000025, 1: 0.000017, 2: 0.000008}
EXPECTED_UKS_STATE_5_HOLE_POP_ALPHA = {0: 0.008449, 1: 0.483229, 2: 0.008468}
EXPECTED_UKS_STATE_5_ELEC_POP_ALPHA = {0: -0.275143, 1: 0.052629, 2: -0.277632}
EXPECTED_UKS_STATE_5_QTA = 0.000049
EXPECTED_UKS_STATE_5_OMEGA = 1.0003
EXPECTED_UKS_STATE_5_OMEGA_ALPHA = 0.5001
EXPECTED_UKS_STATE_5_TWO_ALPHA_BETA = 1.0003
EXPECTED_UKS_STATE_5_TRANS_DIPOLE_TOTAL = 0.003619
EXPECTED_UKS_STATE_5_SEPARATION_TOTAL = 0.764045
EXPECTED_UKS_STATE_5_SEPARATION_ALPHA = 0.764045

# QChem 5.4 UKS TDDFT data - Excited State 2 (from ex-tran-dm.md)
EXPECTED_QC54_STATE_2_NUM = 2
EXPECTED_QC54_STATE_2_MULTIPLICITY = "Excited State 2"
EXPECTED_QC54_STATE_2_OMEGA = 1.0011
EXPECTED_QC54_STATE_2_OMEGA_ALPHA = 0.5005
EXPECTED_QC54_STATE_2_OMEGA_BETA = 0.5005
EXPECTED_QC54_STATE_2_PHE = -0.0230
EXPECTED_QC54_STATE_2_PHE_ALPHA = -0.0230
EXPECTED_QC54_STATE_2_PHE_BETA = -0.0230
# Exciton Total
EXPECTED_QC54_STATE_2_TRANS_DIPOLE_TOTAL = 1.343589
EXPECTED_QC54_STATE_2_TRANS_DIPOLE_COMPONENTS_TOTAL = (0.221046, 1.295819, -0.277890)
EXPECTED_QC54_STATE_2_R_H_TOTAL = (2.311727, 1.559441, -0.071529)
EXPECTED_QC54_STATE_2_R_E_TOTAL = (1.957062, 1.488347, -0.684953)
EXPECTED_QC54_STATE_2_SEPARATION_TOTAL = 0.712131
EXPECTED_QC54_STATE_2_HOLE_SIZE_TOTAL = 0.828081
EXPECTED_QC54_STATE_2_HOLE_SIZE_COMPONENTS_TOTAL = (0.390126, 0.616966, 0.390990)
EXPECTED_QC54_STATE_2_ELEC_SIZE_TOTAL = 1.932792
EXPECTED_QC54_STATE_2_ELEC_SIZE_COMPONENTS_TOTAL = (1.229133, 1.031595, 1.077372)
EXPECTED_QC54_STATE_2_RMS_SEPARATION_TOTAL = 2.219123
EXPECTED_QC54_STATE_2_RMS_SEPARATION_COMPONENTS_TOTAL = (1.336700, 1.203917, 1.299354)
EXPECTED_QC54_STATE_2_COVARIANCE_TOTAL = 0.002016
EXPECTED_QC54_STATE_2_CORRELATION_TOTAL = 0.001259
EXPECTED_QC54_STATE_2_COM_SIZE_TOTAL = 1.051836
EXPECTED_QC54_STATE_2_COM_COMPONENTS_TOTAL = (0.645165, 0.601105, 0.573406)
# Exciton Alpha
EXPECTED_QC54_STATE_2_TRANS_DIPOLE_ALPHA = 0.671794
EXPECTED_QC54_STATE_2_TRANS_DIPOLE_COMPONENTS_ALPHA = (0.110523, 0.647909, -0.138945)
EXPECTED_QC54_STATE_2_SEPARATION_ALPHA = 0.712131
EXPECTED_QC54_STATE_2_COVARIANCE_ALPHA = 0.002016
EXPECTED_QC54_STATE_2_CORRELATION_ALPHA = 0.001259
# Exciton Beta
EXPECTED_QC54_STATE_2_TRANS_DIPOLE_BETA = 0.671794
EXPECTED_QC54_STATE_2_TRANS_DIPOLE_COMPONENTS_BETA = (0.110523, 0.647909, -0.138945)
EXPECTED_QC54_STATE_2_SEPARATION_BETA = 0.712131

# QChem 5.4 UKS TDDFT data - Excited State 7 (from ex-tran-dm.md)
EXPECTED_QC54_STATE_7_NUM = 7
EXPECTED_QC54_STATE_7_MULTIPLICITY = "Excited State 7"
EXPECTED_QC54_STATE_7_OMEGA = 1.0027
EXPECTED_QC54_STATE_7_OMEGA_ALPHA = 0.5014
EXPECTED_QC54_STATE_7_OMEGA_BETA = 0.5014
EXPECTED_QC54_STATE_7_PHE = 0.0432
EXPECTED_QC54_STATE_7_PHE_ALPHA = 0.0432
EXPECTED_QC54_STATE_7_PHE_BETA = 0.0432
# Exciton Total
EXPECTED_QC54_STATE_7_TRANS_DIPOLE_TOTAL = 0.000000
EXPECTED_QC54_STATE_7_TRANS_DIPOLE_COMPONENTS_TOTAL = (-0.000000, 0.000000, 0.000000)
EXPECTED_QC54_STATE_7_R_H_TOTAL = (2.349125, 1.567172, -0.005700)
EXPECTED_QC54_STATE_7_R_E_TOTAL = (1.951527, 1.483352, -0.712621)
EXPECTED_QC54_STATE_7_SEPARATION_TOTAL = 0.815381
EXPECTED_QC54_STATE_7_HOLE_SIZE_TOTAL = 0.861474
EXPECTED_QC54_STATE_7_HOLE_SIZE_COMPONENTS_TOTAL = (0.498740, 0.374148, 0.594482)
EXPECTED_QC54_STATE_7_ELEC_SIZE_TOTAL = 2.078147
EXPECTED_QC54_STATE_7_ELEC_SIZE_COMPONENTS_TOTAL = (1.498453, 0.937763, 1.092674)
EXPECTED_QC54_STATE_7_RMS_SEPARATION_TOTAL = 2.377245
EXPECTED_QC54_STATE_7_RMS_SEPARATION_COMPONENTS_TOTAL = (1.613486, 1.011235, 1.423152)
EXPECTED_QC54_STATE_7_COVARIANCE_TOTAL = 0.037191
EXPECTED_QC54_STATE_7_CORRELATION_TOTAL = 0.020774
EXPECTED_QC54_STATE_7_COM_SIZE_TOTAL = 1.133051
EXPECTED_QC54_STATE_7_COM_COMPONENTS_TOTAL = (0.797332, 0.505767, 0.626311)
# Exciton Alpha
EXPECTED_QC54_STATE_7_TRANS_DIPOLE_ALPHA = 1.095767
EXPECTED_QC54_STATE_7_TRANS_DIPOLE_COMPONENTS_ALPHA = (-0.935082, 0.267658, 0.504665)
EXPECTED_QC54_STATE_7_SEPARATION_ALPHA = 0.815381
EXPECTED_QC54_STATE_7_COVARIANCE_ALPHA = 0.037191
EXPECTED_QC54_STATE_7_CORRELATION_ALPHA = 0.020774
# Exciton Beta
EXPECTED_QC54_STATE_7_TRANS_DIPOLE_BETA = 1.095767
EXPECTED_QC54_STATE_7_TRANS_DIPOLE_COMPONENTS_BETA = (0.935082, -0.267658, -0.504665)
EXPECTED_QC54_STATE_7_SEPARATION_BETA = 0.815381

# Numerical tolerance
FLOAT_TOL = 1e-6


# =============================================================================
# UNIT TESTS: Matches behavior
# =============================================================================


@pytest.mark.unit
def test_trans_dm_parser_exists():
    """Unit test: verify parser can be imported."""
    from calcflow.io.qchem.blocks.tddft.trans_dm import TransitionDensityMatrixParser

    parser = TransitionDensityMatrixParser()
    assert parser is not None


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_trans_dm"])
def test_trans_dm_exists(fixture_name: str, request):
    """Contract test: transition density matrices should exist in TDDFT results."""
    data = request.getfixturevalue(fixture_name)
    assert data.tddft is not None
    assert isinstance(data.tddft, TddftResults)
    assert data.tddft.transition_density_matrices is not None
    assert len(data.tddft.transition_density_matrices) > 0


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_trans_dm"])
def test_trans_dm_structure(fixture_name: str, request):
    """Contract test: each transition DM should have correct structure."""
    data = request.getfixturevalue(fixture_name)
    trans_dms = data.tddft.transition_density_matrices
    assert trans_dms is not None

    for dm in trans_dms:
        assert isinstance(dm, TransitionDensityMatrix)
        assert isinstance(dm.state_number, int)
        assert dm.state_number > 0
        assert isinstance(dm.multiplicity, str)
        assert isinstance(dm.exciton_total, ExcitonAnalysis)


@pytest.mark.contract
def test_rks_has_no_alpha_beta_exciton(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
    """Contract test: RKS should not have alpha/beta exciton analysis."""
    trans_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.transition_density_matrices
    assert trans_dms is not None

    for dm in trans_dms:
        assert dm.exciton_alpha is None, "RKS should not have alpha exciton"
        assert dm.exciton_beta is None, "RKS should not have beta exciton"


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_unrel_dm"])
def test_uks_has_alpha_beta_exciton(fixture_name: str, request):
    """Contract test: UKS should have alpha/beta exciton analysis."""
    data = request.getfixturevalue(fixture_name)
    trans_dms = data.tddft.transition_density_matrices
    assert trans_dms is not None

    for dm in trans_dms:
        assert dm.exciton_alpha is not None, "UKS should have alpha exciton"
        assert dm.exciton_beta is not None, "UKS should have beta exciton"
        assert isinstance(dm.exciton_alpha, ExcitonAnalysis)
        assert isinstance(dm.exciton_beta, ExcitonAnalysis)


@pytest.mark.contract
def test_mulliken_structure_rks(parsed_qchem_62_h2o_rks_tddft_data: CalculationResult):
    """Contract test: mulliken should have transition DM specific fields for RKS."""
    trans_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.transition_density_matrices
    assert trans_dms is not None
    dm = trans_dms[0]

    assert dm.mulliken is not None
    assert dm.mulliken.charges is not None
    assert dm.mulliken.trans_charges is not None
    assert dm.mulliken.hole_populations is not None
    assert dm.mulliken.electron_populations is not None
    assert dm.mulliken.del_q is not None


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_trans_dm"])
def test_exciton_has_transition_fields(fixture_name: str, request):
    """Contract test: exciton should have transition-specific fields."""
    data = request.getfixturevalue(fixture_name)
    trans_dms = data.tddft.transition_density_matrices
    assert trans_dms is not None
    dm = trans_dms[0]
    exciton = dm.exciton_total

    # Check for transition-specific fields
    assert exciton.rms_separation_ang is not None
    assert exciton.covariance is not None
    assert exciton.correlation_coef is not None
    assert exciton.center_of_mass_size_ang is not None


@pytest.mark.contract
@pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_unrel_dm"])
def test_uks_mulliken_optional_but_exciton_present(fixture_name: str, request):
    """Contract test: UKS may have None mulliken, but exciton analysis is always present."""
    data = request.getfixturevalue(fixture_name)
    trans_dms = data.tddft.transition_density_matrices
    assert trans_dms is not None

    for dm in trans_dms:
        # For UKS, mulliken may be None (not parsed), but exciton data is always present
        assert dm.exciton_alpha is not None
        assert dm.exciton_beta is not None
        assert isinstance(dm.exciton_alpha, ExcitonAnalysis)
        assert isinstance(dm.exciton_beta, ExcitonAnalysis)


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
class TestRksTransitionDensityMatrixRegression:
    """Regression tests for RKS transition density matrix values."""

    def test_state_1_mulliken(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken values for RKS Singlet 1."""
        trans_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]

        assert dm.state_number == EXPECTED_RKS_STATE_1_NUM
        assert dm.multiplicity == EXPECTED_RKS_STATE_1_MULTIPLICITY

        for atom_idx, expected in EXPECTED_RKS_STATE_1_TRANS_CHARGES.items():
            actual = dm.mulliken.trans_charges[atom_idx]
            assert actual == pytest.approx(expected, abs=FLOAT_TOL)

        for atom_idx, expected in EXPECTED_RKS_STATE_1_HOLE_POP.items():
            actual = dm.mulliken.hole_populations[atom_idx]
            assert actual == pytest.approx(expected, abs=FLOAT_TOL)

        for atom_idx, expected in EXPECTED_RKS_STATE_1_ELEC_POP.items():
            actual = dm.mulliken.electron_populations[atom_idx]
            assert actual == pytest.approx(expected, abs=FLOAT_TOL)

        for atom_idx, expected in EXPECTED_RKS_STATE_1_DEL_Q.items():
            actual = dm.mulliken.del_q[atom_idx]
            assert actual == pytest.approx(expected, abs=FLOAT_TOL)

    def test_state_1_ct_numbers(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact CT numbers for RKS Singlet 1."""
        trans_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]

        assert dm.sum_abs_trans_charges == pytest.approx(EXPECTED_RKS_STATE_1_QTA, abs=FLOAT_TOL)
        assert dm.sum_squared_trans_charges == pytest.approx(EXPECTED_RKS_STATE_1_QT2, abs=FLOAT_TOL)
        assert dm.omega == pytest.approx(EXPECTED_RKS_STATE_1_OMEGA, abs=FLOAT_TOL)
        assert dm.two_alpha_beta == pytest.approx(EXPECTED_RKS_STATE_1_TWO_ALPHA_BETA, abs=FLOAT_TOL)
        assert dm.loc == pytest.approx(EXPECTED_RKS_STATE_1_LOC, abs=FLOAT_TOL)
        assert dm.loca == pytest.approx(EXPECTED_RKS_STATE_1_LOCA, abs=FLOAT_TOL)
        assert dm.phe == pytest.approx(EXPECTED_RKS_STATE_1_PHE, abs=FLOAT_TOL)

    def test_state_1_exciton(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton values for RKS Singlet 1."""
        trans_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]
        exciton = dm.exciton_total

        assert exciton.r_h_ang[0] == pytest.approx(EXPECTED_RKS_STATE_1_R_H[0], abs=FLOAT_TOL)
        assert exciton.r_e_ang[0] == pytest.approx(EXPECTED_RKS_STATE_1_R_E[0], abs=FLOAT_TOL)
        assert exciton.separation_ang == pytest.approx(EXPECTED_RKS_STATE_1_SEPARATION, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_RKS_STATE_1_HOLE_SIZE, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_RKS_STATE_1_ELEC_SIZE, abs=FLOAT_TOL)
        assert exciton.rms_separation_ang == pytest.approx(EXPECTED_RKS_STATE_1_RMS_SEPARATION, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_RKS_STATE_1_COVARIANCE, abs=FLOAT_TOL)
        assert exciton.correlation_coef == pytest.approx(EXPECTED_RKS_STATE_1_CORRELATION, abs=FLOAT_TOL)
        assert exciton.center_of_mass_size_ang == pytest.approx(EXPECTED_RKS_STATE_1_COM_SIZE, abs=FLOAT_TOL)

    def test_state_5_values(self, parsed_qchem_62_h2o_rks_tddft_data: CalculationResult) -> None:
        """Verify exact values for RKS Singlet 5."""
        trans_dms = parsed_qchem_62_h2o_rks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        assert len(trans_dms) >= 5

        dm = trans_dms[4]  # 0-indexed
        assert dm.state_number == EXPECTED_RKS_STATE_5_NUM
        assert dm.multiplicity == EXPECTED_RKS_STATE_5_MULTIPLICITY

        assert dm.sum_abs_trans_charges == pytest.approx(EXPECTED_RKS_STATE_5_QTA, abs=FLOAT_TOL)
        assert dm.omega == pytest.approx(EXPECTED_RKS_STATE_5_OMEGA, abs=FLOAT_TOL)
        assert dm.loca == pytest.approx(EXPECTED_RKS_STATE_5_LOCA, abs=FLOAT_TOL)
        assert dm.exciton_total.separation_ang == pytest.approx(EXPECTED_RKS_STATE_5_SEPARATION, abs=FLOAT_TOL)


@pytest.mark.regression
class TestUksTransitionDensityMatrixRegression:
    """Regression tests for UKS transition density matrix values."""

    def test_state_1_mulliken(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact Mulliken values for UKS Excited State 1."""
        trans_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]

        assert dm.state_number == EXPECTED_UKS_STATE_1_NUM
        assert dm.multiplicity == EXPECTED_UKS_STATE_1_MULTIPLICITY

        for atom_idx, expected in EXPECTED_UKS_STATE_1_TRANS_CHARGES.items():
            actual = dm.mulliken.trans_charges[atom_idx]
            assert actual == pytest.approx(expected, abs=FLOAT_TOL)

    def test_state_1_ct_numbers(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact CT numbers for UKS Excited State 1."""
        trans_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]

        assert dm.sum_abs_trans_charges == pytest.approx(EXPECTED_UKS_STATE_1_QTA, abs=FLOAT_TOL)
        assert dm.omega == pytest.approx(EXPECTED_UKS_STATE_1_OMEGA, abs=FLOAT_TOL)
        assert dm.omega_alpha == pytest.approx(EXPECTED_UKS_STATE_1_OMEGA_ALPHA, abs=FLOAT_TOL)
        assert dm.omega_beta == pytest.approx(EXPECTED_UKS_STATE_1_OMEGA_BETA, abs=FLOAT_TOL)
        assert dm.two_alpha_beta == pytest.approx(EXPECTED_UKS_STATE_1_TWO_ALPHA_BETA, abs=FLOAT_TOL)
        assert dm.loc == pytest.approx(EXPECTED_UKS_STATE_1_LOC, abs=FLOAT_TOL)
        assert dm.loc_alpha == pytest.approx(EXPECTED_UKS_STATE_1_LOC_ALPHA, abs=FLOAT_TOL)
        assert dm.loc_beta == pytest.approx(EXPECTED_UKS_STATE_1_LOC_BETA, abs=FLOAT_TOL)
        assert dm.loca == pytest.approx(EXPECTED_UKS_STATE_1_LOCA, abs=FLOAT_TOL)
        assert dm.loca_alpha == pytest.approx(EXPECTED_UKS_STATE_1_LOCA_ALPHA, abs=FLOAT_TOL)
        assert dm.loca_beta == pytest.approx(EXPECTED_UKS_STATE_1_LOCA_BETA, abs=FLOAT_TOL)
        assert dm.phe == pytest.approx(EXPECTED_UKS_STATE_1_PHE, abs=FLOAT_TOL)
        assert dm.phe_alpha == pytest.approx(EXPECTED_UKS_STATE_1_PHE_ALPHA, abs=FLOAT_TOL)
        assert dm.phe_beta == pytest.approx(EXPECTED_UKS_STATE_1_PHE_BETA, abs=FLOAT_TOL)

    def test_state_1_exciton_total(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Total) for UKS Excited State 1."""
        trans_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]
        exciton = dm.exciton_total

        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_UKS_STATE_1_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_UKS_STATE_1_ELEC_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.rms_separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_RMS_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_UKS_STATE_1_COVARIANCE_TOTAL, abs=FLOAT_TOL)
        assert exciton.correlation_coef == pytest.approx(EXPECTED_UKS_STATE_1_CORRELATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.center_of_mass_size_ang == pytest.approx(EXPECTED_UKS_STATE_1_COM_SIZE_TOTAL, abs=FLOAT_TOL)

    def test_state_1_exciton_alpha(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Alpha) for UKS Excited State 1."""
        trans_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]

        assert dm.exciton_alpha is not None
        exciton = dm.exciton_alpha
        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_SEPARATION_ALPHA, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_UKS_STATE_1_COVARIANCE_ALPHA, abs=FLOAT_TOL)

    def test_state_1_exciton_beta(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Beta) for UKS Excited State 1."""
        trans_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[0]

        assert dm.exciton_beta is not None
        exciton = dm.exciton_beta
        assert exciton.separation_ang == pytest.approx(EXPECTED_UKS_STATE_1_SEPARATION_BETA, abs=FLOAT_TOL)

    def test_state_5_values(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact values for UKS Excited State 5."""
        trans_dms = parsed_qchem_62_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        assert len(trans_dms) >= 5

        dm = trans_dms[4]  # 0-indexed
        assert dm.state_number == EXPECTED_UKS_STATE_5_NUM
        assert dm.sum_abs_trans_charges == pytest.approx(EXPECTED_UKS_STATE_5_QTA, abs=FLOAT_TOL)
        assert dm.omega == pytest.approx(EXPECTED_UKS_STATE_5_OMEGA, abs=FLOAT_TOL)
        assert dm.exciton_total.separation_ang == pytest.approx(EXPECTED_UKS_STATE_5_SEPARATION_TOTAL, abs=FLOAT_TOL)


@pytest.mark.regression
class TestQc54UksTransitionDensityMatrixRegression:
    """Regression tests for QChem 5.4 UKS transition density matrix values."""

    def test_state_2_ct_numbers(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact CT numbers for QChem 5.4 Excited State 2."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        assert len(trans_dms) >= 2

        dm = trans_dms[1]  # 0-indexed, State 2 is at index 1
        assert dm.state_number == EXPECTED_QC54_STATE_2_NUM
        assert dm.multiplicity == EXPECTED_QC54_STATE_2_MULTIPLICITY
        assert dm.omega == pytest.approx(EXPECTED_QC54_STATE_2_OMEGA, abs=FLOAT_TOL)
        assert dm.omega_alpha == pytest.approx(EXPECTED_QC54_STATE_2_OMEGA_ALPHA, abs=FLOAT_TOL)
        assert dm.omega_beta == pytest.approx(EXPECTED_QC54_STATE_2_OMEGA_BETA, abs=FLOAT_TOL)
        assert dm.phe == pytest.approx(EXPECTED_QC54_STATE_2_PHE, abs=FLOAT_TOL)
        assert dm.phe_alpha == pytest.approx(EXPECTED_QC54_STATE_2_PHE_ALPHA, abs=FLOAT_TOL)
        assert dm.phe_beta == pytest.approx(EXPECTED_QC54_STATE_2_PHE_BETA, abs=FLOAT_TOL)

    def test_state_2_exciton_total(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Total) for QChem 5.4 Excited State 2."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[1]
        exciton = dm.exciton_total

        assert exciton.trans_dipole_moment_debye == pytest.approx(
            EXPECTED_QC54_STATE_2_TRANS_DIPOLE_TOTAL, abs=FLOAT_TOL
        )
        assert exciton.trans_dipole_moment_components_debye[0] == pytest.approx(
            EXPECTED_QC54_STATE_2_TRANS_DIPOLE_COMPONENTS_TOTAL[0], abs=FLOAT_TOL
        )
        assert exciton.trans_dipole_moment_components_debye[1] == pytest.approx(
            EXPECTED_QC54_STATE_2_TRANS_DIPOLE_COMPONENTS_TOTAL[1], abs=FLOAT_TOL
        )
        assert exciton.trans_dipole_moment_components_debye[2] == pytest.approx(
            EXPECTED_QC54_STATE_2_TRANS_DIPOLE_COMPONENTS_TOTAL[2], abs=FLOAT_TOL
        )
        assert exciton.r_h_ang[0] == pytest.approx(EXPECTED_QC54_STATE_2_R_H_TOTAL[0], abs=FLOAT_TOL)
        assert exciton.r_h_ang[1] == pytest.approx(EXPECTED_QC54_STATE_2_R_H_TOTAL[1], abs=FLOAT_TOL)
        assert exciton.r_h_ang[2] == pytest.approx(EXPECTED_QC54_STATE_2_R_H_TOTAL[2], abs=FLOAT_TOL)
        assert exciton.r_e_ang[0] == pytest.approx(EXPECTED_QC54_STATE_2_R_E_TOTAL[0], abs=FLOAT_TOL)
        assert exciton.r_e_ang[1] == pytest.approx(EXPECTED_QC54_STATE_2_R_E_TOTAL[1], abs=FLOAT_TOL)
        assert exciton.r_e_ang[2] == pytest.approx(EXPECTED_QC54_STATE_2_R_E_TOTAL[2], abs=FLOAT_TOL)
        assert exciton.separation_ang == pytest.approx(EXPECTED_QC54_STATE_2_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_QC54_STATE_2_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_QC54_STATE_2_ELEC_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.rms_separation_ang == pytest.approx(EXPECTED_QC54_STATE_2_RMS_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_QC54_STATE_2_COVARIANCE_TOTAL, abs=FLOAT_TOL)
        assert exciton.correlation_coef == pytest.approx(EXPECTED_QC54_STATE_2_CORRELATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.center_of_mass_size_ang == pytest.approx(EXPECTED_QC54_STATE_2_COM_SIZE_TOTAL, abs=FLOAT_TOL)

    def test_state_2_exciton_alpha(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Alpha) for QChem 5.4 Excited State 2."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[1]
        exciton = dm.exciton_alpha

        assert exciton is not None
        assert exciton.trans_dipole_moment_debye == pytest.approx(
            EXPECTED_QC54_STATE_2_TRANS_DIPOLE_ALPHA, abs=FLOAT_TOL
        )
        assert exciton.separation_ang == pytest.approx(EXPECTED_QC54_STATE_2_SEPARATION_ALPHA, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_QC54_STATE_2_COVARIANCE_ALPHA, abs=FLOAT_TOL)
        assert exciton.correlation_coef == pytest.approx(EXPECTED_QC54_STATE_2_CORRELATION_ALPHA, abs=FLOAT_TOL)

    def test_state_2_exciton_beta(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Beta) for QChem 5.4 Excited State 2."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[1]
        exciton = dm.exciton_beta

        assert exciton is not None
        assert exciton.trans_dipole_moment_debye == pytest.approx(
            EXPECTED_QC54_STATE_2_TRANS_DIPOLE_BETA, abs=FLOAT_TOL
        )
        assert exciton.separation_ang == pytest.approx(EXPECTED_QC54_STATE_2_SEPARATION_BETA, abs=FLOAT_TOL)

    def test_state_7_ct_numbers(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact CT numbers for QChem 5.4 Excited State 7."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        assert len(trans_dms) >= 7

        dm = trans_dms[6]  # 0-indexed, State 7 is at index 6
        assert dm.state_number == EXPECTED_QC54_STATE_7_NUM
        assert dm.multiplicity == EXPECTED_QC54_STATE_7_MULTIPLICITY
        assert dm.omega == pytest.approx(EXPECTED_QC54_STATE_7_OMEGA, abs=FLOAT_TOL)
        assert dm.omega_alpha == pytest.approx(EXPECTED_QC54_STATE_7_OMEGA_ALPHA, abs=FLOAT_TOL)
        assert dm.omega_beta == pytest.approx(EXPECTED_QC54_STATE_7_OMEGA_BETA, abs=FLOAT_TOL)
        assert dm.phe == pytest.approx(EXPECTED_QC54_STATE_7_PHE, abs=FLOAT_TOL)
        assert dm.phe_alpha == pytest.approx(EXPECTED_QC54_STATE_7_PHE_ALPHA, abs=FLOAT_TOL)
        assert dm.phe_beta == pytest.approx(EXPECTED_QC54_STATE_7_PHE_BETA, abs=FLOAT_TOL)

    def test_state_7_exciton_total(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Total) for QChem 5.4 Excited State 7."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[6]
        exciton = dm.exciton_total

        assert exciton.trans_dipole_moment_debye == pytest.approx(
            EXPECTED_QC54_STATE_7_TRANS_DIPOLE_TOTAL, abs=FLOAT_TOL
        )
        assert exciton.r_h_ang[0] == pytest.approx(EXPECTED_QC54_STATE_7_R_H_TOTAL[0], abs=FLOAT_TOL)
        assert exciton.r_h_ang[1] == pytest.approx(EXPECTED_QC54_STATE_7_R_H_TOTAL[1], abs=FLOAT_TOL)
        assert exciton.r_h_ang[2] == pytest.approx(EXPECTED_QC54_STATE_7_R_H_TOTAL[2], abs=FLOAT_TOL)
        assert exciton.r_e_ang[0] == pytest.approx(EXPECTED_QC54_STATE_7_R_E_TOTAL[0], abs=FLOAT_TOL)
        assert exciton.r_e_ang[1] == pytest.approx(EXPECTED_QC54_STATE_7_R_E_TOTAL[1], abs=FLOAT_TOL)
        assert exciton.r_e_ang[2] == pytest.approx(EXPECTED_QC54_STATE_7_R_E_TOTAL[2], abs=FLOAT_TOL)
        assert exciton.separation_ang == pytest.approx(EXPECTED_QC54_STATE_7_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.hole_size_ang == pytest.approx(EXPECTED_QC54_STATE_7_HOLE_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.electron_size_ang == pytest.approx(EXPECTED_QC54_STATE_7_ELEC_SIZE_TOTAL, abs=FLOAT_TOL)
        assert exciton.rms_separation_ang == pytest.approx(EXPECTED_QC54_STATE_7_RMS_SEPARATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_QC54_STATE_7_COVARIANCE_TOTAL, abs=FLOAT_TOL)
        assert exciton.correlation_coef == pytest.approx(EXPECTED_QC54_STATE_7_CORRELATION_TOTAL, abs=FLOAT_TOL)
        assert exciton.center_of_mass_size_ang == pytest.approx(EXPECTED_QC54_STATE_7_COM_SIZE_TOTAL, abs=FLOAT_TOL)

    def test_state_7_exciton_alpha(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Alpha) for QChem 5.4 Excited State 7."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[6]
        exciton = dm.exciton_alpha

        assert exciton is not None
        assert exciton.trans_dipole_moment_debye == pytest.approx(
            EXPECTED_QC54_STATE_7_TRANS_DIPOLE_ALPHA, abs=FLOAT_TOL
        )
        assert exciton.separation_ang == pytest.approx(EXPECTED_QC54_STATE_7_SEPARATION_ALPHA, abs=FLOAT_TOL)
        assert exciton.covariance == pytest.approx(EXPECTED_QC54_STATE_7_COVARIANCE_ALPHA, abs=FLOAT_TOL)
        assert exciton.correlation_coef == pytest.approx(EXPECTED_QC54_STATE_7_CORRELATION_ALPHA, abs=FLOAT_TOL)

    def test_state_7_exciton_beta(self, parsed_qchem_54_h2o_uks_tddft_data: CalculationResult) -> None:
        """Verify exact exciton (Beta) for QChem 5.4 Excited State 7."""
        trans_dms = parsed_qchem_54_h2o_uks_tddft_data.tddft.transition_density_matrices
        assert trans_dms is not None
        dm = trans_dms[6]
        exciton = dm.exciton_beta

        assert exciton is not None
        assert exciton.trans_dipole_moment_debye == pytest.approx(
            EXPECTED_QC54_STATE_7_TRANS_DIPOLE_BETA, abs=FLOAT_TOL
        )
        assert exciton.separation_ang == pytest.approx(EXPECTED_QC54_STATE_7_SEPARATION_BETA, abs=FLOAT_TOL)
