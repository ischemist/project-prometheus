"""
Regression tests for Q-Chem XAS (X-ray Absorption Spectroscopy) excitations.

These tests verify that the TDDFT excitations parser correctly extracts
excited state properties from XAS calculations with core orbital excitations.

The test file uses 6.2-mom-xas-smd.out which contains:
- UKS XAS calculation using src1-r1 functional
- MOM (Maximum Overlap Method) for core hole excitations
- 10 excited states computed
- SMD solvation model (water)

Format notes:
- XAS: Core orbital excitations with modified orbital indexing
- TRTYPE=3: Core-excited states with relaxed density
- Includes unrelaxed density matrix and transition density matrix analysis
"""

import pytest

from calcflow.common.results import (
    CalculationResult,
    ExcitedState,
    TddftResults,
    TransitionDensityMatrix,
    UnrelaxedDensityMatrix,
)

# =============================================================================
# HARDCODED REGRESSION TEST DATA
# Extracted from: tests/testing_data/qchem/h2o/6.2-mom-xas-smd.out
# =============================================================================

# State 2 - XAS Excitation Energy
EXPECTED_XAS_STATE_2_EXCITATION_EV = 535.0965
EXPECTED_XAS_STATE_2_TOTAL_ENERGY_AU = -56.87238905
EXPECTED_XAS_STATE_2_STRENGTH = 0.0  # No transition strength for triplet
EXPECTED_XAS_STATE_2_TRANS_MOM_X = 0.0000
EXPECTED_XAS_STATE_2_TRANS_MOM_Y = 0.0000
EXPECTED_XAS_STATE_2_TRANS_MOM_Z = 0.0000

# State 2 - Unrelaxed Density Matrix
EXPECTED_XAS_STATE_2_UNREL_DIPOLE_MOMENT = 0.218622
EXPECTED_XAS_STATE_2_UNREL_DIPOLE_X = 0.061744
EXPECTED_XAS_STATE_2_UNREL_DIPOLE_Y = 0.033967
EXPECTED_XAS_STATE_2_UNREL_DIPOLE_Z = 0.206953
EXPECTED_XAS_STATE_2_UNREL_HOLE_SIZE = 0.122441  # Core hole size
EXPECTED_XAS_STATE_2_UNREL_ELECTRON_SIZE = 1.668019
EXPECTED_XAS_STATE_2_UNREL_R_H_X = 2.328688  # Hole position (oxygen)
EXPECTED_XAS_STATE_2_UNREL_R_H_Y = 1.562923
EXPECTED_XAS_STATE_2_UNREL_R_H_Z = -0.041824
EXPECTED_XAS_STATE_2_UNREL_R_E_X = 2.060655  # Electron position
EXPECTED_XAS_STATE_2_UNREL_R_E_Y = 1.503742
EXPECTED_XAS_STATE_2_UNREL_R_E_Z = -0.530819
EXPECTED_XAS_STATE_2_UNREL_SEPARATION = 0.560768  # Electron-hole separation

# State 8 - XAS Excitation Energy
EXPECTED_XAS_STATE_8_EXCITATION_EV = 542.0062
EXPECTED_XAS_STATE_8_TOTAL_ENERGY_AU = -56.61846296
EXPECTED_XAS_STATE_8_STRENGTH = 0.0449502818  # Allowed transition
EXPECTED_XAS_STATE_8_TRANS_MOM_X = -0.0332
EXPECTED_XAS_STATE_8_TRANS_MOM_Y = -0.0046
EXPECTED_XAS_STATE_8_TRANS_MOM_Z = -0.0476

# State 8 - Unrelaxed Density Matrix
EXPECTED_XAS_STATE_8_UNREL_DIPOLE_MOMENT = 3.163782
EXPECTED_XAS_STATE_8_UNREL_DIPOLE_X = 0.058466
EXPECTED_XAS_STATE_8_UNREL_DIPOLE_Y = -0.673285
EXPECTED_XAS_STATE_8_UNREL_DIPOLE_Z = -3.090759
EXPECTED_XAS_STATE_8_UNREL_HOLE_SIZE = 0.122441
EXPECTED_XAS_STATE_8_UNREL_ELECTRON_SIZE = 1.768996
EXPECTED_XAS_STATE_8_UNREL_R_H_X = 2.328688
EXPECTED_XAS_STATE_8_UNREL_R_H_Y = 1.562923
EXPECTED_XAS_STATE_8_UNREL_R_H_Z = -0.041824
EXPECTED_XAS_STATE_8_UNREL_R_E_X = 2.061337
EXPECTED_XAS_STATE_8_UNREL_R_E_Y = 1.650988
EXPECTED_XAS_STATE_8_UNREL_R_E_Z = 0.155746
EXPECTED_XAS_STATE_8_UNREL_SEPARATION = 0.343898

# Transition Density Matrix Data for State 2
EXPECTED_XAS_STATE_2_TRANS_DM_DIPOLE_MOMENT = 0.000001
EXPECTED_XAS_STATE_2_TRANS_DIPOLE_X = -0.000000
EXPECTED_XAS_STATE_2_TRANS_DIPOLE_Y = -0.000000
EXPECTED_XAS_STATE_2_TRANS_DIPOLE_Z = -0.000001
EXPECTED_XAS_STATE_2_TRANS_HOLE_POS_X = 2.328688
EXPECTED_XAS_STATE_2_TRANS_HOLE_POS_Y = 1.562923
EXPECTED_XAS_STATE_2_TRANS_HOLE_POS_Z = -0.041824
EXPECTED_XAS_STATE_2_TRANS_ELEC_POS_X = 2.060655
EXPECTED_XAS_STATE_2_TRANS_ELEC_POS_Y = 1.503742
EXPECTED_XAS_STATE_2_TRANS_ELEC_POS_Z = -0.530819
EXPECTED_XAS_STATE_2_TRANS_SEPARATION = 0.560768

# Transition Density Matrix Data for State 8
EXPECTED_XAS_STATE_8_TRANS_DM_DIPOLE_MOMENT = 0.147883
EXPECTED_XAS_STATE_8_TRANS_DIPOLE_X = 0.084282
EXPECTED_XAS_STATE_8_TRANS_DIPOLE_Y = 0.011593
EXPECTED_XAS_STATE_8_TRANS_DIPOLE_Z = 0.120961
EXPECTED_XAS_STATE_8_TRANS_HOLE_POS_X = 2.328688
EXPECTED_XAS_STATE_8_TRANS_HOLE_POS_Y = 1.562923
EXPECTED_XAS_STATE_8_TRANS_HOLE_POS_Z = -0.041824
EXPECTED_XAS_STATE_8_TRANS_ELEC_POS_X = 2.061337
EXPECTED_XAS_STATE_8_TRANS_ELEC_POS_Y = 1.650988
EXPECTED_XAS_STATE_8_TRANS_ELEC_POS_Z = 0.155746
EXPECTED_XAS_STATE_8_TRANS_SEPARATION = 0.343898

# Numerical tolerances for regression tests
ENERGY_TOL = 1e-3  # eV
ENERGY_AU_TOL = 1e-7  # atomic units
DIPOLE_TOL = 1e-5  # Debye
POSITION_TOL = 1e-5  # Angstrom
STRENGTH_TOL = 1e-8


# =============================================================================
# REGRESSION TESTS - EXCITATION ENERGIES (STATE 2)
# =============================================================================


@pytest.mark.regression
def test_xas_state_2_excitation_energy(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 2 excitation energy matches expected value."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_2: ExcitedState = tddft.tda_states[1]  # 0-indexed

    assert abs(state_2.excitation_energy_ev - EXPECTED_XAS_STATE_2_EXCITATION_EV) < ENERGY_TOL


@pytest.mark.regression
def test_xas_state_2_total_energy(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 2 total energy matches expected value."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_2: ExcitedState = tddft.tda_states[1]

    assert abs(state_2.total_energy_au - EXPECTED_XAS_STATE_2_TOTAL_ENERGY_AU) < ENERGY_AU_TOL


@pytest.mark.regression
def test_xas_state_2_transition_moment(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 2 transition moment is zero (forbidden transition)."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_2: ExcitedState = tddft.tda_states[1]

    assert abs(state_2.trans_mom_x - EXPECTED_XAS_STATE_2_TRANS_MOM_X) < 1e-4
    assert abs(state_2.trans_mom_y - EXPECTED_XAS_STATE_2_TRANS_MOM_Y) < 1e-4
    assert abs(state_2.trans_mom_z - EXPECTED_XAS_STATE_2_TRANS_MOM_Z) < 1e-4


@pytest.mark.regression
def test_xas_state_2_oscillator_strength(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 2 oscillator strength is zero."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_2: ExcitedState = tddft.tda_states[1]

    assert state_2.oscillator_strength is not None
    assert abs(state_2.oscillator_strength - EXPECTED_XAS_STATE_2_STRENGTH) < STRENGTH_TOL


# =============================================================================
# REGRESSION TESTS - EXCITATION ENERGIES (STATE 8)
# =============================================================================


@pytest.mark.regression
def test_xas_state_8_excitation_energy(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 8 excitation energy."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_8: ExcitedState = tddft.tda_states[7]  # 0-indexed

    assert abs(state_8.excitation_energy_ev - EXPECTED_XAS_STATE_8_EXCITATION_EV) < ENERGY_TOL


@pytest.mark.regression
def test_xas_state_8_total_energy(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 8 total energy."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_8: ExcitedState = tddft.tda_states[7]

    assert abs(state_8.total_energy_au - EXPECTED_XAS_STATE_8_TOTAL_ENERGY_AU) < ENERGY_AU_TOL


@pytest.mark.regression
def test_xas_state_8_transition_moment(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 8 transition moment (allowed transition)."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_8: ExcitedState = tddft.tda_states[7]

    assert abs(state_8.trans_mom_x - EXPECTED_XAS_STATE_8_TRANS_MOM_X) < 1e-4
    assert abs(state_8.trans_mom_y - EXPECTED_XAS_STATE_8_TRANS_MOM_Y) < 1e-4
    assert abs(state_8.trans_mom_z - EXPECTED_XAS_STATE_8_TRANS_MOM_Z) < 1e-4


@pytest.mark.regression
def test_xas_state_8_oscillator_strength(parsed_qchem_62_mom_xas_smd_data: CalculationResult) -> None:
    """Verify state 8 oscillator strength (allowed transition)."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.tda_states is not None
    state_8: ExcitedState = tddft.tda_states[7]

    assert state_8.oscillator_strength is not None
    assert abs(state_8.oscillator_strength - EXPECTED_XAS_STATE_8_STRENGTH) < STRENGTH_TOL


# =============================================================================
# REGRESSION TESTS - UNRELAXED DENSITY MATRIX (STATE 2)
# =============================================================================


@pytest.mark.regression
def test_xas_state_2_unrel_dm_dipole_moment(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 unrelaxed density matrix dipole moment."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    assert abs(unrel.dipole_moment_debye - EXPECTED_XAS_STATE_2_UNREL_DIPOLE_MOMENT) < DIPOLE_TOL


@pytest.mark.regression
def test_xas_state_2_unrel_dm_dipole_components(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 unrelaxed density matrix dipole components."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    x, y, z = unrel.dipole_components_debye
    assert abs(x - EXPECTED_XAS_STATE_2_UNREL_DIPOLE_X) < DIPOLE_TOL
    assert abs(y - EXPECTED_XAS_STATE_2_UNREL_DIPOLE_Y) < DIPOLE_TOL
    assert abs(z - EXPECTED_XAS_STATE_2_UNREL_DIPOLE_Z) < DIPOLE_TOL


@pytest.mark.regression
def test_xas_state_2_unrel_dm_hole_size(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 has correct core hole size."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    assert abs(unrel.exciton_total.hole_size_ang - EXPECTED_XAS_STATE_2_UNREL_HOLE_SIZE) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_2_unrel_dm_electron_size(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 electron cloud size."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    assert abs(unrel.exciton_total.electron_size_ang - EXPECTED_XAS_STATE_2_UNREL_ELECTRON_SIZE) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_2_unrel_dm_separation(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 electron-hole separation."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    assert abs(unrel.exciton_total.separation_ang - EXPECTED_XAS_STATE_2_UNREL_SEPARATION) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_2_unrel_dm_hole_position(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 hole position (should be at oxygen nucleus)."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    r_h_x, r_h_y, r_h_z = unrel.exciton_total.r_h_ang
    assert abs(r_h_x - EXPECTED_XAS_STATE_2_UNREL_R_H_X) < POSITION_TOL
    assert abs(r_h_y - EXPECTED_XAS_STATE_2_UNREL_R_H_Y) < POSITION_TOL
    assert abs(r_h_z - EXPECTED_XAS_STATE_2_UNREL_R_H_Z) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_2_unrel_dm_electron_position(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 electron position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[1]

    r_e_x, r_e_y, r_e_z = unrel.exciton_total.r_e_ang
    assert abs(r_e_x - EXPECTED_XAS_STATE_2_UNREL_R_E_X) < POSITION_TOL
    assert abs(r_e_y - EXPECTED_XAS_STATE_2_UNREL_R_E_Y) < POSITION_TOL
    assert abs(r_e_z - EXPECTED_XAS_STATE_2_UNREL_R_E_Z) < POSITION_TOL


# =============================================================================
# REGRESSION TESTS - UNRELAXED DENSITY MATRIX (STATE 8)
# =============================================================================


@pytest.mark.regression
def test_xas_state_8_unrel_dm_dipole_moment(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 unrelaxed density matrix dipole moment."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    assert abs(unrel.dipole_moment_debye - EXPECTED_XAS_STATE_8_UNREL_DIPOLE_MOMENT) < DIPOLE_TOL


@pytest.mark.regression
def test_xas_state_8_unrel_dm_dipole_components(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 unrelaxed density matrix dipole components."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    x, y, z = unrel.dipole_components_debye
    assert abs(x - EXPECTED_XAS_STATE_8_UNREL_DIPOLE_X) < DIPOLE_TOL
    assert abs(y - EXPECTED_XAS_STATE_8_UNREL_DIPOLE_Y) < DIPOLE_TOL
    assert abs(z - EXPECTED_XAS_STATE_8_UNREL_DIPOLE_Z) < DIPOLE_TOL


@pytest.mark.regression
def test_xas_state_8_unrel_dm_hole_size(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 has correct core hole size."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    assert abs(unrel.exciton_total.hole_size_ang - EXPECTED_XAS_STATE_8_UNREL_HOLE_SIZE) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_8_unrel_dm_electron_size(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 electron cloud size."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    assert abs(unrel.exciton_total.electron_size_ang - EXPECTED_XAS_STATE_8_UNREL_ELECTRON_SIZE) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_8_unrel_dm_separation(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 electron-hole separation."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    assert abs(unrel.exciton_total.separation_ang - EXPECTED_XAS_STATE_8_UNREL_SEPARATION) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_8_unrel_dm_hole_position(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 hole position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    r_h_x, r_h_y, r_h_z = unrel.exciton_total.r_h_ang
    assert abs(r_h_x - EXPECTED_XAS_STATE_8_UNREL_R_H_X) < POSITION_TOL
    assert abs(r_h_y - EXPECTED_XAS_STATE_8_UNREL_R_H_Y) < POSITION_TOL
    assert abs(r_h_z - EXPECTED_XAS_STATE_8_UNREL_R_H_Z) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_8_unrel_dm_electron_position(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 electron position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.unrelaxed_density_matrices is not None
    unrel: UnrelaxedDensityMatrix = tddft.unrelaxed_density_matrices[7]

    r_e_x, r_e_y, r_e_z = unrel.exciton_total.r_e_ang
    assert abs(r_e_x - EXPECTED_XAS_STATE_8_UNREL_R_E_X) < POSITION_TOL
    assert abs(r_e_y - EXPECTED_XAS_STATE_8_UNREL_R_E_Y) < POSITION_TOL
    assert abs(r_e_z - EXPECTED_XAS_STATE_8_UNREL_R_E_Z) < POSITION_TOL


# =============================================================================
# REGRESSION TESTS - TRANSITION DENSITY MATRIX (STATE 2)
# =============================================================================


@pytest.mark.regression
def test_xas_state_2_trans_dm_dipole_moment(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 transition density matrix dipole moment."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[1]

    assert trans.trans_dipole_moment_debye is not None
    assert abs(trans.trans_dipole_moment_debye - EXPECTED_XAS_STATE_2_TRANS_DM_DIPOLE_MOMENT) < DIPOLE_TOL


@pytest.mark.regression
def test_xas_state_2_trans_dm_exciton_hole_pos(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 transition density matrix hole position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[1]

    r_h_x, r_h_y, r_h_z = trans.exciton_total.r_h_ang
    assert abs(r_h_x - EXPECTED_XAS_STATE_2_TRANS_HOLE_POS_X) < POSITION_TOL
    assert abs(r_h_y - EXPECTED_XAS_STATE_2_TRANS_HOLE_POS_Y) < POSITION_TOL
    assert abs(r_h_z - EXPECTED_XAS_STATE_2_TRANS_HOLE_POS_Z) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_2_trans_dm_exciton_electron_pos(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 transition density matrix electron position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[1]

    r_e_x, r_e_y, r_e_z = trans.exciton_total.r_e_ang
    assert abs(r_e_x - EXPECTED_XAS_STATE_2_TRANS_ELEC_POS_X) < POSITION_TOL
    assert abs(r_e_y - EXPECTED_XAS_STATE_2_TRANS_ELEC_POS_Y) < POSITION_TOL
    assert abs(r_e_z - EXPECTED_XAS_STATE_2_TRANS_ELEC_POS_Z) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_2_trans_dm_exciton_separation(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 2 transition density matrix electron-hole separation."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[1]

    assert abs(trans.exciton_total.separation_ang - EXPECTED_XAS_STATE_2_TRANS_SEPARATION) < POSITION_TOL


# =============================================================================
# REGRESSION TESTS - TRANSITION DENSITY MATRIX (STATE 8)
# =============================================================================


@pytest.mark.regression
def test_xas_state_8_trans_dm_dipole_moment(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 transition density matrix dipole moment."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[7]

    assert trans.trans_dipole_moment_debye is not None
    assert abs(trans.trans_dipole_moment_debye - EXPECTED_XAS_STATE_8_TRANS_DM_DIPOLE_MOMENT) < DIPOLE_TOL


@pytest.mark.regression
def test_xas_state_8_trans_dm_exciton_hole_pos(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 transition density matrix hole position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[7]

    r_h_x, r_h_y, r_h_z = trans.exciton_total.r_h_ang
    assert abs(r_h_x - EXPECTED_XAS_STATE_8_TRANS_HOLE_POS_X) < POSITION_TOL
    assert abs(r_h_y - EXPECTED_XAS_STATE_8_TRANS_HOLE_POS_Y) < POSITION_TOL
    assert abs(r_h_z - EXPECTED_XAS_STATE_8_TRANS_HOLE_POS_Z) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_8_trans_dm_exciton_electron_pos(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 transition density matrix electron position."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[7]

    r_e_x, r_e_y, r_e_z = trans.exciton_total.r_e_ang
    assert abs(r_e_x - EXPECTED_XAS_STATE_8_TRANS_ELEC_POS_X) < POSITION_TOL
    assert abs(r_e_y - EXPECTED_XAS_STATE_8_TRANS_ELEC_POS_Y) < POSITION_TOL
    assert abs(r_e_z - EXPECTED_XAS_STATE_8_TRANS_ELEC_POS_Z) < POSITION_TOL


@pytest.mark.regression
def test_xas_state_8_trans_dm_exciton_separation(
    parsed_qchem_62_mom_xas_smd_data: CalculationResult,
) -> None:
    """Verify state 8 transition density matrix electron-hole separation."""
    tddft: TddftResults = parsed_qchem_62_mom_xas_smd_data.tddft
    assert tddft.transition_density_matrices is not None
    trans: TransitionDensityMatrix = tddft.transition_density_matrices[7]

    assert abs(trans.exciton_total.separation_ang - EXPECTED_XAS_STATE_8_TRANS_SEPARATION) < POSITION_TOL
