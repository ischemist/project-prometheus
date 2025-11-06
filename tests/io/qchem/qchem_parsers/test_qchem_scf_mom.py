"""
Contract tests for MOM (Maximum Overlap Method) based SCF parsing.

These tests verify that the SCF parser correctly extracts MOM-specific data
from Q-Chem output files that use the Maximum Overlap Method for initial
orbital guessing in excited state calculations.

MOM Characteristics:
- Appears after "Writing Guess MOs for IMOM"
- SCF iterations are preceded by MOM status lines
- MOM overlap values indicate orbital overlap with target state
- Uses "Maximum Overlap Method Active" and "IMOM method" markers
- Still produces standard SCF iteration output

Test Data Source: qchem/h2o/6.2-mom-sp-smd.out
- Job 1: Regular SCF (baseline for comparison)
- Job 2: MOM-enabled SCF (MOM-specific parsing)

Expected Values (Job 2, MOM iterations):
- 8 total iterations
- MOM overlap range: 5.0 to 4.98 (out of 5 electrons)
- Final energy: -76.16997884 a.u.
- Converged: True
"""

import pytest

from calcflow.common.results import CalculationResult, ScfResults
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# TEST DATA: Hardcoded expected values from qchem/h2o/6.2-mom-sp-smd.out, Job 2 (MOM)
# =============================================================================

# Job 2: MOM-enabled SCF iterations with hardcoded values from ex-mom-sp.md
JOB2_MOM_ITERATIONS = [
    {
        "iteration": 1,
        "energy": -76.0833919808,
        "diis_error": 7.23e-03,
        "mom_active": True,
        "mom_overlap_current": 5.0,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 1,  # Note: iteration counter resets at first DIIS iteration
        "energy": -76.1266046257,
        "diis_error": 6.45e-03,
        "mom_active": True,
        "mom_overlap_current": 4.99,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 2,
        "energy": -76.1420331915,
        "diis_error": 5.03e-03,
        "mom_active": True,
        "mom_overlap_current": 4.98,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 3,
        "energy": -76.1698796185,
        "diis_error": 1.63e-04,
        "mom_active": True,
        "mom_overlap_current": 4.98,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 4,
        "energy": -76.1700093856,
        "diis_error": 7.72e-05,
        "mom_active": True,
        "mom_overlap_current": 4.98,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 5,
        "energy": -76.1706945764,
        "diis_error": 2.70e-04,
        "mom_active": True,
        "mom_overlap_current": 4.98,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 6,
        "energy": -76.1699784600,
        "diis_error": 1.19e-05,
        "mom_active": True,
        "mom_overlap_current": 4.98,
        "mom_overlap_target": 5,
    },
    {
        "iteration": 7,
        "energy": -76.1699788432,
        "diis_error": 3.44e-06,
        "mom_active": True,
        "mom_overlap_current": 4.98,
        "mom_overlap_target": 5,
    },
]

# Summary values for Job 2 (MOM-enabled SCF)
JOB2_SCF_ENERGY = -76.16997884
JOB2_TOTAL_ENERGY = -76.16763138
JOB2_CONVERGED = True
JOB2_N_ITERATIONS = 8  # Note: includes both Roothaan step (iter 1) and DIIS restart (also iter 1)

# SMD results for Job 2
JOB2_SMD_G_PCM_KCAL_MOL = -3.5354
JOB2_SMD_G_CDS_KCAL_MOL = 1.4731
JOB2_SMD_G_ENP_AU = -76.16997884
JOB2_SMD_G_TOT_AU = -76.16763138

# Numerical tolerances
ENERGY_TOL = 1e-8
DIIS_TOL = 1e-10
MOM_OVERLAP_TOL = 0.01


# =============================================================================
# CONTRACT TESTS: MOM Field Presence and Type Validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_iteration_has_mom_active_field(parsed_qchem_data: CalculationResult):
    """Contract test: MOM iterations should have mom_active field present."""
    assert parsed_qchem_data.scf is not None

    for iteration in parsed_qchem_data.scf.iterations:
        # For non-MOM iterations, mom_active should be None or False
        # For MOM iterations, mom_active should be True
        assert iteration.mom_active is None or isinstance(iteration.mom_active, bool)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_iteration_has_overlap_fields(parsed_qchem_data: CalculationResult):
    """Contract test: MOM iterations should have overlap fields present."""
    assert parsed_qchem_data.scf is not None

    for iteration in parsed_qchem_data.scf.iterations:
        if iteration.mom_active:
            # If mom_active is True, overlap fields should be present
            assert iteration.mom_overlap_current is not None
            assert iteration.mom_overlap_target is not None
            assert isinstance(iteration.mom_overlap_current, float)
            assert isinstance(iteration.mom_overlap_target, (float, int))


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_overlap_values_are_floats(parsed_qchem_data: CalculationResult):
    """Contract test: MOM overlap values should be numeric."""
    assert parsed_qchem_data.scf is not None

    for iteration in parsed_qchem_data.scf.iterations:
        if iteration.mom_active:
            assert isinstance(iteration.mom_overlap_current, float)
            assert iteration.mom_overlap_current > 0


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_overlap_target_is_reasonable(parsed_qchem_data: CalculationResult):
    """Contract test: MOM overlap target should be a reasonable electron count."""
    assert parsed_qchem_data.scf is not None

    for iteration in parsed_qchem_data.scf.iterations:
        if iteration.mom_active and iteration.mom_overlap_target is not None:
            # Target should be positive and typically small (number of electrons)
            assert iteration.mom_overlap_target > 0
            assert iteration.mom_overlap_target <= 100  # Reasonable upper bound


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_overlap_current_vs_target_consistency(parsed_qchem_data: CalculationResult):
    """Contract test: MOM overlap current should not exceed target by much."""
    assert parsed_qchem_data.scf is not None

    for iteration in parsed_qchem_data.scf.iterations:
        if iteration.mom_active:
            # Current overlap should be close to or slightly less than target
            target = iteration.mom_overlap_target or 5.0
            assert iteration.mom_overlap_current <= target + 0.05


@pytest.mark.contract
def test_non_mom_iterations_have_none_fields(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """Contract test: Non-MOM iterations should have None for MOM fields."""
    assert parsed_qchem_54_h2o_sp_data.scf is not None

    for iteration in parsed_qchem_54_h2o_sp_data.scf.iterations:
        # Regular SCF (not MOM) should have None for these fields
        assert iteration.mom_active is None
        assert iteration.mom_overlap_current is None
        assert iteration.mom_overlap_target is None


# =============================================================================
# CONTRACT TESTS: MOM SCF Results Structure
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_scf_has_correct_type(parsed_qchem_data: CalculationResult):
    """Contract test: MOM job should have ScfResults instance."""
    assert parsed_qchem_data.scf is not None
    assert isinstance(parsed_qchem_data.scf, ScfResults)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_scf_iterations_is_tuple(parsed_qchem_data: CalculationResult):
    """Contract test: MOM iterations field should be a tuple."""
    assert parsed_qchem_data.scf is not None
    assert isinstance(parsed_qchem_data.scf.iterations, tuple)
    assert len(parsed_qchem_data.scf.iterations) > 0


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_scf_has_required_fields(parsed_qchem_data: CalculationResult):
    """Contract test: MOM SCF results should have all required fields."""
    scf = parsed_qchem_data.scf
    assert scf is not None

    assert isinstance(scf.converged, bool)
    assert isinstance(scf.energy, float)
    assert isinstance(scf.n_iterations, int)
    assert scf.n_iterations > 0


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_scf_n_iterations_matches_length(parsed_qchem_data: CalculationResult):
    """Contract test: n_iterations should match actual iteration count."""
    scf = parsed_qchem_data.scf
    assert scf is not None
    assert scf.n_iterations == len(scf.iterations)


# =============================================================================
# CONTRACT TESTS: MOM vs Non-MOM Comparison
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_job_has_mom_active_iterations(parsed_qchem_data: CalculationResult):
    """Contract test: MOM job should have at least some iterations with mom_active=True."""
    assert parsed_qchem_data.scf is not None

    mom_active_count = sum(1 for it in parsed_qchem_data.scf.iterations if it.mom_active)
    assert mom_active_count > 0, "MOM job should have at least one MOM-active iteration"


@pytest.mark.contract
def test_non_mom_job_has_no_mom_active_iterations(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Contract test: Non-MOM job should have no mom_active iterations."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None

    mom_active_count = sum(1 for it in parsed_qchem_62_h2o_sp_data.scf.iterations if it.mom_active)
    assert mom_active_count == 0, "Non-MOM job should have zero MOM-active iterations"


# =============================================================================
# REGRESSION TESTS: Exact MOM Values from ex-mom-sp.md
# =============================================================================


@pytest.mark.regression
def test_mom_job_converged(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job should be converged."""
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.scf.converged is JOB2_CONVERGED


@pytest.mark.regression
def test_mom_job_final_energy(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job final SCF energy matches expected value."""
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.scf.energy == pytest.approx(JOB2_SCF_ENERGY, abs=ENERGY_TOL)


@pytest.mark.regression
def test_mom_job_final_total_energy(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job final total energy (SMD) matches expected value."""
    assert parsed_qchem_62_h2o_mom_sp_job2.final_energy == pytest.approx(JOB2_TOTAL_ENERGY, abs=ENERGY_TOL)


@pytest.mark.regression
def test_mom_job_num_iterations(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job should have exactly 8 iterations."""
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.scf.n_iterations == JOB2_N_ITERATIONS


@pytest.mark.regression
def test_mom_job_first_iteration_overlap(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: First MOM iteration should have overlap value of 5.0."""
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None
    first_iter = parsed_qchem_62_h2o_mom_sp_job2.scf.iterations[0]

    assert first_iter.mom_active is True
    assert first_iter.mom_overlap_current == pytest.approx(5.0, abs=MOM_OVERLAP_TOL)


@pytest.mark.regression
def test_mom_job_overlap_values_range(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM overlap should range from 5.0 down to ~4.98."""
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None

    overlap_values = [
        it.mom_overlap_current
        for it in parsed_qchem_62_h2o_mom_sp_job2.scf.iterations
        if it.mom_active and it.mom_overlap_current is not None
    ]

    assert len(overlap_values) > 0
    assert max(overlap_values) >= 4.98
    assert min(overlap_values) <= 5.0


@pytest.mark.regression
def test_mom_job_last_iteration_energy(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: Last MOM iteration energy should match SCF energy."""
    scf = parsed_qchem_62_h2o_mom_sp_job2.scf
    assert scf is not None
    assert len(scf.iterations) > 0

    last_iter = scf.iterations[-1]
    assert last_iter.energy == pytest.approx(JOB2_SCF_ENERGY, abs=ENERGY_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "iter_idx,expected_iter_num,expected_energy,expected_diis,expected_overlap",
    [
        (0, 1, -76.0833919808, 7.23e-03, 5.0),
        (1, 1, -76.1266046257, 6.45e-03, 4.99),
        (2, 2, -76.1420331915, 5.03e-03, 4.98),
        (3, 3, -76.1698796185, 1.63e-04, 4.98),
        (4, 4, -76.1700093856, 7.72e-05, 4.98),
        (5, 5, -76.1706945764, 2.70e-04, 4.98),
        (6, 6, -76.1699784600, 1.19e-05, 4.98),
        (7, 7, -76.1699788432, 3.44e-06, 4.98),
    ],
    ids=[
        "mom-iter0-roothaan",
        "mom-iter1-diis-start",
        "mom-iter2",
        "mom-iter3",
        "mom-iter4",
        "mom-iter5",
        "mom-iter6",
        "mom-iter7-converged",
    ],
)
def test_mom_job_iteration_values(
    parsed_qchem_62_h2o_mom_sp_job2: CalculationResult,
    iter_idx: int,
    expected_iter_num: int,
    expected_energy: float,
    expected_diis: float,
    expected_overlap: float,
):
    """Regression test: Exact values for each MOM iteration."""
    scf = parsed_qchem_62_h2o_mom_sp_job2.scf
    assert scf is not None
    assert len(scf.iterations) >= iter_idx + 1

    iteration = scf.iterations[iter_idx]
    assert iteration.iteration == expected_iter_num
    assert iteration.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)
    assert iteration.diis_error == pytest.approx(expected_diis, abs=DIIS_TOL)

    # Check MOM fields (except for the last iteration where MOM status is printed after convergence)
    if iter_idx < 7:  # iterations 0-6 have MOM data
        assert iteration.mom_active is True
        assert iteration.mom_overlap_current == pytest.approx(expected_overlap, abs=MOM_OVERLAP_TOL)
        assert iteration.mom_overlap_target == 5
    else:
        # Iteration 7 has convergence marker, and MOM lines are printed AFTER (not before)
        # So this iteration doesn't capture MOM data
        assert iteration.mom_active is None


# =============================================================================
# REGRESSION TESTS: MOM SMD Results
# =============================================================================


@pytest.mark.regression
def test_mom_job_smd_g_pcm(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job SMD G_PCM value."""
    assert parsed_qchem_62_h2o_mom_sp_job2.smd is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.smd.g_pcm_kcal_mol == pytest.approx(JOB2_SMD_G_PCM_KCAL_MOL, abs=1e-4)


@pytest.mark.regression
def test_mom_job_smd_g_cds(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job SMD G_CDS value."""
    assert parsed_qchem_62_h2o_mom_sp_job2.smd is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.smd.g_cds_kcal_mol == pytest.approx(JOB2_SMD_G_CDS_KCAL_MOL, abs=1e-4)


@pytest.mark.regression
def test_mom_job_smd_g_enp(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job SMD G_ENP value."""
    assert parsed_qchem_62_h2o_mom_sp_job2.smd is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.smd.g_enp_au == pytest.approx(JOB2_SMD_G_ENP_AU, abs=ENERGY_TOL)


@pytest.mark.regression
def test_mom_job_smd_g_tot(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Regression test: MOM job SMD G(tot) value."""
    assert parsed_qchem_62_h2o_mom_sp_job2.smd is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.smd.g_tot_au == pytest.approx(JOB2_SMD_G_TOT_AU, abs=ENERGY_TOL)


# =============================================================================
# INTEGRATION TESTS: MOM SCF Integration with Other Parsers
# =============================================================================


@pytest.mark.integration
def test_mom_scf_works_with_metadata(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Integration test: MOM parser works with metadata."""
    assert parsed_qchem_62_h2o_mom_sp_job2.metadata is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.metadata.software_version is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None


@pytest.mark.integration
def test_mom_scf_works_with_geometry(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Integration test: MOM parser works with geometry parser."""
    # Job 2 has $molecule read, so input_geometry is None (inherited from Job 1)
    # But final_geometry should be present from Standard Nuclear Orientation
    assert parsed_qchem_62_h2o_mom_sp_job2.input_geometry is None
    assert parsed_qchem_62_h2o_mom_sp_job2.final_geometry is not None
    assert len(parsed_qchem_62_h2o_mom_sp_job2.final_geometry) > 0
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None


@pytest.mark.integration
def test_mom_scf_works_with_smd(parsed_qchem_62_h2o_mom_sp_job2: CalculationResult):
    """Integration test: MOM SCF works alongside SMD parsing."""
    assert parsed_qchem_62_h2o_mom_sp_job2.scf is not None
    assert parsed_qchem_62_h2o_mom_sp_job2.smd is not None

    # SCF energy should match SMD G_ENP (SCF + PCM)
    assert parsed_qchem_62_h2o_mom_sp_job2.scf.energy == pytest.approx(
        parsed_qchem_62_h2o_mom_sp_job2.smd.g_enp_au, abs=ENERGY_TOL
    )


# =============================================================================
# CROSS-VERSION TESTS: MOM support across Q-Chem versions
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_scf_present_both_versions(parsed_qchem_data: CalculationResult):
    """Contract test: MOM SCF should be present in both Q-Chem 5.4 and 6.2."""
    assert parsed_qchem_data.scf is not None
    assert isinstance(parsed_qchem_data.scf, ScfResults)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["mom_scf"], indirect=True)
def test_mom_overlap_fields_both_versions(parsed_qchem_data: CalculationResult):
    """Contract test: MOM overlap fields should be present in both versions."""
    assert parsed_qchem_data.scf is not None

    mom_iters = [it for it in parsed_qchem_data.scf.iterations if it.mom_active]
    assert len(mom_iters) > 0

    for iteration in mom_iters:
        assert iteration.mom_overlap_current is not None
        assert iteration.mom_overlap_target is not None
