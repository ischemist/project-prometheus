"""
Tests for the QChem SCF block parser.

These tests verify that the SCF parser correctly extracts SCF iterations,
convergence status, final energies, and SMD solvation model results from
Q-Chem output files. Tests cover both Q-Chem 5.4 and 6.2 formats.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure
- integration: multiple components working together
- regression: exact numerical values match expected

Format notes:
- Q-Chem 5.4: SMD output uses individual lines, "SCF energy in the final basis set"
- Q-Chem 6.2: SMD output uses "Summary of SMD free energies" header, "SCF energy ="
- Both formats: identical SCF iteration values
"""

import pytest

from calcflow.common.results import CalculationResult, ScfIteration, ScfResults, SmdResults
from calcflow.io.qchem.blocks.scf import ScfParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA (Q-Chem 5.4 and 6.2 have identical SCF values)
# =============================================================================

# SCF Iterations - identical in both 5.4 and 6.2
EXPECTED_ITERATIONS = [
    {"iteration": 1, "energy": -75.0734525440, "diis_error": 3.82e-01},
    {"iteration": 2, "energy": -75.2973764631, "diis_error": 4.85e-02},
    {"iteration": 3, "energy": -75.3188175820, "diis_error": 1.63e-02},
    {"iteration": 4, "energy": -75.3207889318, "diis_error": 1.19e-03},
    {"iteration": 5, "energy": -75.3208073942, "diis_error": 1.25e-04},
    {"iteration": 6, "energy": -75.3208076933, "diis_error": 2.28e-05},
    {"iteration": 7, "energy": -75.3208077035, "diis_error": 2.17e-08},
]

EXPECTED_SCF_ENERGY = -75.32080770
EXPECTED_TOTAL_ENERGY = -75.31846024
EXPECTED_CONVERGED = True
EXPECTED_N_ITERATIONS = 7

# SMD Solvation Model Results (identical in both versions)
EXPECTED_SMD_G_PCM_KCAL_MOL = -6.0201
EXPECTED_SMD_G_CDS_KCAL_MOL = 1.4731
EXPECTED_SMD_G_ENP_AU = -75.32080770
EXPECTED_SMD_G_TOT_AU = -75.31846024

# Numerical tolerance for energies (8 decimal places as in output)
ENERGY_TOL = 1e-8
DIIS_TOL = 1e-10


# =============================================================================
# UNIT TESTS: ScfParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_scf_parser_matches_start_line():
    """Unit test: verify ScfParser.matches() recognizes SCF block start."""
    parser = ScfParser()
    state = ParseState(raw_output="")

    start_line = " General SCF calculation program by"
    assert parser.matches(start_line, state) is True


@pytest.mark.unit
def test_scf_parser_matches_with_leading_whitespace():
    """Unit test: verify ScfParser.matches() handles leading whitespace."""
    parser = ScfParser()
    state = ParseState(raw_output="")

    start_line = "    General SCF calculation program by"
    assert parser.matches(start_line, state) is True


@pytest.mark.unit
def test_scf_parser_does_not_match_non_scf_lines():
    """Unit test: verify ScfParser.matches() rejects non-SCF lines."""
    parser = ScfParser()
    state = ParseState(raw_output="")

    # Random lines from QChem output
    assert parser.matches("SCF time:   CPU 0.32s  wall 0.00s", state) is False
    assert parser.matches("Cycle       Energy         DIIS error", state) is False
    assert parser.matches("Random calculation output", state) is False


@pytest.mark.unit
def test_scf_parser_skips_if_already_parsed():
    """Unit test: verify ScfParser.matches() returns False when already parsed."""
    parser = ScfParser()
    state = ParseState(raw_output="")
    state.parsed_scf = True

    start_line = " General SCF calculation program by"
    assert parser.matches(start_line, state) is False


@pytest.mark.unit
def test_scf_parser_does_not_mutate_state_in_matches():
    """
    Unit test: verify that matches() is read-only and does not mutate state.
    Critical for parser-spec compliance.
    """
    parser = ScfParser()
    state = ParseState(raw_output="")

    # Call matches() multiple times
    line = " General SCF calculation program by"
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    # State should be identical after calling matches()
    assert result1 is True
    assert result2 is True
    assert state.parsed_scf is False  # Should NOT be set
    assert state.scf is None  # Should NOT be populated


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["scf"],
    indirect=True,
)
def test_scf_results_has_correct_type(parsed_qchem_data: CalculationResult):
    """Contract test: verify scf field is ScfResults instance."""
    assert parsed_qchem_data.scf is not None
    assert isinstance(parsed_qchem_data.scf, ScfResults)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["scf"], indirect=True)
def test_scf_results_iterations_is_tuple(parsed_qchem_data: CalculationResult):
    """Contract test: verify iterations field is a tuple of ScfIteration objects."""
    assert parsed_qchem_data.scf is not None
    assert isinstance(parsed_qchem_data.scf.iterations, tuple)
    assert len(parsed_qchem_data.scf.iterations) > 0
    assert all(isinstance(it, ScfIteration) for it in parsed_qchem_data.scf.iterations)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["scf"], indirect=True)
def test_scf_results_required_fields_present(parsed_qchem_data: CalculationResult):
    """Contract test: verify all required ScfResults fields are present with correct types."""
    scf = parsed_qchem_data.scf
    assert scf is not None

    # Check required fields
    assert isinstance(scf.converged, bool)
    assert isinstance(scf.energy, float)
    assert isinstance(scf.n_iterations, int)
    assert scf.n_iterations > 0


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["scf"], indirect=True)
def test_scf_iteration_required_fields(parsed_qchem_data: CalculationResult):
    """Contract test: verify each ScfIteration has required fields."""
    scf = parsed_qchem_data.scf
    assert scf is not None

    for iteration in scf.iterations:
        assert isinstance(iteration.iteration, int)
        assert isinstance(iteration.energy, float)
        # DIIS error is QChem-specific, should be present for QChem files
        assert isinstance(iteration.diis_error, float) or iteration.diis_error is None


@pytest.mark.contract
@pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["scf"], indirect=True)
def test_scf_n_iterations_matches_length(parsed_qchem_data: CalculationResult):
    """Contract test: verify n_iterations field matches actual iteration count."""
    scf = parsed_qchem_data.scf
    assert scf is not None
    assert scf.n_iterations == len(scf.iterations)


@pytest.mark.contract
def test_smd_results_has_correct_type(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Contract test: verify smd field is SmdResults instance when present."""
    assert parsed_qchem_62_h2o_sp_data.smd is not None
    assert isinstance(parsed_qchem_62_h2o_sp_data.smd, SmdResults)


@pytest.mark.contract
def test_smd_results_has_expected_fields(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Contract test: verify SmdResults has expected fields populated."""
    smd = parsed_qchem_62_h2o_sp_data.smd
    assert smd is not None

    # All four fields should be present in this SMD calculation
    assert smd.g_pcm_kcal_mol is not None
    assert smd.g_cds_kcal_mol is not None
    assert smd.g_enp_au is not None
    assert smd.g_tot_au is not None

    # All should be floats
    assert isinstance(smd.g_pcm_kcal_mol, float)
    assert isinstance(smd.g_cds_kcal_mol, float)
    assert isinstance(smd.g_enp_au, float)
    assert isinstance(smd.g_tot_au, float)


@pytest.mark.contract
def test_final_energy_set_from_smd_total(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that final_energy is set to SMD total free energy
    when SMD calculation is present.
    """
    # When SMD is present, final_energy should equal smd.g_tot_au
    assert parsed_qchem_62_h2o_sp_data.final_energy is not None
    assert parsed_qchem_62_h2o_sp_data.smd is not None
    # They should be the same value (within tolerance)
    assert parsed_qchem_62_h2o_sp_data.final_energy == pytest.approx(
        parsed_qchem_62_h2o_sp_data.smd.g_tot_au, abs=ENERGY_TOL
    )


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
def test_scf_parsed_alongside_metadata(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify SCF parser works with metadata parser.
    Metadata version should be populated (required for SCF pattern matching).
    """
    assert parsed_qchem_62_h2o_sp_data.metadata is not None
    assert parsed_qchem_62_h2o_sp_data.metadata.software_version is not None

    assert parsed_qchem_62_h2o_sp_data.scf is not None


@pytest.mark.integration
def test_scf_parsed_alongside_geometry(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify SCF parser works with geometry parser.
    All should be present in final result.
    """
    assert parsed_qchem_62_h2o_sp_data.input_geometry is not None
    assert len(parsed_qchem_62_h2o_sp_data.input_geometry) == 3  # H2O

    assert parsed_qchem_62_h2o_sp_data.scf is not None


@pytest.mark.integration
def test_scf_completion_flag_set(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that the SCF parser sets its completion flag.
    This is critical for the parser-spec contract.
    """
    # The parsed result should have SCF data, indicating the flag was set
    assert parsed_qchem_62_h2o_sp_data.scf is not None


@pytest.mark.integration
def test_scf_final_energy_populated(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Integration test: verify final_energy is set after SCF parsing."""
    assert parsed_qchem_62_h2o_sp_data.final_energy is not None
    assert isinstance(parsed_qchem_62_h2o_sp_data.final_energy, float)


@pytest.mark.integration
def test_scf_and_smd_both_present(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify both SCF and SMD results are present and linked.
    """
    assert parsed_qchem_62_h2o_sp_data.scf is not None
    assert parsed_qchem_62_h2o_sp_data.smd is not None

    # SCF energy and SMD G_ENP should be equal (G_ENP = E_SCF + G_PCM)
    assert parsed_qchem_62_h2o_sp_data.scf.energy == pytest.approx(
        parsed_qchem_62_h2o_sp_data.smd.g_enp_au, abs=ENERGY_TOL
    )


@pytest.mark.integration
def test_scf_energy_consistency(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify SCF energy matches the final iteration energy.
    """
    scf = parsed_qchem_62_h2o_sp_data.scf
    assert scf is not None
    assert len(scf.iterations) > 0

    last_iteration = scf.iterations[-1]
    assert scf.energy == pytest.approx(last_iteration.energy, abs=ENERGY_TOL)


# =============================================================================
# REGRESSION TESTS: Exact numerical values from ex-scf.md
# =============================================================================


@pytest.mark.regression
def test_scf_converged_status(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify SCF converged to expected status."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None
    assert parsed_qchem_62_h2o_sp_data.scf.converged is EXPECTED_CONVERGED


@pytest.mark.regression
def test_scf_number_of_iterations(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact number of SCF iterations."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None
    assert parsed_qchem_62_h2o_sp_data.scf.n_iterations == EXPECTED_N_ITERATIONS


@pytest.mark.regression
def test_scf_final_energy_value(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact final SCF energy value."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None
    assert parsed_qchem_62_h2o_sp_data.scf.energy == pytest.approx(EXPECTED_SCF_ENERGY, abs=ENERGY_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "iter_idx,expected_iter,expected_energy,expected_diis",
    [
        (0, 1, -75.0734525440, 3.82e-01),
        (1, 2, -75.2973764631, 4.85e-02),
        (2, 3, -75.3188175820, 1.63e-02),
        (3, 4, -75.3207889318, 1.19e-03),
        (4, 5, -75.3208073942, 1.25e-04),
        (5, 6, -75.3208076933, 2.28e-05),
        (6, 7, -75.3208077035, 2.17e-08),
    ],
)
def test_scf_iteration_values(
    parsed_qchem_62_h2o_sp_data: CalculationResult,
    iter_idx: int,
    expected_iter: int,
    expected_energy: float,
    expected_diis: float,
):
    """Regression test: verify exact values for each SCF iteration."""
    scf = parsed_qchem_62_h2o_sp_data.scf
    assert scf is not None
    assert len(scf.iterations) >= iter_idx + 1

    iteration = scf.iterations[iter_idx]
    assert iteration.iteration == expected_iter
    assert iteration.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)
    assert iteration.diis_error == pytest.approx(expected_diis, abs=DIIS_TOL)


@pytest.mark.regression
def test_all_seven_iterations_present(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify all 7 iterations are parsed, not truncated."""
    assert parsed_qchem_62_h2o_sp_data.scf is not None
    assert len(parsed_qchem_62_h2o_sp_data.scf.iterations) == 7

    # Verify iteration numbers are sequential
    for i, iteration in enumerate(parsed_qchem_62_h2o_sp_data.scf.iterations, start=1):
        assert iteration.iteration == i


@pytest.mark.regression
def test_smd_g_pcm_value(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact SMD polarization energy (G_PCM)."""
    assert parsed_qchem_62_h2o_sp_data.smd is not None
    assert parsed_qchem_62_h2o_sp_data.smd.g_pcm_kcal_mol == pytest.approx(EXPECTED_SMD_G_PCM_KCAL_MOL, abs=1e-4)


@pytest.mark.regression
def test_smd_g_cds_value(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact SMD non-electrostatic energy (G_CDS)."""
    assert parsed_qchem_62_h2o_sp_data.smd is not None
    assert parsed_qchem_62_h2o_sp_data.smd.g_cds_kcal_mol == pytest.approx(EXPECTED_SMD_G_CDS_KCAL_MOL, abs=1e-4)


@pytest.mark.regression
def test_smd_g_enp_value(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact SMD energy in solvent (G_ENP)."""
    assert parsed_qchem_62_h2o_sp_data.smd is not None
    assert parsed_qchem_62_h2o_sp_data.smd.g_enp_au == pytest.approx(EXPECTED_SMD_G_ENP_AU, abs=ENERGY_TOL)


@pytest.mark.regression
def test_smd_g_tot_value(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """Regression test: verify exact SMD total free energy (G(tot))."""
    assert parsed_qchem_62_h2o_sp_data.smd is not None
    assert parsed_qchem_62_h2o_sp_data.smd.g_tot_au == pytest.approx(EXPECTED_SMD_G_TOT_AU, abs=ENERGY_TOL)


@pytest.mark.regression
def test_final_energy_equals_smd_total(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Regression test: verify final energy equals SMD total free energy.
    Final energy should be the most important energy (SMD total in this case).
    """
    assert parsed_qchem_62_h2o_sp_data.final_energy == pytest.approx(EXPECTED_TOTAL_ENERGY, abs=ENERGY_TOL)


@pytest.mark.regression
def test_last_iteration_convergence_marker(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Regression test: verify the last iteration has the convergence marker.
    The last iteration in the parsed list should have extremely low DIIS error.
    """
    scf = parsed_qchem_62_h2o_sp_data.scf
    assert scf is not None
    assert len(scf.iterations) > 0

    last_iteration = scf.iterations[-1]
    # The convergence criterion was met at iteration 7 with DIIS error 2.17e-08
    assert last_iteration.diis_error is not None
    assert last_iteration.diis_error < 1e-05  # Below convergence threshold


# =============================================================================
# PARAMETRIZED TESTS: Q-Chem 5.4 vs 6.2 format compatibility
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_scf_results_present_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Contract test: verify SCF results are present in both Q-Chem 5.4 and 6.2 formats.
    Both versions should parse SCF data correctly despite format differences.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    assert parsed_data.scf is not None
    assert isinstance(parsed_data.scf, ScfResults)
    assert isinstance(parsed_data.scf.iterations, tuple)
    assert len(parsed_data.scf.iterations) > 0


@pytest.mark.regression
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_scf_converged_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Regression test: verify SCF converged status is identical in both versions.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    assert parsed_data.scf is not None
    assert parsed_data.scf.converged is EXPECTED_CONVERGED


@pytest.mark.regression
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_scf_number_of_iterations_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Regression test: verify exact number of SCF iterations in both versions.
    Both 5.4 and 6.2 should produce 7 iterations for this test case.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    assert parsed_data.scf is not None
    assert parsed_data.scf.n_iterations == EXPECTED_N_ITERATIONS


@pytest.mark.regression
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_scf_final_energy_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Regression test: verify exact final SCF energy in both versions.
    Both 5.4 and 6.2 should produce identical SCF energies.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    assert parsed_data.scf is not None
    assert parsed_data.scf.energy == pytest.approx(EXPECTED_SCF_ENERGY, abs=ENERGY_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "parsed_data_fixture,iter_idx,expected_iter,expected_energy,expected_diis",
    [
        ("parsed_qchem_54_h2o_sp_data", 0, 1, -75.0734525405, 3.82e-01),
        ("parsed_qchem_54_h2o_sp_data", 1, 2, -75.2973764603, 4.85e-02),
        ("parsed_qchem_54_h2o_sp_data", 2, 3, -75.3188175792, 1.63e-02),
        ("parsed_qchem_54_h2o_sp_data", 3, 4, -75.3207889290, 1.19e-03),
        ("parsed_qchem_54_h2o_sp_data", 4, 5, -75.3208073914, 1.25e-04),
        ("parsed_qchem_54_h2o_sp_data", 5, 6, -75.3208076904, 2.28e-05),
        ("parsed_qchem_54_h2o_sp_data", 6, 7, -75.3208077007, 2.17e-08),
        ("parsed_qchem_62_h2o_sp_data", 0, 1, -75.0734525440, 3.82e-01),
        ("parsed_qchem_62_h2o_sp_data", 1, 2, -75.2973764631, 4.85e-02),
        ("parsed_qchem_62_h2o_sp_data", 2, 3, -75.3188175820, 1.63e-02),
        ("parsed_qchem_62_h2o_sp_data", 3, 4, -75.3207889318, 1.19e-03),
        ("parsed_qchem_62_h2o_sp_data", 4, 5, -75.3208073942, 1.25e-04),
        ("parsed_qchem_62_h2o_sp_data", 5, 6, -75.3208076933, 2.28e-05),
        ("parsed_qchem_62_h2o_sp_data", 6, 7, -75.3208077035, 2.17e-08),
    ],
    ids=[
        "v54-iter1",
        "v54-iter2",
        "v54-iter3",
        "v54-iter4",
        "v54-iter5",
        "v54-iter6",
        "v54-iter7",
        "v62-iter1",
        "v62-iter2",
        "v62-iter3",
        "v62-iter4",
        "v62-iter5",
        "v62-iter6",
        "v62-iter7",
    ],
)
def test_scf_iteration_values_both_versions(
    parsed_data_fixture: str,
    request: pytest.FixtureRequest,
    iter_idx: int,
    expected_iter: int,
    expected_energy: float,
    expected_diis: float,
):
    """
    Regression test: verify exact SCF iteration values in both 5.4 and 6.2.
    Note: Energy values may differ slightly due to different rounding in output.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    scf = parsed_data.scf
    assert scf is not None
    assert len(scf.iterations) >= iter_idx + 1

    iteration = scf.iterations[iter_idx]
    assert iteration.iteration == expected_iter
    assert iteration.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)
    assert iteration.diis_error == pytest.approx(expected_diis, abs=DIIS_TOL)


@pytest.mark.regression
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_smd_results_present_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Regression test: verify SMD results are present in both versions.
    Both Q-Chem 5.4 and 6.2 should correctly parse SMD solvation model results.

    Note: Q-Chem 5.4 does not output G_PCM explicitly, so it may be None.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    assert parsed_data.smd is not None
    assert isinstance(parsed_data.smd, SmdResults)
    # G_CDS, G_ENP, and G_TOT are present in both versions
    assert parsed_data.smd.g_cds_kcal_mol is not None
    assert parsed_data.smd.g_enp_au is not None
    assert parsed_data.smd.g_tot_au is not None
    # G_PCM is only in 6.2, acceptable as None for 5.4
    if request.node.callspec.id == "qchem-6.2":
        assert parsed_data.smd.g_pcm_kcal_mol is not None


@pytest.mark.regression
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_smd_values_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Regression test: verify exact SMD values in both versions.
    SMD energies should be identical between 5.4 and 6.2.

    Note: G_PCM is only in 6.2; 5.4 format does not output it.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    smd = parsed_data.smd
    assert smd is not None

    # G_CDS, G_ENP, G_TOT should match in both versions
    assert smd.g_cds_kcal_mol == pytest.approx(EXPECTED_SMD_G_CDS_KCAL_MOL, abs=1e-4)
    assert smd.g_enp_au == pytest.approx(EXPECTED_SMD_G_ENP_AU, abs=ENERGY_TOL)
    assert smd.g_tot_au == pytest.approx(EXPECTED_SMD_G_TOT_AU, abs=ENERGY_TOL)

    # G_PCM only in 6.2
    if request.node.callspec.id == "qchem-6.2":
        assert smd.g_pcm_kcal_mol == pytest.approx(EXPECTED_SMD_G_PCM_KCAL_MOL, abs=1e-4)


@pytest.mark.regression
@pytest.mark.parametrize("parsed_data_fixture", ["parsed_qchem_54_h2o_sp_data", "parsed_qchem_62_h2o_sp_data"])
def test_final_energy_both_versions(parsed_data_fixture: str, request: pytest.FixtureRequest):
    """
    Regression test: verify final energy is set correctly in both versions.
    Final energy should equal SMD total free energy in both cases.
    """
    parsed_data = request.getfixturevalue(parsed_data_fixture)
    assert parsed_data.final_energy == pytest.approx(EXPECTED_TOTAL_ENERGY, abs=ENERGY_TOL)
