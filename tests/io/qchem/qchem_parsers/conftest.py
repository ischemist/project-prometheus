from collections.abc import Sequence
from pathlib import Path

import pytest

from calcflow.common.results import CalculationResult
from calcflow.io.qchem import parse_qchem_multi_job_output, parse_qchem_output

# =============================================================================
# FIXTURE FILES: Single source of truth
# =============================================================================
# Maps fixture names to their corresponding output file paths.
# Adding a new test fixture only requires adding one entry here.

# Single-job fixtures - parsed with parse_qchem_output()
FIXTURE_FILES = {
    "parsed_qchem_54_h2o_sp_data": "qchem/h2o/5.4-sp-smd.out",
    "parsed_qchem_62_h2o_sp_data": "qchem/h2o/6.2-sp-smd.out",
    "parsed_qchem_54_h2o_uks_tddft_data": "qchem/h2o/5.4-uks-tddft.out",
    "parsed_qchem_62_h2o_uks_tddft_data": "qchem/h2o/6.2-uks-tddft.out",
    "parsed_qchem_62_h2o_rks_tddft_data": "qchem/h2o/6.2-rks-tddft.out",
}

# Multi-job fixtures - parsed with parse_qchem_multi_job_output()
# Returns Sequence[CalculationResult]
MULTI_JOB_FIXTURE_FILES = {
    "parsed_qchem_54_h2o_mom_sp_multi": "qchem/h2o/5.4-mom-sp-smd.out",
    "parsed_qchem_62_h2o_mom_sp_multi": "qchem/h2o/6.2-mom-sp-smd.out",
    "parsed_qchem_54_h2o_mom_xas_multi": "qchem/h2o/5.4-mom-xas-smd.out",
    "parsed_qchem_62_h2o_mom_xas_multi": "qchem/h2o/6.2-mom-xas-smd.out",
}

# Individual job fixtures - extracts job1 from multi-job files for contract tests
# Returns CalculationResult (just the first job)
JOB1_FIXTURE_FILES = {
    "parsed_qchem_54_h2o_mom_sp_job1": "qchem/h2o/5.4-mom-sp-smd.out",
    "parsed_qchem_62_h2o_mom_sp_job1": "qchem/h2o/6.2-mom-sp-smd.out",
    "parsed_qchem_54_h2o_mom_xas_job1": "qchem/h2o/5.4-mom-xas-smd.out",
    "parsed_qchem_62_h2o_mom_xas_job1": "qchem/h2o/6.2-mom-xas-smd.out",
}

# Job 2 fixtures - extracts job2 from multi-job files (the MOM-enabled job)
# Returns CalculationResult (just the second job, which has MOM)
JOB2_FIXTURE_FILES = {
    "parsed_qchem_54_h2o_mom_sp_job2": "qchem/h2o/5.4-mom-sp-smd.out",
    "parsed_qchem_62_h2o_mom_sp_job2": "qchem/h2o/6.2-mom-sp-smd.out",
    "parsed_qchem_54_h2o_mom_xas_job2": "qchem/h2o/5.4-mom-xas-smd.out",
    "parsed_qchem_62_h2o_mom_xas_job2": "qchem/h2o/6.2-mom-xas-smd.out",
}

# Regression test fixtures - extracts job2 from multi-job XAS files
# Returns CalculationResult (the MOM-enabled XAS job with excitations)
XAS_REGRESSION_FIXTURE_FILES = {
    "parsed_qchem_62_mom_xas_smd_data": "qchem/h2o/6.2-mom-xas-smd.out",
}

# =============================================================================
# FIXTURE SPECIFICATIONS: Organized by parsed block
# =============================================================================
# Maps block names to fixtures that contain that block.
# Contract (and ONLY contract) tests should parametrize using FIXTURE_SPECS[block_name].

FIXTURE_SPECS = {
    # Blocks present in all fixtures (both SP and TDDFT)
    "geometry": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
    "scf": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
    "orbitals": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
    # Blocks present in SP calculations only
    "charges": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
    "multipole": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
    # TDA-only calculations (TDA excitations without full TDDFT)
    "tda_excitations": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    # Full TDDFT calculations (both TDA and TDDFT blocks present)
    "tddft_excitations": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    "tddft_gs_ref": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    "tddft_trans_dm": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    # UKS-specific blocks (unrestricted spin)
    "beta_orbitals": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    "tddft_unrel_dm": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    "tddft_nto": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    "tddft_nto_uks": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
    ],
    # MOM SCF block (present in MOM-enabled jobs)
    "mom_scf": [
        "parsed_qchem_54_h2o_mom_sp_job2",
        "parsed_qchem_62_h2o_mom_sp_job2",
        "parsed_qchem_54_h2o_mom_xas_job2",
        "parsed_qchem_62_h2o_mom_xas_job2",
    ],
    # Timing block (present in all fixtures)
    "timing": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
    # Finalization block (present in all fixtures - all concluded successfully with NORMAL status)
    "finalization": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
        "parsed_qchem_54_h2o_mom_sp_job1",
        "parsed_qchem_62_h2o_mom_sp_job1",
        "parsed_qchem_54_h2o_mom_xas_job1",
        "parsed_qchem_62_h2o_mom_xas_job1",
    ],
}


# =============================================================================
# SESSION-LEVEL CACHE FOR PARSED RESULTS
# =============================================================================

_parsed_cache: dict[str, CalculationResult] = {}
_multi_job_cache: dict[str, Sequence[CalculationResult]] = {}


# =============================================================================
# FIXTURE FACTORY: Dynamically create fixtures based on FIXTURE_FILES
# =============================================================================


def _create_qchem_fixture(fixture_name: str):
    """Factory function to create a session-scoped fixture for single-job files."""

    @pytest.fixture(scope="session", name=fixture_name)
    def _fixture(testing_data_path: Path) -> CalculationResult:
        if fixture_name not in _parsed_cache:
            file_path = testing_data_path / FIXTURE_FILES[fixture_name]
            _parsed_cache[fixture_name] = parse_qchem_output(file_path.read_text())
        return _parsed_cache[fixture_name]

    return _fixture


def _create_multi_job_fixture(fixture_name: str):
    """Factory function to create a session-scoped fixture for multi-job files."""

    @pytest.fixture(scope="session", name=fixture_name)
    def _fixture(testing_data_path: Path) -> Sequence[CalculationResult]:
        if fixture_name not in _multi_job_cache:
            file_path = testing_data_path / MULTI_JOB_FIXTURE_FILES[fixture_name]
            _multi_job_cache[fixture_name] = parse_qchem_multi_job_output(file_path.read_text())
        return _multi_job_cache[fixture_name]

    return _fixture


def _create_job1_fixture(fixture_name: str):
    """Factory function to create a session-scoped fixture that extracts job1 from multi-job files.

    Only parses the first job to avoid parsing errors in subsequent jobs that may not be
    fully implemented yet.
    """

    @pytest.fixture(scope="session", name=fixture_name)
    def _fixture(testing_data_path: Path) -> CalculationResult:
        if fixture_name not in _parsed_cache:
            file_path = testing_data_path / JOB1_FIXTURE_FILES[fixture_name]
            # Only parse the first job, avoiding errors from job 2+ which may have incomplete parsers
            jobs = parse_qchem_multi_job_output(file_path.read_text(), num_jobs=1)
            _parsed_cache[fixture_name] = jobs[0]  # Extract first (and only) job
        return _parsed_cache[fixture_name]

    return _fixture


def _create_job2_fixture(fixture_name: str):
    """Factory function to create a session-scoped fixture that extracts job2 from multi-job files.

    Parses both jobs and returns the second one (which is typically MOM-enabled).
    """

    @pytest.fixture(scope="session", name=fixture_name)
    def _fixture(testing_data_path: Path) -> CalculationResult:
        if fixture_name not in _parsed_cache:
            file_path = testing_data_path / JOB2_FIXTURE_FILES[fixture_name]
            # Parse both jobs, extract the second one (MOM-enabled)
            jobs = parse_qchem_multi_job_output(file_path.read_text(), num_jobs=2)
            _parsed_cache[fixture_name] = jobs[1]  # Extract second job (index 1)
        return _parsed_cache[fixture_name]

    return _fixture


def _create_xas_regression_fixture(fixture_name: str):
    """Factory function to create a session-scoped fixture for XAS regression tests.

    Extracts the second job (MOM-enabled XAS with excitations) from multi-job files.
    """

    @pytest.fixture(scope="session", name=fixture_name)
    def _fixture(testing_data_path: Path) -> CalculationResult:
        if fixture_name not in _parsed_cache:
            file_path = testing_data_path / XAS_REGRESSION_FIXTURE_FILES[fixture_name]
            # Parse both jobs, extract the second one (MOM-enabled with XAS excitations)
            jobs = parse_qchem_multi_job_output(file_path.read_text(), num_jobs=2)
            _parsed_cache[fixture_name] = jobs[1]  # Extract second job (index 1)
        return _parsed_cache[fixture_name]

    return _fixture


# Dynamically create all fixtures from FIXTURE_FILES
for fixture_name in FIXTURE_FILES:
    globals()[fixture_name] = _create_qchem_fixture(fixture_name)

for fixture_name in MULTI_JOB_FIXTURE_FILES:
    globals()[fixture_name] = _create_multi_job_fixture(fixture_name)

for fixture_name in JOB1_FIXTURE_FILES:
    globals()[fixture_name] = _create_job1_fixture(fixture_name)

for fixture_name in JOB2_FIXTURE_FILES:
    globals()[fixture_name] = _create_job2_fixture(fixture_name)

for fixture_name in XAS_REGRESSION_FIXTURE_FILES:
    globals()[fixture_name] = _create_xas_regression_fixture(fixture_name)


# =============================================================================
# INDIRECT FIXTURE: Universal parametrization
# =============================================================================


@pytest.fixture
def parsed_qchem_data(request: pytest.FixtureRequest) -> CalculationResult:
    """
    Universal parametrizable fixture that delegates to session-scoped fixtures.

    Use with FIXTURE_SPECS to parametrize tests:
        @pytest.mark.parametrize("parsed_qchem_data", FIXTURE_SPECS["block_name"], indirect=True)
        def test_something(parsed_qchem_data):
            ...
    """
    return request.getfixturevalue(request.param)
