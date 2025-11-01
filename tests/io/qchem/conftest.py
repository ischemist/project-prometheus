from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.qchem import parse_qchem_output

# =============================================================================
# FIXTURE LISTS: Organized by data content
# =============================================================================
# These lists define which fixtures contain which types of data.
# Use these in parametrized tests to avoid repeating fixture names.

# Fixtures with basic SP calculations (geometry, SCF, charges, multipole, orbitals)
FIXTURES_SP_ONLY = [
    "parsed_qchem_54_h2o_sp_data",
    "parsed_qchem_62_h2o_sp_data",
]

# Fixtures with TDDFT data (geometry, SCF, orbitals, multipole, TDDFT)
FIXTURES_TDDFT = [
    "parsed_qchem_54_h2o_uks_tddft_data",
    "parsed_qchem_62_h2o_uks_tddft_data",
    "parsed_qchem_62_h2o_rks_tddft_data",
]

# Fixtures with UKS TDDFT data only (for unrestricted/beta orbital tests)
FIXTURES_UKS_TDDFT = [
    "parsed_qchem_54_h2o_uks_tddft_data",
    "parsed_qchem_62_h2o_uks_tddft_data",
]

# Fixtures with RKS TDDFT data only
FIXTURES_RKS_TDDFT = [
    "parsed_qchem_62_h2o_rks_tddft_data",
]

# Combined fixtures: SP + TDDFT (for tests that work with both)
FIXTURES_SP_AND_TDDFT = FIXTURES_SP_ONLY + FIXTURES_TDDFT


# =============================================================================
# CONCRETE FIXTURES: Actual parsed data
# =============================================================================


@pytest.fixture(scope="session")
def parsed_qchem_54_h2o_sp_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "5.4-sp-smd.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_sp_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "6.2-sp-smd.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_54_h2o_uks_tddft_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "5.4-uks-tddft.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_uks_tddft_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "6.2-uks-tddft.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_rks_tddft_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "6.2-rks-tddft.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_54_h2o_mom_sp_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "5.4-mom-sp-smd.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_mom_sp_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "6.2-mom-sp-smd.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_54_h2o_mom_xas_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "5.4-mom-xas-smd.out").read_text())


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_mom_xas_data(testing_data_path: Path) -> CalculationResult:
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / "6.2-mom-xas-smd.out").read_text())


# =============================================================================
# INDIRECT FIXTURES: For parametrization
# =============================================================================


@pytest.fixture
def parsed_qchem_h2o_sp_data(request: pytest.FixtureRequest, testing_data_path: Path) -> CalculationResult:
    """Parametrizable fixture for SP calculations (5.4 and 6.2)."""
    fixture_name = request.param
    mapping = {
        "parsed_qchem_54_h2o_sp_data": "5.4-sp-smd.out",
        "parsed_qchem_62_h2o_sp_data": "6.2-sp-smd.out",
    }
    filename = mapping[fixture_name]
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / filename).read_text())


@pytest.fixture
def parsed_qchem_h2o_tddft_data(request: pytest.FixtureRequest, testing_data_path: Path) -> CalculationResult:
    """Parametrizable fixture for TDDFT calculations."""
    fixture_name = request.param
    # Map fixture names to file names
    mapping = {
        "parsed_qchem_54_h2o_uks_tddft_data": "5.4-uks-tddft.out",
        "parsed_qchem_62_h2o_uks_tddft_data": "6.2-uks-tddft.out",
        "parsed_qchem_62_h2o_rks_tddft_data": "6.2-rks-tddft.out",
    }
    filename = mapping[fixture_name]
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / filename).read_text())


@pytest.fixture
def parsed_qchem_h2o_sp_or_tddft_data(request: pytest.FixtureRequest, testing_data_path: Path) -> CalculationResult:
    """Parametrizable fixture for both SP and TDDFT calculations."""
    fixture_name = request.param
    mapping = {
        "parsed_qchem_54_h2o_sp_data": "5.4-sp-smd.out",
        "parsed_qchem_62_h2o_sp_data": "6.2-sp-smd.out",
        "parsed_qchem_54_h2o_uks_tddft_data": "5.4-uks-tddft.out",
        "parsed_qchem_62_h2o_uks_tddft_data": "6.2-uks-tddft.out",
        "parsed_qchem_62_h2o_rks_tddft_data": "6.2-rks-tddft.out",
    }
    filename = mapping[fixture_name]
    return parse_qchem_output((testing_data_path / "qchem" / "h2o" / filename).read_text())
