from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.qchem import parse_qchem_output

# =============================================================================
# FIXTURE SPECIFICATIONS: Organized by parsed block
# =============================================================================
# Single source of truth: maps block names to fixtures that contain that block.
# Tests should parametrize using FIXTURE_SPECS[block_name].

FIXTURE_SPECS = {
    # Blocks present in all fixtures (both SP and TDDFT)
    "geometry": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    "scf": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    "orbitals": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    # Blocks present in SP calculations only
    "charges": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
    ],
    "multipole": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
    ],
    # Blocks present in TDDFT calculations only
    "tddft_excitations": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    "tddft_gs_ref": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    "tddft_trans_dm": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    # UKS-specific blocks (unrestricted spin)
    "beta_orbitals": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
    ],
    "tddft_unrel_dm": [
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
    ],
    # RKS-specific blocks
    "tddft_nto": [
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
    # Timing block (present in all fixtures)
    "timing": [
        "parsed_qchem_54_h2o_sp_data",
        "parsed_qchem_62_h2o_sp_data",
        "parsed_qchem_54_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_uks_tddft_data",
        "parsed_qchem_62_h2o_rks_tddft_data",
    ],
}


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


@pytest.fixture
def timing_fixture(request: pytest.FixtureRequest, testing_data_path: Path) -> CalculationResult:
    """Parametrizable fixture for all timing-related tests."""
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
