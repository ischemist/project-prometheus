from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.qchem import parse_qchem_output

# =============================================================================
# FIXTURE FILES: Single source of truth
# =============================================================================
# Maps fixture names to their corresponding output file paths.
# Adding a new test fixture only requires adding one entry here.

FIXTURE_FILES = {
    "parsed_qchem_54_h2o_sp_data": "qchem/h2o/5.4-sp-smd.out",
    "parsed_qchem_62_h2o_sp_data": "qchem/h2o/6.2-sp-smd.out",
    "parsed_qchem_54_h2o_uks_tddft_data": "qchem/h2o/5.4-uks-tddft.out",
    "parsed_qchem_62_h2o_uks_tddft_data": "qchem/h2o/6.2-uks-tddft.out",
    "parsed_qchem_62_h2o_rks_tddft_data": "qchem/h2o/6.2-rks-tddft.out",
}

# =============================================================================
# FIXTURE SPECIFICATIONS: Organized by parsed block
# =============================================================================
# Maps block names to fixtures that contain that block.
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
# SESSION-LEVEL CACHE FOR PARSED RESULTS
# =============================================================================

_parsed_cache: dict[str, CalculationResult] = {}


# =============================================================================
# FIXTURE FACTORY: Dynamically create fixtures based on FIXTURE_FILES
# =============================================================================


def _create_qchem_fixture(fixture_name: str):
    """Factory function to create a session-scoped fixture for a given fixture name."""

    @pytest.fixture(scope="session", name=fixture_name)
    def _fixture(testing_data_path: Path) -> CalculationResult:
        if fixture_name not in _parsed_cache:
            file_path = testing_data_path / FIXTURE_FILES[fixture_name]
            _parsed_cache[fixture_name] = parse_qchem_output(file_path.read_text())
        return _parsed_cache[fixture_name]

    return _fixture


# Dynamically create all fixtures from FIXTURE_FILES
for fixture_name in FIXTURE_FILES:
    globals()[fixture_name] = _create_qchem_fixture(fixture_name)


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
