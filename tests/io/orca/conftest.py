from pathlib import Path

import pytest

from calcflow.common.results import CalculationResult
from calcflow.io.orca import parse_orca_output

# =============================================================================
# FIXTURE SPECIFICATIONS: Organized by parsed block
# =============================================================================
# Single source of truth: maps block names to fixtures that contain that block.
# Tests should parametrize using FIXTURE_SPECS[block_name].

FIXTURE_SPECS = {
    # Timing block (present in all fixtures)
    "timing": [
        "parsed_orca_h2o_sp_data",
    ],
}


# =============================================================================
# CONCRETE FIXTURES: Actual parsed data
# =============================================================================


@pytest.fixture(scope="session")
def parsed_orca_h2o_sp_data(testing_data_path: Path) -> CalculationResult:
    return parse_orca_output((testing_data_path / "orca" / "h2o" / "sp.out").read_text())
