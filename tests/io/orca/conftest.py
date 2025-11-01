from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.orca import parse_orca_output


@pytest.fixture(scope="session")
def parsed_orca_h2o_sp_data(testing_data_path: Path) -> CalculationResult:
    return parse_orca_output((testing_data_path / "orca" / "h2o" / "sp.out").read_text())
