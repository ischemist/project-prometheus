from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.orca import parse_orca_sp


@pytest.fixture(scope="session")
def orca_h2o_sp_output(testing_data_path: Path) -> str:
    """Fixture that reads the content of the ORCA H2O SP output file."""
    file_path = testing_data_path / "orca" / "h2o" / "sp.out"
    return file_path.read_text()


@pytest.fixture(scope="session")
def parsed_orca_h2o_sp_data(orca_h2o_sp_output: str) -> CalculationResult:
    """Fixture that provides the fully parsed CalculationResult object."""
    return parse_orca_sp(orca_h2o_sp_output)
