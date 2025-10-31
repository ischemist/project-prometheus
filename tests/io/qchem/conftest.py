from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.qchem import parse_qchem_sp


@pytest.fixture(scope="session")
def qchem_h2o_62_sp_output(testing_data_path: Path) -> str:
    """Fixture that reads the content of the QChem H2O SP output file."""
    file_path = testing_data_path / "qchem" / "h2o" / "6.2-sp-smd.out"
    return file_path.read_text()


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_sp_data(qchem_h2o_62_sp_output: str) -> CalculationResult:
    """Fixture that provides the fully parsed CalculationResult object."""
    return parse_qchem_sp(qchem_h2o_62_sp_output)


@pytest.fixture(scope="session")
def qchem_h2o_54_sp_output(testing_data_path: Path) -> str:
    """Fixture that reads the content of the QChem H2O SP output file."""
    file_path = testing_data_path / "qchem" / "h2o" / "5.4-sp-smd.out"
    return file_path.read_text()


@pytest.fixture(scope="session")
def parsed_qchem_54_h2o_sp_data(qchem_h2o_54_sp_output: str) -> CalculationResult:
    """Fixture that provides the fully parsed CalculationResult object."""
    return parse_qchem_sp(qchem_h2o_54_sp_output)


@pytest.fixture(scope="session")
def qchem_h2o_62_uks_tddft_output(testing_data_path: Path) -> str:
    """Fixture that reads the content of the QChem H2O SP output file."""
    file_path = testing_data_path / "qchem" / "h2o" / "6.2-uks-tddft.out"
    return file_path.read_text()


@pytest.fixture(scope="session")
def parsed_qchem_62_h2o_uks_tddft_data(qchem_h2o_62_uks_tddft_output: str) -> CalculationResult:
    """Fixture that provides the fully parsed CalculationResult object."""
    return parse_qchem_sp(qchem_h2o_62_uks_tddft_output)
