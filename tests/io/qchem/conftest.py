from pathlib import Path

import pytest

from calcflow.common.models import CalculationResult
from calcflow.io.qchem import parse_qchem_output


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
