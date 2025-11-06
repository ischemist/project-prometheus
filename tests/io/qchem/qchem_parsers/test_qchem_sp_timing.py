"""
Tests for the QChem timing block parser (CPU and wall time).

These tests verify that timing information is correctly parsed, including:
- Total wall time extraction
- Total CPU time extraction from "Total job time: Xs(wall), Ys(cpu)" line
- Proper data structure and types
"""

import pytest

from calcflow.common.results import CalculationResult, TimingResults
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

TIME_TOL = 0.001  # seconds


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["timing"],
    indirect=True,
)
def test_timing_structure_exists(parsed_qchem_data: CalculationResult):
    """
    Contract test: verify that timing field exists and is of correct type.
    """
    assert parsed_qchem_data.timing is not None
    assert isinstance(parsed_qchem_data.timing, TimingResults)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["timing"],
    indirect=True,
)
def test_wall_time_type(parsed_qchem_data: CalculationResult):
    """
    Contract test: verify total_wall_time_seconds is populated and is a float.
    """
    assert parsed_qchem_data.timing is not None
    assert parsed_qchem_data.timing.total_wall_time_seconds is not None
    assert isinstance(parsed_qchem_data.timing.total_wall_time_seconds, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["timing"],
    indirect=True,
)
def test_cpu_time_type(parsed_qchem_data: CalculationResult):
    """
    Contract test: verify total_cpu_time_seconds is populated and is a float.
    """
    assert parsed_qchem_data.timing is not None
    assert parsed_qchem_data.timing.total_cpu_time_seconds is not None
    assert isinstance(parsed_qchem_data.timing.total_cpu_time_seconds, float)


@pytest.mark.regression
@pytest.mark.parametrize(
    "parsed_qchem_data,expected_wall_time,expected_cpu_time",
    [
        ("parsed_qchem_54_h2o_sp_data", 0.77, 0.22),
        ("parsed_qchem_62_h2o_sp_data", 0.96, 0.41),
    ],
    indirect=["parsed_qchem_data"],
)
def test_sp_time_values(parsed_qchem_data: CalculationResult, expected_wall_time: float, expected_cpu_time: float):
    """
    Regression test: verify wall and CPU time have correct values for SP calculations.
    """
    assert parsed_qchem_data.timing is not None
    assert parsed_qchem_data.timing.total_wall_time_seconds == pytest.approx(expected_wall_time, abs=TIME_TOL)
    assert parsed_qchem_data.timing.total_cpu_time_seconds == pytest.approx(expected_cpu_time, abs=TIME_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "parsed_qchem_data,expected_wall_time,expected_cpu_time",
    [
        ("parsed_qchem_54_h2o_uks_tddft_data", 11.03, 9.68),
        ("parsed_qchem_62_h2o_uks_tddft_data", 19.49, 16.96),
        ("parsed_qchem_62_h2o_rks_tddft_data", 14.09, 11.75),
    ],
    indirect=["parsed_qchem_data"],
    ids=["tddft-uks-5.4", "tddft-uks-6.2", "tddft-rks-6.2"],
)
def test_tddft_time_values(parsed_qchem_data: CalculationResult, expected_wall_time: float, expected_cpu_time: float):
    """
    Regression test: verify wall and CPU time have correct values for TDDFT calculations.
    """
    assert parsed_qchem_data.timing is not None
    assert parsed_qchem_data.timing.total_wall_time_seconds == pytest.approx(expected_wall_time, abs=TIME_TOL)
    assert parsed_qchem_data.timing.total_cpu_time_seconds == pytest.approx(expected_cpu_time, abs=TIME_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["timing"],
    indirect=True,
)
def test_module_times_is_none(parsed_qchem_data: CalculationResult):
    """
    Regression test: verify that module_times is None for QChem
    (QChem doesn't provide module-specific timing).
    """
    assert parsed_qchem_data.timing is not None
    assert parsed_qchem_data.timing.module_times is None


@pytest.mark.integration
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["timing"],
    indirect=True,
)
def test_timing_fields_populated(parsed_qchem_data: CalculationResult):
    """
    Integration test: verify that timing-related fields are properly populated
    in the final CalculationResult.
    """
    assert parsed_qchem_data.timing is not None
    assert parsed_qchem_data.timing.total_wall_time_seconds is not None
    assert parsed_qchem_data.timing.total_cpu_time_seconds is not None
