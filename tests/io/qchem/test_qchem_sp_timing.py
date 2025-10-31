"""
Tests for the QChem timing block parser (CPU and wall time).

These tests verify that timing information is correctly parsed, including:
- Total wall time extraction
- Total CPU time extraction from "Total job time: Xs(wall), Ys(cpu)" line
- Proper data structure and types
"""

import pytest

from calcflow.common.models import CalculationResult, TimingResults

TIME_TOL = 0.001  # seconds


@pytest.mark.contract
def test_timing_structure_exists(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that timing field exists and is of correct type.
    """
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert isinstance(parsed_qchem_54_h2o_sp_data.timing, TimingResults)


@pytest.mark.contract
def test_wall_time_type(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Contract test: verify total_wall_time_seconds is populated and is a float.
    """
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert parsed_qchem_54_h2o_sp_data.timing.total_wall_time_seconds is not None
    assert isinstance(parsed_qchem_54_h2o_sp_data.timing.total_wall_time_seconds, float)


@pytest.mark.contract
def test_cpu_time_type(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Contract test: verify total_cpu_time_seconds is populated and is a float.
    """
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert parsed_qchem_54_h2o_sp_data.timing.total_cpu_time_seconds is not None
    assert isinstance(parsed_qchem_54_h2o_sp_data.timing.total_cpu_time_seconds, float)


@pytest.mark.regression
def test_wall_time_value(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Regression test: verify wall time has correct value.
    QChem output: "Total job time:  0.77s(wall), 0.22s(cpu)"
    Expected wall time: 0.77 seconds
    """
    expected_wall_time = 0.77
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert parsed_qchem_54_h2o_sp_data.timing.total_wall_time_seconds == pytest.approx(expected_wall_time, abs=TIME_TOL)


@pytest.mark.regression
def test_cpu_time_value(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Regression test: verify CPU time has correct value.
    QChem output: "Total job time:  0.77s(wall), 0.22s(cpu)"
    Expected CPU time: 0.22 seconds
    """
    expected_cpu_time = 0.22
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert parsed_qchem_54_h2o_sp_data.timing.total_cpu_time_seconds == pytest.approx(expected_cpu_time, abs=TIME_TOL)


@pytest.mark.regression
def test_module_times_is_none(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Regression test: verify that module_times is None for QChem
    (QChem doesn't provide module-specific timing).
    """
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert parsed_qchem_54_h2o_sp_data.timing.module_times is None


@pytest.mark.integration
def test_timing_fields_populated(parsed_qchem_54_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that timing-related fields are properly populated
    in the final CalculationResult.
    """
    assert parsed_qchem_54_h2o_sp_data.timing is not None
    assert parsed_qchem_54_h2o_sp_data.timing.total_wall_time_seconds is not None
    assert parsed_qchem_54_h2o_sp_data.timing.total_cpu_time_seconds is not None
