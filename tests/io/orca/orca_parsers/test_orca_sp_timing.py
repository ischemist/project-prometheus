"""
Tests for the ORCA timing block parser (total wall time and module-specific times).

These tests verify that timing information is correctly parsed, including:
- Total wall time extraction from "TOTAL RUN TIME" line
- Module-specific timing extraction (Startup, SCF iterations, Property calculations, etc.)
- Proper data structure and types
"""

import pytest

from calcflow.common.results import CalculationResult, TimingResults
from tests.io.orca.orca_parsers.conftest import FIXTURE_SPECS

TIME_TOL = 0.001  # seconds


@pytest.mark.contract
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_timing_structure_exists(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify that timing field exists and is of correct type.
    """
    assert parsed_orca_h2o_sp_data.timing is not None
    assert isinstance(parsed_orca_h2o_sp_data.timing, TimingResults)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_total_wall_time_type(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify total_wall_time_seconds is populated and is a float.
    """
    assert parsed_orca_h2o_sp_data.timing is not None
    assert parsed_orca_h2o_sp_data.timing.total_wall_time_seconds is not None
    assert isinstance(parsed_orca_h2o_sp_data.timing.total_wall_time_seconds, float)


@pytest.mark.contract
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_module_times_type(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Contract test: verify module_times is a Mapping[str, float] if populated.
    """
    assert parsed_orca_h2o_sp_data.timing is not None
    if parsed_orca_h2o_sp_data.timing.module_times is not None:
        # Should be a mapping where keys are strings and values are floats
        for key, value in parsed_orca_h2o_sp_data.timing.module_times.items():
            assert isinstance(key, str)
            assert isinstance(value, float)


@pytest.mark.regression
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_total_wall_time_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify total wall time has correct value.
    ORCA output: "TOTAL RUN TIME: 0 days 0 hours 1 minutes 31 seconds 640 msec"
    Expected: 1*60 + 31 + 0.640 = 91.640 seconds
    """
    expected_wall_time = 91.640
    assert parsed_orca_h2o_sp_data.timing is not None
    assert parsed_orca_h2o_sp_data.timing.total_wall_time_seconds == pytest.approx(expected_wall_time, abs=TIME_TOL)


@pytest.mark.regression
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_module_times_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify module-specific timing values.
    From ORCA output:
    - Startup calculation: 4.753 sec
    - SCF iterations: 76.454 sec
    - Property calculations: 2.003 sec
    """
    assert parsed_orca_h2o_sp_data.timing is not None
    assert parsed_orca_h2o_sp_data.timing.module_times is not None

    expected_modules = {
        "Startup calculation": 4.753,
        "SCF iterations": 76.454,
        "Property calculations": 2.003,
    }

    for module_name, expected_time in expected_modules.items():
        assert module_name in parsed_orca_h2o_sp_data.timing.module_times
        assert parsed_orca_h2o_sp_data.timing.module_times[module_name] == pytest.approx(expected_time, abs=TIME_TOL)


@pytest.mark.regression
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_cpu_time_is_none(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Regression test: verify that CPU time is None for ORCA (ORCA only reports wall time).
    """
    assert parsed_orca_h2o_sp_data.timing is not None
    assert parsed_orca_h2o_sp_data.timing.total_cpu_time_seconds is None


@pytest.mark.integration
@pytest.mark.parametrize("parsed_orca_h2o_sp_data", FIXTURE_SPECS["timing"], indirect=True)
def test_timing_fields_populated(parsed_orca_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that timing-related fields are properly populated
    in the final CalculationResult.
    """
    assert parsed_orca_h2o_sp_data.timing is not None
    assert parsed_orca_h2o_sp_data.timing.total_wall_time_seconds is not None
    assert parsed_orca_h2o_sp_data.timing.module_times is not None
    assert len(parsed_orca_h2o_sp_data.timing.module_times) > 0
