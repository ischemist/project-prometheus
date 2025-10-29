"""
integration and regression tests for the top-level orca single point parser.

these tests operate on a full, real-world orca output file and verify that
the final `calculationresult` object is assembled correctly.
"""

import pytest

from calcflow.common.models import Atom, CalculationResult

# a tolerance for comparing floating point geometry coordinates.
GEOM_TOL = 1e-6


@pytest.mark.integration
def test_orca_sp_parsing_completes(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that parsing a standard orca sp output file runs to completion and
    returns the correct top-level object type.
    """
    assert parsed_orca_h2o_sp_data is not None
    assert isinstance(parsed_orca_h2o_sp_data, CalculationResult)


@pytest.mark.integration
def test_orca_sp_termination_status(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that the parser correctly identifies a normally terminated run.
    """
    assert parsed_orca_h2o_sp_data.termination_status == "NORMAL"


@pytest.mark.integration
def test_orca_sp_input_geometry_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests the structural integrity of the parsed input geometry.
    it checks for the correct number of atoms and their types, but not their
    exact coordinates (which is a regression test).
    """
    geom = parsed_orca_h2o_sp_data.input_geometry
    assert geom is not None
    assert len(geom) == 3
    assert all(isinstance(atom, Atom) for atom in geom)

    symbols = [atom.symbol for atom in geom]
    assert symbols == ["H", "O", "H"]


@pytest.mark.regression
def test_orca_sp_input_geometry_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests for the exact numerical values of the input geometry. this is a
    regression test because it's sensitive to small formatting changes.
    """
    expected_coords = {
        "H": (1.364990, 1.693850, -0.197480),
        "O": (2.328770, 1.562940, -0.041680),
        "H_2": (2.702440, 1.311570, -0.916650),  # using a temp unique key
    }

    parsed_geom = parsed_orca_h2o_sp_data.input_geometry

    # this is a bit verbose but avoids ambiguity with the two H atoms
    h1 = parsed_geom[0]
    o = parsed_geom[1]
    h2 = parsed_geom[2]

    assert h1.symbol == "H"
    assert h1.x == pytest.approx(expected_coords["H"][0], abs=GEOM_TOL)
    assert h1.y == pytest.approx(expected_coords["H"][1], abs=GEOM_TOL)
    assert h1.z == pytest.approx(expected_coords["H"][2], abs=GEOM_TOL)

    assert o.symbol == "O"
    assert o.x == pytest.approx(expected_coords["O"][0], abs=GEOM_TOL)
    assert o.y == pytest.approx(expected_coords["O"][1], abs=GEOM_TOL)
    assert o.z == pytest.approx(expected_coords["O"][2], abs=GEOM_TOL)

    assert h2.symbol == "H"
    assert h2.x == pytest.approx(expected_coords["H_2"][0], abs=GEOM_TOL)
    assert h2.y == pytest.approx(expected_coords["H_2"][1], abs=GEOM_TOL)
    assert h2.z == pytest.approx(expected_coords["H_2"][2], abs=GEOM_TOL)
