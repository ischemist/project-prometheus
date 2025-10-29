"""
Tests for the QChem geometry parser (both input $molecule block and final orientation).

These tests verify that the geometry parser correctly extracts atomic coordinates
from both the input $molecule section and the final "Standard Nuclear Orientation"
output block.
"""

import pytest

from calcflow.common.models import Atom, CalculationResult
from calcflow.io.qchem.blocks.geometry import GeometryParser
from calcflow.io.state import ParseState

# Tolerance for coordinate precision (7 decimal places)
COORD_TOL = 1e-7


# =============================================================================
# UNIT TESTS: GeometryParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_geometry_parser_matches_molecule_block():
    """
    Unit test: verify GeometryParser.matches() recognizes $molecule start line.
    """
    parser = GeometryParser()
    state = ParseState(raw_output="")

    molecule_line = "$molecule"
    assert parser.matches(molecule_line, state) is True


@pytest.mark.unit
def test_geometry_parser_matches_molecule_block_with_whitespace():
    """
    Unit test: verify GeometryParser.matches() handles whitespace before $molecule.
    """
    parser = GeometryParser()
    state = ParseState(raw_output="")

    molecule_line = "  $molecule  "
    assert parser.matches(molecule_line, state) is True


@pytest.mark.unit
def test_geometry_parser_matches_standard_orientation():
    """
    Unit test: verify GeometryParser.matches() recognizes Standard Nuclear Orientation.
    """
    parser = GeometryParser()
    state = ParseState(raw_output="")

    orientation_line = "             Standard Nuclear Orientation (Angstroms)"
    assert parser.matches(orientation_line, state) is True


@pytest.mark.unit
def test_geometry_parser_skips_molecule_if_already_parsed():
    """
    Unit test: verify GeometryParser.matches() returns False when input geometry
    is already parsed (parsed_geometry flag is True).
    """
    parser = GeometryParser()
    state = ParseState(raw_output="")
    state.parsed_geometry = True

    molecule_line = "$molecule"
    assert parser.matches(molecule_line, state) is False


@pytest.mark.unit
def test_geometry_parser_skips_orientation_if_already_parsed():
    """
    Unit test: verify GeometryParser.matches() returns False when final geometry
    is already parsed (final_geometry is not None).
    """
    parser = GeometryParser()
    state = ParseState(raw_output="")
    # Simulate that final_geometry was already parsed
    state.final_geometry = (Atom(symbol="H", x=0.0, y=0.0, z=0.0),)

    orientation_line = "             Standard Nuclear Orientation (Angstroms)"
    assert parser.matches(orientation_line, state) is False


@pytest.mark.unit
def test_geometry_parser_ignores_non_geometry_lines():
    """
    Unit test: verify GeometryParser.matches() returns False for non-geometry lines.
    """
    parser = GeometryParser()
    state = ParseState(raw_output="")

    # Random calculation output lines
    assert parser.matches("SCF converges when DIIS error is below 1.0e-05", state) is False
    assert parser.matches("H        1.36499000      1.69385000     -0.19748000", state) is False
    assert parser.matches("Random line from output", state) is False


# =============================================================================
# CONTRACT TESTS: Geometry structure and type validation
# =============================================================================


@pytest.mark.contract
def test_input_geometry_is_tuple_of_atoms(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify input_geometry is a tuple of Atom objects.
    """
    assert isinstance(parsed_qchem_h2o_sp_data.input_geometry, tuple)
    assert len(parsed_qchem_h2o_sp_data.input_geometry) > 0
    assert all(isinstance(atom, Atom) for atom in parsed_qchem_h2o_sp_data.input_geometry)


@pytest.mark.contract
def test_final_geometry_is_tuple_of_atoms(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify final_geometry is a tuple of Atom objects.
    """
    assert isinstance(parsed_qchem_h2o_sp_data.final_geometry, tuple)
    assert len(parsed_qchem_h2o_sp_data.final_geometry) > 0
    assert all(isinstance(atom, Atom) for atom in parsed_qchem_h2o_sp_data.final_geometry)


@pytest.mark.contract
def test_input_geometry_has_three_atoms(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify input_geometry has exactly 3 atoms for H2O.
    """
    assert len(parsed_qchem_h2o_sp_data.input_geometry) == 3


@pytest.mark.contract
def test_final_geometry_has_three_atoms(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify final_geometry has exactly 3 atoms for H2O.
    """
    assert len(parsed_qchem_h2o_sp_data.final_geometry) == 3


@pytest.mark.contract
def test_input_geometry_atom_symbols(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify input_geometry has correct atom symbols (H, O, H).
    """
    symbols = [atom.symbol for atom in parsed_qchem_h2o_sp_data.input_geometry]
    assert symbols == ["H", "O", "H"]


@pytest.mark.contract
def test_final_geometry_atom_symbols(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify final_geometry has correct atom symbols (H, O, H).
    """
    symbols = [atom.symbol for atom in parsed_qchem_h2o_sp_data.final_geometry]
    assert symbols == ["H", "O", "H"]


@pytest.mark.contract
def test_input_geometry_coordinates_are_floats(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify all input coordinates are floats.
    """
    for atom in parsed_qchem_h2o_sp_data.input_geometry:
        assert isinstance(atom.x, float)
        assert isinstance(atom.y, float)
        assert isinstance(atom.z, float)


@pytest.mark.contract
def test_final_geometry_coordinates_are_floats(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Contract test: verify all final coordinates are floats.
    """
    for atom in parsed_qchem_h2o_sp_data.final_geometry:
        assert isinstance(atom.x, float)
        assert isinstance(atom.y, float)
        assert isinstance(atom.z, float)


# =============================================================================
# REGRESSION TESTS: Exact coordinate values
# =============================================================================


@pytest.mark.regression
def test_input_geometry_h1_x_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify first hydrogen X coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[0].x == pytest.approx(1.36499, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_h1_y_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify first hydrogen Y coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[0].y == pytest.approx(1.69385, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_h1_z_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify first hydrogen Z coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[0].z == pytest.approx(-0.19748, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_oxygen_x_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify oxygen X coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[1].x == pytest.approx(2.32877, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_oxygen_y_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify oxygen Y coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[1].y == pytest.approx(1.56294, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_oxygen_z_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify oxygen Z coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[1].z == pytest.approx(-0.04168, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_h2_x_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify second hydrogen X coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[2].x == pytest.approx(2.70244, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_h2_y_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify second hydrogen Y coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[2].y == pytest.approx(1.31157, abs=COORD_TOL)


@pytest.mark.regression
def test_input_geometry_h2_z_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify second hydrogen Z coordinate."""
    assert parsed_qchem_h2o_sp_data.input_geometry[2].z == pytest.approx(-0.91665, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_h1_x_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify first hydrogen X coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[0].x == pytest.approx(1.36499, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_h1_y_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify first hydrogen Y coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[0].y == pytest.approx(1.69385, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_h1_z_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify first hydrogen Z coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[0].z == pytest.approx(-0.19748, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_oxygen_x_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify oxygen X coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[1].x == pytest.approx(2.32877, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_oxygen_y_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify oxygen Y coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[1].y == pytest.approx(1.56294, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_oxygen_z_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify oxygen Z coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[1].z == pytest.approx(-0.04168, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_h2_x_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify second hydrogen X coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[2].x == pytest.approx(2.70244, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_h2_y_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify second hydrogen Y coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[2].y == pytest.approx(1.31157, abs=COORD_TOL)


@pytest.mark.regression
def test_final_geometry_h2_z_coordinate(parsed_qchem_h2o_sp_data: CalculationResult):
    """Regression test: verify second hydrogen Z coordinate in final geometry."""
    assert parsed_qchem_h2o_sp_data.final_geometry[2].z == pytest.approx(-0.91665, abs=COORD_TOL)


# =============================================================================
# INTEGRATION TESTS: Full geometry parsing
# =============================================================================


@pytest.mark.integration
def test_both_geometries_parsed_together(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that both input and final geometries are parsed
    from the same output file.
    """
    assert parsed_qchem_h2o_sp_data.input_geometry is not None
    assert parsed_qchem_h2o_sp_data.final_geometry is not None
    assert len(parsed_qchem_h2o_sp_data.input_geometry) == 3
    assert len(parsed_qchem_h2o_sp_data.final_geometry) == 3


@pytest.mark.integration
def test_geometries_match_for_single_point(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that input and final geometries are identical
    for a single-point calculation (no optimization).
    """
    input_coords = [(a.x, a.y, a.z) for a in parsed_qchem_h2o_sp_data.input_geometry]
    final_coords = [(a.x, a.y, a.z) for a in parsed_qchem_h2o_sp_data.final_geometry]

    for inp, final in zip(input_coords, final_coords, strict=True):
        assert inp[0] == pytest.approx(final[0], abs=COORD_TOL)
        assert inp[1] == pytest.approx(final[1], abs=COORD_TOL)
        assert inp[2] == pytest.approx(final[2], abs=COORD_TOL)


@pytest.mark.integration
def test_geometry_in_calculation_result(parsed_qchem_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that the CalculationResult object correctly
    contains the parsed geometry data.
    """
    # CalculationResult should have input_geometry as required field
    assert hasattr(parsed_qchem_h2o_sp_data, "input_geometry")
    assert hasattr(parsed_qchem_h2o_sp_data, "final_geometry")

    # Both should be populated
    assert parsed_qchem_h2o_sp_data.input_geometry
    assert parsed_qchem_h2o_sp_data.final_geometry

    # Should contain actual Atom objects
    for atom in parsed_qchem_h2o_sp_data.input_geometry:
        assert isinstance(atom, Atom)
        assert atom.symbol in ["H", "O", "C", "N"]  # Common elements
        assert isinstance(atom.x, float)
        assert isinstance(atom.y, float)
        assert isinstance(atom.z, float)
