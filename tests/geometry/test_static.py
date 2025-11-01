import logging
from pathlib import Path

import pytest

from calcflow.common.exceptions import ParsingError, ValidationError
from calcflow.geometry.static import Atom, Geometry, _parse_energy_from_comment
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)

# --- Fixtures ---


@pytest.fixture
def h2o_atoms() -> tuple[Atom, ...]:
    return (
        Atom(symbol="O", x=0.0, y=0.0, z=0.1173),
        Atom(symbol="H", x=0.0, y=0.7572, z=-0.4692),
        Atom(symbol="H", x=0.0, y=-0.7572, z=-0.4692),
    )


@pytest.fixture
def h2o_geometry(h2o_atoms: tuple[Atom, ...]) -> Geometry:
    return Geometry(comment="Water molecule", atoms=h2o_atoms)


# --- Unit Tests ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "comment, expected_energy",
    [
        ("Coordinates from ORCA-job opt E -981.614502119", -981.614502119),
        ("Energy = -100.5 Hartree", -100.5),
        ("E: -200.123456", -200.123456),
        ("energy=-300.789e-2", -3.00789),
        ("Just a comment", None),
        ("E = not_a_number", None),
    ],
)
def test_parse_energy_from_comment(comment: str, expected_energy: float | None):
    """Tests the standalone energy parsing helper function."""
    assert _parse_energy_from_comment(comment) == expected_energy


# --- Contract Tests (Pydantic Model Behavior) ---


@pytest.mark.contract
def test_atom_model_validation():
    """Tests the validation logic within the Atom model."""
    # Valid symbol (must be capitalized)
    atom = Atom(symbol="H", x=0, y=0, z=0)
    assert atom.symbol == "H"

    # Invalid capitalization
    with pytest.raises(ValidationError, match="element symbol 'h' must be capitalized"):
        Atom(symbol="h", x=0, y=0, z=0)

    # Invalid symbol
    with pytest.raises(ValidationError, match="unknown element symbol: 'Xx'"):
        Atom(symbol="Xx", x=0, y=0, z=0)


@pytest.mark.contract
def test_geometry_computed_fields(h2o_geometry: Geometry):
    """Tests the computed fields of the Geometry model."""
    assert h2o_geometry.num_atoms == 3
    assert h2o_geometry.energy is None  # Comment has no energy

    geom_with_energy = Geometry(
        comment="E = -76.4 Hartree",
        atoms=h2o_geometry.atoms,
    )
    assert geom_with_energy.energy == pytest.approx(-76.4)


@pytest.mark.contract
def test_geometry_cached_properties(h2o_geometry: Geometry):
    """Tests cached properties like unique_elements and total_nuclear_charge."""
    assert h2o_geometry.unique_elements == {"H", "O"}
    assert h2o_geometry.total_nuclear_charge == 10  # 8 (O) + 1 (H) + 1 (H)


@pytest.mark.contract
def test_geometry_total_nuclear_charge_unknown_element(h2o_atoms: tuple[Atom, ...]):
    """tests that creating an atom with an unknown symbol fails."""
    with pytest.raises(ValidationError, match="unknown element symbol: 'Xx'"):
        Atom(symbol="Xx", x=0, y=0, z=0)


@pytest.mark.contract
def test_geometry_to_xyz_str(h2o_geometry: Geometry):
    """Tests the string representation of the Geometry model."""
    output = h2o_geometry.to_xyz_str()
    lines = output.strip().split("\n")
    assert lines[0] == "3"
    assert lines[1] == "Water molecule"
    assert "O        0.00000000      0.00000000      0.11730000" in lines[2]
    assert "H        0.00000000      0.75720000     -0.46920000" in lines[3]
    assert "H        0.00000000     -0.75720000     -0.46920000" in lines[4]


@pytest.mark.contract
@pytest.mark.parametrize(
    "lines, error_match",
    [
        (["1"], "needs at least 2 lines"),
        ("not_an_int\nComment".splitlines(), "first line must be an integer"),
        ("2\nComment\nH 0 0 0".splitlines(), r"declared atom count \(2\) does not match actual atom lines found \(1\)"),
        ("1\nComment\nH 0 0".splitlines(), "expected 4 columns"),
        ("1\nComment\nH 0 bad_coord 0".splitlines(), "could not parse atom"),
    ],
)
def test_geometry_from_lines_validation(lines: list[str], error_match: str):
    """Tests that Geometry.from_lines raises ParsingError for malformed input."""
    with pytest.raises(ParsingError, match=error_match):
        Geometry.from_lines(lines)


# --- Integration Tests (File I/O) ---


@pytest.mark.integration
def test_geometry_from_xyz_file(tmp_path: Path):
    """Tests loading a Geometry from a valid XYZ file."""
    content = """3
Water molecule
O  0.0  0.0  0.1173
H  0.0  0.7572 -0.4692
H  0.0 -0.7572 -0.4692
"""
    file_path = tmp_path / "water.xyz"
    file_path.write_text(content)

    geom = Geometry.from_xyz_file(file_path)
    assert geom.num_atoms == 3
    assert geom.atoms[0].symbol == "O"
    assert geom.atoms[1].x == pytest.approx(0.0)


@pytest.mark.integration
def test_geometry_to_xyz_file(h2o_geometry: Geometry, tmp_path: Path):
    """Tests writing a Geometry to an XYZ file and reading it back."""
    file_path = tmp_path / "output.xyz"
    h2o_geometry.to_xyz_file(file_path)

    assert file_path.is_file()

    # Read it back and verify
    read_geom = Geometry.from_xyz_file(file_path)
    assert read_geom == h2o_geometry
