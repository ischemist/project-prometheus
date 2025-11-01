import logging
from pathlib import Path

import pytest

from calcflow.common.exceptions import ParsingError, ValidationError
from calcflow.geometry.static import Atom, Geometry
from calcflow.geometry.trajectory import Trajectory
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)

# --- Fixtures ---


@pytest.fixture
def frame1() -> Geometry:
    return Geometry(comment="Frame 1", atoms=(Atom(symbol="H", x=0, y=0, z=0),))


@pytest.fixture
def frame2() -> Geometry:
    return Geometry(comment="Frame 2", atoms=(Atom(symbol="H", x=0, y=0, z=1),))


@pytest.fixture
def frame_different_natoms() -> Geometry:
    return Geometry(comment="Frame 3", atoms=(Atom(symbol="H", x=0, y=0, z=0), Atom(symbol="O", x=1, y=0, z=0)))


# --- Contract Tests (Pydantic Model Behavior) ---


@pytest.mark.contract
def test_trajectory_model_validation(frame1: Geometry, frame2: Geometry, frame_different_natoms: Geometry):
    """Tests the Pydantic validator for consistent atom counts."""
    # Valid cases
    Trajectory(frames=())  # empty
    Trajectory(frames=(frame1,))  # single frame
    Trajectory(frames=(frame1, frame2))  # consistent multiple frames

    # Invalid case
    with pytest.raises(ValidationError, match="inconsistent number of atoms"):
        Trajectory(frames=(frame1, frame_different_natoms))


@pytest.mark.contract
def test_trajectory_dunder_methods(frame1: Geometry, frame2: Geometry):
    """Tests basic methods like __len__, __getitem__, __iter__."""
    traj = Trajectory(frames=(frame1, frame2))
    assert len(traj) == 2
    assert traj[0] == frame1
    assert traj[1] == frame2

    iterated = list(traj)
    assert iterated == [frame1, frame2]


# --- Integration Tests (File I/O) ---


@pytest.mark.integration
def test_trajectory_from_xyz_file_valid(tmp_path: Path):
    """Tests loading a Trajectory from a valid multi-frame XYZ file."""
    content = """1
Frame 1
H 0 0 0
1
Frame 2
H 0 0 1
"""
    file_path = tmp_path / "traj.xyz"
    file_path.write_text(content)

    traj = Trajectory.from_xyz_file(file_path)
    assert len(traj) == 2
    assert traj[0].comment == "Frame 1"
    assert traj[1].atoms[0].z == 1.0


@pytest.mark.integration
def test_trajectory_from_xyz_file_empty(tmp_path: Path):
    """Tests loading from an empty file."""
    file_path = tmp_path / "empty.xyz"
    file_path.touch()
    with pytest.raises(ParsingError, match="contains no valid frames"):
        Trajectory.from_xyz_file(file_path)


@pytest.mark.integration
def test_trajectory_from_xyz_file_not_found():
    """Tests loading from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        Trajectory.from_xyz_file("non_existent_file.xyz")


@pytest.mark.integration
@pytest.mark.parametrize(
    "bad_content, error_match",
    [
        ("1\nComment\nH 0 0 0\n2\nComment\nH 0 0 0\nO 1 1 1", "inconsistent number of atoms"),  # Validator error
        ("1\nComment\nH 0 0 0\ninvalid\nComment", "error parsing frame"),  # Parser error
    ],
)
def test_trajectory_from_xyz_file_invalid(tmp_path: Path, bad_content: str, error_match: str):
    """Tests loading from various invalid trajectory files."""
    file_path = tmp_path / "invalid.xyz"
    file_path.write_text(bad_content)
    with pytest.raises((ValidationError, ParsingError), match=error_match):
        Trajectory.from_xyz_file(file_path)
