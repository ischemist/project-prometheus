from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, model_validator

from calcflow.common.exceptions import ParsingError
from calcflow.geometry.static import Geometry


def _iter_xyz_frames(f: TextIO, source: Path | str) -> Iterator[Geometry]:
    """yields Geometry objects frame by frame from an extended xyz stream."""
    while True:
        try:
            num_atoms_line = next(f)
            num_atoms = int(num_atoms_line.strip())
            frame_lines = [num_atoms_line] + [next(f) for _ in range(num_atoms + 1)]
            yield Geometry.from_lines(frame_lines, source)
        except StopIteration:
            break  # clean end of file
        except (ValueError, IndexError) as e:
            raise ParsingError(f"error parsing frame in '{source}': {e}") from e


class Trajectory(BaseModel, frozen=True):
    """
    an immutable, pydantic-based representation of a trajectory.
    validates that all frames have a consistent number of atoms on creation.
    """

    frames: tuple[Geometry, ...]

    @model_validator(mode="after")
    def check_consistent_atom_count(self) -> "Trajectory":
        if not self.frames:
            return self
        first_frame_atoms = self.frames[0].num_atoms
        if not all(frame.num_atoms == first_frame_atoms for frame in self.frames):
            raise ValueError("inconsistent number of atoms across trajectory frames.")
        return self

    @classmethod
    def from_xyz_file(cls, file: Path | str) -> "Trajectory":
        """creates a Trajectory instance from an extended xyz trajectory file."""
        file_path = Path(file)
        if not file_path.is_file():
            raise FileNotFoundError(f"trajectory file not found: {file_path}")

        with file_path.open("r") as f:
            all_frames = tuple(_iter_xyz_frames(f, file_path))

        if not all_frames:
            raise ParsingError(f"trajectory file '{file_path}' contains no valid frames.")

        return cls(frames=all_frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Geometry:
        return self.frames[index]

    def __iter__(self) -> Iterator[Geometry]:
        return iter(self.frames)
