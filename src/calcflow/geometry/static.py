import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import Atom
from calcflow.constants.ptable import ELEMENT_DATA


def _parse_energy_from_comment(comment: str) -> float | None:
    """extracts energy from orca or generic comment lines"""
    # orca: 'Coordinates from ORCA-job opt E -981.614502119079'
    orca_match = re.search(r"E\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", comment)
    if orca_match:
        return float(orca_match.group(1))

    # generic: 'Energy = -100.5' or 'E: -200.1'
    generic_match = re.search(r"(?:E|energy)\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", comment, re.IGNORECASE)
    if generic_match:
        return float(generic_match.group(1))

    return None


@dataclass(frozen=True)
class Geometry:
    """
    an immutable, dataclass-based representation of a molecular geometry.
    the number of atoms and energy are derived properties, not stored state.
    """

    comment: str
    atoms: tuple[Atom, ...]

    @cached_property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @cached_property
    def energy(self) -> float | None:
        return _parse_energy_from_comment(self.comment)

    @cached_property
    def unique_elements(self) -> set[str]:
        """returns a set of unique (uppercase) element symbols."""
        return {atom.symbol.upper() for atom in self.atoms}

    @cached_property
    def total_nuclear_charge(self) -> int:
        """calculates the total nuclear charge (sum of atomic numbers)."""
        return sum(ELEMENT_DATA[atom.symbol.upper()].atomic_number for atom in self.atoms)

    @classmethod
    def from_xyz_file(cls, file: Path | str) -> "Geometry":
        """creates a Geometry instance from an xyz file."""
        file_path = Path(file)
        if not file_path.is_file():
            raise FileNotFoundError(f"xyz file not found: {file_path}")

        with file_path.open("r") as f:
            lines = f.readlines()
        return cls.from_lines(lines, file_path)

    @classmethod
    def from_lines(cls, lines: list[str], source: Path | str = "memory") -> "Geometry":
        """creates a Geometry instance from a list of strings in xyz format."""
        if len(lines) < 2:
            raise ParsingError(f"invalid xyz data in '{source}': needs at least 2 lines.")

        try:
            num_atoms_declared = int(lines[0].strip())
        except ValueError as e:
            raise ParsingError(f"invalid xyz data in '{source}': first line must be an integer.") from e

        comment = lines[1].strip()
        atom_lines = [line.strip() for line in lines[2:] if line.strip()]

        if num_atoms_declared != len(atom_lines):
            raise ParsingError(
                f"invalid xyz data in '{source}': declared atom count ({num_atoms_declared}) "
                f"does not match actual atom lines found ({len(atom_lines)})."
            )

        atoms = []
        for i, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) != 4:
                raise ParsingError(f"invalid xyz data in '{source}' at line {i + 3}: expected 4 columns.")
            try:
                atoms.append(Atom(symbol=parts[0], x=float(parts[1]), y=float(parts[2]), z=float(parts[3])))
            except (ValueError, IndexError) as e:
                raise ParsingError(f"invalid xyz data in '{source}' at line {i + 3}: could not parse atom.") from e

        return cls(comment=comment, atoms=tuple(atoms))

    def to_xyz_str(self) -> str:
        """returns the geometry in xyz file format as a string."""
        return f"{self.num_atoms}\n{self.comment}\n{self.get_coordinate_block()}"

    def __str__(self) -> str:
        return self.to_xyz_str()

    def get_coordinate_block(self) -> str:
        """returns a string block of just the coordinate lines."""
        return "\n".join(f"{atom.symbol:<3} {atom.x:>15.8f} {atom.y:>15.8f} {atom.z:>15.8f}" for atom in self.atoms)

    def to_xyz_file(self, file_path: Path | str) -> None:
        """writes the geometry to an xyz format file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_xyz_str())
