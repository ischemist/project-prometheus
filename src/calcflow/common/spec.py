from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from calcflow.exceptions import ValidationError

# --- Component Specifications ---
# these are the building blocks. if a component is None in the main spec,
# that feature is simply not requested.


@dataclass(frozen=True)
class TddftSpec:
    """specification for a time-dependent dft calculation."""

    nroots: int
    singlets: bool = True
    triplets: bool = False
    use_tda: bool = True  # Tamm-Dancoff Approximation is a common choice
    state_to_optimize: int | None = None  # for geometry optimization of an excited state


@dataclass(frozen=True)
class SolvationSpec:
    """specification for an implicit solvation model."""

    model: str  # e.g., 'smd', 'cpcm'
    solvent: str  # e.g., 'water', 'acetonitrile'


@dataclass(frozen=True)
class OptimizationSpec:
    """specification for geometry optimization tasks."""

    calc_hess_initial: bool = False
    recalc_hess_freq: int | None = None


# --- Main Calculation Specification ---


@dataclass(frozen=True)
class CalculationSpec:
    """
    a pure, program-agnostic data representation of a quantum chemistry calculation.
    this is the canonical "spec" that gets passed to program-specific builders.
    it is intentionally inert and contains no program-specific logic.
    """

    charge: int
    spin_multiplicity: int
    task: Literal["energy", "geometry", "frequency"]
    level_of_theory: str
    basis_set: str | dict[str, str]
    unrestricted: bool = False
    n_cores: int = 1
    memory_per_core_mb: int = 4000

    # optional, modular components of the calculation
    tddft: TddftSpec | None = None
    solvation: SolvationSpec | None = None
    optimization: OptimizationSpec | None = None
    frequency_after_optimization: bool = False

    # the escape hatch for anything program-specific that doesn't fit the generic model.
    # e.g., for orca: {"ri_approx": "RIJCOSX", "aux_basis": "def2/j"}
    program_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """basic, program-agnostic validation."""
        if self.spin_multiplicity < 1:
            raise ValidationError("spin multiplicity must be a positive integer.")
        if self.n_cores < 1:
            raise ValidationError("number of cores must be a positive integer.")
        if self.tddft and self.tddft.nroots < 1:
            raise ValidationError("tddft nroots must be a positive integer.")
        if self.solvation and (not self.solvation.model or not self.solvation.solvent):
            raise ValidationError("solvation model and solvent must both be specified.")
