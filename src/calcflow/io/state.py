"""
Defines the mutable ParseState object used as a "scratchpad" during
the parsing process. It is the mutable counterpart to the final, immutable
CalculationResult model.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from calcflow.common.models import (
    Atom,
    AtomicCharges,
    CalculationMetadata,
    DipoleMoment,
    MultipoleResults,
    OrbitalsSet,
    ScfResults,
    SmdResults,
    TddftResults,
)


class ParseState(BaseModel):
    """
    A mutable Pydantic model serving as the central state object during a parse run.
    BlockParsers read from and write to this object. It is converted to the final
    immutable CalculationResult at the end of the parsing process.
    """

    # --- Raw Data & Metadata ---
    raw_output: str
    metadata: CalculationMetadata = Field(default_factory=lambda: CalculationMetadata(program_name="unknown"))

    # --- Core Results ---
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"] = "UNKNOWN"
    input_geometry: Sequence[Atom] | None = None
    final_geometry: Sequence[Atom] | None = None
    final_energy: float | None = None
    nuclear_repulsion_energy: float | None = None

    # --- Parsed Block Data ---
    scf: ScfResults | None = None
    orbitals: OrbitalsSet | None = None
    atomic_charges: list[AtomicCharges] = Field(default_factory=list)
    multipole: MultipoleResults | None = None
    dipole_moment: DipoleMoment | None = None  # Specific for ORCA's simpler output
    smd: SmdResults | None = None
    tddft: TddftResults | None = None

    # --- Parser Control Flags ---
    parsed_geometry: bool = False
    parsed_scf: bool = False
    parsed_orbitals: bool = False
    parsed_dipole: bool = False
    parsed_dispersion: bool = False
    # Add more as needed for other parsers...

    # --- Communication & Error Handling ---
    parsing_errors: list[str] = Field(default_factory=list)
    parsing_warnings: list[str] = Field(default_factory=list)
    buffered_line: str | None = None  # For parsers that over-read

    model_config = ConfigDict(arbitrary_types_allowed=True)
