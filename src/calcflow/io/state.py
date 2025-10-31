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
    DispersionCorrection,
    MultipoleResults,
    OrbitalsSet,
    ScfResults,
    SmdResults,
    TddftResults,
    TimingResults,
)


class ParseState(BaseModel):
    """
    A mutable Pydantic model serving as the central state object during a parse run.
    BlockParsers read from and write to this object. It is converted to the final
    immutable CalculationResult at the end of the parsing process.
    """

    # --- Raw Data & Metadata ---
    raw_output: str
    metadata: CalculationMetadata = Field(default_factory=lambda: CalculationMetadata(software_name="unknown"))

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
    smd: SmdResults | None = None
    tddft: TddftResults | None = None
    dispersion: DispersionCorrection | None = None
    timing: TimingResults | None = None

    # --- Parser Control Flags ---
    parsed_metadata: bool = False
    parsed_geometry: bool = False
    parsed_scf: bool = False
    parsed_orbitals: bool = False
    parsed_charges: bool = False
    parsed_dipole: bool = False
    parsed_dispersion: bool = False
    parsed_multipole: bool = False
    parsed_timing: bool = False
    parsed_tddft_tda: bool = False
    parsed_tddft_full: bool = False
    parsed_tddft_gs_ref: bool = False
    # Add more as needed for other parsers...

    # --- Communication & Error Handling ---
    parsing_errors: list[str] = Field(default_factory=list)
    parsing_warnings: list[str] = Field(default_factory=list)
    buffered_line: str | None = None  # For parsers that over-read

    model_config = ConfigDict(arbitrary_types_allowed=True)
