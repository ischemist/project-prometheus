"""
Defines the mutable ParseState object used as a "scratchpad" during
the parsing process. It is the mutable counterpart to the final, immutable
CalculationResult model.
"""

from collections.abc import Sequence
from typing import Literal

from calcflow.common.results import (
    Atom,
    AtomicCharges,
    CalculationMetadata,
    CalculationResult,
    DispersionCorrection,
    MultipoleResults,
    OrbitalsSet,
    ScfResults,
    SmdResults,
    TddftResults,
    TimingResults,
)


class ParseState:
    """
    A mutable class serving as the central state object during a parse run.
    BlockParsers read from and write to this object. It is converted to the final
    immutable CalculationResult at the end of the parsing process.
    """

    def __init__(self, raw_output: str):
        # --- Raw Data & Metadata ---
        self.raw_output: str = raw_output
        self.metadata: CalculationMetadata = CalculationMetadata(software_name="unknown")

        # --- Core Results ---
        self.termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"] = "UNKNOWN"
        self.input_geometry: Sequence[Atom] | None = None
        self.final_geometry: Sequence[Atom] | None = None
        self.final_energy: float | None = None
        self.nuclear_repulsion_energy: float | None = None

        # --- Parsed Block Data ---
        self.scf: ScfResults | None = None
        self.orbitals: OrbitalsSet | None = None
        self.atomic_charges: list[AtomicCharges] = []
        self.multipole: MultipoleResults | None = None
        self.smd: SmdResults | None = None
        self.tddft: TddftResults | None = None
        self.dispersion: DispersionCorrection | None = None
        self.timing: TimingResults | None = None

        # --- Parser Control Flags ---
        self.parsed_metadata: bool = False
        self.parsed_geometry: bool = False
        self.parsed_scf: bool = False
        self.parsed_orbitals: bool = False
        self.parsed_charges: bool = False
        self.parsed_dipole: bool = False
        self.parsed_dispersion: bool = False
        self.parsed_multipole: bool = False
        self.parsed_timing: bool = False
        self.parsed_tddft_tda: bool = False
        self.parsed_tddft_full: bool = False
        self.parsed_tddft_gs_ref: bool = False
        self.parsed_tddft_unrelaxed_dm: bool = False
        self.parsed_nto: bool = False
        # Add more as needed for other parsers...

        # --- Communication & Error Handling ---
        self.parsing_errors: list[str] = []
        self.parsing_warnings: list[str] = []
        self.buffered_line: str | None = None  # For parsers that over-read

    def to_calculation_result(self) -> CalculationResult:
        """
        Constructs the final, immutable CalculationResult from the current state.
        This should be the last step of a successful parsing run.
        """
        return CalculationResult(
            raw_output=self.raw_output,
            metadata=self.metadata,
            termination_status=self.termination_status,
            input_geometry=self.input_geometry,
            final_geometry=self.final_geometry,
            final_energy=self.final_energy,
            nuclear_repulsion_energy=self.nuclear_repulsion_energy,
            scf=self.scf,
            orbitals=self.orbitals,
            atomic_charges=self.atomic_charges,
            multipole=self.multipole,
            smd=self.smd,
            tddft=self.tddft,
            dispersion=self.dispersion,
            timing=self.timing,
        )
