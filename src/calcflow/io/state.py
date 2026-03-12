"""
Defines the mutable ParseState object used as a "scratchpad" during
the parsing process. It is the mutable counterpart to the final, immutable
CalculationResult model.
"""

from typing import Literal

from calcflow.common.results import (
    AdcExcitedState,
    AdcGroundState,
    AdcResults,
    AtomicCharges,
    CalculationMetadata,
    CalculationResult,
    DispersionCorrection,
    ExcitedState,
    GroundStateReference,
    MultipoleResults,
    NTOStateAnalysis,
    OrbitalsSet,
    ScfResults,
    SmdResults,
    TddftResults,
    TimingResults,
    TransitionDensityMatrix,
    UnrelaxedDensityMatrix,
)
from calcflow.geometry.static import Geometry

# Block name constants for use with ParseState.parsed_blocks.
# Each constant names a parser block that tracks its own completion state.
BLOCK_METADATA = "metadata"
BLOCK_GEOMETRY = "geometry"
BLOCK_SCF = "scf"
BLOCK_ORBITALS = "orbitals"
BLOCK_MULLIKEN = "mulliken"
BLOCK_LOEWDIN = "loewdin"
BLOCK_HIRSHFELD = "hirshfeld"
BLOCK_CM5 = "cm5"
BLOCK_DIPOLE = "dipole"
BLOCK_DISPERSION = "dispersion"
BLOCK_MULTIPOLE = "multipole"
BLOCK_TIMING = "timing"
BLOCK_TDDFT_TDA = "tddft_tda"
BLOCK_TDDFT_FULL = "tddft_full"
BLOCK_TDDFT_GS_REF = "tddft_gs_ref"
BLOCK_TDDFT_UNRELAXED_DM = "tddft_unrelaxed_dm"
BLOCK_TDDFT_TRANS_DM = "tddft_trans_dm"
BLOCK_NTO = "nto"
BLOCK_ADC_GS = "adc_gs"
BLOCK_ADC_EXCITED = "adc_excited"


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
        self.input_geometry: Geometry | None = None
        self.final_geometry: Geometry | None = None
        self.final_energy: float | None = None
        self.nuclear_repulsion_energy: float | None = None

        # --- Parsed Block Data ---
        self.scf: ScfResults | None = None
        self.orbitals: OrbitalsSet | None = None
        self.atomic_charges: list[AtomicCharges] = []
        self.multipole: MultipoleResults | None = None
        self.smd: SmdResults | None = None
        self.dispersion: DispersionCorrection | None = None
        self.timing: TimingResults | None = None

        # --- ADC sub-results (assembled into AdcResults in to_calculation_result) ---
        self.adc_method: str = "adc(2)"
        self.adc_ground_state: AdcGroundState | None = None
        self.adc_excited_states: list[AdcExcitedState] = []

        # --- TDDFT sub-results (assembled into TddftResults in to_calculation_result) ---
        self.tddft_tda_states: list[ExcitedState] = []
        self.tddft_tddft_states: list[ExcitedState] = []
        self.tddft_nto_analyses: list[NTOStateAnalysis] = []
        self.tddft_ground_state_ref: GroundStateReference | None = None
        self.tddft_unrelaxed_density_matrices: list[UnrelaxedDensityMatrix] = []
        self.tddft_transition_density_matrices: list[TransitionDensityMatrix] = []

        # --- Parser Control Flags ---
        # Set of block names that have been fully parsed. Use BLOCK_* constants as keys.
        self.parsed_blocks: set[str] = set()

        # --- Communication & Error Handling ---
        self.parsing_errors: list[str] = []
        self.parsing_warnings: list[str] = []

    def to_calculation_result(self) -> CalculationResult:
        """
        Constructs the final, immutable CalculationResult from the current state.
        This should be the last step of a successful parsing run.
        """
        # Assemble AdcResults only if any ADC data was parsed.
        adc: AdcResults | None = None
        if self.adc_ground_state is not None or self.adc_excited_states:
            adc = AdcResults(
                method=self.adc_method,
                ground_state=self.adc_ground_state,
                excited_states=self.adc_excited_states,
            )

        # Assemble TddftResults only if any TDDFT data was parsed.
        # Convert empty lists to None to preserve the "None means not parsed" convention.
        tddft: TddftResults | None = None
        if any(
            (
                self.tddft_tda_states,
                self.tddft_tddft_states,
                self.tddft_nto_analyses,
                self.tddft_ground_state_ref is not None,
                self.tddft_unrelaxed_density_matrices,
                self.tddft_transition_density_matrices,
            )
        ):
            tddft = TddftResults(
                tda_states=self.tddft_tda_states or None,
                tddft_states=self.tddft_tddft_states or None,
                nto_analyses=self.tddft_nto_analyses or None,
                ground_state_ref=self.tddft_ground_state_ref,
                unrelaxed_density_matrices=self.tddft_unrelaxed_density_matrices or None,
                transition_density_matrices=self.tddft_transition_density_matrices or None,
            )

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
            tddft=tddft,
            adc=adc,
            dispersion=self.dispersion,
            timing=self.timing,
        )
