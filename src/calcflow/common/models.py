"""
Canonical, Program-Agnostic Data Models for Quantum Chemistry Results.

This module defines the single source of truth for all parsed calculation outputs.
All models are immutable Pydantic models, ensuring data integrity after parsing.
The structure is hierarchical and compositional, building from fundamental concepts
(like atoms and orbitals) up to the complete CalculationResult.

Design Philosophy:
- Pydantic: For robust type validation, serialization, and clear schema definition.
- Immutability: All models are frozen. Once a result is parsed, it cannot be changed.
- Unification: A single set of models represents results from any supported program
  (ORCA, QChem, etc.). Program-specific details are handled by making fields
  optional or by using clearly defined sub-models.
- Clarity: Field names are explicit. Units are Hartree for energy and Angstrom for
  distance unless specified otherwise in the field name.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from calcflow.constants.ptable import ELEMENT_DATA

# --- Core immutable configuration for all models ---
IMMUTABLE_MODEL_CONFIG = ConfigDict(frozen=True)


# =============================================================================
# §1. FUNDAMENTAL BUILDING BLOCKS
# =============================================================================


class Atom(BaseModel):
    """Represents a single atom with its Cartesian coordinates."""

    model_config = IMMUTABLE_MODEL_CONFIG

    symbol: str
    x: float
    y: float
    z: float

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if v.upper() not in ELEMENT_DATA:
            raise ValueError(f"unknown element symbol: '{v}'")
        return v.capitalize()


class Orbital(BaseModel):
    """Represents a single molecular orbital."""

    model_config = IMMUTABLE_MODEL_CONFIG

    index: int  # 0-based index
    energy: float  # Energy in Hartree
    occupation: float | None = None
    energy_ev: float | None = None


# =============================================================================
# §2. GROUND STATE PROPERTY MODELS
# =============================================================================


class ScfIteration(BaseModel):
    """Holds data for a single SCF iteration, unifying ORCA and QChem fields."""

    model_config = IMMUTABLE_MODEL_CONFIG

    iteration: int
    energy: float
    # ORCA-style fields
    delta_e_eh: float | None = None
    rmsdp: float | None = None
    maxdp: float | None = None
    # QChem-style fields
    diis_error: float | None = None
    # MOM-specific fields
    mom_active: bool | None = None
    mom_overlap_current: float | None = None
    mom_overlap_target: float | None = None


class ScfEnergyComponents(BaseModel):
    """Holds the components of the raw SCF energy."""

    model_config = IMMUTABLE_MODEL_CONFIG

    nuclear_repulsion: float
    electronic_eh: float
    one_electron_eh: float
    two_electron_eh: float
    xc_eh: float | None = None  # Exchange-Correlation energy for DFT


class ScfResults(BaseModel):
    """Holds all results from the SCF procedure."""

    model_config = IMMUTABLE_MODEL_CONFIG

    converged: bool
    energy: float  # Final SCF energy in Hartree
    n_iterations: int
    components: ScfEnergyComponents | None = None
    iterations: Sequence[ScfIteration]


class OrbitalsSet(BaseModel):
    """Holds all molecular orbital information."""

    model_config = IMMUTABLE_MODEL_CONFIG

    # For RHF/RKS, only alpha_orbitals will be populated. beta_orbitals will be None.
    alpha_orbitals: Sequence[Orbital]
    beta_orbitals: Sequence[Orbital] | None = None
    alpha_homo_index: int | None = None
    alpha_lumo_index: int | None = None
    beta_homo_index: int | None = None
    beta_lumo_index: int | None = None


class AtomicCharges(BaseModel):
    """Stores atomic charges from a specific population analysis method."""

    model_config = IMMUTABLE_MODEL_CONFIG

    method: str  # e.g., "Mulliken", "Loewdin", "Hirshfeld"
    charges: Mapping[int, float]  # Atom index (0-based) to charge
    spins: Mapping[int, float] | None = None  # Atom index to spin density (UKS only)
    # For excited state difference density analysis (TDDFT unrelaxed DM)
    hole_populations: Mapping[int, float] | None = None  # h+ values (RKS)
    electron_populations: Mapping[int, float] | None = None  # e- values (RKS)
    hole_populations_alpha: Mapping[int, float] | None = None  # h+ alpha (UKS)
    hole_populations_beta: Mapping[int, float] | None = None  # h+ beta (UKS)
    electron_populations_alpha: Mapping[int, float] | None = None  # e- alpha (UKS)
    electron_populations_beta: Mapping[int, float] | None = None  # e- beta (UKS)
    # For transition density matrix analysis
    trans_charges: Mapping[int, float] | None = None  # "Trans. (e)" column
    del_q: Mapping[int, float] | None = None  # "Del q" column


class DipoleMoment(BaseModel):
    """Stores dipole moment components and magnitude in Debye."""

    model_config = IMMUTABLE_MODEL_CONFIG

    x: float
    y: float
    z: float
    magnitude: float


class DispersionCorrection(BaseModel):
    """Stores dispersion correction results from DFT-D3/D4 calculations."""

    model_config = IMMUTABLE_MODEL_CONFIG

    method: str  # e.g., "DFTD3", "DFTD4"
    functional: str | None = None  # e.g., "omegaB97X-D3"
    damping: str | None = None  # e.g., "zero damping"
    molecular_c6_au: float | None = None  # Molecular C6 coefficient in au
    parameters: Mapping[str, float] | None = None  # Scaling and damping parameters
    e_disp_kcal: float | None = None  # Total dispersion energy in kcal/mol
    e_disp_au: float  # Total dispersion energy in Hartree (primary value)
    e6_kcal: float | None = None  # E6 component in kcal/mol
    e8_kcal: float | None = None  # E8 component in kcal/mol
    e8_percentage: float | None = None  # Percentage contribution of E8


class QuadrupoleMoment(BaseModel):
    """Stores Cartesian quadrupole moments in Debye-Ang."""

    model_config = IMMUTABLE_MODEL_CONFIG

    xx: float
    xy: float
    yy: float
    xz: float
    yz: float
    zz: float


class OctopoleMoment(BaseModel):
    """Stores Cartesian octopole moments in Debye-Ang^2."""

    model_config = IMMUTABLE_MODEL_CONFIG

    xxx: float
    xxy: float
    xyy: float
    yyy: float
    xxz: float
    xyz: float
    yyz: float
    xzz: float
    yzz: float
    zzz: float


class HexadecapoleMoment(BaseModel):
    """Stores Cartesian hexadecapole moments in Debye-Ang^3."""

    model_config = IMMUTABLE_MODEL_CONFIG

    xxxx: float
    xxxy: float
    xxyy: float
    xyyy: float
    yyyy: float
    xxxz: float
    xxyz: float
    xyyz: float
    yyyz: float
    xxzz: float
    xyzz: float
    yyzz: float
    xzzz: float
    yzzz: float
    zzzz: float


class MultipoleResults(BaseModel):
    """Container for various electric multipole moments."""

    model_config = IMMUTABLE_MODEL_CONFIG

    charge: float | None = None  # Total charge in ESU x 10^10
    dipole: DipoleMoment | None = None
    quadrupole: QuadrupoleMoment | None = None
    octopole: OctopoleMoment | None = None
    hexadecapole: HexadecapoleMoment | None = None


class SmdResults(BaseModel):
    """Holds results specific to the SMD solvation model."""

    model_config = IMMUTABLE_MODEL_CONFIG

    g_pcm_kcal_mol: float | None = None  # Polarization energy component
    g_cds_kcal_mol: float | None = None  # Non-electrostatic (CDS) component
    g_enp_au: float | None = None  # E_SCF including G_PCM
    g_tot_au: float | None = None  # Total free energy in solution


class TimingResults(BaseModel):
    """Stores timing information from calculation runs."""

    model_config = IMMUTABLE_MODEL_CONFIG

    total_cpu_time_seconds: float | None = None  # QChem: CPU time
    total_wall_time_seconds: float | None = None  # Wall time (clock time)
    module_times: Mapping[str, float] | None = None  # ORCA: module name -> wall time in seconds


# =============================================================================
# §3. TDDFT & EXCITED STATE MODELS
# =============================================================================


class OrbitalTransition(BaseModel):
    """A single orbital transition's contribution to an excited state."""

    model_config = IMMUTABLE_MODEL_CONFIG

    from_idx: int
    to_idx: int
    amplitude: float
    is_alpha_spin: bool | None = None  # True=alpha, False=beta, None=unspecified


class ExcitedState(BaseModel):
    """Core properties of a single excited state."""

    model_config = IMMUTABLE_MODEL_CONFIG

    state_number: int
    multiplicity: Literal["Singlet", "Triplet", "Unknown"]
    excitation_energy_ev: float
    total_energy_au: float
    oscillator_strength: float | None = None
    transitions: Sequence[OrbitalTransition] = Field(default_factory=list)


class NTOContribution(BaseModel):
    """A single Natural Transition Orbital (NTO) contribution."""

    model_config = IMMUTABLE_MODEL_CONFIG

    hole_offset: int  # e.g. -2 for H-2
    electron_offset: int  # e.g. +3 for L+3
    weight_percent: float
    is_alpha_spin: bool


class NTOStateAnalysis(BaseModel):
    """NTO decomposition for a single excited state."""

    model_config = IMMUTABLE_MODEL_CONFIG

    state_number: int
    contributions: Sequence[NTOContribution] = Field(default_factory=list)
    omega_percent: float | None = None  # Total character
    omega_alpha_percent: float | None = None
    omega_beta_percent: float | None = None


class GroundStateReference(BaseModel):
    """Ground state (reference) data from excited state analysis blocks."""

    model_config = IMMUTABLE_MODEL_CONFIG

    frontier_nos: Sequence[float]  # Occupation of frontier NOs
    num_electrons: float  # Total electron count
    num_unpaired_electrons: float | None = None  # n_u value (typically for RKS with unpaired analysis)
    mulliken: AtomicCharges  # Per-atom charges (and optionally spins for UKS) in e
    dipole_moment_debye: float  # Total dipole moment
    dipole_components_debye: tuple[float, float, float]  # (X, Y, Z) in Debye


class NaturalOrbitals(BaseModel):
    """Natural orbital occupations for excited state density matrix analysis."""

    model_config = IMMUTABLE_MODEL_CONFIG

    frontier_occupations: Sequence[float]  # e.g., [0.9992, 1.0006]
    num_electrons: float
    num_unpaired: float | None = None  # n_u value
    num_unpaired_nl: float | None = None  # n_u,nl value
    pr_no: float | None = None  # NO participation ratio


class ExcitonAnalysis(BaseModel):
    """Exciton analysis from density matrix (hole/electron separation)."""

    model_config = IMMUTABLE_MODEL_CONFIG

    r_h_ang: tuple[float, float, float]  # <r_h> in Angstroms (X, Y, Z)
    r_e_ang: tuple[float, float, float]  # <r_e> in Angstroms
    separation_ang: float  # |<r_e - r_h>|
    hole_size_ang: float
    hole_size_components_ang: tuple[float, float, float] | None = None
    electron_size_ang: float
    electron_size_components_ang: tuple[float, float, float] | None = None
    # Fields only present in transition density matrix:
    rms_separation_ang: float | None = None
    rms_separation_components_ang: tuple[float, float, float] | None = None
    covariance: float | None = None
    correlation_coef: float | None = None
    center_of_mass_size_ang: float | None = None
    center_of_mass_components_ang: tuple[float, float, float] | None = None
    # Transition-specific properties (transition dipole moment, etc.)
    trans_dipole_moment_debye: float | None = None  # Trans. dipole moment [D]
    trans_dipole_moment_components_debye: tuple[float, float, float] | None = None  # Cartesian components [D]
    trans_r2_au: float | None = None  # Transition <r^2> [a.u.]
    trans_r2_components_au: tuple[float, float, float] | None = None  # Cartesian components [a.u.]


class UnrelaxedDensityMatrix(BaseModel):
    """Unrelaxed density matrix analysis for a single excited state."""

    model_config = IMMUTABLE_MODEL_CONFIG

    state_number: int
    multiplicity: str | None = None  # "Singlet N" or "Excited State N"
    # Natural Orbitals
    nos_spin_traced: NaturalOrbitals  # Main one (or only one for RKS)
    nos_alpha: NaturalOrbitals | None = None  # UKS only
    nos_beta: NaturalOrbitals | None = None  # UKS only
    # Mulliken (State/Difference DM)
    mulliken: AtomicCharges  # Reuse existing model (charges + optional spins)
    # Multipole moment of density matrix
    molecular_charge: float
    num_electrons: float
    dipole_moment_debye: float
    dipole_components_debye: tuple[float, float, float]
    # Exciton analysis
    exciton_total: ExcitonAnalysis
    exciton_alpha: ExcitonAnalysis | None = None  # UKS only
    exciton_beta: ExcitonAnalysis | None = None  # UKS only


class TransitionDensityMatrix(BaseModel):
    """Transition density matrix analysis for a single excited state."""

    model_config = IMMUTABLE_MODEL_CONFIG

    state_number: int
    multiplicity: str | None = None  # "Singlet N" or "Excited State N"
    # Mulliken Population Analysis (Transition DM)
    # Note: QChem 5.4 doesn't include this section in transition DM analysis
    mulliken: AtomicCharges | None = None  # Reuse existing model
    # CT numbers and transition metrics
    sum_abs_trans_charges: float | None = None  # QTa
    sum_squared_trans_charges: float | None = None  # QT2
    omega: float | None = None
    omega_alpha: float | None = None  # UKS only
    omega_beta: float | None = None  # UKS only
    two_alpha_beta: float | None = None  # 2<alpha|beta>
    loc: float | None = None
    loc_alpha: float | None = None  # UKS only
    loc_beta: float | None = None  # UKS only
    loca: float | None = None  # LOCa
    loca_alpha: float | None = None  # UKS only
    loca_beta: float | None = None  # UKS only
    phe: float | None = None  # <Phe>
    phe_alpha: float | None = None  # UKS only
    phe_beta: float | None = None  # UKS only
    # Transition-specific properties not in standard exciton analysis
    trans_dipole_moment_debye: float | None = None  # Trans. dipole moment [D]
    trans_r2_au: float | None = None  # Transition <r^2> [a.u.]
    trans_dipole_components_debye: tuple[float, float, float] | None = None  # Cartesian components [D]
    trans_r2_components_au: tuple[float, float, float] | None = None  # Cartesian components [a.u.]
    # Exciton analysis (reuse existing model - already has all transition DM fields)
    exciton_total: ExcitonAnalysis
    exciton_alpha: ExcitonAnalysis | None = None  # UKS only
    exciton_beta: ExcitonAnalysis | None = None  # UKS only


class TddftResults(BaseModel):
    """Container for all TDDFT-related parsed data."""

    model_config = IMMUTABLE_MODEL_CONFIG

    # QChem can have both, ORCA usually has one.
    tda_states: Sequence[ExcitedState] | None = None
    tddft_states: Sequence[ExcitedState] | None = None

    # More detailed, program-specific analyses can be added here as needed
    nto_analyses: Sequence[NTOStateAnalysis] | None = None
    ground_state_ref: "GroundStateReference | None" = None
    unrelaxed_density_matrices: Sequence[UnrelaxedDensityMatrix] | None = None
    transition_density_matrices: Sequence[TransitionDensityMatrix] | None = None
    # Add ExcitedStateDetailedAnalysis if the unified model proves necessary


# =============================================================================
# §4. TOP-LEVEL RESULT MODELS
# =============================================================================


class CalculationMetadata(BaseModel):
    """Static metadata about the calculation run."""

    model_config = IMMUTABLE_MODEL_CONFIG

    software_name: str
    software_version: str | None = None


class CalculationResult(BaseModel):
    """
    The canonical, immutable result of a single quantum chemistry calculation.
    This is the final object produced by any successful parser run.
    """

    model_config = IMMUTABLE_MODEL_CONFIG

    # --- Core Information ---
    raw_output: str = Field(repr=False)
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    metadata: CalculationMetadata

    # --- Geometry ---
    input_geometry: Sequence[Atom]
    final_geometry: Sequence[Atom] | None = None

    # --- Energies ---
    final_energy: float | None = None  # The most important energy (e.g., SCF+Dispersion)
    nuclear_repulsion_energy: float | None = None

    # --- Parsed Blocks (Optional) ---
    scf: ScfResults | None = None
    orbitals: OrbitalsSet | None = None
    atomic_charges: Sequence[AtomicCharges] = Field(default_factory=list)
    multipole: MultipoleResults | None = None
    smd: SmdResults | None = None
    tddft: TddftResults | None = None
    nto_analyses: Sequence[NTOStateAnalysis] | None = None
    dispersion: DispersionCorrection | None = None
    timing: TimingResults | None = None

    # --- Program Specific Data ---
    # A catch-all for data that is too program-specific to unify. Use sparingly.
    program_specific: Mapping[str, Any] = Field(default_factory=dict)


class MomCalculationResult(BaseModel):
    """
    Holds the results of a two-step MOM calculation, containing two full
    CalculationResult objects.
    """

    model_config = IMMUTABLE_MODEL_CONFIG

    job1_initial_scf: CalculationResult
    job2_mom_scf: CalculationResult
    raw_output: str = Field(repr=False)


class OptimizationResult(BaseModel):
    """
    The canonical, immutable result of a geometry optimization.
    """

    # TBD, but would likely contain a sequence of CalculationResult-like
    # objects for each optimization cycle.
    pass
