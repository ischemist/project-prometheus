"""
Canonical, Program-Agnostic Data Models for Quantum Chemistry Results.

This module defines the single source of truth for all parsed calculation outputs.
All models are immutable dataclasses, ensuring data integrity after parsing.
The structure is hierarchical and compositional, building from fundamental concepts
(like atoms and orbitals) up to the complete CalculationResult.

Design Philosophy:
- Standard Library: Uses Python's `dataclasses` for zero external dependencies.
- Immutability: All models are frozen. Once a result is parsed, it cannot be changed.
- Unification: A single set of models represents results from any supported program
  (ORCA, QChem, etc.). Program-specific details are handled by making fields
  optional or by using clearly defined sub-models.
- Clarity: Field names are explicit. Units are Hartree for energy and Angstrom for
  distance unless specified otherwise in the field name.
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

from calcflow.common.exceptions import ValidationError
from calcflow.constants.ptable import ELEMENT_DATA

# =============================================================================
# §0. BASE MODEL FOR SERIALIZATION & DESERIALIZATION
# =============================================================================

T = TypeVar("T")


@dataclass(frozen=True)
class FrozenModel:
    """A base class providing to_dict and from_dict for frozen dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Recursively converts the dataclass instance to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """
        Recursively constructs a dataclass instance from a dictionary.
        Ignores extraneous keys in the input dictionary.
        """
        kwargs = {}
        cls_fields = {f.name: f for f in dataclasses.fields(cls)}

        for field_name, field_info in cls_fields.items():
            if field_name in data:
                value = data[field_name]
                kwargs[field_name] = cls._convert_value(value, field_info.type)

        return cls(**kwargs)

    @staticmethod
    def _convert_value(value: Any, target_type: type) -> Any:
        """Helper to recursively convert dictionary values to dataclass fields."""
        if value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle Optional[T] with both old (Union) and new (UnionType | None) syntax
        if (origin is Union or origin is UnionType) and type(None) in args:
            # Assumes Optional[T] is Union[T, NoneType] or T | None
            inner_type = next(t for t in args if t is not type(None))
            return FrozenModel._convert_value(value, inner_type)

        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            # mypy complains here but it's correct; target_type has from_dict
            return target_type.from_dict(value)  # type: ignore

        if origin in (list, Sequence) and isinstance(value, list):
            item_type = args[0]
            return [FrozenModel._convert_value(item, item_type) for item in value]

        if origin in (dict, Mapping) and isinstance(value, dict):
            key_type, val_type = args
            return {
                FrozenModel._convert_value(k, key_type): FrozenModel._convert_value(v, val_type)
                for k, v in value.items()
            }

        return value

    def to_json(self, indent: int = 2) -> str:
        """Serializes the model to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Deserializes a model from a JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# §1. FUNDAMENTAL BUILDING BLOCKS
# =============================================================================


@dataclass(frozen=True)
class Atom(FrozenModel):
    """Represents a single atom with its Cartesian coordinates."""

    symbol: str
    x: float
    y: float
    z: float

    def __post_init__(self):
        """Validates the element symbol after initialization."""
        # On frozen dataclasses, __post_init__ can't modify fields.
        # It can only validate. Parsers are responsible for capitalization.
        if self.symbol.upper() not in ELEMENT_DATA:
            raise ValidationError(f"unknown element symbol: '{self.symbol}'")
        if self.symbol != self.symbol.capitalize():
            raise ValidationError(f"element symbol '{self.symbol}' must be capitalized.")


@dataclass(frozen=True)
class Orbital(FrozenModel):
    """Represents a single molecular orbital."""

    index: int  # 0-based index
    energy: float  # Energy in Hartree
    occupation: float | None = None
    energy_ev: float | None = None


# =============================================================================
# §2. GROUND STATE PROPERTY MODELS
# =============================================================================


@dataclass(frozen=True)
class ScfIteration(FrozenModel):
    """Holds data for a single SCF iteration, unifying ORCA and QChem fields."""

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


@dataclass(frozen=True)
class ScfEnergyComponents(FrozenModel):
    """Holds the components of the raw SCF energy."""

    nuclear_repulsion: float
    electronic_eh: float
    one_electron_eh: float
    two_electron_eh: float
    xc_eh: float | None = None  # Exchange-Correlation energy for DFT


@dataclass(frozen=True)
class ScfResults(FrozenModel):
    """Holds all results from the SCF procedure."""

    converged: bool
    energy: float  # Final SCF energy in Hartree
    n_iterations: int
    iterations: Sequence[ScfIteration]
    components: ScfEnergyComponents | None = None


@dataclass(frozen=True)
class OrbitalsSet(FrozenModel):
    """Holds all molecular orbital information."""

    # For RHF/RKS, only alpha_orbitals will be populated. beta_orbitals will be None.
    alpha_orbitals: Sequence[Orbital]
    beta_orbitals: Sequence[Orbital] | None = None
    alpha_homo_index: int | None = None
    alpha_lumo_index: int | None = None
    beta_homo_index: int | None = None
    beta_lumo_index: int | None = None


@dataclass(frozen=True)
class AtomicCharges(FrozenModel):
    """Stores atomic charges from a specific population analysis method."""

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


@dataclass(frozen=True)
class DipoleMoment(FrozenModel):
    """Stores dipole moment components and magnitude in Debye."""

    x: float
    y: float
    z: float
    magnitude: float


@dataclass(frozen=True)
class DispersionCorrection(FrozenModel):
    """Stores dispersion correction results from DFT-D3/D4 calculations."""

    method: str  # e.g., "DFTD3", "DFTD4"
    e_disp_au: float  # Total dispersion energy in Hartree (primary value)
    functional: str | None = None  # e.g., "omegaB97X-D3"
    damping: str | None = None  # e.g., "zero damping"
    molecular_c6_au: float | None = None  # Molecular C6 coefficient in au
    parameters: Mapping[str, float] | None = None  # Scaling and damping parameters
    e_disp_kcal: float | None = None  # Total dispersion energy in kcal/mol
    e6_kcal: float | None = None  # E6 component in kcal/mol
    e8_kcal: float | None = None  # E8 component in kcal/mol
    e8_percentage: float | None = None  # Percentage contribution of E8


@dataclass(frozen=True)
class QuadrupoleMoment(FrozenModel):
    """Stores Cartesian quadrupole moments in Debye-Ang."""

    xx: float
    xy: float
    yy: float
    xz: float
    yz: float
    zz: float


@dataclass(frozen=True)
class OctopoleMoment(FrozenModel):
    """Stores Cartesian octopole moments in Debye-Ang^2."""

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


@dataclass(frozen=True)
class HexadecapoleMoment(FrozenModel):
    """Stores Cartesian hexadecapole moments in Debye-Ang^3."""

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


@dataclass(frozen=True)
class MultipoleResults(FrozenModel):
    """Container for various electric multipole moments."""

    charge: float | None = None  # Total charge in ESU x 10^10
    dipole: DipoleMoment | None = None
    quadrupole: QuadrupoleMoment | None = None
    octopole: OctopoleMoment | None = None
    hexadecapole: HexadecapoleMoment | None = None


@dataclass(frozen=True)
class SmdResults(FrozenModel):
    """Holds results specific to the SMD solvation model."""

    g_pcm_kcal_mol: float | None = None  # Polarization energy component
    g_cds_kcal_mol: float | None = None  # Non-electrostatic (CDS) component
    g_enp_au: float | None = None  # E_SCF including G_PCM
    g_tot_au: float | None = None  # Total free energy in solution


@dataclass(frozen=True)
class TimingResults(FrozenModel):
    """Stores timing information from calculation runs."""

    total_cpu_time_seconds: float | None = None  # QChem: CPU time
    total_wall_time_seconds: float | None = None  # Wall time (clock time)
    module_times: Mapping[str, float] | None = None  # ORCA: module name -> wall time in seconds


# =============================================================================
# §3. TDDFT & EXCITED STATE MODELS
# =============================================================================


@dataclass(frozen=True)
class OrbitalTransition(FrozenModel):
    """A single orbital transition's contribution to an excited state."""

    from_idx: int
    to_idx: int
    amplitude: float
    is_alpha_spin: bool | None = None  # True=alpha, False=beta, None=unspecified


@dataclass(frozen=True)
class ExcitedState(FrozenModel):
    """Core properties of a single excited state."""

    state_number: int
    multiplicity: Literal["Singlet", "Triplet", "Unknown"]
    excitation_energy_ev: float
    total_energy_au: float
    oscillator_strength: float | None = None
    transitions: Sequence[OrbitalTransition] = field(default_factory=list)
    trans_mom_x: float | None = None  # Transition moment X component
    trans_mom_y: float | None = None  # Transition moment Y component
    trans_mom_z: float | None = None  # Transition moment Z component


@dataclass(frozen=True)
class NTOContribution(FrozenModel):
    """A single Natural Transition Orbital (NTO) contribution."""

    hole_offset: int  # e.g. -2 for H-2
    electron_offset: int  # e.g. +3 for L+3
    weight_percent: float
    is_alpha_spin: bool


@dataclass(frozen=True)
class NTOStateAnalysis(FrozenModel):
    """NTO decomposition for a single excited state."""

    state_number: int
    contributions: Sequence[NTOContribution] = field(default_factory=list)
    omega_percent: float | None = None  # Total character
    omega_alpha_percent: float | None = None
    omega_beta_percent: float | None = None


@dataclass(frozen=True)
class GroundStateReference(FrozenModel):
    """Ground state (reference) data from excited state analysis blocks."""

    frontier_nos: Sequence[float]  # Occupation of frontier NOs
    num_electrons: float  # Total electron count
    mulliken: AtomicCharges  # Per-atom charges (and optionally spins for UKS) in e
    dipole_moment_debye: float  # Total dipole moment
    dipole_components_debye: tuple[float, float, float]  # (X, Y, Z) in Debye
    num_unpaired_electrons: float | None = None  # n_u value (typically for RKS with unpaired analysis)


@dataclass(frozen=True)
class NaturalOrbitals(FrozenModel):
    """Natural orbital occupations for excited state density matrix analysis."""

    frontier_occupations: Sequence[float]  # e.g., [0.9992, 1.0006]
    num_electrons: float
    num_unpaired: float | None = None  # n_u value
    num_unpaired_nl: float | None = None  # n_u,nl value
    pr_no: float | None = None  # NO participation ratio


@dataclass(frozen=True)
class ExcitonAnalysis(FrozenModel):
    """Exciton analysis from density matrix (hole/electron separation)."""

    r_h_ang: tuple[float, float, float]  # <r_h> in Angstroms (X, Y, Z)
    r_e_ang: tuple[float, float, float]  # <r_e> in Angstroms
    separation_ang: float  # |<r_e - r_h>|
    hole_size_ang: float
    electron_size_ang: float
    hole_size_components_ang: tuple[float, float, float] | None = None
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
    trans_dipole_moment_components_debye: tuple[float, float, float] | None = None
    trans_r2_au: float | None = None  # Transition <r^2> [a.u.]
    trans_r2_components_au: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class UnrelaxedDensityMatrix(FrozenModel):
    """Unrelaxed density matrix analysis for a single excited state."""

    state_number: int
    # Natural Orbitals
    nos_spin_traced: NaturalOrbitals  # Main one (or only one for RKS)
    # Mulliken (State/Difference DM)
    mulliken: AtomicCharges  # Reuse existing model (charges + optional spins)
    # Multipole moment of density matrix
    molecular_charge: float
    num_electrons: float
    dipole_moment_debye: float
    dipole_components_debye: tuple[float, float, float]
    # Exciton analysis
    exciton_total: ExcitonAnalysis
    multiplicity: str | None = None  # "Singlet N" or "Excited State N"
    nos_alpha: NaturalOrbitals | None = None  # UKS only
    nos_beta: NaturalOrbitals | None = None  # UKS only
    exciton_alpha: ExcitonAnalysis | None = None  # UKS only
    exciton_beta: ExcitonAnalysis | None = None  # UKS only


@dataclass(frozen=True)
class TransitionDensityMatrix(FrozenModel):
    """Transition density matrix analysis for a single excited state."""

    state_number: int
    # Exciton analysis (reuse existing model - has all transition DM fields)
    exciton_total: ExcitonAnalysis
    multiplicity: str | None = None  # "Singlet N" or "Excited State N"
    # Mulliken Population Analysis (Transition DM)
    mulliken: AtomicCharges | None = None  # Note: QChem 5.4 lacks this section
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
    trans_dipole_components_debye: tuple[float, float, float] | None = None
    trans_r2_components_au: tuple[float, float, float] | None = None
    exciton_alpha: ExcitonAnalysis | None = None  # UKS only
    exciton_beta: ExcitonAnalysis | None = None  # UKS only


@dataclass(frozen=True)
class TddftResults(FrozenModel):
    """Container for all TDDFT-related parsed data."""

    # QChem can have both, ORCA usually has one.
    tda_states: Sequence[ExcitedState] | None = None
    tddft_states: Sequence[ExcitedState] | None = None
    # More detailed, program-specific analyses can be added here
    nto_analyses: Sequence[NTOStateAnalysis] | None = None
    ground_state_ref: GroundStateReference | None = None
    unrelaxed_density_matrices: Sequence[UnrelaxedDensityMatrix] | None = None
    transition_density_matrices: Sequence[TransitionDensityMatrix] | None = None


# =============================================================================
# §4. TOP-LEVEL RESULT MODELS
# =============================================================================


@dataclass(frozen=True)
class CalculationMetadata(FrozenModel):
    """Static metadata about the calculation run."""

    software_name: str
    software_version: str | None = None


@dataclass(frozen=True)
class CalculationResult(FrozenModel):
    """
    The canonical, immutable result of a single quantum chemistry calculation.
    This is the final object produced by any successful parser run.
    """

    # --- Core Information ---
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    metadata: CalculationMetadata
    raw_output: str = field(repr=False)

    # --- Geometry ---
    input_geometry: Sequence[Atom] | None = None
    final_geometry: Sequence[Atom] | None = None

    # --- Energies ---
    final_energy: float | None = None  # e.g., SCF+Dispersion
    nuclear_repulsion_energy: float | None = None

    # --- Parsed Blocks (Optional) ---
    scf: ScfResults | None = None
    orbitals: OrbitalsSet | None = None
    multipole: MultipoleResults | None = None
    smd: SmdResults | None = None
    tddft: TddftResults | None = None
    dispersion: DispersionCorrection | None = None
    timing: TimingResults | None = None
    atomic_charges: Sequence[AtomicCharges] = field(default_factory=list)

    # --- Program Specific Data ---
    program_specific: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts to dictionary, excluding raw_output to save space."""
        data = super().to_dict()
        data.pop("raw_output", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalculationResult":
        """Reconstructs from dictionary, setting raw_output to empty string."""
        if "raw_output" not in data:
            data = {**data, "raw_output": ""}
        return super().from_dict(data)
