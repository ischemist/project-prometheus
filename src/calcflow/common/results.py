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

    @classmethod
    def get_api_docs(cls) -> str:
        """
        returns comprehensive api documentation for llm-assisted code generation.

        this method provides a complete reference of all available data models, fields,
        and usage patterns without requiring access to source code. ideal for sharing
        with llms when you want them to work with parsed quantum chemistry results.

        usage:
            # print documentation for llm consumption
            print(CalculationResult.get_api_docs())

            # save to file
            with open("calcflow_results_api.txt", "w") as f:
                f.write(CalculationResult.get_api_docs())
        """
        return """
CalculationResult API Reference
================================

DESCRIPTION
-----------
Immutable, program-agnostic data models for quantum chemistry calculation results.
All models are frozen dataclasses built with Python's standard library, ensuring
zero external dependencies and data integrity after parsing.

The structure is hierarchical and compositional, building from fundamental concepts
(atoms, orbitals) up to the complete CalculationResult. A single unified data model
represents results from any supported program (ORCA, Q-Chem, etc.).

All energies are in Hartree, distances in Angstrom unless otherwise specified.


TOP-LEVEL MODEL: CalculationResult
-----------------------------------
The canonical result of a quantum chemistry calculation.

Core Fields:
  termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    Whether the calculation completed successfully

  metadata: CalculationMetadata
    Software name and version information

  raw_output: str
    Full text output from the program (excluded from serialization)

Geometry Fields:
  input_geometry: Sequence[Atom] | None
    Starting molecular geometry

  final_geometry: Sequence[Atom] | None
    Optimized geometry (for geometry tasks)

Energy Fields:
  final_energy: float | None
    Final total energy in Hartree (e.g., SCF + dispersion)

  nuclear_repulsion_energy: float | None
    Nuclear repulsion energy in Hartree

Parsed Results (all optional):
  scf: ScfResults | None
    Self-consistent field convergence data

  orbitals: OrbitalsSet | None
    Molecular orbital energies and occupations

  multipole: MultipoleResults | None
    Dipole, quadrupole, octopole, hexadecapole moments

  smd: SmdResults | None
    SMD solvation model results

  tddft: TddftResults | None
    Time-dependent DFT excited state results

  dispersion: DispersionCorrection | None
    DFT-D3/D4 dispersion correction data

  timing: TimingResults | None
    Calculation timing information

  atomic_charges: Sequence[AtomicCharges]
    Population analyses (Mulliken, Loewdin, Hirshfeld, etc.)

  program_specific: Mapping[str, Any]
    Program-specific data not covered by generic models


FUNDAMENTAL MODELS
------------------

Atom:
  symbol: str                    # Element symbol (capitalized, e.g., "O", "H")
  x: float                       # X coordinate in Angstrom
  y: float                       # Y coordinate in Angstrom
  z: float                       # Z coordinate in Angstrom

Orbital:
  index: int                     # 0-based orbital index
  energy: float                  # Orbital energy in Hartree
  occupation: float | None       # Occupation number (0.0, 1.0, 2.0 for RHF)
  energy_ev: float | None        # Orbital energy in eV (optional)


SCF (SELF-CONSISTENT FIELD) MODELS
-----------------------------------

ScfIteration:
  iteration: int                 # Iteration number
  energy: float                  # Energy at this iteration (Hartree)
  # ORCA-style convergence metrics:
  delta_e_eh: float | None       # Energy change
  rmsdp: float | None            # RMS density change
  maxdp: float | None            # Max density change
  # Q-Chem-style convergence:
  diis_error: float | None       # DIIS error
  # MOM-specific fields:
  mom_active: bool | None        # Whether MOM is active
  mom_overlap_current: float | None
  mom_overlap_target: float | None

ScfEnergyComponents:
  nuclear_repulsion: float       # Nuclear repulsion energy (Hartree)
  electronic_eh: float           # Electronic energy (Hartree)
  one_electron_eh: float         # One-electron contribution
  two_electron_eh: float         # Two-electron contribution
  xc_eh: float | None            # Exchange-correlation (DFT only)

ScfResults:
  converged: bool                # Whether SCF converged
  energy: float                  # Final SCF energy (Hartree)
  n_iterations: int              # Number of iterations performed
  iterations: Sequence[ScfIteration]  # Per-iteration data
  components: ScfEnergyComponents | None  # Energy breakdown


MOLECULAR ORBITALS MODEL
-------------------------

OrbitalsSet:
  alpha_orbitals: Sequence[Orbital]
    Alpha spin orbitals (or all orbitals for RHF/RKS)

  beta_orbitals: Sequence[Orbital] | None
    Beta spin orbitals (UHF/UKS only)

  alpha_homo_index: int | None   # Index of alpha HOMO
  alpha_lumo_index: int | None   # Index of alpha LUMO
  beta_homo_index: int | None    # Index of beta HOMO (UHF/UKS)
  beta_lumo_index: int | None    # Index of beta LUMO (UHF/UKS)


PROPERTY MODELS
---------------

AtomicCharges:
  method: str                    # "Mulliken", "Loewdin", "Hirshfeld", etc.
  charges: Mapping[int, float]   # Atom index (0-based) -> charge
  spins: Mapping[int, float] | None  # Atom index -> spin density (UKS)
  # TDDFT excited state analysis fields:
  hole_populations: Mapping[int, float] | None        # h+ (RKS)
  electron_populations: Mapping[int, float] | None    # e- (RKS)
  hole_populations_alpha: Mapping[int, float] | None  # h+ alpha (UKS)
  hole_populations_beta: Mapping[int, float] | None   # h+ beta (UKS)
  electron_populations_alpha: Mapping[int, float] | None  # e- alpha (UKS)
  electron_populations_beta: Mapping[int, float] | None   # e- beta (UKS)
  trans_charges: Mapping[int, float] | None           # Transition charges
  del_q: Mapping[int, float] | None                   # Charge difference

DipoleMoment:
  x: float                       # X component (Debye)
  y: float                       # Y component (Debye)
  z: float                       # Z component (Debye)
  magnitude: float               # Total magnitude (Debye)

QuadrupoleMoment:
  xx, xy, yy, xz, yz, zz: float  # Components in Debye-Ang

OctopoleMoment:
  xxx, xxy, xyy, yyy, xxz, xyz, yyz, xzz, yzz, zzz: float  # Debye-Ang^2

HexadecapoleMoment:
  xxxx, xxxy, xxyy, xyyy, yyyy, xxxz, xxyz, xyyz, yyyz, xxzz, xyzz, yyzz, xzzz, yzzz, zzzz: float  # Debye-Ang^3

MultipoleResults:
  charge: float | None           # Total charge (ESU x 10^10)
  dipole: DipoleMoment | None
  quadrupole: QuadrupoleMoment | None
  octopole: OctopoleMoment | None
  hexadecapole: HexadecapoleMoment | None

DispersionCorrection:
  method: str                    # "DFTD3", "DFTD4"
  e_disp_au: float               # Dispersion energy (Hartree)
  functional: str | None         # e.g., "omegaB97X-D3"
  damping: str | None            # e.g., "zero damping"
  molecular_c6_au: float | None  # C6 coefficient (au)
  parameters: Mapping[str, float] | None  # Scaling/damping parameters
  e_disp_kcal: float | None      # Dispersion energy (kcal/mol)
  e6_kcal: float | None          # E6 component (kcal/mol)
  e8_kcal: float | None          # E8 component (kcal/mol)
  e8_percentage: float | None    # E8 contribution percentage

SmdResults:
  g_pcm_kcal_mol: float | None   # Polarization energy (kcal/mol)
  g_cds_kcal_mol: float | None   # Non-electrostatic CDS (kcal/mol)
  g_enp_au: float | None         # E_SCF + G_PCM (Hartree)
  g_tot_au: float | None         # Total free energy (Hartree)

TimingResults:
  total_cpu_time_seconds: float | None
  total_wall_time_seconds: float | None
  module_times: Mapping[str, float] | None  # Module name -> seconds


TDDFT & EXCITED STATE MODELS
-----------------------------

OrbitalTransition:
  from_idx: int                  # Source orbital index
  to_idx: int                    # Target orbital index
  amplitude: float               # Transition amplitude/coefficient
  is_alpha_spin: bool | None     # True=alpha, False=beta, None=unspecified

ExcitedState:
  state_number: int              # State number (1-based)
  multiplicity: Literal["Singlet", "Triplet", "Unknown"]
  excitation_energy_ev: float    # Excitation energy (eV)
  total_energy_au: float         # Total energy of excited state (Hartree)
  oscillator_strength: float | None  # Oscillator strength (f)
  transitions: Sequence[OrbitalTransition]  # Dominant transitions
  trans_mom_x: float | None      # Transition moment X (au)
  trans_mom_y: float | None      # Transition moment Y (au)
  trans_mom_z: float | None      # Transition moment Z (au)

NTOContribution:
  hole_offset: int               # e.g., -2 for H-2
  electron_offset: int           # e.g., +3 for L+3
  weight_percent: float          # Weight percentage
  is_alpha_spin: bool            # Alpha or beta spin

NTOStateAnalysis:
  state_number: int
  contributions: Sequence[NTOContribution]
  omega_percent: float | None    # Total character
  omega_alpha_percent: float | None
  omega_beta_percent: float | None

GroundStateReference:
  frontier_nos: Sequence[float]  # Natural orbital occupations
  num_electrons: float           # Total electron count
  mulliken: AtomicCharges        # Mulliken charges/spins
  dipole_moment_debye: float     # Total dipole (Debye)
  dipole_components_debye: tuple[float, float, float]  # (X, Y, Z)
  num_unpaired_electrons: float | None  # n_u value

NaturalOrbitals:
  frontier_occupations: Sequence[float]
  num_electrons: float
  num_unpaired: float | None     # n_u
  num_unpaired_nl: float | None  # n_u,nl
  pr_no: float | None            # NO participation ratio

ExcitonAnalysis:
  r_h_ang: tuple[float, float, float]  # Hole position (X, Y, Z) [Ang]
  r_e_ang: tuple[float, float, float]  # Electron position (X, Y, Z) [Ang]
  separation_ang: float          # |r_e - r_h| [Ang]
  hole_size_ang: float           # Hole size [Ang]
  electron_size_ang: float       # Electron size [Ang]
  hole_size_components_ang: tuple[float, float, float] | None
  electron_size_components_ang: tuple[float, float, float] | None
  # Transition density matrix specific:
  rms_separation_ang: float | None
  rms_separation_components_ang: tuple[float, float, float] | None
  covariance: float | None
  correlation_coef: float | None
  center_of_mass_size_ang: float | None
  center_of_mass_components_ang: tuple[float, float, float] | None
  trans_dipole_moment_debye: float | None
  trans_dipole_moment_components_debye: tuple[float, float, float] | None
  trans_r2_au: float | None
  trans_r2_components_au: tuple[float, float, float] | None

UnrelaxedDensityMatrix:
  state_number: int
  nos_spin_traced: NaturalOrbitals  # Main/only (RKS) or spin-traced (UKS)
  mulliken: AtomicCharges        # State/difference density charges
  molecular_charge: float
  num_electrons: float
  dipole_moment_debye: float
  dipole_components_debye: tuple[float, float, float]
  exciton_total: ExcitonAnalysis
  multiplicity: str | None
  nos_alpha: NaturalOrbitals | None  # UKS only
  nos_beta: NaturalOrbitals | None   # UKS only
  exciton_alpha: ExcitonAnalysis | None  # UKS only
  exciton_beta: ExcitonAnalysis | None   # UKS only

TransitionDensityMatrix:
  state_number: int
  exciton_total: ExcitonAnalysis
  multiplicity: str | None
  mulliken: AtomicCharges | None
  # CT numbers and metrics:
  sum_abs_trans_charges: float | None    # QTa
  sum_squared_trans_charges: float | None  # QT2
  omega: float | None
  omega_alpha: float | None      # UKS only
  omega_beta: float | None       # UKS only
  two_alpha_beta: float | None   # 2<alpha|beta>
  loc: float | None              # LOC
  loc_alpha: float | None        # UKS only
  loc_beta: float | None         # UKS only
  loca: float | None             # LOCa
  loca_alpha: float | None       # UKS only
  loca_beta: float | None        # UKS only
  phe: float | None              # <Phe>
  phe_alpha: float | None        # UKS only
  phe_beta: float | None         # UKS only
  trans_dipole_moment_debye: float | None
  trans_r2_au: float | None
  trans_dipole_components_debye: tuple[float, float, float] | None
  trans_r2_components_au: tuple[float, float, float] | None
  exciton_alpha: ExcitonAnalysis | None  # UKS only
  exciton_beta: ExcitonAnalysis | None   # UKS only

TddftResults:
  tda_states: Sequence[ExcitedState] | None          # TDA results
  tddft_states: Sequence[ExcitedState] | None        # Full TDDFT results
  nto_analyses: Sequence[NTOStateAnalysis] | None    # NTO analysis
  ground_state_ref: GroundStateReference | None      # Ground state reference
  unrelaxed_density_matrices: Sequence[UnrelaxedDensityMatrix] | None
  transition_density_matrices: Sequence[TransitionDensityMatrix] | None


METADATA MODEL
--------------

CalculationMetadata:
  software_name: str             # "ORCA", "Q-Chem", etc.
  software_version: str | None   # e.g., "5.0.3", "6.2"


SERIALIZATION METHODS
---------------------
All models inherit from FrozenModel, providing:

  .to_dict() -> dict[str, Any]
    Recursively converts to dictionary
    Note: CalculationResult.to_dict() excludes raw_output to save space

  .from_dict(data: dict[str, Any]) -> Self
    Reconstructs from dictionary
    Note: CalculationResult.from_dict() sets raw_output to "" if missing

  .to_json(indent: int = 2) -> str
    Serializes to JSON string

  .from_json(json_str: str) -> Self
    Deserializes from JSON string


USAGE EXAMPLES
--------------

1. Parse a calculation output:
    from calcflow.io.qchem import parse_qchem_output
    from calcflow.io.orca import parse_orca_output

    # Parse Q-Chem output
    with open("qchem.out") as f:
        result = parse_qchem_output(f.read())

    # Parse ORCA output
    with open("orca.out") as f:
        result = parse_orca_output(f.read())

    # Check if successful
    if result.termination_status == "NORMAL":
        print(f"Final energy: {result.final_energy} Hartree")


2. Access SCF results:
    result = parse_qchem_output(output_text)

    if result.scf:
        print(f"SCF converged: {result.scf.converged}")
        print(f"SCF energy: {result.scf.energy} Hartree")
        print(f"Iterations: {result.scf.n_iterations}")

        # Access last iteration
        last_iter = result.scf.iterations[-1]
        print(f"Final DIIS error: {last_iter.diis_error}")

        # Energy components (if available)
        if result.scf.components:
            print(f"Nuclear repulsion: {result.scf.components.nuclear_repulsion}")
            print(f"Electronic energy: {result.scf.components.electronic_eh}")


3. Work with molecular orbitals:
    if result.orbitals:
        # Find HOMO-LUMO gap
        homo_idx = result.orbitals.alpha_homo_index
        lumo_idx = result.orbitals.alpha_lumo_index

        if homo_idx is not None and lumo_idx is not None:
            homo = result.orbitals.alpha_orbitals[homo_idx]
            lumo = result.orbitals.alpha_orbitals[lumo_idx]
            gap_ev = (lumo.energy - homo.energy) * 27.2114  # Hartree to eV
            print(f"HOMO-LUMO gap: {gap_ev:.2f} eV")

        # List occupied orbitals
        occupied = [orb for orb in result.orbitals.alpha_orbitals
                   if orb.occupation and orb.occupation > 0]
        print(f"Number of occupied orbitals: {len(occupied)}")


4. Extract excited state data:
    if result.tddft and result.tddft.tddft_states:
        for state in result.tddft.tddft_states[:5]:  # First 5 states
            print(f"State {state.state_number}: "
                  f"{state.excitation_energy_ev:.2f} eV "
                  f"({state.multiplicity})")
            if state.oscillator_strength:
                print(f"  f = {state.oscillator_strength:.4f}")

            # Dominant transitions
            for trans in state.transitions[:3]:  # Top 3
                print(f"  {trans.from_idx} -> {trans.to_idx} "
                      f"(amplitude: {trans.amplitude:.2f})")


5. Access population analysis:
    if result.atomic_charges:
        for charges in result.atomic_charges:
            print(f"\n{charges.method} charges:")
            for atom_idx, charge in charges.charges.items():
                atom = result.input_geometry[atom_idx]
                print(f"  {atom.symbol}{atom_idx}: {charge:+.3f}")

            # Spin densities (if UKS)
            if charges.spins:
                print(f"\n{charges.method} spin densities:")
                for atom_idx, spin in charges.spins.items():
                    print(f"  Atom {atom_idx}: {spin:+.3f}")


6. Check dipole moment:
    if result.multipole and result.multipole.dipole:
        dipole = result.multipole.dipole
        print(f"Dipole moment: {dipole.magnitude:.3f} Debye")
        print(f"  X: {dipole.x:.3f}")
        print(f"  Y: {dipole.y:.3f}")
        print(f"  Z: {dipole.z:.3f}")


7. Extract solvation data:
    if result.smd:
        print(f"G_PCM: {result.smd.g_pcm_kcal_mol} kcal/mol")
        print(f"G_CDS: {result.smd.g_cds_kcal_mol} kcal/mol")
        print(f"Total free energy: {result.smd.g_tot_au} Hartree")


8. Get dispersion correction:
    if result.dispersion:
        print(f"Method: {result.dispersion.method}")
        print(f"E_disp: {result.dispersion.e_disp_au} Hartree")
        if result.dispersion.e_disp_kcal:
            print(f"E_disp: {result.dispersion.e_disp_kcal} kcal/mol")


9. Save results to JSON:
    result = parse_qchem_output(output_text)

    # Serialize to JSON (raw_output automatically excluded)
    with open("result.json", "w") as f:
        f.write(result.to_json())

    # Load from JSON
    with open("result.json") as f:
        loaded_result = CalculationResult.from_json(f.read())

    # Verify (note: raw_output will be empty string in loaded_result)
    assert loaded_result.final_energy == result.final_energy
    assert loaded_result.scf.converged == result.scf.converged


10. Work with geometry:
    if result.input_geometry:
        print("Initial geometry:")
        for atom in result.input_geometry:
            print(f"  {atom.symbol:2s} {atom.x:10.6f} {atom.y:10.6f} {atom.z:10.6f}")

    if result.final_geometry:
        print("\nOptimized geometry:")
        for atom in result.final_geometry:
            print(f"  {atom.symbol:2s} {atom.x:10.6f} {atom.y:10.6f} {atom.z:10.6f}")


11. Analyze exciton properties (Q-Chem TDDFT):
    if result.tddft and result.tddft.transition_density_matrices:
        for tdm in result.tddft.transition_density_matrices:
            print(f"\nState {tdm.state_number}:")
            exciton = tdm.exciton_total
            print(f"  Hole-electron separation: {exciton.separation_ang:.3f} Å")
            print(f"  Hole size: {exciton.hole_size_ang:.3f} Å")
            print(f"  Electron size: {exciton.electron_size_ang:.3f} Å")

            # CT metrics
            if tdm.omega:
                print(f"  Omega: {tdm.omega:.3f}")
            if tdm.loc:
                print(f"  LOC: {tdm.loc:.3f}")


12. Handle MOM calculations (multi-job outputs):
    # MOM outputs are parsed as a single CalculationResult
    # with the final job's data
    result = parse_qchem_output(mom_output_text)

    # Check program_specific for MOM-related data
    if "mom" in result.program_specific:
        print("MOM calculation detected")
        print(result.program_specific["mom"])


13. Extract timing information:
    if result.timing:
        if result.timing.total_wall_time_seconds:
            minutes = result.timing.total_wall_time_seconds / 60
            print(f"Total wall time: {minutes:.1f} minutes")

        if result.timing.module_times:
            print("\nModule timings:")
            for module, seconds in result.timing.module_times.items():
                print(f"  {module}: {seconds:.1f} s")


14. Complete workflow - parse, analyze, save:
    from calcflow.io.qchem import parse_qchem_output
    from pathlib import Path

    # Parse output
    output_path = Path("calculation.out")
    result = parse_qchem_output(output_path.read_text())

    # Check status
    if result.termination_status != "NORMAL":
        print("Calculation failed!")
        exit(1)

    # Extract key data
    data = {
        "energy": result.final_energy,
        "converged": result.scf.converged if result.scf else None,
        "dipole": result.multipole.dipole.magnitude if result.multipole and result.multipole.dipole else None,
    }

    # Add excited state data
    if result.tddft and result.tddft.tddft_states:
        data["excitations"] = [
            {
                "state": s.state_number,
                "energy_ev": s.excitation_energy_ev,
                "f": s.oscillator_strength,
            }
            for s in result.tddft.tddft_states
        ]

    # Save to JSON
    import json
    with open("analysis.json", "w") as f:
        json.dump(data, f, indent=2)

    # Save full result (without raw_output)
    with open("full_result.json", "w") as f:
        f.write(result.to_json())


IMPORTANT NOTES
---------------
1. All models are immutable (frozen dataclasses). Once created, they cannot be modified.

2. CalculationResult.to_dict() and .to_json() automatically EXCLUDE raw_output
   to save space. The raw output is typically very large and not needed for
   serialization.

3. When deserializing with .from_dict() or .from_json(), raw_output is set to
   an empty string ("").

4. Optional fields default to None. Check for None before accessing nested attributes.

5. Units are standardized:
   - Energy: Hartree (unless field name specifies otherwise like _ev or _kcal_mol)
   - Distance: Angstrom
   - Dipole moments: Debye
   - Time: seconds

6. For UHF/UKS calculations, beta_orbitals will be populated. For RHF/RKS,
   only alpha_orbitals is used.

7. TDDFT results can contain multiple types of analyses:
   - tda_states/tddft_states: Basic excitation data
   - nto_analyses: Natural transition orbital analysis
   - ground_state_ref: Ground state reference data
   - unrelaxed_density_matrices: Excited state density analysis
   - transition_density_matrices: Transition density with CT metrics

8. Atom indices are 0-based throughout all models.

9. State numbers in TDDFT are 1-based (S1, S2, etc.).

10. All models support perfect roundtrip serialization (except raw_output):
    reconstructed = Model.from_dict(original.to_dict())
    assert reconstructed == original  # True (ignoring raw_output)
""".strip()
