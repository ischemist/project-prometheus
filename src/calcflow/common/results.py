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
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from calcflow._version import __version__ as _CALCFLOW_VERSION
from calcflow.common.models import Atom, FrozenModel
from calcflow.common.types import AdcSpin, Matrix3x3
from calcflow.geometry.static import Geometry

# Re-export so that existing ``from calcflow.common.results import Atom``
# and ``from calcflow.common.results import FrozenModel`` imports keep working.
__all__ = ["Atom", "FrozenModel"]

logger = logging.getLogger(__name__)

# Schema version for CalculationResult serialization format.
# Increment this when the serialized structure changes (field renames, removals,
# type changes) and add a corresponding migration step in CalculationResult._migrate().
#
# v1 → v2: input_geometry / final_geometry changed from a bare list of atom
#           dicts  [{"symbol": "O", "x": 0, ...}, ...]  to a Geometry object
#           {"comment": "", "atoms": [{"symbol": "O", ...}, ...]}.
RESULT_SCHEMA_VERSION: int = 2

# =============================================================================
# §1. FUNDAMENTAL BUILDING BLOCKS
# =============================================================================


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

    def to_list(self, n_atoms: int) -> list[float]:
        """Returns charges as a list of length n_atoms, indexed by atom position.

        Missing indices (sparse charges) default to 0.0. n_atoms must be provided
        explicitly since AtomicCharges has no geometry reference.

        Args:
            n_atoms: expected number of atoms; defines the output list length.

        Returns:
            List of floats of length n_atoms.

        Raises:
            ValueError: if any charge index is >= n_atoms.
        """
        out_of_range = [i for i in self.charges if i >= n_atoms]
        if out_of_range:
            raise ValueError(f"charge indices {sorted(out_of_range)} are out of range for n_atoms={n_atoms}")
        return [self.charges.get(i, 0.0) for i in range(n_atoms)]


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
# §4. ADC MODELS
# =============================================================================


@dataclass(frozen=True)
class TwoPhotonAbsorption(FrozenModel):
    """Two-photon absorption data for a single excited state."""

    cross_section_au: float
    matrix_au: Matrix3x3  # 3x3 TPA tensor in atomic units


@dataclass(frozen=True)
class AdcAmplitude(FrozenModel):
    """A single amplitude contribution in an ADC excited state."""

    occ_i: int  # occupied orbital index (1-based as in output)
    vir_a: int  # virtual orbital index (1-based as in output)
    amplitude: float
    spin: AdcSpin | None = None  # "A" (alpha) or "B" (beta) for the occ_i/vir_a pair
    occ_j: int | None = None  # second occupied index for 2p2h configurations
    vir_b: int | None = None  # second virtual index for 2p2h configurations


@dataclass(frozen=True)
class AdcGroundState(FrozenModel):
    """Ground state (HF + MP2) data from an ADC calculation."""

    hf_energy_au: float
    mp2_correlation_energy_au: float
    total_energy_au: float  # HF + MP2 correlation
    # Natural orbital analysis of the ground-state density matrix
    nos_alpha: NaturalOrbitals | None = None
    nos_beta: NaturalOrbitals | None = None
    nos_spin_traced: NaturalOrbitals | None = None
    # Mulliken population analysis
    mulliken: AtomicCharges | None = None
    # Multipole
    dipole_moment_debye: float | None = None
    dipole_components_debye: tuple[float, float, float] | None = None
    # Exciton analysis of the difference density matrix
    exciton_total: ExcitonAnalysis | None = None
    exciton_alpha: ExcitonAnalysis | None = None
    exciton_beta: ExcitonAnalysis | None = None


@dataclass(frozen=True)
class AdcExcitedState(FrozenModel):
    """All data for a single ADC(2) excited state."""

    state_number: int
    total_energy_au: float
    excitation_energy_ev: float
    oscillator_strength: float | None = None
    trans_dip_moment_au: tuple[float, float, float] | None = None
    r2_au: tuple[float, float, float] | None = None
    two_photon_absorption: TwoPhotonAbsorption | None = None
    dip_moment_au: tuple[float, float, float] | None = None
    total_dipole_debye: float | None = None
    v1_squared: float | None = None
    v2_squared: float | None = None
    amplitudes: Sequence[AdcAmplitude] = field(default_factory=list)
    # Density matrix analysis (difference DM)
    nos_alpha: NaturalOrbitals | None = None
    nos_beta: NaturalOrbitals | None = None
    nos_spin_traced: NaturalOrbitals | None = None
    mulliken: AtomicCharges | None = None
    dipole_dm_debye: float | None = None
    dipole_dm_components_debye: tuple[float, float, float] | None = None
    exciton_diff_total: ExcitonAnalysis | None = None
    exciton_diff_alpha: ExcitonAnalysis | None = None
    exciton_diff_beta: ExcitonAnalysis | None = None
    # CT numbers (transition DM)
    ct_omega: float | None = None
    ct_omega_alpha: float | None = None
    ct_omega_beta: float | None = None
    ct_phe: float | None = None
    ct_phe_alpha: float | None = None
    ct_phe_beta: float | None = None
    # Exciton analysis of the transition DM
    exciton_trans_total: ExcitonAnalysis | None = None
    exciton_trans_alpha: ExcitonAnalysis | None = None
    exciton_trans_beta: ExcitonAnalysis | None = None
    # NTO decomposition (state-averaged)
    nto_alpha: Sequence[NTOContribution] | None = None
    nto_beta: Sequence[NTOContribution] | None = None


@dataclass(frozen=True)
class AdcResults(FrozenModel):
    """Container for all ADC(2) parsed data."""

    method: str  # e.g. "adc(2)"
    ground_state: AdcGroundState | None = None
    excited_states: Sequence[AdcExcitedState] = field(default_factory=list)


# =============================================================================
# §4b. SCHEMA MIGRATION HELPERS
# =============================================================================


def _migrate_result_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v1 CalculationResult dict to v2.

    v1 stored ``input_geometry`` / ``final_geometry`` as bare lists of atom
    dicts: ``[{"symbol": "O", "x": 0.0, "y": 0.0, "z": 0.0}, ...]``.

    v2 stores them as Geometry dicts: ``{"comment": "", "atoms": [...]}``.
    """
    data = {**data}
    for field_name in ("input_geometry", "final_geometry"):
        raw = data.get(field_name)
        if isinstance(raw, list):
            data[field_name] = {"comment": "", "atoms": raw}
        # None or already a dict → leave as-is
    return data


# =============================================================================
# §5. TOP-LEVEL RESULT MODELS
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
    input_geometry: Geometry | None = None
    final_geometry: Geometry | None = None

    # --- Energies ---
    final_energy: float | None = None  # e.g., SCF+Dispersion
    nuclear_repulsion_energy: float | None = None

    # --- Parsed Blocks (Optional) ---
    scf: ScfResults | None = None
    orbitals: OrbitalsSet | None = None
    multipole: MultipoleResults | None = None
    smd: SmdResults | None = None
    tddft: TddftResults | None = None
    adc: AdcResults | None = None
    dispersion: DispersionCorrection | None = None
    timing: TimingResults | None = None
    atomic_charges: Sequence[AtomicCharges] = field(default_factory=list)

    # --- Program Specific Data ---
    program_specific: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts to dictionary, excluding raw_output to save space.

        Includes version metadata for compatibility tracking:
        - ``calcflow_version``: package semver string (provenance/auditing).
        - ``schema_version``: integer format version (compatibility/migration).
        """
        data = super().to_dict()
        data.pop("raw_output", None)
        data["calcflow_version"] = _CALCFLOW_VERSION
        data["schema_version"] = RESULT_SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalculationResult":
        """Reconstructs from dictionary, setting raw_output to empty string.

        Handles version metadata:
        - ``calcflow_version`` is stripped (provenance only, not a dataclass field).
        - ``schema_version`` is read and used to apply any necessary migrations
          before constructing the instance.  Dumps without ``schema_version``
          (produced before this feature existed) are treated as version 1.
        """
        data = {**data}  # copy to avoid mutating input
        data.pop("calcflow_version", None)
        incoming_version = data.pop("schema_version", 1)
        data = cls._migrate(data, incoming_version)
        if "raw_output" not in data:
            data["raw_output"] = ""
        return super().from_dict(data)

    @staticmethod
    def _migrate(data: dict[str, Any], from_version: int) -> dict[str, Any]:
        """Apply sequential migrations from *from_version* to the current schema.

        Each migration step transforms the dict from version *N* to *N+1*.
        Add new migration blocks here when ``RESULT_SCHEMA_VERSION`` is bumped::

            if from_version < 3:
                data = _migrate_result_v2_to_v3(data)
        """
        if from_version > RESULT_SCHEMA_VERSION:
            logger.warning(
                "CalculationResult dump was created with schema version %d, "
                "but this code only understands version %d. "
                "Some fields may be missing or misinterpreted.",
                from_version,
                RESULT_SCHEMA_VERSION,
            )
        elif from_version < RESULT_SCHEMA_VERSION:
            logger.warning(
                "Migrating CalculationResult from schema version %d to %d.",
                from_version,
                RESULT_SCHEMA_VERSION,
            )
        if from_version < 2:
            data = _migrate_result_v1_to_v2(data)
        return data

    def get_charges(self, method: str) -> "AtomicCharges | None":
        """Returns the AtomicCharges for the given population analysis method, or None.

        Method names are case-sensitive and match how parsers store them
        (e.g. "Mulliken", "Hirshfeld", "CM5", "Loewdin").
        """
        return next((ac for ac in self.atomic_charges if ac.method == method), None)

    @classmethod
    def get_schema(cls) -> str:
        """Returns an auto-generated structural map of every result field.

        Walks the dataclass hierarchy recursively and emits an indented tree of
        field names with their types and inline docstring annotations where available.
        Use this to navigate result fields without reading source code.

        units convention (unless field name specifies otherwise):
          energy: Hartree  |  _ev suffix: eV  |  _kcal_mol: kcal/mol
          distance/size: Angstrom  |  dipole/trans moment: Debye  |  time: seconds
        """
        _seen: set[type] = set()

        def _type_str(t: Any) -> str:
            """compact representation of a type annotation."""
            origin = get_origin(t)
            args = get_args(t)
            # Literal["A", "B"] → Literal["A", "B"]
            if origin is Literal:
                return "Literal[" + ", ".join(repr(a) for a in args) + "]"
            # handle X | Y (python 3.10+ UnionType) and Union[X, Y]
            if isinstance(t, UnionType) or origin is Union:
                non_none = [a for a in args if a is not type(None)]
                has_none = type(None) in args
                parts = " | ".join(_type_str(a) for a in non_none)
                return parts + (" | None" if has_none else "")
            # Sequence / list / tuple
            if (origin is not None and getattr(origin, "__name__", "") in ("Sequence", "List")) or origin in (
                list,
                tuple,
            ):
                return f"list[{_type_str(args[0])}]" if args else "list"
            # Mapping / dict
            if (origin is not None and getattr(origin, "__name__", "") in ("Mapping", "Dict")) or origin is dict:
                k = _type_str(args[0]) if args else "?"
                v = _type_str(args[1]) if len(args) > 1 else "?"
                return f"dict[{k}, {v}]"
            if hasattr(t, "__name__"):
                return t.__name__
            return repr(t)

        def _render(klass: type, indent: int = 0) -> list[str]:
            pad = "  " * indent
            lines: list[str] = []
            if not dataclasses.is_dataclass(klass):
                return lines
            if klass in _seen:
                lines.append(f"{pad}  (see {klass.__name__} above)")
                return lines
            _seen.add(klass)

            hints = {f.name: f.type for f in dataclasses.fields(klass)}

            # resolve string annotations where possible
            try:
                resolved = get_type_hints(klass)
            except NameError:
                resolved = hints

            for f in dataclasses.fields(klass):
                raw_type = resolved.get(f.name, f.type)
                tstr = _type_str(raw_type)
                default = ""
                if f.default is not dataclasses.MISSING:
                    default = f" = {f.default!r}"
                elif f.default_factory is not dataclasses.MISSING:
                    try:
                        default = f" = {f.default_factory()!r}"
                    except Exception:
                        default = " = ..."
                # inline comment from field default comment if field name hints units
                lines.append(f"{pad}  .{f.name}: {tstr}{default}")

                # recurse into nested dataclasses (unwrap Optional/list first)
                inner = raw_type
                args = get_args(raw_type)
                if args:
                    # pick first non-None arg
                    candidates = [a for a in args if a is not type(None)]
                    inner = candidates[0] if candidates else raw_type
                    # unwrap list/Sequence
                    inner_args = get_args(inner)
                    if inner_args:
                        inner = inner_args[0]
                if dataclasses.is_dataclass(inner) and inner not in _seen:
                    lines.extend(_render(cast(type, inner), indent + 1))

            return lines

        out: list[str] = [
            "CalculationResult — field schema (auto-generated)",
            "=" * 55,
            "",
            "units: energy=Hartree (_ev=eV, _kcal_mol=kcal/mol) | distance/size=Angstrom | dipole=Debye | time=seconds",
            "",
            "parse functions:",
            "  from calcflow.io.qchem import parse_qchem_output, parse_qchem_multi_job_output",
            "  from calcflow.io.orca import parse_orca_output",
            "  result = parse_qchem_output(Path('calc.out').read_text())",
            "  result = parse_orca_output(Path('calc.out').read_text())",
            "  jobs   = parse_qchem_multi_job_output(text)  # list[CalculationResult]",
            "",
            "gzip-compressed files (.json.gz):",
            "  import gzip, json",
            "  raw = json.loads(gzip.decompress(Path('result.json.gz').read_bytes()))",
            "  # single result:     result = CalculationResult.from_dict(raw)",
            "  # list of results:   jobs   = [CalculationResult.from_dict(j) for j in raw]",
            "",
            "serialization:",
            "  result.to_json()                  # str; save with Path('out.json').write_text(...)",
            "  CalculationResult.from_json(s)    # load from json string",
            "  CalculationResult.from_dict(d)    # load from dict",
            "  # multi-job list roundtrip:",
            "  import json",
            "  json.dumps([r.to_dict() for r in jobs])          # serialize list",
            "  [CalculationResult.from_dict(d) for d in data]   # deserialize list",
            "",
            "annotated views (atom-centric access, co-locates geometry + result):",
            "  from calcflow import AnnotatedGeometry",
            "  ag = AnnotatedGeometry.from_result(result)",
            "  ag[0].charges                         # {'Mulliken': -0.42, ...}  ground state",
            "  ag[0].spins                           # {'Mulliken': 0.1, ...}    UKS only",
            "  ag.get_unrelaxed_state(1)[0].hole_population    # TDDFT unrelaxed DM",
            "  ag.get_transition_state(1)[0].trans_charge      # TDDFT transition DM",
            "  ag.get_adc_state(1)[0].hole_population          # ADC excited state",
            "  # all three return None if the data was not parsed",
            "",
            "CalculationResult",
        ]
        out.extend(_render(cls))
        out += [
            "",
            "notes:",
            "  - all optional fields default to None; guard with 'if result.field:'",
            "  - atom indices are 0-based; tddft state numbers are 1-based",
            "  - raw_output is excluded from to_dict()/to_json() serialization",
            "  - for UHF/UKS: beta_orbitals populated; for RHF/RKS: only alpha_orbitals",
        ]
        return "\n".join(out)

    @classmethod
    def get_api_docs(cls) -> str:
        """compatibility alias — returns get_schema() output.

        prefer get_schema() directly.
        """
        return cls.get_schema()
