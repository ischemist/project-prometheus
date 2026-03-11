from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any, Literal, TypeVar, cast

from calcflow._version import __version__ as _CALCFLOW_VERSION
from calcflow.common.exceptions import ConfigurationError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.io.orca.builder import OrcaBuilder
from calcflow.io.qchem.builder import QchemBuilder

logger = logging.getLogger(__name__)

T_CalculationInput = TypeVar("T_CalculationInput", bound="CalculationInput")
T_SpecSerializable = TypeVar("T_SpecSerializable", bound="_SpecSerializable")
type TASK_TYPES = Literal["energy", "geometry", "frequency"]

# Schema version for CalculationInput serialization format.
# Increment this when the serialized structure changes (field renames, removals,
# type changes) and add a corresponding migration step in CalculationInput._migrate().
INPUT_SCHEMA_VERSION: int = 1

# registry of program-specific builders.
BUILDERS = {"orca": OrcaBuilder(), "qchem": QchemBuilder()}

# --- Component Specifications ---
# these are the building blocks. if a component is None in the main spec,
# that feature is simply not requested.


class _SpecSerializable:
    """Shared strict serialization helpers for user-facing input specs.

    Unknown keys are rejected in ``from_dict`` to catch misspelled user options early.
    """

    def to_dict(self) -> dict[str, Any]:
        """serializes to a dictionary."""
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass to use _SpecSerializable.")
        return asdict(cast(Any, self))

    @classmethod
    def from_dict(cls: type[T_SpecSerializable], data: dict[str, Any]) -> T_SpecSerializable:
        """deserializes from a dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class TddftSpec(_SpecSerializable):
    """specification for a time-dependent dft calculation."""

    nroots: int
    singlets: bool = True
    triplets: bool = False
    use_tda: bool = True  # Tamm-Dancoff Approximation is a common choice
    state_to_optimize: int | None = None  # for geometry optimization of an excited state


@dataclass(frozen=True)
class SolvationSpec(_SpecSerializable):
    """specification for an implicit solvation model."""

    model: str  # e.g., 'smd', 'cpcm'
    solvent: str = ""  # named solvent (e.g., 'water'). empty when using custom dielectrics.
    dielectric: float | None = None  # static dielectric constant (overrides named solvent for pcm)
    optical_dielectric: float | None = None  # high-frequency dielectric constant


@dataclass(frozen=True)
class ChargesSpec(_SpecSerializable):
    """which charge partitioning schemes to compute."""

    mulliken: bool = True
    hirshfeld: bool = False
    cm5: bool = False
    hirshiter: bool = False  # Hirshfeld-I (iterative)
    hirshiter_thresh: int = 5

    def __post_init__(self) -> None:
        if self.hirshiter_thresh < 1:
            raise ValidationError("hirshiter_thresh must be a positive integer.")
        # cm5 requires hirshfeld — auto-enable it rather than silently producing wrong output
        if self.cm5 and not self.hirshfeld:
            object.__setattr__(self, "hirshfeld", True)
        # Hirshfeld-I (hirshiter) also requires Hirshfeld to be active
        if self.hirshiter and not self.hirshfeld:
            object.__setattr__(self, "hirshfeld", True)


@dataclass(frozen=True)
class ScfSpec(_SpecSerializable):
    """scf convergence parameters."""

    algorithm: str = "diis"
    max_cycles: int = 100
    convergence: int = 8  # threshold exponent: scf converges when error < 10^-N


@dataclass(frozen=True)
class OptimizationSpec(_SpecSerializable):
    """specification for geometry optimization tasks."""

    calc_hess_initial: bool = False
    recalc_hess_freq: int | None = None


@dataclass(frozen=True)
class MomSpec(_SpecSerializable):
    """
    specification for maximum overlap method (mom) calculations.

    mom is a technique to converge scf to excited states or non-aufbau configurations
    by guiding orbital occupations. requires a two-job input: job1 computes ground state
    orbitals, job2 uses those orbitals with modified occupations to reach the target state.

    transition notation supports both symbolic and numeric orbital specifications:
    - symbolic: "HOMO->LUMO", "HOMO-1->LUMO+1", "HOMO-2->LUMO"
    - numeric: "5->6", "3->LUMO", "HOMO->7" (absolute orbital indices)
    - ionization: "HOMO->vac", "5->vac" (remove electron)
    - spin-specific: "HOMO(beta)->LUMO(alpha)", "5(alpha)->vac"
    - multiple transitions: "HOMO->LUMO; HOMO-1->LUMO+1" (semicolon-separated)

    for ionization, job2_charge and job2_spin_multiplicity should be set to match
    the ionized state (e.g., charge +1, multiplicity 2 for a neutral singlet -> cation doublet).
    """

    transition: str
    method: str = "IMOM"  # or "MOM"

    # for ionization: override charge/spin in second job
    job2_charge: int | None = None
    job2_spin_multiplicity: int | None = None

    # manual override for advanced users (bypasses symbolic transition parsing)
    alpha_occupation: str | None = None
    beta_occupation: str | None = None


# --- Main Calculation Specification ---


@dataclass(frozen=True)
class CalculationInput:
    """
    the fluent, user-facing api for building a quantum chemistry calculation.

    this is an immutable dataclass with a fluent api for progressive construction.
    each 'set' method returns a new instance with the updated field.
    """

    charge: int
    spin_multiplicity: int
    task: Literal["energy", "geometry", "frequency"]
    level_of_theory: str
    basis_set: str | dict[str, str]
    unrestricted: bool = False
    n_cores: int = 1
    memory_per_core_mb: int = 4000

    total_memory_mb: int | None = None  # program-level memory (e.g., Q-Chem MEM_TOTAL)

    # optional, modular components of the calculation
    tddft: TddftSpec | None = None
    solvation: SolvationSpec | None = None
    optimization: OptimizationSpec | None = None
    mom: MomSpec | None = None
    charges: ChargesSpec | None = None
    scf: ScfSpec | None = None
    frequency_after_optimization: bool = False

    # the escape hatch for anything program-specific that doesn't fit the generic model.
    # e.g., for orca: {"ri_approx": "RIJCOSX", "aux_basis": "def2/j"}
    program_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """basic, program-agnostic validation."""
        if self.spin_multiplicity < 1:
            raise ValidationError("spin multiplicity must be a positive integer.")
        if self.n_cores < 1:
            raise ValidationError("number of cores must be a positive integer.")
        if self.tddft and self.tddft.nroots < 1:
            raise ValidationError("tddft nroots must be a positive integer.")
        if self.solvation and not self.solvation.model:
            raise ValidationError("solvation model must be specified.")
        if self.solvation and not self.solvation.solvent and self.solvation.dielectric is None:
            raise ValidationError("solvation requires either a named solvent or a dielectric constant.")
        if self.solvation and self.solvation.dielectric is not None and self.solvation.model.lower() != "pcm":
            raise ConfigurationError(
                f"custom dielectric constants are only supported for the 'pcm' model, got '{self.solvation.model}'."
            )
        if self.mom:
            if not self.unrestricted:
                raise ValidationError("mom requires an unrestricted calculation.")
            if not self.mom.transition and not (self.mom.alpha_occupation and self.mom.beta_occupation):
                raise ValidationError(
                    "mom requires either 'transition' or both 'alpha_occupation' and 'beta_occupation'."
                )

    @property
    def requires_multiple_jobs(self) -> bool:
        """returns true if this calculation requires multiple sequential jobs (e.g., for mom)."""
        return self.mom is not None

    # --- Core Parameter Setters ---

    def set_level_of_theory(self: T_CalculationInput, lot: str) -> T_CalculationInput:
        """updates the level of theory (method/functional)."""
        return replace(self, level_of_theory=lot)

    def set_basis_set(self: T_CalculationInput, basis: str | dict[str, str]) -> T_CalculationInput:
        """updates the basis set."""
        return replace(self, basis_set=basis)

    def set_task(self: T_CalculationInput, task: TASK_TYPES) -> T_CalculationInput:
        """updates the main calculation task."""
        return replace(self, task=task)

    def set_unrestricted(self: T_CalculationInput, unrestricted: bool = True) -> T_CalculationInput:
        """sets the calculation to be unrestricted (uks/uhf) or restricted (rks/rhf)."""
        return replace(self, unrestricted=unrestricted)

    # --- Computational Resource Setters ---

    def set_cores(self: T_CalculationInput, n_cores: int) -> T_CalculationInput:
        """sets the number of cpu cores to use."""
        return replace(self, n_cores=n_cores)

    def set_memory_per_core(self: T_CalculationInput, mb: int) -> T_CalculationInput:
        """sets the memory per core in megabytes."""
        return replace(self, memory_per_core_mb=mb)

    # --- Calculation Component Setters ---

    def set_solvation(self: T_CalculationInput, model: str, solvent: str) -> T_CalculationInput:
        """adds or updates the implicit solvation model."""
        solv_spec = SolvationSpec(model=model.lower(), solvent=solvent.lower())
        return replace(self, solvation=solv_spec)

    def set_tddft(
        self: T_CalculationInput,
        nroots: int,
        singlets: bool = True,
        triplets: bool = False,
        use_tda: bool = True,
        state_to_optimize: int | None = None,
    ) -> T_CalculationInput:
        """adds or updates the tddft calculation parameters."""
        if state_to_optimize and self.task != "geometry":
            raise ConfigurationError("`state_to_optimize` is only valid for 'geometry' tasks.")
        tddft_spec = TddftSpec(
            nroots=nroots,
            singlets=singlets,
            triplets=triplets,
            use_tda=use_tda,
            state_to_optimize=state_to_optimize,
        )
        return replace(self, tddft=tddft_spec)

    def set_optimization(
        self: T_CalculationInput,
        calc_hess_initial: bool = False,
        recalc_hess_freq: int | None = None,
    ) -> T_CalculationInput:
        """adds or updates geometry optimization parameters."""
        if self.task != "geometry":
            raise ConfigurationError("optimization settings are only valid for 'geometry' tasks.")
        opt_spec = OptimizationSpec(
            calc_hess_initial=calc_hess_initial,
            recalc_hess_freq=recalc_hess_freq,
        )
        return replace(self, optimization=opt_spec)

    def run_frequency_after_opt(self: T_CalculationInput) -> T_CalculationInput:
        """enables a frequency calculation to be run after a successful geometry optimization."""
        if self.task != "geometry":
            raise ConfigurationError("frequency calculation can only follow a 'geometry' task.")
        return replace(self, frequency_after_optimization=True)

    def set_custom_solvation(
        self: T_CalculationInput,
        model: str,
        dielectric: float,
        optical_dielectric: float | None = None,
    ) -> T_CalculationInput:
        """adds or updates pcm solvation with explicit dielectric constants (no named solvent)."""
        if model.lower() != "pcm":
            raise ConfigurationError(
                f"set_custom_solvation only supports model='pcm', got '{model}'. "
                "Use set_solvation() for named-solvent non-PCM models."
            )
        solv_spec = SolvationSpec(
            model=model.lower(),
            dielectric=dielectric,
            optical_dielectric=optical_dielectric,
        )
        return replace(self, solvation=solv_spec)

    def set_charges(
        self: T_CalculationInput,
        mulliken: bool = True,
        hirshfeld: bool = False,
        cm5: bool = False,
        hirshiter: bool = False,
        hirshiter_thresh: int = 5,
    ) -> T_CalculationInput:
        """configures charge partitioning schemes. cm5=True auto-enables hirshfeld."""
        return replace(
            self,
            charges=ChargesSpec(
                mulliken=mulliken,
                hirshfeld=hirshfeld,
                cm5=cm5,
                hirshiter=hirshiter,
                hirshiter_thresh=hirshiter_thresh,
            ),
        )

    def set_scf(
        self: T_CalculationInput,
        algorithm: str = "diis",
        max_cycles: int = 100,
        convergence: int = 8,
    ) -> T_CalculationInput:
        """sets scf convergence parameters."""
        return replace(self, scf=ScfSpec(algorithm=algorithm, max_cycles=max_cycles, convergence=convergence))

    def set_total_memory(self: T_CalculationInput, mb: int) -> T_CalculationInput:
        """sets the total program memory allocation in megabytes (e.g., Q-Chem MEM_TOTAL)."""
        return replace(self, total_memory_mb=mb)

    def set_mom(
        self: T_CalculationInput,
        transition: str,
        method: str = "IMOM",
        job2_charge: int | None = None,
        job2_spin_multiplicity: int | None = None,
    ) -> T_CalculationInput:
        """
        adds or updates maximum overlap method (mom) settings for excited state calculations.

        mom requires a two-job calculation and unrestricted wavefunctions.

        args:
            transition: transition string supporting symbolic (e.g., "HOMO->LUMO") or
                numeric (e.g., "5->6", "3->LUMO") orbital specifications.
                use "->vac" for ionization (e.g., "HOMO->vac", "5->vac").
            method: mom variant ("MOM" or "IMOM", default "IMOM")
            job2_charge: charge for second job (for ionization, e.g., +1)
            job2_spin_multiplicity: spin multiplicity for second job (for ionization, e.g., 2)
        """
        mom_spec = MomSpec(
            transition=transition,
            method=method,
            job2_charge=job2_charge,
            job2_spin_multiplicity=job2_spin_multiplicity,
        )
        return replace(self, mom=mom_spec)

    # --- Program-Specific Options ---

    def set_options(self: T_CalculationInput, **kwargs: Any) -> T_CalculationInput:
        """
        the 'escape hatch': sets program-specific options that don't have a generic equivalent.

        these options are passed directly to the program-specific builder, which is
        responsible for validating and interpreting them.

        example:
            .set_options(ri_approx="RIJCOSX", aux_basis="def2/j")
        """
        # use a copy to ensure immutability
        new_opts = {**self.program_options, **kwargs}
        return replace(self, program_options=new_opts)

    # --- Convenience Wrappers for Program-Specific Options ---
    # these methods provide a discoverable, type-safe api for common
    # program-specific features, but just call `set_options` under the hood.

    def enable_ri_for_orca(self: T_CalculationInput, approx: str, aux_basis: str) -> T_CalculationInput:
        """
        convenience method to enable ri approximation for orca.
        this is a wrapper around `set_options`.
        """
        return self.set_options(ri_approx=approx.upper(), aux_basis=aux_basis)

    def set_basis(self: T_CalculationInput, basis: str | dict[str, str]) -> T_CalculationInput:
        """
        sets basis set, supporting element-specific basis sets.

        args:
            basis: either a uniform basis set string (e.g., "def2-tzvp") or
                   a dict mapping element symbols to basis sets (e.g., {"H": "pc-2", "O": "pcX-2"})

        example:
            .set_basis("def2-svp")  # uniform basis
            .set_basis({"H": "pc-2", "O": "pcX-2"})  # element-specific (Q-Chem)
        """
        if isinstance(basis, str):
            return replace(self, basis_set=basis)
        else:
            return self.set_options(element_basis=basis)

    def set_reduced_excitation_space(self: T_CalculationInput, initial_orbitals: list[int]) -> T_CalculationInput:
        """
        sets up reduced excitation space (TRNSS) for core-level spectroscopy.

        this restricts tddft excitations to originate from specific orbitals,
        useful for X-ray absorption spectroscopy (XAS) calculations.

        args:
            initial_orbitals: list of orbital indices (1-based) from which excitations originate.
                             typically core orbitals for XAS.

        example:
            .set_reduced_excitation_space(initial_orbitals=[1])  # excitations from orbital 1 only

        note: this is primarily for Q-Chem. requires tddft to be enabled.
        """
        return self.set_options(reduced_excitation_space_orbitals=initial_orbitals)

    # --- Exporter ---

    def export(self, program: str, geometry: Geometry) -> str:
        """
        the main export entrypoint. dispatches to the correct program builder.

        args:
            program: the name of the qc program (e.g., "orca", "qchem").
            geometry: the molecular geometry object.

        returns:
            a string containing the formatted input file.
        """
        program_lower = program.lower()
        if program_lower not in BUILDERS:
            raise NotImplementedError(
                f"no builder registered for program '{program}'. available: {list(BUILDERS.keys())}"
            )
        builder = BUILDERS[program_lower]
        return builder.build(self, geometry)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """
        serializes the calculation input to a dictionary.

        nested spec objects are also converted to dicts for clean json serialization.

        includes version metadata for compatibility tracking:
        - ``calcflow_version``: package semver string (provenance/auditing).
        - ``schema_version``: integer format version (compatibility/migration).
        """
        data = asdict(self)
        # asdict already recursively converts nested dataclasses
        data["calcflow_version"] = _CALCFLOW_VERSION
        data["schema_version"] = INPUT_SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculationInput:
        """
        deserializes a calculation input from a dictionary.

        reconstructs nested spec objects from their dict representations.

        handles version metadata:
        - ``calcflow_version`` is stripped (provenance only, not a dataclass field).
        - ``schema_version`` is read and used to apply any necessary migrations
          before constructing the instance.  Dumps without ``schema_version``
          (produced before this feature existed) are treated as version 1.
        """
        # create a copy to avoid mutating the input
        data = dict(data)

        # remove version metadata (not dataclass fields)
        data.pop("calcflow_version", None)
        incoming_version = data.pop("schema_version", 1)
        data = cls._migrate(data, incoming_version)

        # reconstruct nested specs if present
        if data.get("tddft") is not None:
            data["tddft"] = TddftSpec.from_dict(data["tddft"])
        if data.get("solvation") is not None:
            data["solvation"] = SolvationSpec.from_dict(data["solvation"])
        if data.get("optimization") is not None:
            data["optimization"] = OptimizationSpec.from_dict(data["optimization"])
        if data.get("mom") is not None:
            data["mom"] = MomSpec.from_dict(data["mom"])
        if data.get("charges") is not None:
            data["charges"] = ChargesSpec.from_dict(data["charges"])
        if data.get("scf") is not None:
            data["scf"] = ScfSpec.from_dict(data["scf"])

        return cls(**data)

    @staticmethod
    def _migrate(data: dict[str, Any], from_version: int) -> dict[str, Any]:
        """Apply sequential migrations from *from_version* to the current schema.

        Each migration step transforms the dict from version *N* to *N+1*.
        Add new migration blocks here when ``INPUT_SCHEMA_VERSION`` is bumped::

            if from_version < 2:
                data = _migrate_input_v1_to_v2(data)
            if from_version < 3:
                data = _migrate_input_v2_to_v3(data)
        """
        if from_version > INPUT_SCHEMA_VERSION:
            logger.warning(
                "CalculationInput dump was created with schema version %d, "
                "but this code only understands version %d. "
                "Some fields may be missing or misinterpreted.",
                from_version,
                INPUT_SCHEMA_VERSION,
            )
        elif from_version < INPUT_SCHEMA_VERSION:
            logger.warning(
                "Migrating CalculationInput from schema version %d to %d.",
                from_version,
                INPUT_SCHEMA_VERSION,
            )
        # --- future migrations go here ---
        return data

    def to_json(self, indent: int = 2) -> str:
        """serializes the calculation input to a json string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> CalculationInput:
        """deserializes a calculation input from a json string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def get_api_docs(cls) -> str:
        """
        returns comprehensive api documentation for llm-assisted code generation.

        this method provides a complete reference of all available fields, methods,
        and usage patterns without requiring access to source code. ideal for sharing
        with llms when you want them to generate code using calcflow.

        usage:
            # print documentation for llm consumption
            print(CalculationInput.get_api_docs())

            # save to file
            with open("calcflow_api.txt", "w") as f:
                f.write(CalculationInput.get_api_docs())
        """
        return """
CalculationInput API Reference
==============================

DESCRIPTION
-----------
Immutable specification for quantum chemistry calculations with a fluent API.
All setter methods return a new instance (the original is unchanged).

CONSTRUCTOR
-----------
Required parameters:
  charge: int
    Molecular charge (0 for neutral, -1 for anion, +1 for cation)

  spin_multiplicity: int
    Spin multiplicity: 1=singlet, 2=doublet, 3=triplet, etc.
    Must be >= 1

  task: Literal["energy", "geometry", "frequency"]
    Type of calculation:
      - "energy": single-point energy calculation
      - "geometry": geometry optimization
      - "frequency": vibrational frequency analysis

  level_of_theory: str
    DFT functional or ab initio method
    Examples: "B3LYP", "wB97X-D3", "PBE0", "M06-2X", "MP2", "CCSD(T)"

  basis_set: str | dict[str, str]
    Basis set specification
    - String for uniform basis: "def2-tzvp", "6-31G*", "cc-pvtz"
    - Dict for element-specific: {"O": "aug-cc-pvtz", "H": "cc-pvdz"}

Optional parameters:
  unrestricted: bool = False
    If True, use unrestricted (UKS/UHF) wavefunctions
    If False, use restricted (RKS/RHF) wavefunctions
    Required for open-shell systems and MOM calculations

  n_cores: int = 1
    Number of CPU cores to use

  memory_per_core_mb: int = 4000
    Memory allocation per core in megabytes

  tddft: TddftSpec | None = None
    TDDFT configuration (set via .set_tddft() method)

  solvation: SolvationSpec | None = None
    Implicit solvation configuration (set via .set_solvation() method)

  optimization: OptimizationSpec | None = None
    Optimization settings (set via .set_optimization() method)

  mom: MomSpec | None = None
    Maximum Overlap Method settings (set via .set_mom() method)

  frequency_after_optimization: bool = False
    Run frequency calculation after geometry optimization

  program_options: dict[str, Any] = {}
    Program-specific options not covered by generic API


FLUENT API METHODS
------------------

Core Settings:
  .set_level_of_theory(lot: str) -> CalculationInput
    Update the level of theory (functional or method)
    Example: .set_level_of_theory("wB97X-D3")

  .set_basis_set(basis: str | dict[str, str]) -> CalculationInput
    Update the basis set
    Examples:
      .set_basis_set("def2-tzvp")
      .set_basis_set({"O": "pcX-2", "H": "pc-2"})

  .set_task(task: Literal["energy", "geometry", "frequency"]) -> CalculationInput
    Update the calculation task type
    Example: .set_task("geometry")

  .set_unrestricted(unrestricted: bool = True) -> CalculationInput
    Enable/disable unrestricted wavefunctions
    Example: .set_unrestricted(True)

Computational Resources:
  .set_cores(n_cores: int) -> CalculationInput
    Set number of CPU cores
    Example: .set_cores(16)

  .set_memory_per_core(mb: int) -> CalculationInput
    Set memory per core in MB
    Example: .set_memory_per_core(8000)

TDDFT (Excited States):
  .set_tddft(
      nroots: int,
      singlets: bool = True,
      triplets: bool = False,
      use_tda: bool = True,
      state_to_optimize: int | None = None
  ) -> CalculationInput
    Configure time-dependent DFT for excited states

    Parameters:
      nroots: Number of excited states to compute (must be >= 1)
      singlets: Include singlet excitations
      triplets: Include triplet excitations
      use_tda: Use Tamm-Dancoff approximation (faster, usually adequate)
      state_to_optimize: For geometry optimizations, which excited state to optimize
                        (e.g., 1 for S1, 2 for S2). Only valid with task="geometry"

    Examples:
      .set_tddft(nroots=10)
      .set_tddft(nroots=20, singlets=True, triplets=True)
      .set_tddft(nroots=5, state_to_optimize=1)  # optimize S1 geometry

Solvation:
  .set_solvation(model: str, solvent: str) -> CalculationInput
    Add implicit solvation model

    Parameters:
      model: Solvation model name (case-insensitive)
             Common: "smd", "cpcm", "pcm", "cosmo"
      solvent: Solvent name (case-insensitive)
               Common: "water", "acetonitrile", "dmso", "methanol", "acetone"

    Example:
      .set_solvation(model="smd", solvent="water")
      .set_solvation(model="cpcm", solvent="acetonitrile")

Optimization:
  .set_optimization(
      calc_hess_initial: bool = False,
      recalc_hess_freq: int | None = None
  ) -> CalculationInput
    Configure geometry optimization settings (only valid for task="geometry")

    Parameters:
      calc_hess_initial: Calculate Hessian at the start of optimization
      recalc_hess_freq: Recalculate Hessian every N steps

    Example:
      .set_optimization(calc_hess_initial=True)
      .set_optimization(recalc_hess_freq=5)

  .run_frequency_after_opt() -> CalculationInput
    Run frequency calculation after successful geometry optimization
    Only valid for task="geometry"

    Example:
      .run_frequency_after_opt()

MOM (Maximum Overlap Method for Excited States):
  .set_mom(
      transition: str,
      method: str = "IMOM",
      job2_charge: int | None = None,
      job2_spin_multiplicity: int | None = None
  ) -> CalculationInput
    Configure Maximum Overlap Method for non-Aufbau electronic configurations
    Requires unrestricted=True

    Parameters:
      transition: Orbital transition specification
        Symbolic notation:
          - "HOMO->LUMO": promote electron from HOMO to LUMO
          - "HOMO-1->LUMO": from HOMO-1 to LUMO
          - "HOMO->LUMO+1": from HOMO to LUMO+1
        Numeric notation:
          - "5->6": promote from orbital 5 to orbital 6
          - "3->LUMO": from orbital 3 to LUMO
        Ionization:
          - "HOMO->vac": remove electron from HOMO
          - "5->vac": remove electron from orbital 5
        Spin-specific:
          - "HOMO(beta)->LUMO(alpha)"
        Multiple transitions:
          - "HOMO->LUMO; HOMO-1->LUMO+1" (semicolon-separated)

      method: "MOM" or "IMOM" (Initial Maximum Overlap Method, recommended)

      job2_charge: Override charge for second job (used for ionization)
      job2_spin_multiplicity: Override spin multiplicity for second job

    Examples:
      .set_unrestricted().set_mom("HOMO->LUMO")
      .set_unrestricted().set_mom("HOMO->vac", job2_charge=1, job2_spin_multiplicity=2)
      .set_unrestricted().set_mom("5->6", method="IMOM")

Program-Specific Options:
  .set_options(**kwargs: Any) -> CalculationInput
    Escape hatch for program-specific options not covered by generic API
    Options are passed directly to the program builder

    Example:
      .set_options(ri_approx="RIJCOSX", aux_basis="def2/j")
      .set_options(scf_convergence=1e-8, max_scf_cycles=200)

  .enable_ri_for_orca(approx: str, aux_basis: str) -> CalculationInput
    Convenience method for enabling RI approximation in ORCA
    This is a wrapper around .set_options()

    Parameters:
      approx: RI approximation type (e.g., "RIJCOSX", "RIJK", "RI")
      aux_basis: Auxiliary basis set (e.g., "def2/j", "cc-pvtz/c")

    Example:
      .enable_ri_for_orca("RIJCOSX", "def2/j")


EXPORT AND SERIALIZATION
-------------------------
  .export(program: str, geometry: Geometry) -> str
    Generate program-specific input file content

    Parameters:
      program: Target program name ("orca", "qchem")
      geometry: Molecular geometry (use Geometry.from_xyz_file() or Geometry())

    Returns: String containing the formatted input file

    Example:
      from calcflow.geometry.static import Geometry
      geom = Geometry.from_xyz_file("molecule.xyz")
      input_content = calc.export("qchem", geom)

  .to_dict() -> dict[str, Any]
    Serialize to dictionary

  .to_json(indent: int = 2) -> str
    Serialize to JSON string

  .from_dict(data: dict[str, Any]) -> CalculationInput
    Deserialize from dictionary

  .from_json(json_str: str) -> CalculationInput
    Deserialize from JSON string


PROPERTIES
----------
  .requires_multiple_jobs -> bool
    Returns True if calculation requires multiple sequential jobs (e.g., MOM)


USAGE EXAMPLES
--------------

1. Basic single-point energy calculation:
    from calcflow.common.input import CalculationInput
    from calcflow.geometry.static import Geometry

    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="B3LYP",
        basis_set="def2-tzvp"
    )

    geom = Geometry.from_xyz_file("molecule.xyz")
    with open("qchem.in", "w") as f:
        f.write(calc.export("qchem", geom))


2. TDDFT with solvation:
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wB97X-D3",
        basis_set="def2-tzvp",
        n_cores=16
    ).set_tddft(
        nroots=10,
        singlets=True,
        triplets=False
    ).set_solvation(
        model="smd",
        solvent="water"
    )


3. Element-specific basis for XAS:
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wB97X-D3",
        basis_set={"O": "pcX-2", "H": "pc-2"}
    ).set_tddft(nroots=10)


4. Geometry optimization with frequency:
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="geometry",
        level_of_theory="wB97X-D3",
        basis_set="def2-svp",
        n_cores=16
    ).run_frequency_after_opt()


5. Excited state optimization (S1):
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="geometry",
        level_of_theory="PBE0",
        basis_set="def2-svp"
    ).set_tddft(
        nroots=5,
        state_to_optimize=1  # optimize S1
    )


6. MOM calculation for ionization:
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wB97X-D3",
        basis_set="def2-tzvp",
        unrestricted=True
    ).set_mom(
        transition="HOMO->vac",
        job2_charge=1,
        job2_spin_multiplicity=2
    ).set_solvation("smd", "water")


7. ORCA calculation with RI approximation:
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="geometry",
        level_of_theory="wB97X-D3",
        basis_set="def2-svp",
        n_cores=16
    ).enable_ri_for_orca(
        approx="RIJCOSX",
        aux_basis="def2/j"
    ).run_frequency_after_opt()


8. Fluent API chaining:
    calc = (
        CalculationInput(
            charge=-1,
            spin_multiplicity=2,
            task="energy",
            level_of_theory="M06-2X",
            basis_set="def2-tzvp"
        )
        .set_unrestricted()
        .set_tddft(nroots=20, singlets=True, triplets=True)
        .set_solvation("smd", "acetonitrile")
        .set_cores(32)
        .set_memory_per_core(8000)
    )


9. Save and load calculation spec:
    # save
    with open("calc_spec.json", "w") as f:
        f.write(calc.to_json())

    # load
    loaded_calc = CalculationInput.from_json(
        open("calc_spec.json").read()
    )
    assert loaded_calc == calc  # perfect roundtrip!


10. Complete workflow:
    from calcflow.common.input import CalculationInput
    from calcflow.geometry.static import Geometry

    # define calculation
    calc = (
        CalculationInput(
            charge=0, spin_multiplicity=1, task="energy",
            level_of_theory="wB97X-D3", basis_set="def2-tzvp", n_cores=16
        )
        .set_tddft(nroots=10)
        .set_solvation("smd", "water")
    )

    # load geometry
    geom = Geometry.from_xyz_file("molecule.xyz")

    # export input files
    qchem_input = calc.export("qchem", geom)
    orca_input = calc.export("orca", geom)

    # write to disk
    with open("qchem.in", "w") as f:
        f.write(qchem_input)
    with open("orca.inp", "w") as f:
        f.write(orca_input)

    # save spec for reproducibility
    with open("calc_spec.json", "w") as f:
        f.write(calc.to_json())


VALIDATION
----------
The constructor performs validation on initialization:
  - spin_multiplicity must be >= 1
  - tddft.nroots must be >= 1 (if tddft is set)
  - solvation model and solvent must both be specified (if solvation is set)
  - mom requires unrestricted=True
  - mom requires either transition or manual occupation specification
  - state_to_optimize only valid for task="geometry"
  - optimization settings only valid for task="geometry"
  - frequency_after_optimization only valid for task="geometry"

Program-specific builders perform additional validation when .export() is called.


RELATED CLASSES
---------------
TddftSpec: TDDFT configuration (nroots, singlets, triplets, use_tda, state_to_optimize)
SolvationSpec: Solvation configuration (model, solvent)
OptimizationSpec: Optimization configuration (calc_hess_initial, recalc_hess_freq)
MomSpec: MOM configuration (transition, method, job2_charge, job2_spin_multiplicity)
Geometry: Molecular geometry (use Geometry.from_xyz_file() or construct manually)
""".strip()
