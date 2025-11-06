from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar

from calcflow.common.exceptions import ConfigurationError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.io.orca.builder import OrcaBuilder
from calcflow.io.qchem.builder import QchemBuilder

T_CalculationInput = TypeVar("T_CalculationInput", bound="CalculationInput")
type TASK_TYPES = Literal["energy", "geometry", "frequency"]

# lazy-loaded registry to prevent circular imports.
BUILDERS = {"orca": OrcaBuilder(), "qchem": QchemBuilder()}

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


@dataclass(frozen=True)
class MomSpec:
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

    # optional, modular components of the calculation
    tddft: TddftSpec | None = None
    solvation: SolvationSpec | None = None
    optimization: OptimizationSpec | None = None
    mom: MomSpec | None = None
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
