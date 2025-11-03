from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar

from calcflow.exceptions import ConfigurationError
from calcflow.geometry.static import Geometry
from calcflow.io.orca.builder import OrcaBuilder
from calcflow.io.qchem.builder import QchemBuilder
from calcflow.io.spec import CalculationSpec, OptimizationSpec, SolvationSpec, TddftSpec

T_CalculationInput = TypeVar("T_CalculationInput", bound="CalculationInput")

# lazy-loaded registry to prevent circular imports.
BUILDERS: dict[str, OrcaBuilder | QchemBuilder] = {}


def _register_builders():
    """lazy registration to prevent circular imports at module level."""
    if BUILDERS:  # only register once
        return
    from calcflow.io.orca.builder import OrcaBuilder
    from calcflow.io.qchem.builder import QchemBuilder

    BUILDERS["orca"] = OrcaBuilder()
    BUILDERS["qchem"] = QchemBuilder()


@dataclass(frozen=True)
class CalculationInput:
    """
    the fluent, user-facing api for building a calculation.

    this class is an immutable factory for `CalculationSpec` objects.
    each 'set' method returns a new instance with an updated spec.
    """

    spec: CalculationSpec

    def __init__(
        self,
        charge: int,
        spin_multiplicity: int,
        task: Literal["energy", "geometry", "frequency"],
        level_of_theory: str,
        basis_set: str | dict[str, str],
        **kwargs: Any,
    ):
        """
        convenience constructor that builds the initial `CalculationSpec`.
        this hides the `spec` object from the user on first creation.
        """
        # this is the sanctioned way to initialize a frozen dataclass
        # with a custom __init__ signature. it bypasses the frozen check.
        object.__setattr__(
            self,
            "spec",
            CalculationSpec(
                charge=charge,
                spin_multiplicity=spin_multiplicity,
                task=task,
                level_of_theory=level_of_theory,
                basis_set=basis_set,
                **kwargs,
            ),
        )

    # --- Core Parameter Setters ---

    def set_level_of_theory(self: T_CalculationInput, lot: str) -> T_CalculationInput:
        """updates the level of theory (method/functional)."""
        new_spec = replace(self.spec, level_of_theory=lot)
        return self.__class__(**new_spec.__dict__)

    def set_basis_set(self: T_CalculationInput, basis: str | dict[str, str]) -> T_CalculationInput:
        """updates the basis set."""
        new_spec = replace(self.spec, basis_set=basis)
        return self.__class__(**new_spec.__dict__)

    def set_task(self: T_CalculationInput, task: Literal["energy", "geometry", "frequency"]) -> T_CalculationInput:
        """updates the main calculation task."""
        new_spec = replace(self.spec, task=task)
        return self.__class__(**new_spec.__dict__)

    def set_unrestricted(self: T_CalculationInput, unrestricted: bool = True) -> T_CalculationInput:
        """sets the calculation to be unrestricted (uks/uhf) or restricted (rks/rhf)."""
        new_spec = replace(self.spec, unrestricted=unrestricted)
        return self.__class__(**new_spec.__dict__)

    # --- Computational Resource Setters ---

    def set_cores(self: T_CalculationInput, n_cores: int) -> T_CalculationInput:
        """sets the number of cpu cores to use."""
        new_spec = replace(self.spec, n_cores=n_cores)
        return self.__class__(**new_spec.__dict__)

    def set_memory_per_core(self: T_CalculationInput, mb: int) -> T_CalculationInput:
        """sets the memory per core in megabytes."""
        new_spec = replace(self.spec, memory_per_core_mb=mb)
        return self.__class__(**new_spec.__dict__)

    # --- Calculation Component Setters ---

    def set_solvation(self: T_CalculationInput, model: str, solvent: str) -> T_CalculationInput:
        """adds or updates the implicit solvation model."""
        solv_spec = SolvationSpec(model=model.lower(), solvent=solvent.lower())
        new_spec = replace(self.spec, solvation=solv_spec)
        return self.__class__(**new_spec.__dict__)

    def set_tddft(
        self: T_CalculationInput,
        nroots: int,
        singlets: bool = True,
        triplets: bool = False,
        use_tda: bool = True,
        state_to_optimize: int | None = None,
    ) -> T_CalculationInput:
        """adds or updates the tddft calculation parameters."""
        if state_to_optimize and self.spec.task != "geometry":
            raise ConfigurationError("`state_to_optimize` is only valid for 'geometry' tasks.")
        tddft_spec = TddftSpec(
            nroots=nroots,
            singlets=singlets,
            triplets=triplets,
            use_tda=use_tda,
            state_to_optimize=state_to_optimize,
        )
        new_spec = replace(self.spec, tddft=tddft_spec)
        return self.__class__(**new_spec.__dict__)

    def set_optimization(
        self: T_CalculationInput,
        calc_hess_initial: bool = False,
        recalc_hess_freq: int | None = None,
    ) -> T_CalculationInput:
        """adds or updates geometry optimization parameters."""
        if self.spec.task != "geometry":
            raise ConfigurationError("optimization settings are only valid for 'geometry' tasks.")
        opt_spec = OptimizationSpec(
            calc_hess_initial=calc_hess_initial,
            recalc_hess_freq=recalc_hess_freq,
        )
        new_spec = replace(self.spec, optimization=opt_spec)
        return self.__class__(**new_spec.__dict__)

    def run_frequency_after_opt(self: T_CalculationInput) -> T_CalculationInput:
        """enables a frequency calculation to be run after a successful geometry optimization."""
        if self.spec.task != "geometry":
            raise ConfigurationError("frequency calculation can only follow a 'geometry' task.")
        new_spec = replace(self.spec, frequency_after_optimization=True)
        return self.__class__(**new_spec.__dict__)

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
        new_opts = {**self.spec.program_options, **kwargs}
        new_spec = replace(self.spec, program_options=new_opts)
        return self.__class__(**new_spec.__dict__)

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
        _register_builders()  # ensure builders are loaded
        program_lower = program.lower()
        if program_lower not in BUILDERS:
            raise NotImplementedError(
                f"no builder registered for program '{program}'. available: {list(BUILDERS.keys())}"
            )
        builder = BUILDERS[program_lower]
        return builder.build(self.spec, geometry)
