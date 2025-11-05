from __future__ import annotations

from typing import TYPE_CHECKING

from calcflow.common.exceptions import NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry

if TYPE_CHECKING:
    from calcflow.common.input import CalculationInput

# fmt:off
SUPPORTED_FUNCTIONALS = {"b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x", "wb97x-d3"}
HF_VARIANTS = {"uhf", "rhf", "hf"}
MP2_VARIANTS = {"ump2", "mp2", "ri-mp2", "ri-ump2"}
CC_VARIANTS = {"ccsd", "ccsd(t)"}
# fmt:on


class OrcaBuilder:
    """translates a generic `CalculationInput` into an orca input file."""

    def build(self, spec: CalculationInput, geometry: Geometry) -> str:
        """
        the main build method. orchestrates the creation of the input file.
        """
        self._validate_spec(spec)

        blocks = [
            self._build_keywords(spec),
            self._build_procs(spec),
            self._build_mem(spec),
            self._build_solvent(spec),
            self._build_tddft(spec),
            self._build_geom(spec),
            f"* xyz {spec.charge} {spec.spin_multiplicity}",
            geometry.get_coordinate_block(),
            "*",
        ]
        return "\n".join(block for block in blocks if block).strip() + "\n"

    def _validate_spec(self, spec: CalculationInput):
        """performs orca-specific validation on the spec."""
        if isinstance(spec.basis_set, dict):
            raise NotSupportedError("orca builder does not support dictionary basis sets.")
        if spec.solvation and spec.solvation.model not in {"smd", "cpcm"}:
            raise NotSupportedError(f"orca only supports 'smd' and 'cpcm' solvation, not '{spec.solvation.model}'.")
        # validate program-specific options
        opts = spec.program_options
        if "ri_approx" in opts and "aux_basis" not in opts:
            raise ValidationError("if 'ri_approx' is set, 'aux_basis' must also be provided.")

    def _build_keywords(self, spec: CalculationInput) -> str:
        keywords: list[str] = []

        # task
        task_map = {"energy": "SP", "geometry": "Opt", "frequency": "Freq"}
        keywords.append(task_map[spec.task])
        if spec.frequency_after_optimization and spec.task == "geometry":
            keywords.append("Freq")

        # level of theory
        keywords.extend(self._handle_level_of_theory(spec))

        # basis set
        if isinstance(spec.basis_set, str):
            keywords.append(spec.basis_set)

        # program options (ri, etc.)
        opts = spec.program_options
        if "ri_approx" in opts:
            keywords.append(opts["ri_approx"])
            keywords.append(opts["aux_basis"])

        # solvation
        if spec.solvation and spec.solvation.model == "cpcm":
            keywords.append(f'CPCM("{spec.solvation.solvent}")')

        return f"! {' '.join(keywords)}"

    def _handle_level_of_theory(self, spec: CalculationInput) -> list[str]:
        raw_method = spec.level_of_theory.lower()
        if raw_method in SUPPORTED_FUNCTIONALS:
            return ["UKS" if spec.unrestricted else "RKS", raw_method]
        if raw_method in HF_VARIANTS:
            return ["UHF" if spec.unrestricted else "RHF"]
        if raw_method in MP2_VARIANTS:
            return ["UMP2" if spec.unrestricted else "MP2"]  # simplified for brevity
        if raw_method in CC_VARIANTS:
            return [raw_method.upper()]
        raise NotSupportedError(f"level of theory '{spec.level_of_theory}' not supported by orca builder.")

    def _build_procs(self, spec: CalculationInput) -> str:
        return f"%pal nprocs {spec.n_cores} end" if spec.n_cores > 1 else ""

    def _build_mem(self, spec: CalculationInput) -> str:
        return f"%maxcore {spec.memory_per_core_mb}"

    def _build_solvent(self, spec: CalculationInput) -> str:
        if spec.solvation and spec.solvation.model == "smd":
            return f'%cpcm\n    smd true\n    SMDsolvent "{spec.solvation.solvent}"\nend'
        return ""

    def _build_tddft(self, spec: CalculationInput) -> str:
        if not spec.tddft:
            return ""
        lines = ["%tddft"]
        lines.append(f"    NRoots {spec.tddft.nroots}")
        lines.append(f"    Triplets {str(spec.tddft.triplets).lower()}")
        lines.append(f"    TDA {str(spec.tddft.use_tda).lower()}")
        if spec.tddft.state_to_optimize:
            lines.append(f"    IRoot {spec.tddft.state_to_optimize}")
        lines.append("end")
        return "\n".join(lines)

    def _build_geom(self, spec: CalculationInput) -> str:
        if not spec.optimization:
            return ""
        lines = ["%geom"]
        if spec.optimization.calc_hess_initial:
            lines.append("    Calc_Hess true")
        if spec.optimization.recalc_hess_freq:
            lines.append(f"    Recalc_Hess {spec.optimization.recalc_hess_freq}")
        lines.append("end")
        return "\n".join(lines) if len(lines) > 1 else ""
