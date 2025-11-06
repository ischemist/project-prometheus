from __future__ import annotations

import re
from typing import TYPE_CHECKING

from calcflow.common.exceptions import ConfigurationError, NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry

if TYPE_CHECKING:
    from calcflow.common.input import CalculationInput

# fmt:off
SUPPORTED_FUNCTIONALS = {"b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x", "wb97x-d3", "src1-r1"}
# fmt:on


class QchemBuilder:
    """translates a generic `CalculationInput` into a q-chem input file."""

    def build(self, spec: CalculationInput, geometry: Geometry) -> str:
        """
        the main build method. orchestrates the creation of the input file.
        delegates to specialized builders based on whether multiple jobs are required.
        """
        self._validate_spec(spec)

        if spec.requires_multiple_jobs:
            return self._build_multi_job_input(spec, geometry)
        else:
            return self._build_single_job_input(spec, geometry)

    def _build_single_job_input(self, spec: CalculationInput, geometry: Geometry) -> str:
        """builds a standard single-job input file."""
        blocks = [
            self._build_molecule(spec, geometry),
            self._build_rem(spec),
            self._build_basis(spec),
            self._build_solvation_blocks(spec),
            self._build_solute(spec),
        ]
        return "\n\n".join(block for block in blocks if block).strip() + "\n"

    def _build_multi_job_input(self, spec: CalculationInput, geometry: Geometry) -> str:
        """
        builds a multi-job input file (e.g., for mom).
        currently only supports mom, but structured to allow future extensions.
        """
        if spec.mom:
            return self._build_mom_input(spec, geometry)
        else:
            raise NotSupportedError("multi-job calculations are currently only supported for mom.")

    def _build_mom_input(self, spec: CalculationInput, geometry: Geometry) -> str:
        """
        builds a two-job mom input file.
        job 1: ground state calculation to generate reference orbitals.
        job 2: excited state calculation with modified orbital occupations.
        """
        job1 = self._build_mom_job1(spec, geometry)
        job2 = self._build_mom_job2(spec, geometry)
        return f"{job1}\n\n@@@\n\n{job2}\n"

    def _build_mom_job1(self, spec: CalculationInput, geometry: Geometry) -> str:
        """builds the first job for mom: ground state reference calculation."""
        blocks = [
            self._build_molecule(spec, geometry),
            self._build_rem(spec, task_override="energy"),
            self._build_basis(spec),
            self._build_solvation_blocks(spec),
            # no $solute block in job1 (tddft only runs in job2)
        ]
        return "\n\n".join(block for block in blocks if block).strip()

    def _build_mom_job2(self, spec: CalculationInput, geometry: Geometry) -> str:
        """builds the second job for mom: excited state with target occupations."""
        assert spec.mom is not None

        # determine charge/spin for job2 (may differ for ionization)
        charge_job2 = spec.mom.job2_charge if spec.mom.job2_charge is not None else spec.charge
        mult_job2 = (
            spec.mom.job2_spin_multiplicity if spec.mom.job2_spin_multiplicity is not None else spec.spin_multiplicity
        )

        blocks = [
            self._build_molecule(spec, geometry, charge_override=charge_job2, mult_override=mult_job2, read_geom=True),
            self._build_rem(spec, scf_guess="read", mom_start=True, task_override="energy"),
            self._build_basis(spec),
            self._build_solvation_blocks(spec),
            self._build_occupied_block(spec, geometry),
            self._build_solute(spec),  # tddft excitations from the mom state
        ]
        return "\n\n".join(block for block in blocks if block).strip()

    def _validate_spec(self, spec: CalculationInput):
        """performs q-chem-specific validation on the spec."""
        if spec.solvation and spec.solvation.model not in {"pcm", "smd", "isosvp", "cpcm"}:
            raise NotSupportedError(f"q-chem builder does not support solvation model '{spec.solvation.model}'.")

        # mom validation (should already be caught by CalculationInput.__post_init__, but double-check)
        if spec.mom and not spec.unrestricted:
            raise ConfigurationError("mom requires an unrestricted calculation. use .set_unrestricted()")

        # program_options validation (for backward compatibility during transition)
        opts = spec.program_options
        if opts.get("run_mom", False):
            raise ConfigurationError("program_options['run_mom'] is deprecated. use .set_mom() instead.")
        if opts.get("reduced_excitation_space_orbitals") and not spec.tddft:
            raise ConfigurationError("reduced excitation space requires tddft to be enabled.")

    def _build_molecule(
        self,
        spec: CalculationInput,
        geometry: Geometry,
        charge_override: int | None = None,
        mult_override: int | None = None,
        read_geom: bool = False,
    ) -> str:
        if read_geom:
            return "$molecule\nread\n$end"

        charge = charge_override if charge_override is not None else spec.charge
        multiplicity = mult_override if mult_override is not None else spec.spin_multiplicity
        return f"$molecule\n{charge} {multiplicity}\n{geometry.get_coordinate_block().strip()}\n$end"

    def _build_rem(
        self,
        spec: CalculationInput,
        scf_guess: str | None = None,
        mom_start: bool = False,
        task_override: str | None = None,
    ) -> str:
        rem_vars: dict[str, str | bool | int] = {}
        opts = spec.program_options

        # --- job type ---
        task = task_override or spec.task
        task_map = {"energy": "sp", "geometry": "opt", "frequency": "freq"}
        if task not in task_map:
            raise NotSupportedError(f"task '{task}' not supported by q-chem builder.")
        rem_vars["JOBTYPE"] = task_map[task]

        # --- level of theory & basis ---
        method = spec.level_of_theory.lower()
        if method in SUPPORTED_FUNCTIONALS or method == "hf":
            rem_vars["METHOD"] = method
        else:
            raise NotSupportedError(f"level of theory '{method}' not supported by q-chem builder.")

        rem_vars["BASIS"] = "gen" if isinstance(spec.basis_set, dict) else spec.basis_set
        rem_vars["UNRESTRICTED"] = spec.unrestricted
        rem_vars["SYMMETRY"] = False
        rem_vars["SYM_IGNORE"] = True

        # --- scf & mom ---
        if scf_guess:
            rem_vars["SCF_GUESS"] = scf_guess
        if mom_start:
            rem_vars["MOM_START"] = 1
            rem_vars["MOM_METHOD"] = spec.mom.method if spec.mom else "IMOM"

        # --- tddft ---
        if spec.tddft:  # noqa: SIM102
            # tddft block is only added to the *final* calculation step
            # i.e., the first job if not mom, or the second job if mom
            if not spec.mom or mom_start:
                rem_vars["CIS_N_ROOTS"] = spec.tddft.nroots
                rem_vars["CIS_SINGLETS"] = spec.tddft.singlets
                rem_vars["CIS_TRIPLETS"] = spec.tddft.triplets
                rem_vars["RPA"] = not spec.tddft.use_tda

                # reduced excitation space (trnss)
                trnss_orbs = opts.get("reduced_excitation_space_orbitals")
                if trnss_orbs:
                    rem_vars["TRNSS"] = True
                    rem_vars["TRTYPE"] = 3
                    rem_vars["N_SOL"] = len(trnss_orbs)

        # --- solvation ---
        if spec.solvation:
            rem_vars["SOLVENT_METHOD"] = spec.solvation.model

        # --- format block ---
        lines = ["$rem"]
        max_key_len = max(len(k) for k in rem_vars) if rem_vars else 0
        for key, value in rem_vars.items():
            lines.append(f"    {key:<{max_key_len}}    {value}")
        lines.append("$end")
        return "\n".join(lines)

    def _build_basis(self, spec: CalculationInput) -> str:
        if not isinstance(spec.basis_set, dict):
            return ""
        lines = ["$basis"]
        for element, basis_name in spec.basis_set.items():
            lines.append(f"{element.capitalize()} 0")
            lines.append(basis_name)
            lines.append("****")
        lines.append("$end")
        return "\n".join(lines)

    def _build_solvation_blocks(self, spec: CalculationInput) -> str:
        if not spec.solvation:
            return ""
        if spec.solvation.model == "pcm":
            return f"$solvent\n    SolventName {spec.solvation.solvent}\n$end"
        if spec.solvation.model == "smd":
            return f"$smx\n    solvent {spec.solvation.solvent}\n$end"
        return ""

    def _build_solute(self, spec: CalculationInput) -> str:
        """builds the $solute block for trnss calculations."""
        trnss_orbs = spec.program_options.get("reduced_excitation_space_orbitals")
        if not spec.tddft or not trnss_orbs:
            return ""
        return f"$solute\n{' '.join(map(str, trnss_orbs))}\n$end"

    def _build_occupied_block(self, spec: CalculationInput, geometry: Geometry) -> str:
        """generates the $occupied block for mom calculations."""
        if not spec.mom:
            return ""

        if not spec.unrestricted:
            # this check is redundant with _validate_spec, but good for defense
            raise ConfigurationError("mom requires an unrestricted calculation. use .set_unrestricted()")

        # determine charge/multiplicity for this specific job (may differ for ionization)
        effective_charge = spec.mom.job2_charge if spec.mom.job2_charge is not None else spec.charge
        effective_multiplicity = (
            spec.mom.job2_spin_multiplicity if spec.mom.job2_spin_multiplicity is not None else spec.spin_multiplicity
        )
        total_electrons = geometry.total_nuclear_charge - effective_charge

        # check for manual occupation override
        if spec.mom.alpha_occupation and spec.mom.beta_occupation:
            alpha_occ = spec.mom.alpha_occupation
            beta_occ = spec.mom.beta_occupation
        elif spec.mom.transition == "GROUND_STATE":
            # special case: ground state occupation (for testing/validation)
            if total_electrons % 2 != 0:
                raise ConfigurationError("mom 'GROUND_STATE' requires an even number of electrons.")
            n_occ = total_electrons // 2
            alpha_occ = beta_occ = f"1:{n_occ}" if n_occ > 1 else "1"
        elif spec.mom.transition:
            # handle symbolic transitions like "HOMO->LUMO" or "5(beta)->vac"
            alpha_occ, beta_occ = self._convert_extended_transitions_to_occupations(spec.mom.transition, spec, geometry)
        else:
            raise ConfigurationError(
                "mom requires either 'transition' or both 'alpha_occupation' and 'beta_occupation'."
            )

        self._validate_mom_occupations(alpha_occ, beta_occ, total_electrons, effective_multiplicity)
        return f"$occupied\n{alpha_occ}\n{beta_occ}\n$end"

    def _validate_mom_occupations(self, alpha_occ: str, beta_occ: str, total_electrons: int, multiplicity: int) -> None:
        """validates occupation strings against expected electron count and spin."""
        n_alpha = _count_electrons_in_qchem_occupation(alpha_occ)
        n_beta = _count_electrons_in_qchem_occupation(beta_occ)
        if (n_alpha + n_beta) != total_electrons:
            raise ValidationError(
                f"total electrons in occupation ({n_alpha + n_beta}) does not match "
                f"expected {total_electrons} from geometry and charge."
            )
        expected_alpha, expected_beta = _calculate_expected_electron_distribution(total_electrons, multiplicity)
        if n_alpha != expected_alpha or n_beta != expected_beta:
            raise ValidationError(
                f"spin multiplicity {multiplicity} is inconsistent with occupation "
                f"({n_alpha} alpha, {n_beta} beta). expected ({expected_alpha} alpha, {expected_beta} beta)."
            )

    def _convert_extended_transitions_to_occupations(
        self, transition_string: str, spec: CalculationInput, geometry: Geometry
    ) -> tuple[str, str]:
        """converts symbolic transition string to alpha/beta occupation strings."""
        # ground state of the *first* job determines the reference homo
        initial_total_electrons = geometry.total_nuclear_charge - spec.charge
        if initial_total_electrons % 2 != 0:
            raise ConfigurationError(
                f"symbolic mom transitions require an even number of electrons in the initial state, got {initial_total_electrons}."
            )
        initial_homo = initial_total_electrons // 2

        alpha_occupied = set(range(1, initial_homo + 1))
        beta_occupied = set(range(1, initial_homo + 1))

        operations = [op.strip() for op in transition_string.split(";")]
        for operation in operations:
            self._apply_single_operation(operation, alpha_occupied, beta_occupied, initial_homo)

        return self._format_occupation_set(alpha_occupied), self._format_occupation_set(beta_occupied)

    def _apply_single_operation(
        self, operation: str, alpha_occupied: set[int], beta_occupied: set[int], initial_homo: int
    ) -> None:
        """applies a single transition (e.g., "HOMO->LUMO") to the occupation sets."""
        if "->" not in operation:
            raise ValidationError(f"invalid transition format: '{operation}'. expected 'source->target'.")
        source_str, target_str = [x.strip() for x in operation.split("->")]

        source_orb, source_spin = self._parse_spin_specification(source_str)
        source_idx = self._resolve_orbital_index(source_orb, initial_homo)

        if target_str.upper() == "VAC":  # ionization
            self._apply_ionization(source_idx, source_spin, alpha_occupied, beta_occupied)
        else:  # excitation
            target_orb, target_spin = self._parse_spin_specification(target_str)
            target_idx = self._resolve_orbital_index(target_orb, initial_homo)
            self._apply_excitation(
                source_idx, source_spin, target_idx, target_spin, alpha_occupied, beta_occupied, initial_homo
            )

    def _resolve_orbital_index(self, orbital_spec: str, initial_homo: int) -> int:
        """resolves "HOMO-1", "LUMO", etc., to a numerical index."""
        if orbital_spec.isdigit():
            return int(orbital_spec)
        match = re.fullmatch(r"(HOMO|LUMO)(?:([+-])(\d+))?", orbital_spec, re.IGNORECASE)
        if not match:
            raise ValidationError(f"invalid orbital specification: '{orbital_spec}'")
        orb_type, operator, offset_str = match.groups()
        offset = int(offset_str) if offset_str else 0
        if orb_type.upper() == "HOMO":
            return initial_homo - offset if operator == "-" else initial_homo
        else:  # LUMO
            return initial_homo + 1 + offset if operator == "+" else initial_homo + 1

    def _apply_ionization(self, source_idx: int, spin: str | None, alpha_occ: set[int], beta_occ: set[int]) -> None:
        """
        removes an electron from the specified orbital for ionization transitions (e.g., "HOMO->vac").

        spin channel selection logic:
        1. if spin is explicitly specified ("alpha" or "beta"), remove from that channel
        2. if spin is unspecified (none), prefer removing from beta, then fall back to alpha
        3. raise error if the orbital is unoccupied in the target channel

        rationale for preferring beta when spin is unspecified:
        - for closed-shell ground states (alpha == beta initially), ionizing from beta
          leaves the unpaired electron in alpha (spin-up by convention)
        - this matches the typical chemical interpretation: cation radicals from
          closed-shell neutrals are usually represented with alpha (spin-up) unpaired electrons
        - example: h2o (singlet, 10e) -> h2o+ (doublet, 9e) with unpaired electron in alpha

        """
        if spin == "alpha":
            if source_idx not in alpha_occ:
                raise ValidationError(f"cannot ionize from unoccupied alpha orbital {source_idx}")
            alpha_occ.remove(source_idx)
        elif spin == "beta":
            if source_idx not in beta_occ:
                raise ValidationError(f"cannot ionize from unoccupied beta orbital {source_idx}")
            beta_occ.remove(source_idx)
        else:
            # no spin specified: prefer beta (closed-shell convention), fall back to alpha
            if source_idx in beta_occ:
                beta_occ.remove(source_idx)
            elif source_idx in alpha_occ:
                alpha_occ.remove(source_idx)
            else:
                raise ValidationError(f"cannot ionize from unoccupied orbital {source_idx}")

    def _apply_excitation(
        self,
        source_idx: int,
        source_spin: str | None,
        target_idx: int,
        target_spin: str | None,
        alpha_occ: set[int],
        beta_occ: set[int],
        initial_homo: int,
    ) -> None:
        if source_idx > initial_homo or target_idx <= initial_homo:
            raise ValidationError(
                f"invalid excitation: source must be <= HOMO ({initial_homo}) and target must be > HOMO."
            )

        source_channel = source_spin or "alpha"
        target_channel = target_spin or source_channel

        # remove from source
        if source_channel == "alpha":
            alpha_occ.remove(source_idx)
        else:
            beta_occ.remove(source_idx)

        # add to target
        if target_channel == "alpha":
            alpha_occ.add(target_idx)
        else:
            beta_occ.add(target_idx)

    def _parse_spin_specification(self, orbital_spec: str) -> tuple[str, str | None]:
        """parses '5(beta)' into ('5', 'beta')."""
        match = re.match(r"^(.+?)\((\w+)\)$", orbital_spec.strip(), re.IGNORECASE)
        if match:
            orb, spin = match.group(1).strip(), match.group(2).lower()
            if spin not in ("alpha", "beta"):
                raise ValidationError(f"invalid spin '{spin}'. must be 'alpha' or 'beta'.")
            return orb, spin
        return orbital_spec.strip(), None

    def _format_occupation_set(self, occupied: set[int]) -> str:
        """formats a set of indices like {1,2,3,5} into a q-chem string "1:3 5"."""
        if not occupied:
            return ""
        nums = sorted(list(occupied))
        parts = []
        start = end = nums[0]
        for num in nums[1:]:
            if num == end + 1:
                end = num
            else:
                parts.append(str(start) if start == end else f"{start}:{end}")
                start = end = num
        parts.append(str(start) if start == end else f"{start}:{end}")
        return " ".join(parts)


def _count_electrons_in_qchem_occupation(occupation_string: str) -> int:
    """counts electrons in a q-chem occupation string like "1:5 7"."""
    if not occupation_string.strip():
        return 0
    count = 0
    for part in occupation_string.strip().split():
        if ":" in part:
            start, end = map(int, part.split(":"))
            count += end - start + 1
        else:
            count += 1
    return count


def _calculate_expected_electron_distribution(n_electrons: int, multiplicity: int) -> tuple[int, int]:
    """calculates n_alpha and n_beta from total electrons and multiplicity."""
    if (n_electrons + multiplicity - 1) % 2 != 0:
        raise ValidationError(f"cannot achieve multiplicity {multiplicity} with {n_electrons} electrons.")
    n_alpha = (n_electrons + multiplicity - 1) // 2
    n_beta = n_electrons - n_alpha
    if n_beta < 0:
        raise ValidationError(f"multiplicity {multiplicity} is impossible for {n_electrons} electrons.")
    return n_alpha, n_beta
