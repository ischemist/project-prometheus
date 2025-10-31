"""
Parser for QChem unrelaxed density matrix analysis in TDDFT calculations.

This parser extracts per-state analysis of unrelaxed density matrices including:
- Natural orbital occupations (frontier NOs, electron counts, PR_NO)
- Mulliken population analysis (charges, hole/electron populations)
- Multipole moment analysis (charge centers, dipole moment, RMS size)
- Exciton analysis (hole/electron positions and sizes, separation)

The parser handles both RKS (restricted) and UKS (unrestricted) calculations.
"""

import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.models import (
    AtomicCharges,
    ExcitonAnalysis,
    NaturalOrbitals,
    TddftResults,
    UnrelaxedDensityMatrix,
)
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Pattern constants ---
# Start marker: identifies the beginning of unrelaxed density matrices section
UNRELAXED_DM_START_PAT = re.compile(r"^\s+Analysis of Unrelaxed Density Matrices")

# State header patterns (RKS vs UKS)
RKS_STATE_HEADER_PAT = re.compile(r"^\s+(Singlet|Triplet)\s+(\d+)\s+:")
UKS_STATE_HEADER_PAT = re.compile(r"^\s+Excited State\s+(\d+)\s+:")

# Natural orbitals section markers
NOS_SECTION_PAT = re.compile(r"^\s+NOs$")  # RKS: just "NOs"
NOS_ALPHA_PAT = re.compile(r"^\s+NOs \(alpha\)")  # UKS marker
NOS_SPIN_TRACED_PAT = re.compile(r"^\s+NOs \(spin-traced\)")  # UKS: use this section

# NOs data patterns
FRONTIER_NOS_PAT = re.compile(r"^\s+Occupation of frontier NOs:")
OCCUPATION_VAL_PAT = re.compile(r"^\s+([\d.-]+)(?:\s+([\d.-]+))?")
NUM_ELECTRONS_PAT = re.compile(r"^\s+Number of electrons:\s+([\d.]+)")
NUM_UNPAIRED_PAT = re.compile(r"^\s+Number of unpaired electrons:\s+n_u\s*=\s*([-\d.]+),\s+n_u,nl\s*=\s*([-\d.]+)")
PR_NO_PAT = re.compile(r"^\s+NO participation ratio \(PR_NO\):\s+([\d.]+)")

# Mulliken analysis section marker
MULLIKEN_START_PAT = re.compile(r"^\s+Mulliken Population Analysis \(State/Difference DM\)")

# Mulliken atom line patterns (RKS vs UKS)
# RKS: "  1 H       -0.283169        0.016896       -0.530620       -0.513723"
# UKS: "  1 H   -0.272264   -0.000000    0.008484    0.008484   -0.259893   -0.259893"
MULLIKEN_ATOM_RKS_PAT = re.compile(r"^\s+(\d+)\s+[A-Z][a-z]?\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
MULLIKEN_ATOM_UKS_PAT = re.compile(
    r"^\s+(\d+)\s+[A-Z][a-z]?\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)

# Multipole moment analysis patterns
MULTIPOLE_START_PAT = re.compile(r"^\s+Multipole moment analysis of the density matrix")
MOLECULAR_CHARGE_PAT = re.compile(r"^\s+Molecular charge:\s+([-\d.]+)")
TOTAL_ELECTRONS_PAT = re.compile(r"^\s+Number of electrons:\s+([-\d.]+)")
ELECTRONIC_CHARGE_CENTER_PAT = re.compile(
    r"^\s+Center of electronic charge \[Ang\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]"
)
NUCLEAR_CHARGE_CENTER_PAT = re.compile(
    r"^\s+Center of nuclear charge \[Ang\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]"
)
DIPOLE_MOMENT_PAT = re.compile(r"^\s+Dipole moment \[D\]:\s+([-\d.]+)")
DIPOLE_COMPONENTS_PAT = re.compile(r"^\s+Cartesian components \[D\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]")
RMS_SIZE_PAT = re.compile(r"^\s+RMS size of the density \[Ang\]:\s+([-\d.]+)")
RMS_COMPONENTS_PAT = re.compile(r"^\s+Cartesian components \[Ang\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]")

# Exciton analysis patterns
EXCITON_START_PAT = re.compile(r"^\s+Exciton analysis of the difference density matrix")
EXCITON_TOTAL_PAT = re.compile(r"^\s+Total:")  # UKS marker
EXCITON_ALPHA_PAT = re.compile(r"^\s+Alpha spin:")  # UKS marker
EXCITON_BETA_PAT = re.compile(r"^\s+Beta spin:")  # UKS marker
HOLE_POS_PAT = re.compile(r"^\s+<r_h> \[Ang\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]")
ELECTRON_POS_PAT = re.compile(r"^\s+<r_e> \[Ang\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]")
SEPARATION_PAT = re.compile(r"^\s+\|<r_e - r_h>\| \[Ang\]:\s+([-\d.]+)")
HOLE_SIZE_PAT = re.compile(r"^\s+Hole size \[Ang\]:\s+([-\d.]+)")
ELECTRON_SIZE_PAT = re.compile(r"^\s+Electron size \[Ang\]:\s+([-\d.]+)")
HOLE_SIZE_COMP_PAT = re.compile(r"^\s+Cartesian components \[Ang\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]")

# End-of-block markers
END_OF_BLOCK_PATS = [
    re.compile(r"^\s+Transition Density Matrix Analysis"),  # Next section
    re.compile(r"^\s+-{70,}"),  # Long dashed separator
]


class UnrelaxedDensityMatrixParser:
    """Parses unrelaxed density matrix analysis from TDDFT calculations."""

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line starts the unrelaxed density matrices block.

        Matches "Analysis of Unrelaxed Density Matrices" and only if not already parsed.
        """
        if state.parsed_tddft_unrelaxed_dm:
            return False

        return bool(UNRELAXED_DM_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse unrelaxed density matrices block.

        Consumes lines from iterator until end of block, populating state.tddft.unrelaxed_density_matrices.
        """
        logger.debug("Starting unrelaxed density matrices block parsing.")

        unrelaxed_dms: list[UnrelaxedDensityMatrix] = []
        is_uks: bool | None = None  # Detect RKS vs UKS

        # Skip separator/blank lines until first state
        # Don't check for dashed separators here - there's one right after the header
        for line in iterator:
            # Detect first state header
            rks_match = RKS_STATE_HEADER_PAT.search(line)
            uks_match = UKS_STATE_HEADER_PAT.search(line)

            if rks_match:
                is_uks = False
                state_num = int(rks_match.group(2))
                multiplicity = rks_match.group(1)
                dm = self._parse_state_rks(iterator, state_num, multiplicity)
                unrelaxed_dms.append(dm)
                break
            elif uks_match:
                is_uks = True
                state_num = int(uks_match.group(1))
                dm = self._parse_state_uks(iterator, state_num)
                unrelaxed_dms.append(dm)
                break

            # Only check for "Transition Density Matrix Analysis" as true end marker
            # (dashed separators appear after header, so we can't use them here)
            if "Transition Density Matrix Analysis" in line:
                state.buffered_line = line
                state.parsing_warnings.append("Unrelaxed DM block ended before any states parsed.")
                return

        if is_uks is None:
            raise ParsingError("Could not determine RKS/UKS type in unrelaxed DM block")

        # Continue parsing remaining states
        for line in iterator:
            # Check for end of block
            if any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                logger.debug(f"Unrelaxed DM parser ended on terminator: {line.strip()}")
                state.buffered_line = line
                break

            # Parse next state
            if is_uks:
                uks_match = UKS_STATE_HEADER_PAT.search(line)
                if uks_match:
                    state_num = int(uks_match.group(1))
                    dm = self._parse_state_uks(iterator, state_num)
                    unrelaxed_dms.append(dm)
            else:
                rks_match = RKS_STATE_HEADER_PAT.search(line)
                if rks_match:
                    state_num = int(rks_match.group(2))
                    multiplicity = rks_match.group(1)
                    dm = self._parse_state_rks(iterator, state_num, multiplicity)
                    unrelaxed_dms.append(dm)

        if not unrelaxed_dms:
            raise ParsingError("No unrelaxed density matrices parsed")

        # --- Update ParseState ---
        state.parsed_tddft_unrelaxed_dm = True

        # Merge with existing TDDFT data if present
        if state.tddft is None:
            state.tddft = TddftResults(unrelaxed_density_matrices=unrelaxed_dms)
        else:
            # Preserve existing fields
            state.tddft = TddftResults(
                tda_states=state.tddft.tda_states,
                tddft_states=state.tddft.tddft_states,
                nto_analyses=state.tddft.nto_analyses,
                ground_state_ref=state.tddft.ground_state_ref,
                unrelaxed_density_matrices=unrelaxed_dms,
            )

        logger.debug(f"Parsed {len(unrelaxed_dms)} unrelaxed density matrices.")

    def _parse_state_rks(self, iterator: Iterator[str], state_num: int, multiplicity: str) -> UnrelaxedDensityMatrix:
        """Parse a single RKS state's unrelaxed density matrix analysis."""
        logger.debug(f"Parsing RKS state {state_num}")

        # Data containers
        nos_data: dict = {}
        mulliken_charges: dict[int, float] = {}
        hole_pops: dict[int, float] = {}
        electron_pops: dict[int, float] = {}
        multipole_data: dict = {}
        exciton_data: dict = {}

        # State flags
        in_nos = False
        awaiting_occupation_values = False
        in_mulliken = False
        in_multipole = False
        in_exciton = False
        next_line_is_hole_size_components = False
        next_line_is_electron_size_components = False
        next_line_is_rms_components = False

        for line in iterator:
            # Check for next state or end of block
            if RKS_STATE_HEADER_PAT.search(line) or any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                # We've read too far - return the line
                # Note: We don't set buffered_line here because we're in a sub-parser
                # Instead, we'll need to handle this differently
                # Actually, we should just break and let the main loop handle it
                break

            # --- Parse NOs section ---
            if NOS_SECTION_PAT.search(line):
                in_nos = True
                continue

            if in_nos and FRONTIER_NOS_PAT.search(line):
                awaiting_occupation_values = True
                continue

            if awaiting_occupation_values:
                val_match = OCCUPATION_VAL_PAT.search(line)
                if val_match:
                    values = [float(v) for v in val_match.groups() if v is not None]
                    if values:
                        nos_data["frontier_nos"] = values
                        awaiting_occupation_values = False
                        continue

            if in_nos:
                electrons_match = NUM_ELECTRONS_PAT.search(line)
                if electrons_match:
                    nos_data["num_electrons"] = float(electrons_match.group(1))
                    continue

                unpaired_match = NUM_UNPAIRED_PAT.search(line)
                if unpaired_match:
                    nos_data["n_u"] = float(unpaired_match.group(1))
                    nos_data["n_u_nl"] = float(unpaired_match.group(2))
                    continue

                pr_no_match = PR_NO_PAT.search(line)
                if pr_no_match:
                    nos_data["pr_no"] = float(pr_no_match.group(1))
                    in_nos = False  # End of NOs section
                    continue

            # --- Parse Mulliken section ---
            if MULLIKEN_START_PAT.search(line):
                in_mulliken = True
                continue

            if in_mulliken:
                atom_match = MULLIKEN_ATOM_RKS_PAT.search(line)
                if atom_match:
                    atom_idx = int(atom_match.group(1)) - 1  # 0-indexed
                    charge = float(atom_match.group(2))
                    hole = float(atom_match.group(3))
                    electron = float(atom_match.group(4))
                    mulliken_charges[atom_idx] = charge
                    hole_pops[atom_idx] = hole
                    electron_pops[atom_idx] = electron
                    continue

                if line.strip().startswith("Sum:"):
                    in_mulliken = False
                    continue

            # --- Parse Multipole section ---
            if MULTIPOLE_START_PAT.search(line):
                in_multipole = True
                continue

            if in_multipole:
                charge_match = MOLECULAR_CHARGE_PAT.search(line)
                if charge_match:
                    multipole_data["molecular_charge"] = float(charge_match.group(1))
                    continue

                total_elec_match = TOTAL_ELECTRONS_PAT.search(line)
                if total_elec_match:
                    multipole_data["total_electrons"] = float(total_elec_match.group(1))
                    continue

                elec_center_match = ELECTRONIC_CHARGE_CENTER_PAT.search(line)
                if elec_center_match:
                    multipole_data["electronic_charge_center"] = (
                        float(elec_center_match.group(1)),
                        float(elec_center_match.group(2)),
                        float(elec_center_match.group(3)),
                    )
                    continue

                nuc_center_match = NUCLEAR_CHARGE_CENTER_PAT.search(line)
                if nuc_center_match:
                    multipole_data["nuclear_charge_center"] = (
                        float(nuc_center_match.group(1)),
                        float(nuc_center_match.group(2)),
                        float(nuc_center_match.group(3)),
                    )
                    continue

                dipole_match = DIPOLE_MOMENT_PAT.search(line)
                if dipole_match:
                    multipole_data["dipole_moment"] = float(dipole_match.group(1))
                    continue

                dipole_comp_match = DIPOLE_COMPONENTS_PAT.search(line)
                if dipole_comp_match:
                    multipole_data["dipole_components"] = (
                        float(dipole_comp_match.group(1)),
                        float(dipole_comp_match.group(2)),
                        float(dipole_comp_match.group(3)),
                    )
                    continue

                rms_match = RMS_SIZE_PAT.search(line)
                if rms_match:
                    multipole_data["rms_size"] = float(rms_match.group(1))
                    next_line_is_rms_components = True
                    continue

                if next_line_is_rms_components:
                    rms_comp_match = RMS_COMPONENTS_PAT.search(line)
                    if rms_comp_match:
                        multipole_data["rms_components"] = (
                            float(rms_comp_match.group(1)),
                            float(rms_comp_match.group(2)),
                            float(rms_comp_match.group(3)),
                        )
                        next_line_is_rms_components = False
                        in_multipole = False  # End of multipole section
                        continue

            # --- Parse Exciton section ---
            if EXCITON_START_PAT.search(line):
                in_exciton = True
                continue

            if in_exciton:
                hole_pos_match = HOLE_POS_PAT.search(line)
                if hole_pos_match:
                    exciton_data["hole_position"] = (
                        float(hole_pos_match.group(1)),
                        float(hole_pos_match.group(2)),
                        float(hole_pos_match.group(3)),
                    )
                    continue

                elec_pos_match = ELECTRON_POS_PAT.search(line)
                if elec_pos_match:
                    exciton_data["electron_position"] = (
                        float(elec_pos_match.group(1)),
                        float(elec_pos_match.group(2)),
                        float(elec_pos_match.group(3)),
                    )
                    continue

                sep_match = SEPARATION_PAT.search(line)
                if sep_match:
                    exciton_data["separation"] = float(sep_match.group(1))
                    continue

                hole_size_match = HOLE_SIZE_PAT.search(line)
                if hole_size_match:
                    exciton_data["hole_size"] = float(hole_size_match.group(1))
                    next_line_is_hole_size_components = True
                    continue

                if next_line_is_hole_size_components:
                    comp_match = HOLE_SIZE_COMP_PAT.search(line)
                    if comp_match:
                        exciton_data["hole_size_components"] = (
                            float(comp_match.group(1)),
                            float(comp_match.group(2)),
                            float(comp_match.group(3)),
                        )
                        next_line_is_hole_size_components = False
                        continue

                elec_size_match = ELECTRON_SIZE_PAT.search(line)
                if elec_size_match:
                    exciton_data["electron_size"] = float(elec_size_match.group(1))
                    next_line_is_electron_size_components = True
                    continue

                if next_line_is_electron_size_components:
                    comp_match = HOLE_SIZE_COMP_PAT.search(line)
                    if comp_match:
                        exciton_data["electron_size_components"] = (
                            float(comp_match.group(1)),
                            float(comp_match.group(2)),
                            float(comp_match.group(3)),
                        )
                        next_line_is_electron_size_components = False
                        in_exciton = False  # End of exciton section
                        continue

        # Validate and construct models
        if not nos_data.get("frontier_nos"):
            raise ParsingError(f"State {state_num}: no frontier NOs parsed")

        nos = NaturalOrbitals(
            frontier_occupations=nos_data["frontier_nos"],
            num_electrons=nos_data.get("num_electrons"),
            num_unpaired=nos_data.get("n_u"),
            num_unpaired_nl=nos_data.get("n_u_nl"),
            pr_no=nos_data.get("pr_no"),
        )

        mulliken = AtomicCharges(
            method="Mulliken", charges=mulliken_charges, hole_populations=hole_pops, electron_populations=electron_pops
        )

        exciton = ExcitonAnalysis(
            r_h_ang=exciton_data.get("hole_position"),
            r_e_ang=exciton_data.get("electron_position"),
            separation_ang=exciton_data.get("separation"),
            hole_size_ang=exciton_data.get("hole_size"),
            hole_size_components_ang=exciton_data.get("hole_size_components"),
            electron_size_ang=exciton_data.get("electron_size"),
            electron_size_components_ang=exciton_data.get("electron_size_components"),
        )

        return UnrelaxedDensityMatrix(
            state_number=state_num,
            multiplicity=multiplicity,
            nos_spin_traced=nos,
            mulliken=mulliken,
            molecular_charge=multipole_data.get("molecular_charge"),
            num_electrons=multipole_data.get("total_electrons"),
            dipole_moment_debye=multipole_data.get("dipole_moment"),
            dipole_components_debye=multipole_data.get("dipole_components"),
            exciton_total=exciton,
        )

    def _parse_state_uks(self, iterator: Iterator[str], state_num: int) -> UnrelaxedDensityMatrix:
        """Parse a single UKS state's unrelaxed density matrix analysis."""
        logger.debug(f"Parsing UKS state {state_num}")

        # Data containers
        nos_data: dict = {}
        mulliken_charges: dict[int, float] = {}
        mulliken_spins: dict[int, float] = {}
        hole_pops_alpha: dict[int, float] = {}
        hole_pops_beta: dict[int, float] = {}
        electron_pops_alpha: dict[int, float] = {}
        electron_pops_beta: dict[int, float] = {}
        multipole_data: dict = {}
        exciton_data: dict = {}
        exciton_alpha_data: dict = {}
        exciton_beta_data: dict = {}

        # State flags
        in_nos_spin_traced = False
        awaiting_occupation_values = False
        in_mulliken = False
        in_multipole = False
        in_exciton = False
        exciton_section: str | None = None  # "total", "alpha", "beta"
        next_line_is_hole_size_components = False
        next_line_is_electron_size_components = False
        next_line_is_rms_components = False

        for line in iterator:
            # Check for next state or end of block
            if UKS_STATE_HEADER_PAT.search(line) or any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                break

            # --- Detect NOs spin-traced section ---
            if NOS_SPIN_TRACED_PAT.search(line):
                in_nos_spin_traced = True
                continue

            if in_nos_spin_traced and FRONTIER_NOS_PAT.search(line):
                awaiting_occupation_values = True
                continue

            if awaiting_occupation_values:
                val_match = OCCUPATION_VAL_PAT.search(line)
                if val_match:
                    values = [float(v) for v in val_match.groups() if v is not None]
                    if values:
                        nos_data["frontier_nos"] = values
                        awaiting_occupation_values = False
                        continue

            if in_nos_spin_traced:
                electrons_match = NUM_ELECTRONS_PAT.search(line)
                if electrons_match:
                    nos_data["num_electrons"] = float(electrons_match.group(1))
                    continue

                unpaired_match = NUM_UNPAIRED_PAT.search(line)
                if unpaired_match:
                    nos_data["n_u"] = float(unpaired_match.group(1))
                    nos_data["n_u_nl"] = float(unpaired_match.group(2))
                    continue

                pr_no_match = PR_NO_PAT.search(line)
                if pr_no_match:
                    nos_data["pr_no"] = float(pr_no_match.group(1))
                    in_nos_spin_traced = False
                    continue

            # --- Parse Mulliken section ---
            if MULLIKEN_START_PAT.search(line):
                in_mulliken = True
                continue

            if in_mulliken:
                atom_match = MULLIKEN_ATOM_UKS_PAT.search(line)
                if atom_match:
                    atom_idx = int(atom_match.group(1)) - 1  # 0-indexed
                    charge = float(atom_match.group(2))
                    spin = float(atom_match.group(3))
                    hole_alpha = float(atom_match.group(4))
                    hole_beta = float(atom_match.group(5))
                    elec_alpha = float(atom_match.group(6))
                    elec_beta = float(atom_match.group(7))
                    mulliken_charges[atom_idx] = charge
                    mulliken_spins[atom_idx] = spin
                    hole_pops_alpha[atom_idx] = hole_alpha
                    hole_pops_beta[atom_idx] = hole_beta
                    electron_pops_alpha[atom_idx] = elec_alpha
                    electron_pops_beta[atom_idx] = elec_beta
                    continue

                if line.strip().startswith("Sum:"):
                    in_mulliken = False
                    continue

            # --- Parse Multipole section ---
            if MULTIPOLE_START_PAT.search(line):
                in_multipole = True
                continue

            if in_multipole:
                charge_match = MOLECULAR_CHARGE_PAT.search(line)
                if charge_match:
                    multipole_data["molecular_charge"] = float(charge_match.group(1))
                    continue

                total_elec_match = TOTAL_ELECTRONS_PAT.search(line)
                if total_elec_match:
                    multipole_data["total_electrons"] = float(total_elec_match.group(1))
                    continue

                elec_center_match = ELECTRONIC_CHARGE_CENTER_PAT.search(line)
                if elec_center_match:
                    multipole_data["electronic_charge_center"] = (
                        float(elec_center_match.group(1)),
                        float(elec_center_match.group(2)),
                        float(elec_center_match.group(3)),
                    )
                    continue

                nuc_center_match = NUCLEAR_CHARGE_CENTER_PAT.search(line)
                if nuc_center_match:
                    multipole_data["nuclear_charge_center"] = (
                        float(nuc_center_match.group(1)),
                        float(nuc_center_match.group(2)),
                        float(nuc_center_match.group(3)),
                    )
                    continue

                dipole_match = DIPOLE_MOMENT_PAT.search(line)
                if dipole_match:
                    multipole_data["dipole_moment"] = float(dipole_match.group(1))
                    continue

                dipole_comp_match = DIPOLE_COMPONENTS_PAT.search(line)
                if dipole_comp_match:
                    multipole_data["dipole_components"] = (
                        float(dipole_comp_match.group(1)),
                        float(dipole_comp_match.group(2)),
                        float(dipole_comp_match.group(3)),
                    )
                    continue

                rms_match = RMS_SIZE_PAT.search(line)
                if rms_match:
                    multipole_data["rms_size"] = float(rms_match.group(1))
                    next_line_is_rms_components = True
                    continue

                if next_line_is_rms_components:
                    rms_comp_match = RMS_COMPONENTS_PAT.search(line)
                    if rms_comp_match:
                        multipole_data["rms_components"] = (
                            float(rms_comp_match.group(1)),
                            float(rms_comp_match.group(2)),
                            float(rms_comp_match.group(3)),
                        )
                        next_line_is_rms_components = False
                        in_multipole = False
                        continue

            # --- Parse Exciton section ---
            if EXCITON_START_PAT.search(line):
                in_exciton = True
                continue

            if in_exciton:
                # Detect subsection markers
                if EXCITON_TOTAL_PAT.search(line):
                    exciton_section = "total"
                    continue
                if EXCITON_ALPHA_PAT.search(line):
                    exciton_section = "alpha"
                    continue
                if EXCITON_BETA_PAT.search(line):
                    exciton_section = "beta"
                    continue

                # Determine which dict to populate
                target_dict = exciton_data
                if exciton_section == "alpha":
                    target_dict = exciton_alpha_data
                elif exciton_section == "beta":
                    target_dict = exciton_beta_data

                hole_pos_match = HOLE_POS_PAT.search(line)
                if hole_pos_match:
                    target_dict["hole_position"] = (
                        float(hole_pos_match.group(1)),
                        float(hole_pos_match.group(2)),
                        float(hole_pos_match.group(3)),
                    )
                    continue

                elec_pos_match = ELECTRON_POS_PAT.search(line)
                if elec_pos_match:
                    target_dict["electron_position"] = (
                        float(elec_pos_match.group(1)),
                        float(elec_pos_match.group(2)),
                        float(elec_pos_match.group(3)),
                    )
                    continue

                sep_match = SEPARATION_PAT.search(line)
                if sep_match:
                    target_dict["separation"] = float(sep_match.group(1))
                    continue

                hole_size_match = HOLE_SIZE_PAT.search(line)
                if hole_size_match:
                    target_dict["hole_size"] = float(hole_size_match.group(1))
                    next_line_is_hole_size_components = True
                    continue

                if next_line_is_hole_size_components:
                    comp_match = HOLE_SIZE_COMP_PAT.search(line)
                    if comp_match:
                        target_dict["hole_size_components"] = (
                            float(comp_match.group(1)),
                            float(comp_match.group(2)),
                            float(comp_match.group(3)),
                        )
                        next_line_is_hole_size_components = False
                        continue

                elec_size_match = ELECTRON_SIZE_PAT.search(line)
                if elec_size_match:
                    target_dict["electron_size"] = float(elec_size_match.group(1))
                    next_line_is_electron_size_components = True
                    continue

                if next_line_is_electron_size_components:
                    comp_match = HOLE_SIZE_COMP_PAT.search(line)
                    if comp_match:
                        target_dict["electron_size_components"] = (
                            float(comp_match.group(1)),
                            float(comp_match.group(2)),
                            float(comp_match.group(3)),
                        )
                        next_line_is_electron_size_components = False
                        # Check if we've finished beta section (last one)
                        if exciton_section == "beta":
                            in_exciton = False
                        continue

        # Validate and construct models
        if not nos_data.get("frontier_nos"):
            raise ParsingError(f"State {state_num}: no frontier NOs parsed")

        nos = NaturalOrbitals(
            frontier_occupations=nos_data["frontier_nos"],
            num_electrons=nos_data.get("num_electrons"),
            num_unpaired=nos_data.get("n_u"),
            num_unpaired_nl=nos_data.get("n_u_nl"),
            pr_no=nos_data.get("pr_no"),
        )

        mulliken = AtomicCharges(
            method="Mulliken",
            charges=mulliken_charges,
            spins=mulliken_spins,
            hole_populations_alpha=hole_pops_alpha,
            hole_populations_beta=hole_pops_beta,
            electron_populations_alpha=electron_pops_alpha,
            electron_populations_beta=electron_pops_beta,
        )

        exciton = ExcitonAnalysis(
            r_h_ang=exciton_data.get("hole_position"),
            r_e_ang=exciton_data.get("electron_position"),
            separation_ang=exciton_data.get("separation"),
            hole_size_ang=exciton_data.get("hole_size"),
            hole_size_components_ang=exciton_data.get("hole_size_components"),
            electron_size_ang=exciton_data.get("electron_size"),
            electron_size_components_ang=exciton_data.get("electron_size_components"),
        )

        # Build alpha/beta exciton analyses if data present
        exciton_alpha = None
        if exciton_alpha_data:
            exciton_alpha = ExcitonAnalysis(
                r_h_ang=exciton_alpha_data.get("hole_position"),
                r_e_ang=exciton_alpha_data.get("electron_position"),
                separation_ang=exciton_alpha_data.get("separation"),
                hole_size_ang=exciton_alpha_data.get("hole_size"),
                hole_size_components_ang=exciton_alpha_data.get("hole_size_components"),
                electron_size_ang=exciton_alpha_data.get("electron_size"),
                electron_size_components_ang=exciton_alpha_data.get("electron_size_components"),
            )

        exciton_beta = None
        if exciton_beta_data:
            exciton_beta = ExcitonAnalysis(
                r_h_ang=exciton_beta_data.get("hole_position"),
                r_e_ang=exciton_beta_data.get("electron_position"),
                separation_ang=exciton_beta_data.get("separation"),
                hole_size_ang=exciton_beta_data.get("hole_size"),
                hole_size_components_ang=exciton_beta_data.get("hole_size_components"),
                electron_size_ang=exciton_beta_data.get("electron_size"),
                electron_size_components_ang=exciton_beta_data.get("electron_size_components"),
            )

        # UKS doesn't have multiplicity in state header - will be determined from excited state info
        return UnrelaxedDensityMatrix(
            state_number=state_num,
            multiplicity=None,  # UKS doesn't specify in this section
            nos_spin_traced=nos,
            mulliken=mulliken,
            molecular_charge=multipole_data.get("molecular_charge"),
            num_electrons=multipole_data.get("total_electrons"),
            dipole_moment_debye=multipole_data.get("dipole_moment"),
            dipole_components_debye=multipole_data.get("dipole_components"),
            exciton_total=exciton,
            exciton_alpha=exciton_alpha,
            exciton_beta=exciton_beta,
        )
