"""
Parser for QChem ground state reference data in excited state analysis.

This parser extracts ground state (reference) data from the TDDFT Excited State
Analysis block, including:
- Frontier natural orbital occupations
- Electron counts and unpaired electron info
- Mulliken population analysis (charges and optionally spins for UKS)
- Dipole moment and Cartesian components

The parser handles both RKS (restricted) and UKS (unrestricted) calculations.
"""

import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import AtomicCharges, GroundStateReference, TddftResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Pattern constants ---
# Start marker: identifies the beginning of ground state reference section
GS_REF_START_PAT = re.compile(r"^\s+Ground State \(Reference\) :")

# Frontier NOs occupation pattern (handles "0.0000   2.0000" or similar)
FRONTIER_NOS_PAT = re.compile(r"^\s+Occupation of frontier NOs:")

# NOs spin-traced marker (UKS only)
NOS_SPIN_TRACED_PAT = re.compile(r"^\s+NOs \(spin-traced\)")

# Occupation values pattern (captures floating point numbers from indented line)
# Matches lines with leading spaces followed by numbers
OCCUPATION_VAL_PAT = re.compile(r"^\s+([\d.]+)(?:\s+([\d.]+))?(?:\s+([\d.]+))?")  # Up to 3 values per line

# Electron count patterns
NUM_ELECTRONS_PAT = re.compile(r"^\s+Number of electrons:\s+([\d.]+)")
NUM_UNPAIRED_PAT = re.compile(r"^\s+Number of unpaired electrons:\s+n_u\s*=\s*([-\d.]+)")

# Mulliken analysis section marker (matches both 5.4 without suffix and 6.2 with (State DM))
MULLIKEN_START_PAT = re.compile(r"^\s+Mulliken Population Analysis")

# Atom line in Mulliken table (with optional spin column)
# Format: " 1 H        0.230554" or " 1 H        0.230554        0.000000"
MULLIKEN_ATOM_PAT = re.compile(r"^\s+(\d+)\s+[A-Z][a-z]?\s+([-\d.]+)(?:\s+([-\d.]+))?")

# Dipole moment patterns
DIPOLE_MOMENT_PAT = re.compile(r"^\s+Dipole moment \[D\]:\s+([-\d.]+)")
DIPOLE_COMPONENTS_PAT = re.compile(r"^\s+Cartesian components \[D\]:\s+\[\s+([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\s*\]")

# End-of-block markers (must have seen some content first)
END_OF_BLOCK_PATS = [
    re.compile(r"^\s+Analysis of Unrelaxed Density Matrices"),  # Start of next section
    re.compile(r"^\s+-{70,}"),  # Long dashed separator (at least 70 dashes)
]


class GroundStateRefParser:
    """Parses ground state reference data from TDDFT excited state analysis block."""

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line starts the ground state reference block.

        Matches "Ground State (Reference) :" and only if not already parsed.
        """
        if state.parsed_tddft_gs_ref:
            return False

        return bool(GS_REF_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse ground state reference block.

        Consumes lines from iterator until end of block, populating state.tddft.ground_state_ref.
        """
        logger.debug("Starting ground state reference block parsing.")

        gs_data: dict = {
            "frontier_nos": [],
            "num_electrons": None,
            "num_unpaired_electrons": None,
            "mulliken_charges": {},  # atom_index -> charge
            "mulliken_spins": {},  # atom_index -> spin
            "dipole_moment_debye": None,
            "dipole_components_debye": None,
        }

        atom_index = -1  # Track which atom we're parsing in Mulliken table

        seen_frontier_nos = False
        in_mulliken_table = False
        seen_nos_alpha = False  # Track if we've seen NOs (alpha) - indicates UKS
        in_spin_traced_section = False  # Track if we're in the spin-traced section for UKS

        for line in iterator:
            # --- Check for end of block ---
            if any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                logger.debug(f"Ground state ref parser ended on terminator: {line.strip()}")
                state.buffered_line = line
                break

            # --- Detect NOs sections (to identify UKS vs RKS) ---
            # UKS has "NOs (alpha)", "NOs (beta)", "NOs (spin-traced)"
            # RKS has just "NOs"
            if "NOs (alpha)" in line:
                seen_nos_alpha = True
                continue

            if NOS_SPIN_TRACED_PAT.search(line):
                in_spin_traced_section = True
                continue

            # --- Parse frontier NOs occupations ---
            # For UKS, only capture from spin-traced section
            # For RKS, capture the first one we see
            if FRONTIER_NOS_PAT.search(line):
                # Only capture if we haven't already and conditions are met
                if not gs_data["frontier_nos"] and (in_spin_traced_section or (not seen_nos_alpha)):
                    seen_frontier_nos = True
                    # Next line(s) should contain occupation values
                continue

            # Capture occupation values (after frontier NOs line)
            if seen_frontier_nos and not gs_data["frontier_nos"]:
                # Try to parse numeric values from this line
                val_match = OCCUPATION_VAL_PAT.search(line)
                if val_match:
                    # Collect all non-empty groups
                    values = [float(v) for v in val_match.groups() if v is not None]
                    if values:
                        gs_data["frontier_nos"] = values
                        seen_frontier_nos = False
                        in_spin_traced_section = False  # Reset after capturing
                        continue
                # Empty line or non-numeric - skip
                if not line.strip():
                    continue

            # --- Parse number of electrons ---
            electrons_match = NUM_ELECTRONS_PAT.search(line)
            if electrons_match:
                gs_data["num_electrons"] = float(electrons_match.group(1))
                continue

            # --- Parse number of unpaired electrons ---
            unpaired_match = NUM_UNPAIRED_PAT.search(line)
            if unpaired_match:
                gs_data["num_unpaired_electrons"] = float(unpaired_match.group(1))
                continue

            # --- Detect start of Mulliken analysis ---
            if MULLIKEN_START_PAT.search(line):
                in_mulliken_table = True
                continue

            # --- Parse Mulliken atomic charges/spins ---
            if in_mulliken_table:
                # Try to match an atom line
                atom_match = MULLIKEN_ATOM_PAT.search(line)
                if atom_match:
                    atom_index = int(atom_match.group(1)) - 1  # Convert to 0-based indexing
                    charge = float(atom_match.group(2))
                    gs_data["mulliken_charges"][atom_index] = charge

                    # If spin value present, add it
                    if atom_match.group(3) is not None:
                        spin = float(atom_match.group(3))
                        gs_data["mulliken_spins"][atom_index] = spin
                    continue

                # Check for end of Mulliken table (separator or Sum line)
                if line.strip().startswith("Sum:"):
                    in_mulliken_table = False
                    continue

                # Skip header/separator lines within the table
                if line.strip().startswith("---") or line.strip().startswith("Atom"):
                    continue

            # --- Parse dipole moment ---
            dipole_match = DIPOLE_MOMENT_PAT.search(line)
            if dipole_match:
                gs_data["dipole_moment_debye"] = float(dipole_match.group(1))
                continue

            # --- Parse dipole components ---
            components_match = DIPOLE_COMPONENTS_PAT.search(line)
            if components_match:
                x = float(components_match.group(1))
                y = float(components_match.group(2))
                z = float(components_match.group(3))
                gs_data["dipole_components_debye"] = (x, y, z)
                continue

        # --- Validate and construct GroundStateReference ---
        if not gs_data["frontier_nos"]:
            raise ParsingError("Ground state reference: no frontier NOs parsed")
        if gs_data["num_electrons"] is None:
            raise ParsingError("Ground state reference: no electron count parsed")
        if not gs_data["mulliken_charges"]:
            raise ParsingError("Ground state reference: no Mulliken charges parsed")
        if gs_data["dipole_moment_debye"] is None:
            raise ParsingError("Ground state reference: no dipole moment parsed")
        if gs_data["dipole_components_debye"] is None:
            raise ParsingError("Ground state reference: no dipole components parsed")

        # Convert mulliken_spins to None if empty (RKS case)
        mulliken_spins = gs_data["mulliken_spins"] if gs_data["mulliken_spins"] else None

        # Create AtomicCharges model
        mulliken = AtomicCharges(
            method="Mulliken",
            charges=gs_data["mulliken_charges"],
            spins=mulliken_spins,
        )

        gs_ref = GroundStateReference(
            frontier_nos=gs_data["frontier_nos"],
            num_electrons=gs_data["num_electrons"],
            num_unpaired_electrons=gs_data["num_unpaired_electrons"],
            mulliken=mulliken,
            dipole_moment_debye=gs_data["dipole_moment_debye"],
            dipole_components_debye=gs_data["dipole_components_debye"],
        )

        # --- Update ParseState ---
        state.parsed_tddft_gs_ref = True

        # Merge with existing TDDFT data if present
        if state.tddft is None:
            state.tddft = TddftResults(ground_state_ref=gs_ref)
        else:
            # Preserve existing states and analyses
            state.tddft = TddftResults(
                tda_states=state.tddft.tda_states,
                tddft_states=state.tddft.tddft_states,
                nto_analyses=state.tddft.nto_analyses,
                ground_state_ref=gs_ref,
            )

        logger.debug("Ground state reference parsing completed successfully.")
