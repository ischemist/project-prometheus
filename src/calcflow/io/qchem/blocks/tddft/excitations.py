"""
Parser for QChem TDDFT and TDA excitation energies.

This parser handles two types of blocks:
1. TDDFT/TDA Excitation Energies (Tamm-Dancoff Approximation initial guess)
2. TDDFT Excitation Energies (full TDDFT)

Both blocks contain excited state information including:
- Excitation energies in eV and total energies in au
- Multiplicity or <S**2> values
- Transition moments (X, Y, Z components)
- Oscillator strengths
- Orbital transitions with amplitudes (possibly with spin labels for UKS)
"""

import re
from collections.abc import Iterator

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import ExcitedState, OrbitalTransition, TddftResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Pattern constants ---
# Start patterns: identify the beginning of a TDDFT block
TDA_START_PAT = re.compile(r"^\s*TDDFT/TDA\s+Excitation\s+Energies")
TDDFT_START_PAT = re.compile(r"^\s+TDDFT\s+Excitation\s+Energies")

# Block structure patterns
STATE_START_PAT = re.compile(r"^\s*Excited state\s+(\d+):\s+excitation energy \(eV\)\s+=\s+([\d.]+)")
TOTAL_ENERGY_PAT = re.compile(r"^\s*Total energy for state\s+\d+:\s+([-\d.]+)\s+au")
MULTIPLICITY_PAT = re.compile(r"^\s*Multiplicity:\s+(Singlet|Triplet)")
S_SQUARED_PAT = re.compile(r"^\s*<S\*\*2>\s*:\s*([-\d.]+)")
TRANS_MOM_PAT = re.compile(r"^\s*Trans\.\s+Mom\.\s*:\s+([-\d.]+)\s+X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z")
STRENGTH_PAT = re.compile(r"^\s*Strength\s+:\s+([\d.eE+-]+)")
# Transition pattern: handles both "D(5) --> V(1)" and "X: D(5) --> V(1)" with optional spin labels
TRANSITION_PAT = re.compile(
    r"^\s*(?:X:\s)?D\(\s*(\d+)\)\s*-->\s*V\(\s*(\d+)\)\s+amplitude\s*=\s*([-\d.]+)(?:\s+(alpha|beta))?"
)

# End-of-block markers (after first state is parsed)
END_OF_BLOCK_PATS = [
    re.compile(r"^\s*-+\s*$"),  # Dashed separator line (end of block)
    re.compile(r"^\s*Exchange:"),  # Start of convergence iteration info
    re.compile(r"^\s*Direct TDDFT calculation"),
]


class ExcitationsParser:
    """Parses TDDFT/TDA excitation energies blocks."""

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line starts a TDDFT excitation block.

        Matches either TDA or full TDDFT block start, and only if not already parsed.
        """
        if state.parsed_tddft_tda and state.parsed_tddft_full:
            return False

        return bool(TDA_START_PAT.search(line) or TDDFT_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse TDDFT/TDA excitation energies block.

        Consumes lines from iterator until end of block, populating state.tddft.
        """
        logger.debug("Starting TDDFT excitations block parsing.")

        # Determine block type
        is_tda = bool(TDA_START_PAT.search(start_line))
        is_tddft = bool(TDDFT_START_PAT.search(start_line))

        if not (is_tda or is_tddft):
            raise ParsingError(f"Unexpected start line for excitations parser: {start_line.strip()}")

        excited_states: list[ExcitedState] = []
        current_state: dict | None = None
        seen_first_state = False

        for line in iterator:
            # --- Check for end of block (only after we've seen at least one state) ---
            if seen_first_state and any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                logger.debug(f"Excitations parser ended on terminator: {line.strip()}")
                state.buffered_line = line
                break

            # --- Parse new excited state ---
            state_match = STATE_START_PAT.search(line)
            if state_match:
                # Save previous state if exists
                if current_state is not None:
                    excited_states.append(self._build_excited_state(current_state))

                # Start new state
                state_number = int(state_match.group(1))
                excitation_energy_ev = float(state_match.group(2))
                current_state = {
                    "state_number": state_number,
                    "excitation_energy_ev": excitation_energy_ev,
                    "total_energy_au": None,
                    "multiplicity": "Unknown",
                    "oscillator_strength": None,
                    "transitions": [],
                }
                seen_first_state = True
                continue

            if current_state is None:
                # Skip lines before first state
                continue

            # --- Parse total energy ---
            energy_match = TOTAL_ENERGY_PAT.search(line)
            if energy_match:
                current_state["total_energy_au"] = float(energy_match.group(1))
                continue

            # --- Parse multiplicity ---
            mult_match = MULTIPLICITY_PAT.search(line)
            if mult_match:
                current_state["multiplicity"] = mult_match.group(1)
                continue

            # --- Parse <S**2> (for UKS) ---
            s2_match = S_SQUARED_PAT.search(line)
            if s2_match:
                # For UKS, store S**2 value in multiplicity field
                # (not ideal but follows existing pattern in models)
                s2_value = float(s2_match.group(1))
                if s2_value > 1.5:
                    current_state["multiplicity"] = "Triplet"
                else:
                    current_state["multiplicity"] = "Singlet"
                continue

            # --- Parse transition moment ---
            tm_match = TRANS_MOM_PAT.search(line)
            if tm_match:
                current_state["trans_mom_x"] = float(tm_match.group(1))
                current_state["trans_mom_y"] = float(tm_match.group(2))
                current_state["trans_mom_z"] = float(tm_match.group(3))
                continue

            # --- Parse oscillator strength ---
            strength_match = STRENGTH_PAT.search(line)
            if strength_match:
                current_state["oscillator_strength"] = float(strength_match.group(1))
                continue

            # --- Parse orbital transitions ---
            trans_match = TRANSITION_PAT.search(line)
            if trans_match:
                from_idx = int(trans_match.group(1)) - 1  # Convert to 0-indexed
                to_idx = int(trans_match.group(2)) - 1
                amplitude = float(trans_match.group(3))
                spin_label = trans_match.group(4)

                is_alpha = None
                if spin_label:
                    is_alpha = spin_label == "alpha"

                transition = OrbitalTransition(
                    from_idx=from_idx, to_idx=to_idx, amplitude=amplitude, is_alpha_spin=is_alpha
                )
                current_state["transitions"].append(transition)
                continue

        # --- Save final state ---
        if current_state is not None:
            excited_states.append(self._build_excited_state(current_state))

        if not excited_states:
            state.parsing_warnings.append("TDDFT/TDA excitations block found but no states parsed.")
            return

        # --- Update ParseState ---
        # Determine which field to populate based on block type
        if is_tda:
            state.parsed_tddft_tda = True
            if state.tddft is None:
                state.tddft = TddftResults(tda_states=excited_states)
            else:
                # Merge with existing TDDFT states if present
                state.tddft = TddftResults(tda_states=excited_states, tddft_states=state.tddft.tddft_states)
        else:  # is_tddft
            state.parsed_tddft_full = True
            if state.tddft is None:
                state.tddft = TddftResults(tddft_states=excited_states)
            else:
                # Merge with existing TDA states if present
                state.tddft = TddftResults(tda_states=state.tddft.tda_states, tddft_states=excited_states)

        logger.debug(f"Parsed {len(excited_states)} excited states.")

    def _build_excited_state(self, state_data: dict) -> ExcitedState:
        """Build an ExcitedState model from parsed dictionary."""
        return ExcitedState(
            state_number=state_data["state_number"],
            multiplicity=state_data["multiplicity"],
            excitation_energy_ev=state_data["excitation_energy_ev"],
            total_energy_au=state_data["total_energy_au"] or 0.0,  # Fallback if not found (shouldn't happen)
            oscillator_strength=state_data.get("oscillator_strength"),
            transitions=state_data.get("transitions", []),
            trans_mom_x=state_data.get("trans_mom_x"),
            trans_mom_y=state_data.get("trans_mom_y"),
            trans_mom_z=state_data.get("trans_mom_z"),
        )
