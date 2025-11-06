"""
Parser for QChem SA-NTO (State-Averaged Natural Transition Orbital) decomposition.

This parser handles the NTO decomposition block that appears in TDDFT output,
providing information on the hole-electron character of each excited state.

Format:
- RKS: States labeled as "Singlet N :" or "Triplet N :"
- UKS: States labeled as "Excited State N :" with separate "Alpha spin:" and "Beta spin:" sections
- Each contribution: "H- X -> L+ Y: [±]0.XXXX ( XX.X%)"
- Omega: "omega = XXX.X%" (or "omega =  XX.X%" for UKS per spin)
"""

import re
from collections.abc import Iterator

from calcflow.common.results import NTOContribution, NTOStateAnalysis, TddftResults
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Pattern constants ---
# Block start
NTO_START_PAT = re.compile(r"^\s*SA-NTO\s+Decomposition")

# State headers
RKS_STATE_PAT = re.compile(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:")
UKS_STATE_PAT = re.compile(r"^\s*Excited\s+State\s+(\d+)\s*:")

# Spin section headers (for UKS)
SPIN_SECTION_PAT = re.compile(r"^\s*(Alpha|Beta)\s+spin:")

# NTO contribution line
# Matches: "H- 0 -> L+ 0: -0.7067 ( 99.9%)"
CONTRIBUTION_PAT = re.compile(r"^\s*H-\s*(\d+)\s*->\s*L\+\s*(\d+):\s*([-\d.]+)\s*\(\s*([\d.]+)%\)")

# Omega line (total character)
# Matches: "omega = 100.1%" or "omega =  50.1%"
OMEGA_PAT = re.compile(r"^\s*omega\s*=\s*([\d.]+)%")

# End-of-block patterns
END_OF_BLOCK_PATS = [
    re.compile(r"^\s*={50,}\s*$"),  # Long equals line (marks end of NTO section, at least 50 =)
]


class NTOParser:
    """Parses SA-NTO decomposition blocks from QChem TDDFT output."""

    def matches(self, line: str, state: ParseState) -> bool:
        """
        Check if this line starts an SA-NTO decomposition block.

        Returns False if already parsed.
        """
        if state.parsed_nto:
            return False

        return bool(NTO_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        """
        Parse SA-NTO decomposition block.

        Consumes lines from iterator until end of block, populating state.tddft.nto_analyses.
        """
        logger.debug("Starting SA-NTO decomposition block parsing.")

        nto_states: list[NTOStateAnalysis] = []
        current_state: dict | None = None
        current_spin: str | None = None  # Track 'alpha' or 'beta' for UKS
        last_state_number = 0  # Track expected state sequence

        for line in iterator:
            # --- Check for end of block ---
            if any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                logger.debug(f"NTO parser ended on terminator: {line.strip()}")
                state.buffered_line = line
                break

            # --- Try RKS state header ---
            rks_match = RKS_STATE_PAT.search(line)
            if rks_match:
                # Save previous state
                if current_state is not None:
                    nto_states.append(self._build_nto_state(current_state))

                multiplicity = rks_match.group(1)
                state_number = int(rks_match.group(2))

                # Check for sequential state numbers - stop if we skip states
                if last_state_number > 0 and state_number != last_state_number + 1:
                    logger.debug(
                        f"Non-sequential state number {state_number} after {last_state_number}, ending NTO parsing"
                    )
                    state.buffered_line = line
                    break

                last_state_number = state_number
                current_state = {
                    "state_number": state_number,
                    "multiplicity": multiplicity,
                    "contributions": [],
                    "omega_percent": None,
                    "is_uks": False,
                }
                current_spin = None
                continue

            # --- Try UKS state header ---
            uks_match = UKS_STATE_PAT.search(line)
            if uks_match:
                # Save previous state
                if current_state is not None:
                    nto_states.append(self._build_nto_state(current_state))

                state_number = int(uks_match.group(1))

                # Check for sequential state numbers - stop if we skip states
                if last_state_number > 0 and state_number != last_state_number + 1:
                    logger.debug(
                        f"Non-sequential state number {state_number} after {last_state_number}, ending NTO parsing"
                    )
                    state.buffered_line = line
                    break

                last_state_number = state_number
                current_state = {
                    "state_number": state_number,
                    "multiplicity": "Unknown",
                    "contributions": [],
                    "omega_alpha_percent": None,
                    "omega_beta_percent": None,
                    "is_uks": True,
                }
                current_spin = None
                continue

            if current_state is None:
                continue

            # --- Try spin section header (UKS only) ---
            if current_state.get("is_uks"):
                spin_match = SPIN_SECTION_PAT.search(line)
                if spin_match:
                    current_spin = spin_match.group(1).lower()
                    continue

            # --- Parse NTO contribution ---
            contrib_match = CONTRIBUTION_PAT.search(line)
            if contrib_match:
                hole_offset = int(contrib_match.group(1))
                electron_offset = int(contrib_match.group(2))
                weight_percent = float(contrib_match.group(4))

                # Determine spin
                is_alpha = True  # Default for RKS
                if current_state.get("is_uks"):
                    is_alpha = current_spin == "alpha"

                contribution = NTOContribution(
                    hole_offset=hole_offset,
                    electron_offset=electron_offset,
                    weight_percent=weight_percent,
                    is_alpha_spin=is_alpha,
                )
                current_state["contributions"].append(contribution)
                continue

            # --- Parse omega line ---
            omega_match = OMEGA_PAT.search(line)
            if omega_match:
                omega_value = float(omega_match.group(1))

                if current_state.get("is_uks"):
                    # For UKS, store by spin
                    if current_spin == "alpha":
                        current_state["omega_alpha_percent"] = omega_value
                    elif current_spin == "beta":
                        current_state["omega_beta_percent"] = omega_value
                else:
                    # For RKS, single omega
                    current_state["omega_percent"] = omega_value
                continue

        # --- Save final state ---
        if current_state is not None:
            nto_states.append(self._build_nto_state(current_state))

        if not nto_states:
            state.parsing_warnings.append("SA-NTO decomposition block found but no states parsed.")
            state.parsed_nto = True
            return

        # --- Update ParseState ---
        # Integrate into existing TddftResults or create new one
        existing_tddft = state.tddft.to_dict() if state.tddft else {}
        existing_tddft["nto_analyses"] = nto_states
        state.tddft = TddftResults.from_dict(existing_tddft)
        state.parsed_nto = True

        logger.debug(f"Parsed {len(nto_states)} NTO states.")

    def _build_nto_state(self, state_data: dict) -> NTOStateAnalysis:
        """Build an NTOStateAnalysis model from parsed dictionary."""
        return NTOStateAnalysis(
            state_number=state_data["state_number"],
            contributions=state_data.get("contributions", []),
            omega_percent=state_data.get("omega_percent"),
            omega_alpha_percent=state_data.get("omega_alpha_percent"),
            omega_beta_percent=state_data.get("omega_beta_percent"),
        )
