import re
from collections.abc import Iterator

from calcflow.common.exceptions import InternalCodeError, ParsingError
from calcflow.common.patterns import VersionSpec
from calcflow.common.results import ScfIteration, ScfResults, SmdResults
from calcflow.io.qchem.blocks.patterns import QCHEM_PATTERNS
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Non-versioned patterns specific to the SCF block's structure ---
SCF_START_PAT = re.compile(r"^\s*General SCF calculation program by")
SCF_ITER_HEADER_PAT = re.compile(r"^\s*Cycle\s+Energy\s+DIIS error")
# Pattern for iteration lines: must have 3+ whitespace-separated fields, where first is small int
# Energy field is a float (with optional sign), and last field is scientific notation
# More restrictive: iteration is 1-3 digits, energy has decimal point, diis_error has 'e' or 'E'
SCF_ITER_PAT = re.compile(r"^\s*(\d{1,3})\s+(-?\d+\.\d+)\s+([\d\.eE+-]+)")
SCF_CONVERGENCE_PAT = re.compile(r"Convergence criterion met")
SMD_SUMMARY_START_PAT = re.compile(r"^\s*Summary of SMD free energies:")
# MOM (Maximum Overlap Method) patterns
MOM_ACTIVE_PAT = re.compile(r"^\s*Maximum Overlap Method Active")
MOM_METHOD_PAT = re.compile(r"^\s*(?:IMOM|MOM) method")
MOM_OVERLAP_PAT = re.compile(r"^\s*MOM overlap:\s+([\d.]+)\s+/\s+([\d.]+)")
# Heuristic end-of-block markers
END_OF_BLOCK_PATS = [
    re.compile(r"^\s*Orbital Energies \(a\.u\.\)"),
    re.compile(r"^\s*Mulliken Net Atomic Charges"),
    re.compile(r"^\s*TDDFT/TDA\s+Excitation\s+Energies"),
]


class ScfParser:
    """Parses the main SCF block, including iterations, final energies, and SMD summaries."""

    def matches(self, line: str, state: ParseState) -> bool:
        return not state.parsed_scf and bool(SCF_START_PAT.search(line))

    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None:
        logger.debug("Starting SCF block parsing.")

        iterations: list[ScfIteration] = []
        converged = False
        in_smd_summary = False
        in_iter_block = False  # Track whether we've seen the iteration header
        scf_energy_from_iter_block: float | None = None

        # MOM (Maximum Overlap Method) buffering
        # MOM status lines appear before iteration lines, so we buffer them
        mom_active: bool | None = None
        mom_overlap_current: float | None = None
        mom_overlap_target: float | None = None

        # Temp storage for version-parsed fields before model creation
        smd_data: dict[str, float] = {}
        final_energy_data: dict[str, float] = {}

        for line in iterator:
            # --- 1. Check for End-of-Block Conditions ---
            if any(pat.search(line) for pat in END_OF_BLOCK_PATS):
                logger.debug(f"SCF parser ended on terminator line: {line.strip()}")
                state.buffered_line = line
                break

            # --- 2. Handle Block State ---
            if SMD_SUMMARY_START_PAT.search(line):
                in_smd_summary = True
                continue

            # --- 3. Detect SCF iteration block header ---
            if SCF_ITER_HEADER_PAT.search(line):
                in_iter_block = True
                logger.debug("Detected SCF iteration block header")
                continue

            # --- 4. Parse Iteration Data (only after seeing header) ---
            if in_iter_block:
                # --- 4a. Check for MOM status lines (appear before iteration lines) ---
                if MOM_ACTIVE_PAT.search(line):
                    mom_active = True
                    continue

                if MOM_METHOD_PAT.search(line):
                    # IMOM or MOM method line, just note it's active
                    continue

                mom_overlap_match = MOM_OVERLAP_PAT.search(line)
                if mom_overlap_match:
                    try:
                        # MOM overlap appears twice (for alpha/beta), take the first value
                        if mom_overlap_current is None:
                            mom_overlap_current = float(mom_overlap_match.group(1))
                            mom_overlap_target = float(mom_overlap_match.group(2))
                    except (ValueError, IndexError):
                        state.parsing_warnings.append(f"Could not parse MOM overlap line: {line.strip()}")
                    continue

                # --- 4b. Parse iteration line with buffered MOM data ---
                iter_match = SCF_ITER_PAT.search(line)
                if iter_match:
                    try:
                        iteration = int(iter_match.group(1))
                        energy = float(iter_match.group(2))
                        diis_error = float(iter_match.group(3))

                        # Create iteration with MOM data if available
                        iterations.append(
                            ScfIteration(
                                iteration=iteration,
                                energy=energy,
                                diis_error=diis_error,
                                mom_active=mom_active,
                                mom_overlap_current=mom_overlap_current,
                                mom_overlap_target=mom_overlap_target,
                            )
                        )
                        scf_energy_from_iter_block = energy

                        # Clear MOM buffer for next iteration
                        mom_active = None
                        mom_overlap_current = None
                        mom_overlap_target = None

                        if SCF_CONVERGENCE_PAT.search(line):
                            converged = True
                    except (ValueError, IndexError):
                        state.parsing_warnings.append(f"Could not parse SCF iteration line: {line.strip()}")
                    continue
                elif line.strip().startswith("-------"):
                    # Separator line between header and iterations
                    continue
                elif not line.strip():
                    # Empty line
                    continue
                else:
                    # End of iteration block (only if not a known pattern)
                    # Check if this is actually the end or just an unrecognized line
                    if not (MOM_ACTIVE_PAT.search(line) or MOM_METHOD_PAT.search(line) or MOM_OVERLAP_PAT.search(line)):
                        in_iter_block = False

            # --- 5. Process Version-Dependent Patterns ---
            self._process_versioned_patterns(line, state, in_smd_summary, smd_data, final_energy_data)

        # --- 5. Finalize and Store Results ---
        if not iterations:
            raise ParsingError("SCF block found, but no SCF iterations were parsed.")

        # Determine the final SCF energy
        final_scf_energy = final_energy_data.get("scf_energy", scf_energy_from_iter_block)
        if final_scf_energy is None:
            raise ParsingError("Could not determine final SCF energy.")

        state.scf = ScfResults(
            converged=converged,
            energy=final_scf_energy,
            n_iterations=len(iterations),
            iterations=tuple(iterations),
        )

        if smd_data:
            state.smd = SmdResults(**smd_data)

        # Set the top-level final_energy for the whole calculation
        # Prioritize the total SMD energy if available, otherwise use the explicitly parsed
        # final energy, falling back to the SCF energy.
        state.final_energy = smd_data.get("g_tot_au", final_energy_data.get("final_energy", final_scf_energy))

        state.parsed_scf = True
        logger.info(f"Parsed SCF data. Converged: {converged}, Energy: {final_scf_energy:.8f}")

    def _process_versioned_patterns(
        self,
        line: str,
        state: ParseState,
        in_smd_block: bool,
        smd_storage: dict[str, float],
        energy_storage: dict[str, float],
    ) -> None:
        """
        Helper to iterate through versioned patterns and populate storage dicts.

        For SMD patterns:
        - Q-Chem 6.2+: Only match within "Summary of SMD free energies:" block
        - Q-Chem 5.4: Parse SMD lines anywhere (no dedicated summary block header)
        """
        if state.metadata.software_version is None:
            # This should not happen if MetadataParser runs first, but is a safeguard.
            raise InternalCodeError("Cannot process versioned patterns: QChem version not yet parsed.")

        qchem_version = VersionSpec.from_str(state.metadata.software_version)

        for p_def in QCHEM_PATTERNS:
            # Context filtering for SMD patterns
            # Q-Chem 6.2+ requires "Summary of SMD" header, but 5.4 has inline SMD lines
            if p_def.block_type == "smd_summary":  # noqa: SIM102
                # For 6.2+, only parse SMD within summary block
                if qchem_version >= VersionSpec.from_str("6.0.0") and not in_smd_block:
                    continue
                # For 5.4, parse SMD patterns anywhere (no header in 5.4 format)

            # Get version-appropriate pattern and match
            versioned_pattern = p_def.get_matching_pattern(qchem_version)
            if not versioned_pattern:
                continue

            match = versioned_pattern.pattern.search(line)
            if not match:
                continue

            # Process match and store result
            value = versioned_pattern.transform(match)
            if p_def.block_type == "smd_summary":
                smd_storage[p_def.field_name] = value
            else:
                energy_storage[p_def.field_name] = value

            logger.debug(f"Matched pattern '{p_def.description}': {value}")
            break  # Assume only one pattern matches per line
