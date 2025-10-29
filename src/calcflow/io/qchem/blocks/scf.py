import re
from collections.abc import Iterator

from calcflow.common.exceptions import InternalCodeError, ParsingError
from calcflow.common.models import ScfIteration, ScfResults, SmdResults
from calcflow.common.patterns import VersionSpec
from calcflow.io.qchem.blocks.patterns import QCHEM_PATTERNS
from calcflow.io.state import ParseState
from calcflow.utils import logger

# --- Non-versioned patterns specific to the SCF block's structure ---
SCF_START_PAT = re.compile(r"^\s*General SCF calculation program by")
SCF_ITER_PAT = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+([\d\.eE+-]+)")
SCF_CONVERGENCE_PAT = re.compile(r"Convergence criterion met")
SMD_SUMMARY_START_PAT = re.compile(r"^\s*Summary of SMD free energies:")
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
        scf_energy_from_iter_block: float | None = None

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

            # --- 3. Parse Iteration Data ---
            iter_match = SCF_ITER_PAT.search(line)
            if iter_match:
                try:
                    iteration = int(iter_match.group(1))
                    energy = float(iter_match.group(2))
                    diis_error = float(iter_match.group(3))
                    iterations.append(ScfIteration(iteration=iteration, energy=energy, diis_error=diis_error))
                    scf_energy_from_iter_block = energy
                    if SCF_CONVERGENCE_PAT.search(line):
                        converged = True
                except (ValueError, IndexError):
                    state.parsing_warnings.append(f"Could not parse SCF iteration line: {line.strip()}")
                continue

            # --- 4. Process Version-Dependent Patterns ---
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
        """
        if state.metadata.software_version is None:
            # This should not happen if MetadataParser runs first, but is a safeguard.
            raise InternalCodeError("Cannot process versioned patterns: QChem version not yet parsed.")

        qchem_version = VersionSpec.from_str(state.metadata.software_version)

        for p_def in QCHEM_PATTERNS:
            # Context filtering
            if p_def.block_type == "smd_summary" and not in_smd_block:
                continue

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
