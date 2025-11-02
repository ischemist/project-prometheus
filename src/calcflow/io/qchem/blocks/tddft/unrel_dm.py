# src/calcflow/io/qchem/blocks/unrel_dm.py

"""
Parser for the "Analysis of Unrelaxed Density Matrices" block in Q-Chem outputs.

This block appears in TDDFT calculations and provides detailed analysis for each
excited state, including Natural Transition Orbitals (NTOs), Mulliken population
analysis of the state/difference density, multipole moments, and exciton analysis.

The structure is complex as it repeats for each state and has different formats
for Restricted (RKS) and Unrestricted (UKS) calculations.
"""

import logging
import re
from collections.abc import Iterator as LineIterator
from itertools import chain
from typing import Any, ClassVar

from calcflow.common.results import (
    AtomicCharges,
    ExcitonAnalysis,
    NaturalOrbitals,
    TddftResults,
    UnrelaxedDensityMatrix,
)
from calcflow.io.core import BlockParser, ParseState

logger = logging.getLogger(__name__)


def _to_float(val: str | None) -> float | None:
    """Safely convert a string to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class UnrelaxedDensityMatrixParser(BlockParser):
    """
    Parses the entire "Analysis of Unrelaxed Density Matrices" section.

    This parser orchestrates the parsing of multiple sub-blocks, one for each
    excited state listed. It handles both RKS and UKS output formats.
    """

    START_PAT: ClassVar[re.Pattern] = re.compile(r"Analysis of Unrelaxed Density Matrices")
    STATE_HEADER_PAT: ClassVar[re.Pattern] = re.compile(r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:")
    END_PAT: ClassVar[re.Pattern] = re.compile(r"^\s*-{5,}\s*$")

    def matches(self, line: str, state: ParseState) -> bool:
        if state.tddft and state.tddft.unrelaxed_density_matrices is not None:
            return False
        return bool(self.START_PAT.search(line))

    def parse(self, iterator: LineIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing 'Analysis of Unrelaxed Density Matrices' block.")
        all_analyses: list[UnrelaxedDensityMatrix] = []

        line_buffer = None
        # Consume until we find the first actual state header
        for line in chain([start_line], iterator):
            if self.STATE_HEADER_PAT.match(line):
                line_buffer = line
                break

        while line_buffer is not None:
            state_match = self.STATE_HEADER_PAT.match(line_buffer)
            if not state_match:
                state.buffered_line = line_buffer
                break

            multiplicity, state_num_str = state_match.groups()
            state_num = int(state_num_str)
            # FIX: Construct the full multiplicity string as expected by tests.
            full_multiplicity_str = f"{multiplicity} {state_num_str}"
            logger.debug(f"Parsing DM analysis for state {state_num}.")

            analysis, line_buffer = self._parse_single_state_block(iterator, state_num, full_multiplicity_str)
            if analysis:
                all_analyses.append(analysis)
            else:
                # If a block fails to parse, we need to find the next one to avoid an infinite loop
                for line in iterator:
                    if self.STATE_HEADER_PAT.match(line) or "---" in line:
                        line_buffer = line
                        break
                else:
                    line_buffer = None

        if not all_analyses:
            msg = "Found 'Unrelaxed DM' block but parsed no states."
            state.parsing_warnings.append(msg)
            logger.warning(msg)
            return

        existing_tddft = state.tddft.to_dict() if state.tddft else {}
        existing_tddft["unrelaxed_density_matrices"] = all_analyses
        state.tddft = TddftResults.from_dict(existing_tddft)
        logger.debug("Finished parsing 'Analysis of Unrelaxed Density Matrices'.")

    def _parse_single_state_block(
        self, iterator: LineIterator, state_number: int, multiplicity: str
    ) -> tuple[UnrelaxedDensityMatrix | None, str | None]:
        data: dict[str, Any] = {"state_number": state_number, "multiplicity": multiplicity}
        line_buffer = None

        # Consume the separator line after the state header
        next(iterator, None)

        for line in iterator:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if self.STATE_HEADER_PAT.match(line):
                line_buffer = line
                break

            # Dispatch to sub-parsers
            if "NOs" in line:
                nos_data, line_buffer = self._parse_nos_section(iterator, line)
                data.update(nos_data)
            elif "Mulliken Population Analysis" in line:
                mulliken_data, line_buffer = self._parse_mulliken_section(iterator)
                data["mulliken"] = mulliken_data
            elif "Multipole moment analysis" in line:
                multipole_data, line_buffer = self._parse_multipole_section(iterator)
                data.update(multipole_data)
            elif "Exciton analysis" in line:
                exciton_data, line_buffer = self._parse_exciton_section(iterator)
                data.update(exciton_data)
            else:
                # Unrecognized line, likely end of all DM blocks
                line_buffer = line
                break

            if line_buffer:  # noqa: SIM102
                # A sub-parser over-read. We need to process the buffered line.
                # This is unlikely with the new design but is a safeguard.
                if self.STATE_HEADER_PAT.match(line_buffer):
                    break

        try:
            model = UnrelaxedDensityMatrix.from_dict(data)
            return model, line_buffer
        except Exception as e:
            logger.error(f"Model creation failed for state {state_number}: {e}", exc_info=True)
            return None, line_buffer

    def _parse_key_value_line(self, line: str, key: str) -> float | None:
        match = re.search(rf"{re.escape(key)}.*?:?\s+(-?[\d.]+)", line)
        return _to_float(match.group(1)) if match else None

    def _parse_vector_line(self, line: str) -> tuple[float, ...] | None:
        match = re.search(r"\[\s*([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\]", line)
        return tuple(map(float, match.groups())) if match else None

    def _parse_nos_section(
        self, iterator: LineIterator, start_line: str
    ) -> tuple[dict[str, NaturalOrbitals | None], str | None]:
        nos: dict[str, NaturalOrbitals | None] = {}
        line_buffer = start_line

        while line_buffer and "NOs" in line_buffer:
            current_spin_key = "nos_spin_traced"
            if "(alpha)" in line_buffer:
                current_spin_key = "nos_alpha"
            elif "(beta)" in line_buffer:
                current_spin_key = "nos_beta"

            sub_data: dict[str, Any] = {}
            line_buffer = None
            for line in iterator:
                if not line.strip() or "---" in line:
                    line_buffer = None
                    break
                if "NOs (" in line or "Mulliken" in line or "Multipole" in line:
                    line_buffer = line
                    break

                if "Occupation of frontier NOs:" in line:
                    vals = [float(v) for v in next(iterator).strip().split()]
                    sub_data["frontier_occupations"] = vals
                elif "Number of electrons:" in line:
                    sub_data["num_electrons"] = self._parse_key_value_line(line, "Number of electrons")
                elif "Number of unpaired electrons:" in line:
                    match = re.search(r"n_u\s*=\s*([\d.-]+),\s*n_u,nl\s*=\s*([\d.-]+)", line)
                    if match:
                        sub_data["num_unpaired"] = _to_float(match.group(1))
                        sub_data["num_unpaired_nl"] = _to_float(match.group(2))
                elif "NO participation ratio" in line:
                    sub_data["pr_no"] = self._parse_key_value_line(line, "PR_NO")

            if sub_data:
                nos[current_spin_key] = NaturalOrbitals.from_dict(sub_data)
        return nos, line_buffer

    def _parse_mulliken_section(self, iterator: LineIterator) -> tuple[AtomicCharges | None, str | None]:
        header_line = next(iterator)
        next(iterator)  # separator

        is_uks = "Spin (e)" in header_line
        data: dict[str, Any] = {
            "method": "Mulliken (Unrelaxed DM)",
            "charges": {},
            "spins": {} if is_uks else None,
            "hole_populations": {} if not is_uks else None,
            "electron_populations": {} if not is_uks else None,
            "hole_populations_alpha": {} if is_uks else None,
            "hole_populations_beta": {} if is_uks else None,
            "electron_populations_alpha": {} if is_uks else None,
            "electron_populations_beta": {} if is_uks else None,
        }

        line_buffer = None
        for line in iterator:
            if "---" in line or "Sum:" in line:
                continue
            if not line.strip():
                break

            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                line_buffer = line
                break

            idx = int(parts[0]) - 1
            try:
                if is_uks:
                    data["charges"][idx] = float(parts[2])
                    data["spins"][idx] = float(parts[3])
                    data["hole_populations_alpha"][idx] = float(parts[4])
                    data["hole_populations_beta"][idx] = float(parts[5])
                    data["electron_populations_alpha"][idx] = float(parts[6])
                    data["electron_populations_beta"][idx] = float(parts[7])
                else:
                    data["charges"][idx] = float(parts[2])
                    data["hole_populations"][idx] = float(parts[3])
                    data["electron_populations"][idx] = float(parts[4])
            except (ValueError, IndexError):
                logger.warning(f"Could not parse Mulliken line: {line.strip()}")

        return (AtomicCharges.from_dict(data) if data["charges"] else None), line_buffer

    def _parse_multipole_section(self, iterator: LineIterator) -> tuple[dict[str, Any], str | None]:
        data: dict[str, Any] = {}
        line_buffer = None
        for line in iterator:
            if not line.strip():
                break
            if "Molecular charge:" in line:
                data["molecular_charge"] = self._parse_key_value_line(line, "Molecular charge")
            elif "Number of electrons:" in line:
                data["num_electrons"] = self._parse_key_value_line(line, "Number of electrons")
            elif "Dipole moment [D]:" in line:
                data["dipole_moment_debye"] = self._parse_key_value_line(line, "Dipole moment")
            elif "Cartesian components [D]:" in line:
                data["dipole_components_debye"] = self._parse_vector_line(line)
            elif "Exciton analysis" in line:
                line_buffer = line
                break
        return data, line_buffer

    def _parse_exciton_section(self, iterator: LineIterator) -> tuple[dict[str, ExcitonAnalysis | None], str | None]:
        """
        Orchestrates parsing of the potentially multi-part exciton block, now
        with robust handling of transitions between Total, Alpha, and Beta sections.
        """
        exciton: dict[str, ExcitonAnalysis | None] = {}
        line_buffer = None

        # Find the first meaningful line to determine if this is an RKS or UKS block.
        for line in iterator:
            if line.strip():
                line_buffer = line
                break
        else:
            return {}, None  # Reached end of iterator

        # RKS case: the first line is data, not a "Total:" header.
        if "Total:" not in line_buffer:
            exciton["exciton_total"], line_buffer = self._parse_exciton_sub_block(iterator, initial_line=line_buffer)
            return exciton, line_buffer

        # UKS case: the line we found was "Total:".
        # We've consumed the header, so we can now parse the sub-block.
        exciton["exciton_total"], line_buffer = self._parse_exciton_sub_block(iterator)

        # CRITICAL FIX: After parsing a sub-block, if the buffer is empty (due to
        # ending on a blank line), we must actively search for the next header.
        if line_buffer is None:
            for line in iterator:
                if line.strip():
                    line_buffer = line
                    break

        if line_buffer and "Alpha spin:" in line_buffer:
            exciton["exciton_alpha"], line_buffer = self._parse_exciton_sub_block(iterator)

            # Repeat the search logic for the Beta block.
            if line_buffer is None:
                for line in iterator:
                    if line.strip():
                        line_buffer = line
                        break

        if line_buffer and "Beta spin:" in line_buffer:
            exciton["exciton_beta"], line_buffer = self._parse_exciton_sub_block(iterator)

        # Return the final state of the buffer so the main loop can continue correctly.
        return exciton, line_buffer

    def _parse_exciton_sub_block(
        self, iterator: LineIterator, initial_line: str | None = None
    ) -> tuple[ExcitonAnalysis, str | None]:
        """
        FIXED: Reworked to be a robust state machine that correctly identifies
        terminating lines and returns them, preventing the parent loop from breaking.
        """
        data: dict[str, Any] = {}
        expecting_hole_components = False
        expecting_electron_components = False

        line_source = chain([initial_line], iterator) if initial_line else iterator

        for line in line_source:
            stripped = line.strip()

            # Check for terminators FIRST. This is the crucial change.
            if (
                any(kw in stripped for kw in ["Alpha spin:", "Beta spin:"])
                or any(kw in line for kw in ["Mulliken Population", "Multipole moment"])
                or self.STATE_HEADER_PAT.match(line)
            ):
                return ExcitonAnalysis.from_dict(data), line

            if not stripped:  # A blank line terminates the current sub-block.
                break

            # State machine for component lines that appear on the *next* line.
            if expecting_hole_components:
                if "Cartesian components" in line:
                    data["hole_size_components_ang"] = self._parse_vector_line(line)
                expecting_hole_components = False
                continue  # This was a component line, skip other parsing for it.
            if expecting_electron_components:
                if "Cartesian components" in line:
                    data["electron_size_components_ang"] = self._parse_vector_line(line)
                expecting_electron_components = False
                continue

            # Parse regular key-value/vector lines
            if "<r_h> [Ang]:" in line:
                data["r_h_ang"] = self._parse_vector_line(line)
            elif "<r_e> [Ang]:" in line:
                data["r_e_ang"] = self._parse_vector_line(line)
            elif "|<r_e - r_h>| [Ang]:" in line:
                data["separation_ang"] = self._parse_key_value_line(line, "|<r_e - r_h>|")
            elif "Hole size [Ang]:" in line:
                data["hole_size_ang"] = self._parse_key_value_line(line, "Hole size")
                expecting_hole_components = True
            elif "Electron size [Ang]:" in line:
                data["electron_size_ang"] = self._parse_key_value_line(line, "Electron size")
                expecting_electron_components = True

        return ExcitonAnalysis.from_dict(data), None
