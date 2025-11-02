# src/calcflow/io/qchem/blocks/tddft/trans_dm.py

"""
Parser for the "Transition Density Matrix Analysis" block in Q-Chem outputs.

This block appears in TDDFT calculations and provides detailed analysis for each
excited state, including Mulliken population analysis, CT numbers, and exciton analysis.

The structure is similar to unrelaxed density matrices but WITHOUT Natural Orbitals
analysis, and WITH additional CT numbers and transition-specific exciton metrics.

Different formats for Restricted (RKS) and Unrestricted (UKS) calculations.
"""

import logging
import re
from collections.abc import Iterator as LineIterator
from itertools import chain
from typing import Any, ClassVar

from calcflow.common.results import (
    AtomicCharges,
    ExcitonAnalysis,
    TddftResults,
    TransitionDensityMatrix,
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


class TransitionDensityMatrixParser(BlockParser):
    """
    Parses the entire "Transition Density Matrix Analysis" section.

    This parser orchestrates the parsing of multiple sub-blocks, one for each
    excited state listed. It handles both RKS and UKS output formats.
    """

    START_PAT: ClassVar[re.Pattern] = re.compile(r"Transition Density Matrix Analysis")
    STATE_HEADER_PAT: ClassVar[re.Pattern] = re.compile(r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:")
    END_PAT: ClassVar[re.Pattern] = re.compile(r"^\s*-{5,}\s*$")

    def matches(self, line: str, state: ParseState) -> bool:
        if state.tddft and state.tddft.transition_density_matrices is not None:
            return False
        return bool(self.START_PAT.search(line))

    def parse(self, iterator: LineIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing 'Transition Density Matrix Analysis' block.")
        all_analyses: list[TransitionDensityMatrix] = []

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
            # Construct the full multiplicity string as expected by tests
            full_multiplicity_str = f"{multiplicity} {state_num_str}"
            logger.debug(f"Parsing trans DM analysis for state {state_num}.")

            analysis, line_buffer = self._parse_single_state_block(iterator, state_num, full_multiplicity_str)
            if analysis:
                all_analyses.append(analysis)
            else:
                # If a block fails to parse, find the next one to avoid an infinite loop
                for line in iterator:
                    if self.STATE_HEADER_PAT.match(line) or "---" in line:
                        line_buffer = line
                        break
                else:
                    line_buffer = None

        if not all_analyses:
            msg = "Found 'Transition Density Matrix' block but parsed no states."
            state.parsing_warnings.append(msg)
            logger.warning(msg)
            return

        existing_tddft = state.tddft.to_dict() if state.tddft else {}
        existing_tddft["transition_density_matrices"] = all_analyses
        state.tddft = TddftResults.from_dict(existing_tddft)
        logger.debug("Finished parsing 'Transition Density Matrix Analysis'.")

    def _parse_single_state_block(
        self, iterator: LineIterator, state_number: int, multiplicity: str
    ) -> tuple[TransitionDensityMatrix | None, str | None]:
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
            if "Mulliken Population Analysis" in line:
                mulliken_data, line_buffer = self._parse_mulliken_section(iterator)
                data["mulliken"] = mulliken_data
                # After mulliken, look for QTa and QT2 values
                if line_buffer is None:
                    for qt_line in iterator:
                        if not qt_line.strip():
                            continue
                        if "Sum of absolute trans. charges, QTa" in qt_line:
                            val = self._parse_key_value_line(qt_line, "QTa")
                            if val is not None:
                                data["sum_abs_trans_charges"] = val
                        elif "Sum of squared  trans. charges, QT2" in qt_line:
                            val = self._parse_key_value_line(qt_line, "QT2")
                            if val is not None:
                                data["sum_squared_trans_charges"] = val
                        elif "CT numbers" in qt_line:
                            line_buffer = qt_line
                            break
                        elif not qt_line.strip():
                            continue
                        elif self.STATE_HEADER_PAT.match(qt_line):
                            line_buffer = qt_line
                            break
            elif "CT numbers" in line:
                ct_data, line_buffer = self._parse_ct_numbers_section(iterator)
                data.update(ct_data)
            elif "Exciton analysis of the transition density matrix" in line:
                exciton_data, line_buffer = self._parse_exciton_section(iterator)
                data.update(exciton_data)
            else:
                # Unrecognized line, likely end of all TDM blocks
                line_buffer = line
                break

            # If a sub-parser over-read, check what it found
            if line_buffer:
                # If it's the next state header, we're done with this state
                if self.STATE_HEADER_PAT.match(line_buffer):
                    break
                # Otherwise, re-process the buffered line through dispatch logic
                line = line_buffer
                line_buffer = None
                # Re-check against all dispatch conditions
                if "CT numbers" in line:
                    ct_data, line_buffer = self._parse_ct_numbers_section(iterator)
                    data.update(ct_data)
                    if line_buffer and self.STATE_HEADER_PAT.match(line_buffer):
                        break
                elif "Exciton analysis of the transition density matrix" in line:
                    exciton_data, line_buffer = self._parse_exciton_section(iterator)
                    data.update(exciton_data)
                    if line_buffer and self.STATE_HEADER_PAT.match(line_buffer):
                        break

        try:
            model = TransitionDensityMatrix.from_dict(data)
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

    def _parse_mulliken_section(self, iterator: LineIterator) -> tuple[AtomicCharges | None, str | None]:
        """
        Parse Mulliken population analysis section.

        RKS format: Atom, Trans. (e), h+, e-, Del q
        UKS format: Atom, Trans. (e), h+ (alpha), h+ (beta), e- (alpha), e- (beta)
        """
        header_line = next(iterator)
        next(iterator)  # separator

        is_uks = "alpha" in header_line.lower()
        data: dict[str, Any] = {
            "method": "Mulliken (Transition DM)",
            "charges": {},
            "trans_charges": {},
            "hole_populations": {} if not is_uks else None,
            "electron_populations": {} if not is_uks else None,
            "hole_populations_alpha": {} if is_uks else None,
            "hole_populations_beta": {} if is_uks else None,
            "electron_populations_alpha": {} if is_uks else None,
            "electron_populations_beta": {} if is_uks else None,
            "del_q": {},
        }

        line_buffer = None
        for line in iterator:
            if "---" in line or "Sum:" in line:
                continue
            if not line.strip():
                break

            parts = line.split()
            if not parts or not parts[0].isdigit():
                line_buffer = line
                break

            idx = int(parts[0]) - 1
            try:
                # Use float() to parse numeric columns
                # After the atom label (parts[0]) and element (parts[1]), we have the numeric data
                if is_uks:
                    # UKS: Atom Element Trans(e), h+ alpha, h+ beta, e- alpha, e- beta, [Del q]
                    # parts: [atom_num, element, trans_charge, h_alpha, h_beta, e_alpha, e_beta, ...]
                    data["trans_charges"][idx] = float(parts[2])
                    data["charges"][idx] = float(parts[2])  # Trans charge as "charges"
                    data["hole_populations_alpha"][idx] = float(parts[3])
                    data["hole_populations_beta"][idx] = float(parts[4])
                    data["electron_populations_alpha"][idx] = float(parts[5])
                    data["electron_populations_beta"][idx] = float(parts[6])
                    # Del q might not be present in UKS
                    if len(parts) > 7:
                        data["del_q"][idx] = float(parts[7])
                else:
                    # RKS: Atom Element Trans(e), h+, e-, Del q
                    # parts: [atom_num, element, trans_charge, h_pop, e_pop, del_q]
                    data["trans_charges"][idx] = float(parts[2])
                    data["charges"][idx] = float(parts[2])  # Trans charge as "charges"
                    data["hole_populations"][idx] = float(parts[3])
                    data["electron_populations"][idx] = float(parts[4])
                    data["del_q"][idx] = float(parts[5])
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse Mulliken line: {line.strip()} ({e})")

        return (AtomicCharges.from_dict(data) if data["trans_charges"] else None), line_buffer

    def _parse_ct_numbers_section(self, iterator: LineIterator) -> tuple[dict[str, Any], str | None]:
        """
        Parse CT numbers section.

        RKS has: omega, 2<alpha|beta>, LOC, LOCa, <Phe>
        UKS has: omega (total + alpha + beta), 2<alpha|beta>, LOC (total + alpha + beta),
                 LOCa (total + alpha + beta), <Phe> (total + alpha + beta)
        """
        data: dict[str, Any] = {}
        line_buffer = None

        for line in iterator:
            if not line.strip():
                break
            if self.STATE_HEADER_PAT.match(line):
                line_buffer = line
                break

            # Parse various CT metrics
            if "omega" in line and "=" in line:
                # RKS: omega        =     1.0011
                # UKS: omega        =     1.0016 (alpha:     0.5008, beta:     0.5008)
                match = re.search(r"omega\s*=\s*([\d.-]+)", line)
                if match:
                    data["omega"] = _to_float(match.group(1))

                # Check for alpha/beta values
                alpha_match = re.search(r"alpha:\s*([\d.-]+)", line)
                beta_match = re.search(r"beta:\s*([\d.-]+)", line)
                if alpha_match:
                    data["omega_alpha"] = _to_float(alpha_match.group(1))
                if beta_match:
                    data["omega_beta"] = _to_float(beta_match.group(1))

            elif "2<alpha|beta>" in line and "=" in line:
                match = re.search(r"2<alpha\|beta>\s*=\s*([-\d.]+)", line)
                if match:
                    data["two_alpha_beta"] = _to_float(match.group(1))

            elif "LOC" in line and "=" in line and "LOCa" not in line:
                # LOC          =     0.0010 (alpha:     0.0007, beta:     0.0007)
                match = re.search(r"LOC\s*=\s*([\d.-]+)", line)
                if match:
                    data["loc"] = _to_float(match.group(1))

                alpha_match = re.search(r"alpha:\s*([\d.-]+)", line)
                beta_match = re.search(r"beta:\s*([\d.-]+)", line)
                if alpha_match:
                    data["loc_alpha"] = _to_float(alpha_match.group(1))
                if beta_match:
                    data["loc_beta"] = _to_float(beta_match.group(1))

            elif "LOCa" in line and "=" in line:
                # LOCa         =     0.1492 (alpha:     0.0841, beta:     0.0841)
                match = re.search(r"LOCa\s*=\s*([\d.-]+)", line)
                if match:
                    data["loca"] = _to_float(match.group(1))

                alpha_match = re.search(r"alpha:\s*([\d.-]+)", line)
                beta_match = re.search(r"beta:\s*([\d.-]+)", line)
                if alpha_match:
                    data["loca_alpha"] = _to_float(alpha_match.group(1))
                if beta_match:
                    data["loca_beta"] = _to_float(beta_match.group(1))

            elif "<Phe>" in line and "=" in line:
                # <Phe>        =    -0.0230 (alpha:     0.0381, beta:     0.0381)
                match = re.search(r"<Phe>\s*=\s*([-\d.]+)", line)
                if match:
                    data["phe"] = _to_float(match.group(1))

                alpha_match = re.search(r"alpha:\s*([-\d.]+)", line)
                beta_match = re.search(r"beta:\s*([-\d.]+)", line)
                if alpha_match:
                    data["phe_alpha"] = _to_float(alpha_match.group(1))
                if beta_match:
                    data["phe_beta"] = _to_float(beta_match.group(1))

        return data, line_buffer

    def _parse_exciton_section(self, iterator: LineIterator) -> tuple[dict[str, Any], str | None]:
        """
        Orchestrates parsing of the exciton block, handling both RKS and UKS cases.
        Returns a dict with transition metrics + exciton analyses.
        """
        exciton_data: dict[str, Any] = {}
        line_buffer = None

        # Find the first meaningful line to determine if this is an RKS or UKS block
        for line in iterator:
            if line.strip():
                line_buffer = line
                break
        else:
            return {}, None  # Reached end of iterator

        # RKS case: the first line is data, not a "Total:" header
        if "Total:" not in line_buffer:
            trans_metrics, exciton, line_buffer = self._parse_exciton_sub_block(iterator, initial_line=line_buffer)
            exciton_data.update(trans_metrics)
            exciton_data["exciton_total"] = exciton
            return exciton_data, line_buffer

        # UKS case: the line we found was "Total:"
        # We've consumed the header, so we can now parse the sub-block
        trans_metrics, exciton, line_buffer = self._parse_exciton_sub_block(iterator)
        exciton_data.update(trans_metrics)
        exciton_data["exciton_total"] = exciton

        # After parsing a sub-block, if buffer is empty, search for next header
        if line_buffer is None:
            for line in iterator:
                if line.strip():
                    line_buffer = line
                    break

        if line_buffer and "Alpha spin:" in line_buffer:
            _, exciton_alpha, line_buffer = self._parse_exciton_sub_block(iterator)
            exciton_data["exciton_alpha"] = exciton_alpha

            # Repeat search logic for the Beta block
            if line_buffer is None:
                for line in iterator:
                    if line.strip():
                        line_buffer = line
                        break

        if line_buffer and "Beta spin:" in line_buffer:
            _, exciton_beta, line_buffer = self._parse_exciton_sub_block(iterator)
            exciton_data["exciton_beta"] = exciton_beta

        return exciton_data, line_buffer

    def _parse_exciton_sub_block(
        self, iterator: LineIterator, initial_line: str | None = None
    ) -> tuple[dict[str, Any], ExcitonAnalysis, str | None]:
        """
        Parses a single exciton analysis sub-block with all transition-specific fields.

        Returns (transition_metrics_dict, exciton_analysis, buffered_line)

        This handles the enhanced exciton fields that appear in transition density matrices:
        - Trans. dipole moment
        - Transition <r^2>
        - RMS electron-hole separation
        - Covariance
        - Correlation coefficient
        - Center-of-mass size
        """
        exciton_data: dict[str, Any] = {}
        trans_metrics: dict[str, Any] = {}
        expecting_hole_components = False
        expecting_electron_components = False
        expecting_rms_components = False
        expecting_com_components = False
        expecting_trans_components = False
        expecting_trans_r2_components = False

        line_source = chain([initial_line], iterator) if initial_line else iterator

        for line in line_source:
            stripped = line.strip()

            # Check for terminators FIRST
            if (
                any(kw in stripped for kw in ["Alpha spin:", "Beta spin:"])
                or any(kw in line for kw in ["Mulliken Population", "Multipole moment"])
                or self.STATE_HEADER_PAT.match(line)
            ):
                exciton_data.update(trans_metrics)
                return trans_metrics, ExcitonAnalysis.from_dict(exciton_data), line

            if not stripped:  # A blank line terminates the current sub-block
                break

            # State machine for component lines that appear on the *next* line
            if expecting_hole_components:
                if "Cartesian components" in line:
                    exciton_data["hole_size_components_ang"] = self._parse_vector_line(line)
                expecting_hole_components = False
                continue
            if expecting_electron_components:
                if "Cartesian components" in line:
                    exciton_data["electron_size_components_ang"] = self._parse_vector_line(line)
                expecting_electron_components = False
                continue
            if expecting_rms_components:
                if "Cartesian components" in line:
                    exciton_data["rms_separation_components_ang"] = self._parse_vector_line(line)
                expecting_rms_components = False
                continue
            if expecting_com_components:
                if "Cartesian components" in line:
                    exciton_data["center_of_mass_components_ang"] = self._parse_vector_line(line)
                expecting_com_components = False
                continue
            if expecting_trans_components:
                if "Cartesian components [D]:" in line:
                    trans_metrics["trans_dipole_moment_components_debye"] = self._parse_vector_line(line)
                expecting_trans_components = False
                continue
            if expecting_trans_r2_components:
                if "Cartesian components [a.u.]:" in line:
                    trans_metrics["trans_r2_components_au"] = self._parse_vector_line(line)
                expecting_trans_r2_components = False
                continue

            # Parse regular key-value/vector lines
            if "Trans. dipole moment" in line and "[D]:" in line:
                val = self._parse_key_value_line(line, "Trans. dipole moment")
                if val is not None:
                    trans_metrics["trans_dipole_moment_debye"] = val
                expecting_trans_components = True
            elif "Transition <r^2>" in line and "[a.u.]:" in line:
                val = self._parse_key_value_line(line, "Transition <r^2>")
                if val is not None:
                    trans_metrics["trans_r2_au"] = val
                expecting_trans_r2_components = True
            elif "<r_h> [Ang]:" in line:
                exciton_data["r_h_ang"] = self._parse_vector_line(line)
            elif "<r_e> [Ang]:" in line:
                exciton_data["r_e_ang"] = self._parse_vector_line(line)
            elif "|<r_e - r_h>| [Ang]:" in line:
                exciton_data["separation_ang"] = self._parse_key_value_line(line, "|<r_e - r_h>|")
            elif "Hole size [Ang]:" in line:
                exciton_data["hole_size_ang"] = self._parse_key_value_line(line, "Hole size")
                expecting_hole_components = True
            elif "Electron size [Ang]:" in line:
                exciton_data["electron_size_ang"] = self._parse_key_value_line(line, "Electron size")
                expecting_electron_components = True
            elif "RMS electron-hole separation [Ang]:" in line:
                exciton_data["rms_separation_ang"] = self._parse_key_value_line(line, "RMS electron-hole separation")
                expecting_rms_components = True
            elif "Covariance(r_h, r_e) [Ang^2]:" in line:
                exciton_data["covariance"] = self._parse_key_value_line(line, "Covariance")
            elif "Correlation coefficient:" in line:
                exciton_data["correlation_coef"] = self._parse_key_value_line(line, "Correlation coefficient")
            elif "Center-of-mass size [Ang]:" in line:
                exciton_data["center_of_mass_size_ang"] = self._parse_key_value_line(line, "Center-of-mass size")
                expecting_com_components = True

        exciton_data.update(trans_metrics)
        return trans_metrics, ExcitonAnalysis.from_dict(exciton_data), None
