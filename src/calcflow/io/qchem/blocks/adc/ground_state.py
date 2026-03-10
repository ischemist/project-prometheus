"""
Parser for the ADC ground state block in Q-Chem output files.

This block starts with the 'A D C  M A N' banner and contains:
- HF Summary (energy, dipole)
- MP(2) Summary (correlation energy, total energy)
- Density matrix analysis (NOs for alpha, beta, spin-traced)
- Mulliken Population Analysis (UKS format)
- Multipole moment analysis (dipole)
- Exciton analysis of the difference density matrix (Total + Alpha + Beta)

The block ends at "Starting Davidson for excited states" which is buffered.
"""

import logging
import re
from collections.abc import Iterator as LineIterator
from itertools import chain
from typing import Any, ClassVar

from calcflow.common.results import (
    AdcGroundState,
    AdcResults,
    AtomicCharges,
    ExcitonAnalysis,
    NaturalOrbitals,
)
from calcflow.io.core import BlockParser, ParseState

logger = logging.getLogger(__name__)


def _to_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_kv(line: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}.*?:\s+(-?[\d.]+)", line)
    return _to_float(m.group(1)) if m else None


def _parse_vec(line: str) -> tuple[float, float, float] | None:
    m = re.search(r"\[\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]", line)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


class AdcGroundStateParser(BlockParser):
    """Parses the ADC ground state block (HF + MP2 + density matrix analysis)."""

    BANNER_PAT: ClassVar[re.Pattern] = re.compile(r"A D C\s+M A N")
    HF_SECTION_PAT: ClassVar[re.Pattern] = re.compile(r"^\s*-+\s*$")
    DAVIDSON_PAT: ClassVar[re.Pattern] = re.compile(r"Starting Davidson for excited states")

    def matches(self, line: str, state: ParseState) -> bool:
        if state.parsed_adc_gs:
            return False
        return bool(self.BANNER_PAT.search(line))

    def parse(self, iterator: LineIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing ADC ground state block.")
        data: dict[str, Any] = {}

        line_source = chain([start_line], iterator)
        line_buffer: str | None = None

        while True:
            if line_buffer is not None:
                line = line_buffer
                line_buffer = None
            else:
                try:
                    line = next(line_source)
                except StopIteration:
                    break

            if self.DAVIDSON_PAT.search(line):
                state.buffered_line = line
                break

            if "Energy:" in line and "hf_energy_au" not in data and "a.u." in line:
                m = re.search(r"Energy:\s+([-\d.]+)\s+a\.u\.", line)
                if m:
                    data["hf_energy_au"] = float(m.group(1))
                continue

            if "MP energy contribution:" in line:
                m = re.search(r"MP energy contribution:\s+([-\d.]+)\s+a\.u\.", line)
                if m:
                    data["mp2_correlation_energy_au"] = float(m.group(1))
                continue

            if "Total energy:" in line and "mp2_correlation_energy_au" in data and "total_energy_au" not in data:
                m = re.search(r"Total energy:\s+([-\d.]+)\s+a\.u\.", line)
                if m:
                    data["total_energy_au"] = float(m.group(1))
                continue

            if "NOs" in line and "Density matrix analysis" not in line:
                nos_data, line_buffer = self._parse_nos_section(iterator, line)
                data.update(nos_data)
                continue

            if "Mulliken Population Analysis" in line:
                mulliken, line_buffer = self._parse_mulliken_section(iterator)
                data["mulliken"] = mulliken
                continue

            if "Multipole moment analysis" in line:
                mp_data, line_buffer = self._parse_multipole_section(iterator)
                data.update(mp_data)
                continue

            if "Exciton analysis of the difference density matrix" in line:
                exciton_data, line_buffer = self._parse_exciton_section(iterator)
                data.update(exciton_data)
                continue

        try:
            gs = AdcGroundState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to create AdcGroundState: {e}", exc_info=True)
            state.parsing_warnings.append(f"AdcGroundState creation failed: {e}")
            return

        state.adc = AdcResults(method="adc(2)", ground_state=gs)
        state.parsed_adc_gs = True
        logger.debug("Finished parsing ADC ground state block.")

    # ------------------------------------------------------------------
    # Sub-parsers (minimal, mirror unrel_dm.py approach)
    # ------------------------------------------------------------------

    def _parse_nos_section(
        self, iterator: LineIterator, start_line: str
    ) -> tuple[dict[str, NaturalOrbitals | None], str | None]:
        nos: dict[str, NaturalOrbitals | None] = {}
        line_buffer: str | None = start_line

        while line_buffer and "NOs" in line_buffer:
            if "(alpha)" in line_buffer:
                key = "nos_alpha"
            elif "(beta)" in line_buffer:
                key = "nos_beta"
            else:
                key = "nos_spin_traced"

            sub: dict[str, Any] = {}
            line_buffer = None
            for line in iterator:
                stripped = line.strip()
                if not stripped or "---" in line:
                    line_buffer = None
                    break
                if "NOs (" in line or "Mulliken" in line or "Multipole" in line or "Exciton" in line:
                    line_buffer = line
                    break
                if "Occupation of frontier NOs:" in line:
                    vals_line = next(iterator, "")
                    sub["frontier_occupations"] = [float(v) for v in vals_line.strip().split()]
                elif "Number of electrons:" in line:
                    sub["num_electrons"] = _parse_kv(line, "Number of electrons")
                elif "Number of unpaired electrons:" in line:
                    m = re.search(r"n_u\s*=\s*([\d.-]+),\s*n_u,nl\s*=\s*([\d.-]+)", line)
                    if m:
                        sub["num_unpaired"] = float(m.group(1))
                        sub["num_unpaired_nl"] = float(m.group(2))
                elif "NO participation ratio" in line:
                    sub["pr_no"] = _parse_kv(line, "PR_NO")

            if sub:
                nos[key] = NaturalOrbitals.from_dict(sub)

        return nos, line_buffer

    def _parse_mulliken_section(self, iterator: LineIterator) -> tuple[AtomicCharges | None, str | None]:
        header_line = next(iterator, "")
        next(iterator, None)  # separator

        is_uks = "Spin (e)" in header_line
        data: dict[str, Any] = {
            "method": "Mulliken (ADC GS)",
            "charges": {},
            "spins": {} if is_uks else None,
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
                data["charges"][idx] = float(parts[2])
                if is_uks:
                    data["spins"][idx] = float(parts[3])
                    data["hole_populations_alpha"][idx] = float(parts[4])
                    data["hole_populations_beta"][idx] = float(parts[5])
                    data["electron_populations_alpha"][idx] = float(parts[6])
                    data["electron_populations_beta"][idx] = float(parts[7])
            except (ValueError, IndexError):
                logger.warning(f"Could not parse Mulliken line: {line.strip()}")

        return (AtomicCharges.from_dict(data) if data["charges"] else None), line_buffer

    def _parse_multipole_section(self, iterator: LineIterator) -> tuple[dict[str, Any], str | None]:
        data: dict[str, Any] = {}
        line_buffer = None
        for line in iterator:
            if not line.strip():
                break
            if "Dipole moment [D]:" in line:
                data["dipole_moment_debye"] = _parse_kv(line, "Dipole moment")
            elif "Cartesian components [D]:" in line:
                data["dipole_components_debye"] = _parse_vec(line)
            elif "Exciton analysis" in line:
                line_buffer = line
                break
        return data, line_buffer

    def _parse_exciton_section(self, iterator: LineIterator) -> tuple[dict[str, ExcitonAnalysis | None], str | None]:
        exciton: dict[str, ExcitonAnalysis | None] = {}
        line_buffer: str | None = None

        # Find first meaningful line
        for line in iterator:
            if line.strip():
                line_buffer = line
                break
        else:
            return {}, None

        # RKS (first line is data, not "Total:")
        if "Total:" not in line_buffer:
            exciton["exciton_total"], line_buffer = self._parse_exciton_sub(iterator, initial_line=line_buffer)
            return exciton, line_buffer

        # UKS: Total: header consumed
        exciton["exciton_total"], line_buffer = self._parse_exciton_sub(iterator)

        if line_buffer is None:
            for line in iterator:
                if line.strip():
                    line_buffer = line
                    break

        if line_buffer and "Alpha spin:" in line_buffer:
            exciton["exciton_alpha"], line_buffer = self._parse_exciton_sub(iterator)
            if line_buffer is None:
                for line in iterator:
                    if line.strip():
                        line_buffer = line
                        break

        if line_buffer and "Beta spin:" in line_buffer:
            exciton["exciton_beta"], line_buffer = self._parse_exciton_sub(iterator)

        return exciton, line_buffer

    def _parse_exciton_sub(
        self, iterator: LineIterator, initial_line: str | None = None
    ) -> tuple[ExcitonAnalysis, str | None]:
        data: dict[str, Any] = {}
        expecting_hole_comp = False
        expecting_electron_comp = False

        src = chain([initial_line], iterator) if initial_line else iterator
        for line in src:
            stripped = line.strip()
            if any(kw in stripped for kw in ["Alpha spin:", "Beta spin:", "Starting Davidson"]):
                return ExcitonAnalysis.from_dict(data), line
            if any(kw in line for kw in ["Mulliken Population", "Multipole moment", "Transition density"]):
                return ExcitonAnalysis.from_dict(data), line
            if not stripped:
                break

            if expecting_hole_comp:
                if "Cartesian components" in line:
                    data["hole_size_components_ang"] = _parse_vec(line)
                expecting_hole_comp = False
                continue
            if expecting_electron_comp:
                if "Cartesian components" in line:
                    data["electron_size_components_ang"] = _parse_vec(line)
                expecting_electron_comp = False
                continue

            if "<r_h> [Ang]:" in line:
                data["r_h_ang"] = _parse_vec(line)
            elif "<r_e> [Ang]:" in line:
                data["r_e_ang"] = _parse_vec(line)
            elif "|<r_e - r_h>| [Ang]:" in line:
                data["separation_ang"] = _parse_kv(line, "|<r_e - r_h>|")
            elif "Hole size [Ang]:" in line:
                data["hole_size_ang"] = _parse_kv(line, "Hole size")
                expecting_hole_comp = True
            elif "Electron size [Ang]:" in line:
                data["electron_size_ang"] = _parse_kv(line, "Electron size")
                expecting_electron_comp = True

        return ExcitonAnalysis.from_dict(data), None
