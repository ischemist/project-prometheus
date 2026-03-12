"""
Parser for the "Excited State Summary" block in Q-Chem ADC(2) output files.

Each excited state contains (in order):
  - State header / term symbol / energies
  - Osc. strength, trans dip moment, <i|r^2|0>
  - TPA cross-section + 3x3 TPA matrix
  - Dip. moment, V1^2/V2^2
  - Important amplitudes table
  - Density matrix analysis (NOs alpha/beta/spin-traced)
  - Mulliken Population Analysis (UKS format)
  - Multipole moment analysis (dipole)
  - Exciton analysis of the difference density matrix (Total + Alpha + Beta)
  - Transition density matrix analysis (CT numbers)
  - Exciton analysis of the transition density matrix (Total + Alpha + Beta)
  - Decomposition into state-averaged NTOs (Alpha + Beta)
  - State separator: "----" line

The block is terminated by "Time of ADC calculation" which is buffered.
"""

import logging
import re
from itertools import chain
from typing import Any, ClassVar

from calcflow.common.exceptions import ParsingError
from calcflow.common.patterns import extract_index
from calcflow.common.results import (
    AdcAmplitude,
    AdcExcitedState,
    AtomicCharges,
    ExcitonAnalysis,
    NaturalOrbitals,
    NTOContribution,
    TwoPhotonAbsorption,
)
from calcflow.common.types import SpinChannel
from calcflow.io.core import BlockParser, ParseState
from calcflow.io.peekable import PeekableIterator
from calcflow.io.qchem.blocks.parse_helpers import parse_key_value as _parse_kv
from calcflow.io.qchem.blocks.parse_helpers import parse_vector as _parse_vec

logger = logging.getLogger(__name__)

_TPA_ROW_PAT = re.compile(r"\|\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\|")
_STATE_HEADER_PAT = re.compile(r"^\s*Excited state\s+(\d+)\s*\(A\)")
_END_PAT = re.compile(r"Time of ADC calculation")
_AMP_ORB_PAT = re.compile(r"(\d+)\s*\(A\)\s*(A|B)")
_NTO_LINE_PAT = re.compile(r"H-\s*(\d+)\s*->\s*L\+\s*(\d+):\s*([-\d.]+)\s*\(\s*([\d.]+)%\)")


class AdcExcitedStatesParser(BlockParser):
    """Parses the full 'Excited State Summary' block from Q-Chem ADC output."""

    START_PAT: ClassVar[re.Pattern] = re.compile(r"Excited State Summary")

    def matches(self, line: str, state: ParseState) -> bool:
        if state.parsed_adc_excited:
            return False
        return bool(self.START_PAT.search(line))

    def parse(self, iterator: PeekableIterator, start_line: str, state: ParseState) -> None:
        logger.debug("Parsing ADC Excited State Summary block.")
        excited_states: list[AdcExcitedState] = []

        line_buffer: str | None = None
        # Advance to the first state header
        for line in chain([start_line], iterator):
            if _END_PAT.search(line):
                iterator.push_back(line)
                line_buffer = None
                break
            if _STATE_HEADER_PAT.match(line):
                line_buffer = line
                break

        while line_buffer is not None:
            m = _STATE_HEADER_PAT.match(line_buffer)
            if not m:
                # Not a state header; could be END line or unexpected content
                iterator.push_back(line_buffer)
                break

            state_num = int(m.group(1))
            es, line_buffer = self._parse_single_state(iterator, state_num)
            if es is not None:
                excited_states.append(es)
            # If line_buffer is the END line, push it back and stop
            if line_buffer is not None and _END_PAT.search(line_buffer):
                iterator.push_back(line_buffer)
                line_buffer = None
                break

        if not excited_states:
            logger.warning("AdcExcitedStatesParser: no states parsed.")

        state.adc_excited_states = excited_states
        state.parsed_adc_excited = True
        logger.debug(f"Finished parsing {len(excited_states)} ADC excited states.")

    # ------------------------------------------------------------------
    # Per-state parser
    # ------------------------------------------------------------------

    def _parse_single_state(
        self, iterator: PeekableIterator, state_number: int
    ) -> tuple[AdcExcitedState | None, str | None]:
        data: dict[str, Any] = {"state_number": state_number, "amplitudes": []}
        line_buffer: str | None = None
        tpa_rows: list[tuple[float, float, float]] = []
        tpa_cross: float | None = None
        in_amplitudes = False
        nto_alpha: list[NTOContribution] = []
        nto_beta: list[NTOContribution] = []
        current_nto_spin: SpinChannel | None = None
        in_nto_section = False

        while True:
            if line_buffer is not None:
                line = line_buffer
                line_buffer = None
            else:
                try:
                    line = next(iterator)
                except StopIteration:
                    break

            # Termination: END pattern or next state header — return line as buffer
            if _END_PAT.search(line):
                line_buffer = line
                break
            if _STATE_HEADER_PAT.match(line):
                line_buffer = line
                break

            stripped = line.strip()

            # --- Core properties ---
            if "Total energy:" in line and "a.u." in line:
                m = re.search(r"Total energy:\s+([-\d.]+)\s+a\.u\.", line)
                if m:
                    data["total_energy_au"] = float(m.group(1))

            elif "Excitation energy:" in line and "eV" in line:
                m = re.search(r"Excitation energy:\s+([\d.]+)\s+eV", line)
                if m:
                    data["excitation_energy_ev"] = float(m.group(1))

            elif "Osc. strength:" in line:
                m = re.search(r"Osc\. strength:\s+([\d.]+)", line)
                if m:
                    data["oscillator_strength"] = float(m.group(1))

            elif "Trans. dip. moment [a.u.]:" in line:
                data["trans_dip_moment_au"] = _parse_vec(line)

            elif "<i|r^2|0> [a.u.]:" in line:
                data["r2_au"] = _parse_vec(line)

            elif "Two-photon absorption cross-section" in line:
                m = re.search(r"cross-section\s*\[a\.u\.\]:\s+([\d.]+)", line)
                if m:
                    tpa_cross = float(m.group(1))

            elif "Two-photon absorption matrix" in line:
                # Parse next 3 lines for TPA matrix rows
                for _ in range(3):
                    row_line = next(iterator, "")
                    rm = _TPA_ROW_PAT.search(row_line)
                    if rm:
                        tpa_rows.append((float(rm.group(1)), float(rm.group(2)), float(rm.group(3))))

            elif "Dip. moment [a.u.]:" in line:
                data["dip_moment_au"] = _parse_vec(line)

            elif "Total dipole [Debye]:" in line:
                data["total_dipole_debye"] = _parse_kv(line, "Total dipole")

            elif "V1^2" in line:
                m1 = re.search(r"V1\^2\s*=\s*([\d.]+)", line)
                m2 = re.search(r"V2\^2\s*=\s*([\d.]+)", line)
                if m1:
                    data["v1_squared"] = float(m1.group(1))
                if m2:
                    data["v2_squared"] = float(m2.group(1))

            # --- Amplitude table ---
            elif "Important amplitudes:" in line:
                in_amplitudes = True
                iterator.skip(2)  # header line (occ i  occ j  ...) and separator ---

            elif in_amplitudes:
                if "---" in stripped:
                    in_amplitudes = False
                    continue
                # Try to parse amplitude line
                amp = self._parse_amplitude_line(line)
                if amp is not None:
                    data["amplitudes"].append(amp)

            # --- Density matrix analysis ---
            elif "NOs" in line and "Density matrix analysis" not in line and "Decomposition" not in line:
                nos_data, line_buffer = self._parse_nos_section(iterator, line)
                data.update(nos_data)

            elif "Mulliken Population Analysis" in line:
                mulliken, line_buffer = self._parse_mulliken_section(iterator)
                data["mulliken"] = mulliken

            elif "Multipole moment analysis" in line:
                mp_data, line_buffer = self._parse_multipole_section(iterator)
                data.update(mp_data)

            elif "Exciton analysis of the difference density matrix" in line:
                exciton_data, line_buffer = self._parse_exciton_section(iterator, prefix="exciton_diff")
                data.update(exciton_data)

            elif "Transition density matrix analysis:" in line:
                ct_data, line_buffer = self._parse_ct_section(iterator)
                data.update(ct_data)

            elif "Exciton analysis of the transition density matrix" in line:
                exciton_trans, line_buffer = self._parse_exciton_section(iterator, prefix="exciton_trans")
                data.update(exciton_trans)

            # --- NTO decomposition ---
            elif "Decomposition into state-averaged NTOs" in line:
                current_nto_spin = None
                in_nto_section = True

            elif in_nto_section and "Alpha spin:" in line and current_nto_spin is None:
                current_nto_spin = "alpha"

            elif in_nto_section and "Beta spin:" in line and current_nto_spin in (None, "alpha"):
                current_nto_spin = "beta"

            elif in_nto_section and current_nto_spin is not None:
                nm = _NTO_LINE_PAT.search(line)
                if nm:
                    hole_offset = -int(nm.group(1))
                    electron_offset = int(nm.group(2))
                    weight = float(nm.group(4))
                    is_alpha = current_nto_spin == "alpha"
                    contrib = NTOContribution(
                        hole_offset=hole_offset,
                        electron_offset=electron_offset,
                        weight_percent=weight,
                        is_alpha_spin=is_alpha,
                    )
                    if is_alpha:
                        nto_alpha.append(contrib)
                    else:
                        nto_beta.append(contrib)

        # Build TPA object if we got data
        if tpa_cross is not None and len(tpa_rows) == 3:
            data["two_photon_absorption"] = TwoPhotonAbsorption(
                cross_section_au=tpa_cross,
                matrix_au=(tpa_rows[0], tpa_rows[1], tpa_rows[2]),
            )
        elif tpa_cross is not None:
            data["two_photon_absorption"] = TwoPhotonAbsorption(
                cross_section_au=tpa_cross,
                matrix_au=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            )

        if nto_alpha:
            data["nto_alpha"] = nto_alpha
        if nto_beta:
            data["nto_beta"] = nto_beta

        try:
            return AdcExcitedState.from_dict(data), line_buffer
        except (TypeError, ValueError, KeyError) as exc:
            raise ParsingError(f"Invalid ADC excited state {state_number}") from exc

    # ------------------------------------------------------------------
    # Amplitude parser
    # ------------------------------------------------------------------

    def _parse_amplitude_line(self, line: str) -> AdcAmplitude | None:
        """Parse a line like: ' 5 (A) B              6 (A) B                 0.6841'"""
        orbs = _AMP_ORB_PAT.findall(line)
        # last float is amplitude
        nums = re.findall(r"[-\d.]+", line)
        if not orbs or not nums:
            return None
        try:
            amplitude = float(nums[-1])
        except (ValueError, IndexError):
            return None

        if len(orbs) == 2:
            # 1h1p: occ_i -> vir_a
            occ_i = int(orbs[0][0])
            spin_i = orbs[0][1]  # A or B
            vir_a = int(orbs[1][0])
            return AdcAmplitude(occ_i=occ_i, vir_a=vir_a, amplitude=amplitude, spin=spin_i)
        if len(orbs) == 4:
            # 2h2p: occ_i, occ_j -> vir_a, vir_b
            occ_i = int(orbs[0][0])
            spin_i = orbs[0][1]
            occ_j = int(orbs[1][0])
            vir_a = int(orbs[2][0])
            vir_b = int(orbs[3][0])
            return AdcAmplitude(occ_i=occ_i, vir_a=vir_a, amplitude=amplitude, spin=spin_i, occ_j=occ_j, vir_b=vir_b)
        return None

    # ------------------------------------------------------------------
    # Sub-parsers (adapted from ground_state.py helpers)
    # ------------------------------------------------------------------

    def _parse_nos_section(
        self, iterator: PeekableIterator, start_line: str
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

    def _parse_mulliken_section(self, iterator: PeekableIterator) -> tuple[AtomicCharges | None, str | None]:
        header_line = next(iterator, "")
        iterator.skip()  # separator

        is_uks = "Spin (e)" in header_line
        data: dict[str, Any] = {
            "method": "Mulliken (ADC state)",
            "charges": {},
            "spins": {} if is_uks else None,
            "hole_populations_alpha": {} if is_uks else None,
            "hole_populations_beta": {} if is_uks else None,
            "electron_populations_alpha": {} if is_uks else None,
            "electron_populations_beta": {} if is_uks else None,
        }

        charges: dict[int, float] = data["charges"]
        spins: dict[int, float] | None = data["spins"]
        hole_populations_alpha: dict[int, float] | None = data["hole_populations_alpha"]
        hole_populations_beta: dict[int, float] | None = data["hole_populations_beta"]
        electron_populations_alpha: dict[int, float] | None = data["electron_populations_alpha"]
        electron_populations_beta: dict[int, float] | None = data["electron_populations_beta"]

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
            try:
                idx = extract_index(parts[0])
                charges[idx] = float(parts[2])
                if is_uks:
                    assert spins is not None and hole_populations_alpha is not None
                    assert hole_populations_beta is not None and electron_populations_alpha is not None
                    assert electron_populations_beta is not None
                    spins[idx] = float(parts[3])
                    hole_populations_alpha[idx] = float(parts[4])
                    hole_populations_beta[idx] = float(parts[5])
                    electron_populations_alpha[idx] = float(parts[6])
                    electron_populations_beta[idx] = float(parts[7])
            except (ValueError, IndexError):
                logger.warning(f"Could not parse Mulliken line: {line.strip()}")

        return (AtomicCharges.from_dict(data) if data["charges"] else None), line_buffer

    def _parse_multipole_section(self, iterator: PeekableIterator) -> tuple[dict[str, Any], str | None]:
        data: dict[str, Any] = {}
        line_buffer = None
        for line in iterator:
            if not line.strip():
                break
            if "Dipole moment [D]:" in line:
                data["dipole_dm_debye"] = _parse_kv(line, "Dipole moment")
            elif "Cartesian components [D]:" in line:
                data["dipole_dm_components_debye"] = _parse_vec(line)
            elif "Exciton analysis" in line or "Transition density" in line:
                line_buffer = line
                break
        return data, line_buffer

    def _parse_ct_section(self, iterator: PeekableIterator) -> tuple[dict[str, Any], str | None]:
        """Parse CT numbers: omega and Phe with alpha/beta decomposition."""
        data: dict[str, Any] = {}
        line_buffer = None
        for line in iterator:
            if not line.strip():
                continue
            if "Exciton analysis" in line or "Decomposition" in line:
                line_buffer = line
                break
            if "CT numbers" in line:
                continue
            if "omega" in line and "=" in line:
                m = re.search(r"omega\s*=\s*([\d.]+)\s*\(alpha:\s*([\d.]+),\s*beta:\s*([\d.]+)\)", line)
                if m:
                    data["ct_omega"] = float(m.group(1))
                    data["ct_omega_alpha"] = float(m.group(2))
                    data["ct_omega_beta"] = float(m.group(3))
            elif "<Phe>" in line and "=" in line:
                m = re.search(r"<Phe>\s*=\s*([\d.]+)\s*\(alpha:\s*([\d.]+),\s*beta:\s*([\d.]+)\)", line)
                if m:
                    data["ct_phe"] = float(m.group(1))
                    data["ct_phe_alpha"] = float(m.group(2))
                    data["ct_phe_beta"] = float(m.group(3))
        return data, line_buffer

    def _parse_exciton_section(
        self, iterator: PeekableIterator, prefix: str = "exciton"
    ) -> tuple[dict[str, ExcitonAnalysis | None], str | None]:
        result: dict[str, ExcitonAnalysis | None] = {}

        # Find first meaningful line
        iterator.take_while(lambda ln: not ln.strip())
        line_buffer = iterator.peek()
        if line_buffer is None:
            return {}, None
        next(iterator)  # consume the peeked line

        # RKS (no Total: header)
        if "Total:" not in line_buffer:
            result[f"{prefix}_total"], line_buffer = self._parse_exciton_sub(iterator, initial_line=line_buffer)
            return result, line_buffer

        # UKS: Total: header consumed, parse sub
        result[f"{prefix}_total"], line_buffer = self._parse_exciton_sub(iterator)

        if line_buffer is None:
            iterator.take_while(lambda ln: not ln.strip())
            line_buffer = iterator.peek()
            if line_buffer is not None:
                next(iterator)

        if line_buffer and "Alpha spin:" in line_buffer:
            result[f"{prefix}_alpha"], line_buffer = self._parse_exciton_sub(iterator)
            if line_buffer is None:
                iterator.take_while(lambda ln: not ln.strip())
                line_buffer = iterator.peek()
                if line_buffer is not None:
                    next(iterator)

        if line_buffer and "Beta spin:" in line_buffer:
            result[f"{prefix}_beta"], line_buffer = self._parse_exciton_sub(iterator)

        return result, line_buffer

    def _parse_exciton_sub(
        self, iterator: PeekableIterator, initial_line: str | None = None
    ) -> tuple[ExcitonAnalysis, str | None]:
        data: dict[str, Any] = {}
        expecting_hole_comp = False
        expecting_electron_comp = False
        expecting_rms_comp = False
        expecting_com_comp = False

        src = chain([initial_line], iterator) if initial_line else iterator
        for line in src:
            stripped = line.strip()

            # Termination conditions
            if any(kw in stripped for kw in ["Alpha spin:", "Beta spin:", "Decomposition"]):
                return ExcitonAnalysis.from_dict(data), line
            if any(kw in line for kw in ["Transition density matrix", "Mulliken Population", "Multipole moment"]):
                return ExcitonAnalysis.from_dict(data), line
            if _STATE_HEADER_PAT.match(line) or _END_PAT.search(line):
                return ExcitonAnalysis.from_dict(data), line
            if not stripped:
                break

            # Component flags
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
            if expecting_rms_comp:
                if "Cartesian components" in line:
                    data["rms_separation_components_ang"] = _parse_vec(line)
                expecting_rms_comp = False
                continue
            if expecting_com_comp:
                if "Cartesian components" in line:
                    data["center_of_mass_components_ang"] = _parse_vec(line)
                expecting_com_comp = False
                continue

            # Parse fields
            if "Trans. dipole moment [D]:" in line:
                data["trans_dipole_moment_debye"] = _parse_kv(line, "Trans. dipole moment")
            elif "Cartesian components [D]:" in line:
                data["trans_dipole_moment_components_debye"] = _parse_vec(line)
            elif "<r_h> [Ang]:" in line:
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
            elif "RMS electron-hole separation [Ang]:" in line:
                data["rms_separation_ang"] = _parse_kv(line, "RMS electron-hole separation")
                expecting_rms_comp = True
            elif "Covariance(r_h, r_e)" in line:
                data["covariance"] = _parse_kv(line, "Covariance")
            elif "Correlation coefficient:" in line:
                data["correlation_coef"] = _parse_kv(line, "Correlation coefficient")
            elif "Center-of-mass size [Ang]:" in line:
                data["center_of_mass_size_ang"] = _parse_kv(line, "Center-of-mass size")
                expecting_com_comp = True

        return ExcitonAnalysis.from_dict(data), None
