"""
Post-processing utilities for stick spectra from quantum chemistry calculations.

Two-layer API:
  Layer 1 (core)  — array-based, type-agnostic:
      make_energy_grid()        uniform grid from plain energy values
      lorentzian_broadening()   Lorentzian convolution of arbitrary (energy, weight) pairs

  Layer 2 (typed wrappers) — extract energies/weights from calcflow result objects:
      spectrum_from_excited_states()      TDDFT, oscillator-strength weighted
      opa_spectrum_from_adc_states()      ADC,  oscillator-strength weighted
      tpa_spectrum_from_adc_states()      ADC,  TPA cross-section weighted

  For energy grids, call make_energy_grid([s.excitation_energy_ev for s in states], ...)
  directly — the extraction is a trivial one-liner not worth a dedicated wrapper.

Rendering (Plotly, matplotlib) stays in the calling application.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from calcflow.common.exceptions import ValidationError
from calcflow.common.results import AdcExcitedState, ExcitedState

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)


# =============================================================================
# Layer 1: core, array-based
# =============================================================================


def make_energy_grid(
    energies: Sequence[float],
    padding: float = 5.0,
    n_points: int = 2000,
) -> npt.NDArray[np.float64]:
    """Return a uniform energy grid spanning *energies* ± padding (eV).

    Args:
        energies:  Excitation energies in eV.
        padding:   Extra range added on each side of the min/max energy, in eV.
        n_points:  Number of grid points.
    """
    import numpy as np

    if not energies:
        raise ValidationError("energies must not be empty; cannot construct energy grid without at least one value")
    return np.linspace(min(energies) - padding, max(energies) + padding, n_points)


def lorentzian_broadening(
    energies: Sequence[float],
    weights: Sequence[float],
    energy_grid: Sequence[float] | npt.NDArray[np.float64],
    fwhm: float,
    energy_shift: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return a Lorentzian-broadened spectrum on *energy_grid*.

    Each (energy, weight) pair contributes a Lorentzian centred at its energy,
    scaled by its weight:

        I(E) = Σ_i  w_i · (1/π) · hwhm / ((E - E_i)² + hwhm²)

    The Lorentzian integrates to 1 over all E, so each peak contributes
    area = w_i to the total spectrum.  hwhm = fwhm / 2.

    Args:
        energies:      Peak positions in eV.
        weights:       Intensity weights (oscillator strengths, TPA cross sections, …).
        energy_grid:   Energy axis in eV at which to evaluate the spectrum.
        fwhm:          Full width at half maximum in eV.
        energy_shift:  Rigid shift applied to all energies before broadening, in eV.

    Returns:
        Spectrum intensities (same length as energy_grid).
        All-zeros when *energies* / *weights* are empty.
    """
    import numpy as np

    grid = np.asarray(energy_grid, dtype=np.float64)
    if fwhm <= 0.0:
        raise ValidationError(f"fwhm must be positive, got {fwhm}")
    if len(energies) != len(weights):
        raise ValidationError(f"energies and weights must have the same length, got {len(energies)} vs {len(weights)}")
    if not energies:
        return np.zeros_like(grid)

    hwhm = fwhm / 2.0
    peak_e = np.array(energies, dtype=np.float64)[:, np.newaxis] + energy_shift  # (N, 1)
    w = np.array(weights, dtype=np.float64)[:, np.newaxis]  # (N, 1)
    return np.sum(w * (1.0 / np.pi) * hwhm / ((grid - peak_e) ** 2 + hwhm**2), axis=0)


# =============================================================================
# Layer 2: typed wrappers — TDDFT ExcitedState
# =============================================================================


def spectrum_from_excited_states(
    states: Sequence[ExcitedState],
    energy_grid: Sequence[float] | npt.NDArray[np.float64],
    fwhm: float,
    energy_shift: float = 0.0,
    only_singlets: bool = False,
) -> npt.NDArray[np.float64]:
    """Lorentzian-broadened absorption spectrum from TDDFT excited states.

    Each state is weighted by its oscillator strength.

    Args:
        states:        Parsed TDDFT excited states.
        energy_grid:   Energy axis in eV.
        fwhm:          Full width at half maximum in eV.
        energy_shift:  Rigid shift applied to all state energies, in eV.
        only_singlets: When True, triplet states are excluded.

    Returns:
        Spectrum intensities (same length as energy_grid). All-zeros if every
        state is filtered out or has oscillator_strength=None.
    """
    energies: list[float] = []
    weights: list[float] = []
    for state in states:
        if only_singlets and state.multiplicity != "Singlet":
            continue
        if state.oscillator_strength is None:
            logger.warning("state %d has oscillator_strength=None, skipping", state.state_number)
            continue
        energies.append(state.excitation_energy_ev)
        weights.append(state.oscillator_strength)
    return lorentzian_broadening(energies, weights, energy_grid, fwhm=fwhm, energy_shift=energy_shift)


# =============================================================================
# Layer 2: typed wrappers — ADC AdcExcitedState
# =============================================================================


def opa_spectrum_from_adc_states(
    states: Sequence[AdcExcitedState],
    energy_grid: Sequence[float] | npt.NDArray[np.float64],
    fwhm: float,
    energy_shift: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Lorentzian-broadened OPA spectrum from ADC excited states.

    Each state is weighted by its oscillator strength.  States with
    oscillator_strength=None are skipped with a warning.

    Args:
        states:        Parsed ADC excited states.
        energy_grid:   Energy axis in eV.
        fwhm:          Full width at half maximum in eV.
        energy_shift:  Rigid shift applied to all state energies, in eV.
    """
    energies: list[float] = []
    weights: list[float] = []
    for state in states:
        if state.oscillator_strength is None:
            logger.warning("ADC state %d has oscillator_strength=None, skipping", state.state_number)
            continue
        energies.append(state.excitation_energy_ev)
        weights.append(state.oscillator_strength)
    return lorentzian_broadening(energies, weights, energy_grid, fwhm=fwhm, energy_shift=energy_shift)


def tpa_spectrum_from_adc_states(
    states: Sequence[AdcExcitedState],
    energy_grid: Sequence[float] | npt.NDArray[np.float64],
    fwhm: float,
    energy_shift: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Lorentzian-broadened TPA spectrum from ADC excited states.

    Each state is weighted by its two-photon absorption cross section
    (two_photon_absorption.cross_section_au).  States where
    two_photon_absorption is None are skipped with a warning.

    Args:
        states:        Parsed ADC excited states.
        energy_grid:   Energy axis in eV.
        fwhm:          Full width at half maximum in eV.
        energy_shift:  Rigid shift applied to all state energies, in eV.
    """
    energies: list[float] = []
    weights: list[float] = []
    for state in states:
        if state.two_photon_absorption is None:
            logger.warning("ADC state %d has two_photon_absorption=None, skipping", state.state_number)
            continue
        energies.append(state.excitation_energy_ev)
        weights.append(state.two_photon_absorption.cross_section_au)
    return lorentzian_broadening(energies, weights, energy_grid, fwhm=fwhm, energy_shift=energy_shift)
