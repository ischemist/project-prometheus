"""
Post-processing utilities for TDDFT stick spectra.

Operates on parsed ExcitedState sequences; returns plain numpy arrays.
Rendering (Plotly, matplotlib) stays in the calling application.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from calcflow.common.exceptions import ValidationError
from calcflow.common.results import ExcitedState

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def make_energy_grid(
    states: Sequence[ExcitedState],
    padding: float = 5.0,
    n_points: int = 2000,
) -> npt.NDArray[np.float64]:
    """Return a uniform energy grid spanning the state energies ± padding (eV)."""
    import numpy as np

    energies = [s.excitation_energy_ev for s in states]
    if not energies:
        raise ValidationError("states must not be empty; cannot construct energy grid without at least one state")
    return np.linspace(min(energies) - padding, max(energies) + padding, n_points)


def lorentzian_spectrum(
    states: Sequence[ExcitedState],
    energy_grid: Sequence[float] | npt.NDArray[np.float64],
    fwhm: float,
    energy_shift: float = 0.0,
    only_singlets: bool = False,
) -> npt.NDArray[np.float64]:
    """Return the Lorentzian-broadened absorption spectrum on energy_grid.

    Each excited state contributes a Lorentzian centred at its excitation energy,
    weighted by its oscillator strength:

        I(E) = Σ_i  f_i · (1/π) · hwhm / ((E - E_i)² + hwhm²)

    The Lorentzian (1/π) · hwhm / ((E - E₀)² + hwhm²) integrates to 1 over all E,
    so each state contributes area = f_i to the total spectrum.
    hwhm = fwhm / 2 (half-width at half maximum); at E = E₀ ± hwhm the Lorentzian
    drops to half its peak value, which is how FWHM is defined.

    Args:
        states:        Parsed excited states (pass tddft.tda_states directly).
        energy_grid:   Energy axis in eV at which to evaluate the spectrum.
        fwhm:          Full width at half maximum of each Lorentzian, in eV.
        energy_shift:  Rigid shift applied to all state energies before broadening,
                       in eV. Use to align calculated energies with experiment.
        only_singlets: When True, triplet states are excluded from the sum.

    Returns:
        Spectrum intensities (same length as energy_grid). All-zeros if every
        state is filtered out or has oscillator_strength=None.

    Notes:
        States with oscillator_strength=None are skipped and logged as warnings.
    """
    import numpy as np

    grid = np.asarray(energy_grid, dtype=np.float64)
    if fwhm <= 0.0:
        raise ValidationError(f"fwhm must be positive, got {fwhm}")
    hwhm = fwhm / 2.0  # half-width at half maximum; Lorentzian parameter γ = FWHM/2

    peak_energies_ev: list[float] = []
    oscillator_strengths: list[float] = []
    for state in states:
        if only_singlets and state.multiplicity != "Singlet":
            continue
        if state.oscillator_strength is None:
            logger.warning("state %d has oscillator_strength=None, skipping", state.state_number)
            continue
        peak_energies_ev.append(state.excitation_energy_ev + energy_shift)
        oscillator_strengths.append(state.oscillator_strength)

    if not peak_energies_ev:
        return np.zeros_like(grid)

    # Evaluate each Lorentzian on the full grid and sum over states.
    # (N_states, 1) broadcasts against grid (N_grid,) → (N_states, N_grid); sum axis=0 → (N_grid,).
    peak_energies = np.array(peak_energies_ev, dtype=np.float64)[:, np.newaxis]  # (N_states, 1)
    strengths = np.array(oscillator_strengths, dtype=np.float64)[:, np.newaxis]  # (N_states, 1)
    return np.sum(strengths * (1.0 / np.pi) * hwhm / ((grid - peak_energies) ** 2 + hwhm**2), axis=0)
