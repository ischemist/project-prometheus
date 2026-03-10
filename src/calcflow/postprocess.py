"""
Post-processing utilities for TDDFT stick spectra.

Operates on parsed ExcitedState sequences; returns plain numpy arrays.
Rendering (Plotly, matplotlib) stays in the calling application.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from calcflow.common.results import ExcitedState

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def make_energy_grid(
    states: Sequence[ExcitedState],
    padding: float = 5.0,
    n_points: int = 2000,
) -> np.ndarray:
    """Return a uniform energy grid spanning the state energies ± padding (eV)."""
    import numpy as np

    energies = [s.excitation_energy_ev for s in states]
    return np.linspace(min(energies) - padding, max(energies) + padding, n_points)


def lorentzian_spectrum(
    states: Sequence[ExcitedState],
    energy_grid: Sequence[float] | np.ndarray,
    fwhm: float,
    energy_shift: float = 0.0,
    only_singlets: bool = False,
) -> np.ndarray:
    """Lorentzian-broaden a stick spectrum onto energy_grid.

    Intensity = Σ_i  f_i · (1/π) · γ / ((E - (E_i + shift))² + γ²)
    where γ = fwhm / 2.

    States with oscillator_strength=None are skipped with a warning.
    """
    import numpy as np

    grid = np.asarray(energy_grid, dtype=float)
    gamma = fwhm / 2.0

    active = []
    for state in states:
        if only_singlets and state.multiplicity != "Singlet":
            continue
        if state.oscillator_strength is None:
            logger.warning("state %d has oscillator_strength=None, skipping", state.state_number)
            continue
        active.append((state.excitation_energy_ev + energy_shift, state.oscillator_strength))

    if not active:
        return np.zeros_like(grid)

    # broadcast: e0 (N,1), f (N,1) against grid (M,) → sum over N → (M,)
    e0 = np.array([e for e, _ in active])[:, np.newaxis]  # (N, 1)
    f = np.array([s for _, s in active])[:, np.newaxis]  # (N, 1)
    return np.sum(f * (1.0 / np.pi) * gamma / ((grid - e0) ** 2 + gamma**2), axis=0)
