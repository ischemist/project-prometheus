"""
Unit tests for calcflow.postprocess.

All tests use hand-crafted ExcitedState objects — no real output files.
"""

from typing import Literal

import numpy as np
import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.results import ExcitedState
from calcflow.postprocess import lorentzian_spectrum, make_energy_grid


def _state(
    state_number: int,
    energy_ev: float,
    oscillator_strength: float | None = 1.0,
    multiplicity: Literal["Singlet", "Triplet", "Unknown"] = "Singlet",
) -> ExcitedState:
    return ExcitedState(
        state_number=state_number,
        multiplicity=multiplicity,
        excitation_energy_ev=energy_ev,
        total_energy_au=-100.0,
        oscillator_strength=oscillator_strength,
    )


# ---------------------------------------------------------------------------
# make_energy_grid
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_make_energy_grid_span():
    states = [_state(1, 10.0), _state(2, 15.0)]
    grid = make_energy_grid(states, padding=2.0, n_points=500)
    assert grid[0] == pytest.approx(8.0)
    assert grid[-1] == pytest.approx(17.0)
    assert len(grid) == 500


@pytest.mark.unit
def test_make_energy_grid_default_params():
    states = [_state(1, 5.0), _state(2, 10.0)]
    grid = make_energy_grid(states)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(15.0)
    assert len(grid) == 2000


@pytest.mark.unit
def test_make_energy_grid_empty_states_raises():
    with pytest.raises(ValidationError, match="states must not be empty"):
        make_energy_grid([])


# ---------------------------------------------------------------------------
# lorentzian_spectrum
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lorentzian_single_state_peak_at_energy():
    """Peak of the broadened spectrum should be at the state energy."""
    state = _state(1, energy_ev=10.0, oscillator_strength=1.0)
    grid = np.linspace(5.0, 15.0, 5000)
    spectrum = lorentzian_spectrum([state], grid, fwhm=0.5)
    assert grid[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.01)


@pytest.mark.unit
def test_lorentzian_only_singlets_skips_triplets():
    """only_singlets=True: triplet contributes nothing; singlet peak survives."""
    singlet = _state(1, energy_ev=10.0, oscillator_strength=1.0, multiplicity="Singlet")
    triplet = _state(2, energy_ev=12.0, oscillator_strength=1.0, multiplicity="Triplet")
    grid = np.linspace(8.0, 15.0, 2000)

    filtered = lorentzian_spectrum([singlet, triplet], grid, fwhm=0.5, only_singlets=True)
    unfiltered = lorentzian_spectrum([singlet], grid, fwhm=0.5)

    np.testing.assert_allclose(filtered, unfiltered)


@pytest.mark.unit
def test_lorentzian_energy_shift_moves_peak():
    """energy_shift=+1 should move the peak by exactly 1 eV."""
    state = _state(1, energy_ev=10.0, oscillator_strength=1.0)
    grid = np.linspace(5.0, 20.0, 5000)

    shifted = lorentzian_spectrum([state], grid, fwhm=0.5, energy_shift=1.0)
    peak_energy = grid[np.argmax(shifted)]
    assert peak_energy == pytest.approx(11.0, abs=0.01)


@pytest.mark.unit
def test_lorentzian_all_filtered_returns_zeros():
    """If every state is filtered out the result should be all zeros."""
    triplet = _state(1, energy_ev=10.0, oscillator_strength=1.0, multiplicity="Triplet")
    grid = np.linspace(5.0, 15.0, 100)
    spectrum = lorentzian_spectrum([triplet], grid, fwhm=0.5, only_singlets=True)
    np.testing.assert_array_equal(spectrum, np.zeros(100))


@pytest.mark.unit
def test_lorentzian_none_oscillator_skipped(caplog):
    """States with oscillator_strength=None are skipped without raising."""
    import logging

    none_state = _state(1, energy_ev=10.0, oscillator_strength=None)
    grid = np.linspace(5.0, 15.0, 100)
    with caplog.at_level(logging.WARNING, logger="calcflow.postprocess"):
        spectrum = lorentzian_spectrum([none_state], grid, fwhm=0.5)
    np.testing.assert_array_equal(spectrum, np.zeros(100))
    assert "oscillator_strength=None" in caplog.text


@pytest.mark.unit
def test_lorentzian_output_length_matches_grid():
    """Output array length must equal the input energy_grid length."""
    states = [_state(1, 10.0), _state(2, 12.0)]
    for n in (50, 200, 2000):
        grid = np.linspace(5.0, 20.0, n)
        spectrum = lorentzian_spectrum(states, grid, fwhm=0.5)
        assert len(spectrum) == n


@pytest.mark.unit
def test_lorentzian_intensity_positive():
    """Spectrum values should be non-negative for positive oscillator strengths."""
    states = [_state(i, float(e), oscillator_strength=0.5) for i, e in enumerate([8.0, 10.0, 12.0], start=1)]
    grid = np.linspace(3.0, 18.0, 1000)
    spectrum = lorentzian_spectrum(states, grid, fwhm=0.5)
    assert np.all(spectrum >= 0.0)


@pytest.mark.unit
def test_lorentzian_nonpositive_fwhm_raises():
    grid = np.linspace(5.0, 15.0, 100)
    with pytest.raises(ValidationError, match="fwhm must be positive"):
        lorentzian_spectrum([_state(1, energy_ev=10.0, oscillator_strength=1.0)], grid, fwhm=0.0)
