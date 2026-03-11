"""
Unit tests for calcflow.postprocess.

All tests use hand-crafted ExcitedState / AdcExcitedState objects — no real output files.
"""

from typing import Literal

import numpy as np
import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.results import AdcExcitedState, ExcitedState, TwoPhotonAbsorption
from calcflow.postprocess import (
    lorentzian_broadening,
    make_energy_grid,
    opa_spectrum_from_adc_states,
    spectrum_from_excited_states,
    tpa_spectrum_from_adc_states,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _adc_state(
    state_number: int,
    energy_ev: float,
    oscillator_strength: float | None = 1.0,
    tpa_cross_section: float | None = None,
) -> AdcExcitedState:
    tpa = (
        TwoPhotonAbsorption(cross_section_au=tpa_cross_section, matrix_au=((0.0,) * 3,) * 3)
        if tpa_cross_section is not None
        else None
    )  # type: ignore[arg-type]
    return AdcExcitedState(
        state_number=state_number,
        total_energy_au=-100.0,
        excitation_energy_ev=energy_ev,
        oscillator_strength=oscillator_strength,
        two_photon_absorption=tpa,
    )


# ---------------------------------------------------------------------------
# make_energy_grid (core, accepts plain floats)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_make_energy_grid_span():
    grid = make_energy_grid([10.0, 15.0], padding=2.0, n_points=500)
    assert grid[0] == pytest.approx(8.0)
    assert grid[-1] == pytest.approx(17.0)
    assert len(grid) == 500


@pytest.mark.unit
def test_make_energy_grid_default_params():
    grid = make_energy_grid([5.0, 10.0])
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(15.0)
    assert len(grid) == 2000


@pytest.mark.unit
def test_make_energy_grid_empty_raises():
    with pytest.raises(ValidationError, match="energies must not be empty"):
        make_energy_grid([])


# ---------------------------------------------------------------------------
# lorentzian_broadening (core)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lorentzian_broadening_peak_at_energy():
    grid = np.linspace(5.0, 15.0, 5000)
    spectrum = lorentzian_broadening([10.0], [1.0], grid, fwhm=0.5)
    assert grid[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.01)


@pytest.mark.unit
def test_lorentzian_broadening_energy_shift():
    grid = np.linspace(5.0, 20.0, 5000)
    spectrum = lorentzian_broadening([10.0], [1.0], grid, fwhm=0.5, energy_shift=1.0)
    assert grid[np.argmax(spectrum)] == pytest.approx(11.0, abs=0.01)


@pytest.mark.unit
def test_lorentzian_broadening_empty_returns_zeros():
    grid = np.linspace(5.0, 15.0, 100)
    spectrum = lorentzian_broadening([], [], grid, fwhm=0.5)
    np.testing.assert_array_equal(spectrum, np.zeros(100))


@pytest.mark.unit
def test_lorentzian_broadening_output_length_matches_grid():
    for n in (50, 200, 2000):
        grid = np.linspace(5.0, 20.0, n)
        spectrum = lorentzian_broadening([10.0, 12.0], [1.0, 0.5], grid, fwhm=0.5)
        assert len(spectrum) == n


@pytest.mark.unit
def test_lorentzian_broadening_nonpositive_fwhm_raises():
    grid = np.linspace(5.0, 15.0, 100)
    with pytest.raises(ValidationError, match="fwhm must be positive"):
        lorentzian_broadening([10.0], [1.0], grid, fwhm=0.0)


@pytest.mark.unit
def test_lorentzian_broadening_mismatched_lengths_raises():
    grid = np.linspace(5.0, 15.0, 100)
    with pytest.raises(ValidationError, match="same length"):
        lorentzian_broadening([10.0, 11.0], [1.0], grid, fwhm=0.5)


@pytest.mark.unit
def test_lorentzian_broadening_intensity_positive():
    grid = np.linspace(3.0, 18.0, 1000)
    spectrum = lorentzian_broadening([8.0, 10.0, 12.0], [0.5, 0.5, 0.5], grid, fwhm=0.5)
    assert np.all(spectrum >= 0.0)


# ---------------------------------------------------------------------------
# spectrum_from_excited_states (TDDFT OPA)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spectrum_from_excited_states_peak_at_energy():
    state = _state(1, energy_ev=10.0, oscillator_strength=1.0)
    grid = np.linspace(5.0, 15.0, 5000)
    spectrum = spectrum_from_excited_states([state], grid, fwhm=0.5)
    assert grid[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.01)


@pytest.mark.unit
def test_spectrum_from_excited_states_only_singlets_skips_triplets():
    singlet = _state(1, energy_ev=10.0, oscillator_strength=1.0, multiplicity="Singlet")
    triplet = _state(2, energy_ev=12.0, oscillator_strength=1.0, multiplicity="Triplet")
    grid = np.linspace(8.0, 15.0, 2000)
    filtered = spectrum_from_excited_states([singlet, triplet], grid, fwhm=0.5, only_singlets=True)
    unfiltered = spectrum_from_excited_states([singlet], grid, fwhm=0.5)
    np.testing.assert_allclose(filtered, unfiltered)


@pytest.mark.unit
def test_spectrum_from_excited_states_energy_shift_moves_peak():
    state = _state(1, energy_ev=10.0, oscillator_strength=1.0)
    grid = np.linspace(5.0, 20.0, 5000)
    shifted = spectrum_from_excited_states([state], grid, fwhm=0.5, energy_shift=1.0)
    assert grid[np.argmax(shifted)] == pytest.approx(11.0, abs=0.01)


@pytest.mark.unit
def test_spectrum_from_excited_states_all_filtered_returns_zeros():
    triplet = _state(1, energy_ev=10.0, oscillator_strength=1.0, multiplicity="Triplet")
    grid = np.linspace(5.0, 15.0, 100)
    spectrum = spectrum_from_excited_states([triplet], grid, fwhm=0.5, only_singlets=True)
    np.testing.assert_array_equal(spectrum, np.zeros(100))


@pytest.mark.unit
def test_spectrum_from_excited_states_none_oscillator_skipped(caplog):
    import logging

    none_state = _state(1, energy_ev=10.0, oscillator_strength=None)
    grid = np.linspace(5.0, 15.0, 100)
    with caplog.at_level(logging.WARNING, logger="calcflow.postprocess"):
        spectrum = spectrum_from_excited_states([none_state], grid, fwhm=0.5)
    np.testing.assert_array_equal(spectrum, np.zeros(100))
    assert "oscillator_strength=None" in caplog.text


@pytest.mark.unit
def test_spectrum_from_excited_states_nonpositive_fwhm_raises():
    grid = np.linspace(5.0, 15.0, 100)
    with pytest.raises(ValidationError, match="fwhm must be positive"):
        spectrum_from_excited_states([_state(1, energy_ev=10.0, oscillator_strength=1.0)], grid, fwhm=0.0)


# ---------------------------------------------------------------------------
# opa_spectrum_from_adc_states
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_opa_spectrum_from_adc_states_peak_at_energy():
    state = _adc_state(1, energy_ev=10.0, oscillator_strength=1.0)
    grid = np.linspace(5.0, 15.0, 5000)
    spectrum = opa_spectrum_from_adc_states([state], grid, fwhm=0.5)
    assert grid[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.01)


@pytest.mark.unit
def test_opa_spectrum_from_adc_states_none_oscillator_skipped(caplog):
    import logging

    state = _adc_state(1, energy_ev=10.0, oscillator_strength=None)
    grid = np.linspace(5.0, 15.0, 100)
    with caplog.at_level(logging.WARNING, logger="calcflow.postprocess"):
        spectrum = opa_spectrum_from_adc_states([state], grid, fwhm=0.5)
    np.testing.assert_array_equal(spectrum, np.zeros(100))
    assert "oscillator_strength=None" in caplog.text


@pytest.mark.unit
def test_opa_spectrum_from_adc_states_energy_shift():
    state = _adc_state(1, energy_ev=10.0, oscillator_strength=1.0)
    grid = np.linspace(5.0, 20.0, 5000)
    spectrum = opa_spectrum_from_adc_states([state], grid, fwhm=0.5, energy_shift=1.0)
    assert grid[np.argmax(spectrum)] == pytest.approx(11.0, abs=0.01)


# ---------------------------------------------------------------------------
# tpa_spectrum_from_adc_states
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tpa_spectrum_from_adc_states_peak_at_energy():
    state = _adc_state(1, energy_ev=10.0, tpa_cross_section=2.5)
    grid = np.linspace(5.0, 15.0, 5000)
    spectrum = tpa_spectrum_from_adc_states([state], grid, fwhm=0.5)
    assert grid[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.01)


@pytest.mark.unit
def test_tpa_spectrum_from_adc_states_weighted_by_cross_section():
    """A state with 2× cross section should produce 2× the peak height."""
    grid = np.linspace(5.0, 25.0, 5000)
    state_1x = _adc_state(1, energy_ev=10.0, tpa_cross_section=1.0)
    state_2x = _adc_state(2, energy_ev=20.0, tpa_cross_section=2.0)
    spectrum = tpa_spectrum_from_adc_states([state_1x, state_2x], grid, fwhm=0.5)
    peak_1x = spectrum[np.argmin(np.abs(grid - 10.0))]
    peak_2x = spectrum[np.argmin(np.abs(grid - 20.0))]
    assert peak_2x == pytest.approx(2.0 * peak_1x, rel=1e-3)


@pytest.mark.unit
def test_tpa_spectrum_from_adc_states_none_tpa_skipped(caplog):
    import logging

    state = _adc_state(1, energy_ev=10.0, tpa_cross_section=None)
    grid = np.linspace(5.0, 15.0, 100)
    with caplog.at_level(logging.WARNING, logger="calcflow.postprocess"):
        spectrum = tpa_spectrum_from_adc_states([state], grid, fwhm=0.5)
    np.testing.assert_array_equal(spectrum, np.zeros(100))
    assert "two_photon_absorption=None" in caplog.text


@pytest.mark.unit
def test_tpa_spectrum_from_adc_states_energy_shift():
    state = _adc_state(1, energy_ev=10.0, tpa_cross_section=1.0)
    grid = np.linspace(5.0, 20.0, 5000)
    spectrum = tpa_spectrum_from_adc_states([state], grid, fwhm=0.5, energy_shift=1.0)
    assert grid[np.argmax(spectrum)] == pytest.approx(11.0, abs=0.01)


@pytest.mark.unit
def test_tpa_and_opa_independent_from_same_states():
    """OPA and TPA spectra from the same ADC states should differ when weights differ."""
    states = [
        _adc_state(1, energy_ev=10.0, oscillator_strength=0.5, tpa_cross_section=2.0),
        _adc_state(2, energy_ev=12.0, oscillator_strength=1.0, tpa_cross_section=0.5),
    ]
    grid = np.linspace(5.0, 20.0, 2000)
    opa = opa_spectrum_from_adc_states(states, grid, fwhm=0.3)
    tpa = tpa_spectrum_from_adc_states(states, grid, fwhm=0.3)
    # OPA peak should be at 12 eV (higher oscillator strength), TPA at 10 eV (higher cross section)
    assert grid[np.argmax(opa)] == pytest.approx(12.0, abs=0.05)
    assert grid[np.argmax(tpa)] == pytest.approx(10.0, abs=0.05)
