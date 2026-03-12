"""Tests for AnnotatedAtom, AnnotatedGeometry, and AnnotatedTrajectory."""

import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.results import (
    AdcExcitedState,
    AdcGroundState,
    AdcResults,
    AtomicCharges,
    CalculationMetadata,
    CalculationResult,
    ExcitonAnalysis,
    NaturalOrbitals,
    TransitionDensityMatrix,
    UnrelaxedDensityMatrix,
)
from calcflow.geometry.annotated import (
    AnnotatedAtom,
    AnnotatedGeometry,
    AnnotatedStateAtom,
    AnnotatedStateView,
    AnnotatedTrajectory,
)
from calcflow.geometry.static import Atom, Geometry

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def h2o_atoms() -> tuple[Atom, ...]:
    return (
        Atom(symbol="O", x=0.0, y=0.0, z=0.1173),
        Atom(symbol="H", x=0.0, y=0.7572, z=-0.4692),
        Atom(symbol="H", x=0.0, y=-0.7572, z=-0.4692),
    )


@pytest.fixture
def h2o_geometry(h2o_atoms: tuple[Atom, ...]) -> Geometry:
    return Geometry(comment="", atoms=h2o_atoms)


@pytest.fixture
def alt_geometry(h2o_atoms: tuple[Atom, ...]) -> Geometry:
    """A distinct geometry object (simulates input vs. final geometry)."""
    shifted = tuple(Atom(symbol=a.symbol, x=a.x + 0.1, y=a.y, z=a.z) for a in h2o_atoms)
    return Geometry(comment="input", atoms=shifted)


@pytest.fixture
def mulliken() -> AtomicCharges:
    return AtomicCharges(method="Mulliken", charges={0: -0.8, 1: 0.4, 2: 0.4})


@pytest.fixture
def hirshfeld() -> AtomicCharges:
    return AtomicCharges(method="Hirshfeld", charges={0: -0.5, 1: 0.25, 2: 0.25})


@pytest.fixture
def mulliken_with_spins() -> AtomicCharges:
    return AtomicCharges(method="Mulliken", charges={0: -0.8, 1: 0.4, 2: 0.4}, spins={0: 0.1, 1: -0.05, 2: -0.05})


@pytest.fixture
def excited_state_charges() -> AtomicCharges:
    """AtomicCharges from TDDFT unrelaxed DM — should NOT appear on AnnotatedAtom.charges."""
    return AtomicCharges(
        method="Mulliken",
        charges={0: -0.1, 1: 0.05, 2: 0.05},
        hole_populations={0: 0.9, 1: 0.05, 2: 0.05},
    )


def _make_result(
    *,
    final_geometry: Geometry | None = None,
    input_geometry: Geometry | None = None,
    atomic_charges: list[AtomicCharges] | None = None,
    final_energy: float | None = None,
) -> CalculationResult:
    """Minimal CalculationResult for testing."""
    return CalculationResult(
        termination_status="NORMAL",
        metadata=CalculationMetadata(software_name="test"),
        raw_output="",
        final_geometry=final_geometry,
        input_geometry=input_geometry,
        atomic_charges=atomic_charges or [],
        final_energy=final_energy,
    )


# =============================================================================
# AnnotatedGeometry.from_result()
# =============================================================================


@pytest.mark.unit
def test_from_result_uses_final_geometry(h2o_geometry: Geometry, alt_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry, input_geometry=alt_geometry)
    ag = AnnotatedGeometry.from_result(result)
    assert ag.geometry is h2o_geometry


@pytest.mark.unit
def test_from_result_falls_back_to_input_geometry(h2o_geometry: Geometry):
    result = _make_result(input_geometry=h2o_geometry)
    ag = AnnotatedGeometry.from_result(result)
    assert ag.geometry is h2o_geometry


@pytest.mark.unit
def test_from_result_use_input_geometry_flag(h2o_geometry: Geometry, alt_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry, input_geometry=alt_geometry)
    ag = AnnotatedGeometry.from_result(result, use_input_geometry=True)
    assert ag.geometry is alt_geometry


@pytest.mark.unit
def test_from_result_raises_when_no_geometry():
    result = _make_result()
    with pytest.raises(ValidationError, match="neither final_geometry nor input_geometry"):
        AnnotatedGeometry.from_result(result)


@pytest.mark.unit
def test_from_result_raises_use_input_geometry_when_missing():
    result = _make_result(final_geometry=Geometry(comment="", atoms=(Atom(symbol="H", x=0, y=0, z=0),)))
    with pytest.raises(ValidationError, match="no input_geometry"):
        AnnotatedGeometry.from_result(result, use_input_geometry=True)


# =============================================================================
# AnnotatedGeometry.__getitem__ / AnnotatedAtom
# =============================================================================


@pytest.mark.unit
def test_getitem_returns_annotated_atom(h2o_geometry: Geometry, mulliken: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    atom = ag[0]
    assert isinstance(atom, AnnotatedAtom)
    assert atom.symbol == "O"
    assert atom.x == pytest.approx(0.0)
    assert atom.y == pytest.approx(0.0)
    assert atom.z == pytest.approx(0.1173)
    assert atom.index == 0


@pytest.mark.unit
def test_getitem_charges_populated(h2o_geometry: Geometry, mulliken: AtomicCharges, hirshfeld: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken, hirshfeld])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    atom = ag[0]
    assert atom.charges == {"Mulliken": pytest.approx(-0.8), "Hirshfeld": pytest.approx(-0.5)}


@pytest.mark.unit
def test_getitem_spins_populated(h2o_geometry: Geometry, mulliken_with_spins: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken_with_spins])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    atom = ag[1]
    assert atom.spins == {"Mulliken": pytest.approx(-0.05)}


@pytest.mark.unit
def test_getitem_spins_empty_for_rks(h2o_geometry: Geometry, mulliken: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    assert ag[0].spins == {}


@pytest.mark.unit
def test_getitem_no_charges_at_all(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    atom = ag[0]
    assert atom.charges == {}
    assert atom.spins == {}


@pytest.mark.unit
def test_getitem_excited_state_charges_excluded(h2o_geometry: Geometry, excited_state_charges: AtomicCharges):
    """AtomicCharges with hole_populations set must not appear in AnnotatedAtom.charges."""
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[excited_state_charges])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    assert ag[0].charges == {}


@pytest.mark.unit
def test_getitem_negative_index(h2o_geometry: Geometry, mulliken: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    atom = ag[-1]
    assert atom.index == 2
    assert atom.symbol == "H"
    assert atom.charges == {"Mulliken": pytest.approx(0.4)}


@pytest.mark.unit
def test_getitem_out_of_range(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    with pytest.raises(IndexError):
        _ = ag[99]


# =============================================================================
# AnnotatedGeometry iteration / length
# =============================================================================


@pytest.mark.unit
def test_len(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert len(ag) == 3


@pytest.mark.unit
def test_iter_yields_all_atoms_in_order(h2o_geometry: Geometry, mulliken: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    atoms = list(ag)
    assert len(atoms) == 3
    assert [a.index for a in atoms] == [0, 1, 2]
    assert [a.symbol for a in atoms] == ["O", "H", "H"]


# =============================================================================
# AnnotatedGeometry property delegates
# =============================================================================


@pytest.mark.unit
def test_get_charges_delegates(h2o_geometry: Geometry, mulliken: AtomicCharges):
    result = _make_result(final_geometry=h2o_geometry, atomic_charges=[mulliken])
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    assert ag.get_charges("Mulliken") is mulliken
    assert ag.get_charges("Unknown") is None


@pytest.mark.unit
def test_energy_property(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry, final_energy=-76.1234)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.energy == pytest.approx(-76.1234)


@pytest.mark.unit
def test_energy_none_when_absent(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.energy is None


@pytest.mark.unit
def test_tddft_property_delegates(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.tddft is result.tddft  # both None


@pytest.mark.unit
def test_adc_property_delegates(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.adc is result.adc  # both None


# =============================================================================
# AnnotatedTrajectory.from_results()
# =============================================================================


@pytest.fixture
def three_frame_results(h2o_geometry: Geometry) -> list[CalculationResult]:
    return [_make_result(final_geometry=h2o_geometry, final_energy=float(-76 - i)) for i in range(3)]


@pytest.mark.unit
def test_from_results_happy_path(three_frame_results: list[CalculationResult]):
    traj = AnnotatedTrajectory.from_results(three_frame_results)
    assert len(traj) == 3


@pytest.mark.unit
def test_from_results_empty_raises():
    with pytest.raises(ValidationError, match="empty"):
        AnnotatedTrajectory.from_results([])


@pytest.mark.unit
def test_from_results_single_missing_geometry():
    result = _make_result()  # no geometry
    with pytest.raises(ValidationError, match=r"indices \[0\]"):
        AnnotatedTrajectory.from_results([result])


@pytest.mark.unit
def test_from_results_multiple_missing_geometry(h2o_geometry: Geometry):
    results = [
        _make_result(final_geometry=h2o_geometry),
        _make_result(),  # missing
        _make_result(final_geometry=h2o_geometry),
        _make_result(),  # missing
    ]
    with pytest.raises(ValidationError, match=r"indices \[1, 3\]"):
        AnnotatedTrajectory.from_results(results)


@pytest.mark.unit
def test_from_results_inconsistent_atom_counts(h2o_geometry: Geometry):
    single_atom = Geometry(comment="", atoms=(Atom(symbol="H", x=0, y=0, z=0),))
    results = [
        _make_result(final_geometry=h2o_geometry),
        _make_result(final_geometry=single_atom),
    ]
    with pytest.raises(ValidationError, match="inconsistent number of atoms"):
        AnnotatedTrajectory.from_results(results)


# =============================================================================
# AnnotatedTrajectory dunder methods
# =============================================================================


@pytest.mark.unit
def test_trajectory_len(three_frame_results: list[CalculationResult]):
    traj = AnnotatedTrajectory.from_results(three_frame_results)
    assert len(traj) == 3


@pytest.mark.unit
def test_trajectory_getitem(three_frame_results: list[CalculationResult]):
    traj = AnnotatedTrajectory.from_results(three_frame_results)
    assert traj[0].energy == pytest.approx(-76.0)
    assert traj[2].energy == pytest.approx(-78.0)


@pytest.mark.unit
def test_trajectory_iter(three_frame_results: list[CalculationResult]):
    traj = AnnotatedTrajectory.from_results(three_frame_results)
    energies = [frame.energy for frame in traj]
    assert energies == [pytest.approx(-76.0), pytest.approx(-77.0), pytest.approx(-78.0)]


# =============================================================================
# Fixtures for excited-state views
# =============================================================================

# Minimal required args for NaturalOrbitals and ExcitonAnalysis (used in
# UnrelaxedDensityMatrix which has several required fields).
_NOS = NaturalOrbitals(frontier_occupations=[1.0, 0.0], num_electrons=10.0)
_EXCITON = ExcitonAnalysis(
    r_h_ang=(0.0, 0.0, 0.0),
    r_e_ang=(1.0, 0.0, 0.0),
    separation_ang=1.0,
    hole_size_ang=0.5,
    electron_size_ang=0.5,
)


@pytest.fixture
def unrelaxed_dm_s1(h2o_atoms: tuple[Atom, ...]) -> UnrelaxedDensityMatrix:
    """Unrelaxed DM for state 1 with hole/electron populations (RKS)."""
    mulliken = AtomicCharges(
        method="Mulliken",
        charges={0: -0.1, 1: 0.05, 2: 0.05},
        hole_populations={0: 0.8, 1: 0.1, 2: 0.1},
        electron_populations={0: 0.2, 1: 0.4, 2: 0.4},
    )
    return UnrelaxedDensityMatrix(
        state_number=1,
        nos_spin_traced=_NOS,
        mulliken=mulliken,
        molecular_charge=0.0,
        num_electrons=10.0,
        dipole_moment_debye=1.5,
        dipole_components_debye=(0.0, 0.0, 1.5),
        exciton_total=_EXCITON,
    )


@pytest.fixture
def unrelaxed_dm_s1_uks(h2o_atoms: tuple[Atom, ...]) -> UnrelaxedDensityMatrix:
    """Unrelaxed DM for state 1 with alpha/beta hole/electron populations (UKS)."""
    mulliken = AtomicCharges(
        method="Mulliken",
        charges={0: -0.1, 1: 0.05, 2: 0.05},
        hole_populations_alpha={0: 0.6, 1: 0.2, 2: 0.2},
        hole_populations_beta={0: 0.3, 1: 0.1, 2: 0.1},
        electron_populations_alpha={0: 0.1, 1: 0.3, 2: 0.3},
        electron_populations_beta={0: 0.05, 1: 0.15, 2: 0.15},
    )
    return UnrelaxedDensityMatrix(
        state_number=1,
        nos_spin_traced=_NOS,
        mulliken=mulliken,
        molecular_charge=0.0,
        num_electrons=10.0,
        dipole_moment_debye=1.5,
        dipole_components_debye=(0.0, 0.0, 1.5),
        exciton_total=_EXCITON,
    )


@pytest.fixture
def transition_dm_s2() -> TransitionDensityMatrix:
    """Transition DM for state 2 with trans_charges and del_q."""
    mulliken = AtomicCharges(
        method="Mulliken",
        charges={0: 0.0, 1: 0.0, 2: 0.0},
        trans_charges={0: -0.3, 1: 0.15, 2: 0.15},
        del_q={0: 0.1, 1: -0.05, 2: -0.05},
    )
    return TransitionDensityMatrix(state_number=2, exciton_total=_EXCITON, mulliken=mulliken)


@pytest.fixture
def adc_state_3() -> AdcExcitedState:
    """ADC excited state 3 with difference-DM Mulliken charges."""
    mulliken = AtomicCharges(
        method="Mulliken",
        charges={0: -0.2, 1: 0.1, 2: 0.1},
        hole_populations={0: 0.7, 1: 0.15, 2: 0.15},
        electron_populations={0: 0.3, 1: 0.35, 2: 0.35},
    )
    return AdcExcitedState(
        state_number=3,
        total_energy_au=-76.0,
        excitation_energy_ev=4.5,
        mulliken=mulliken,
    )


def _make_tddft_result(
    h2o_geometry: Geometry,
    unrelaxed: list[UnrelaxedDensityMatrix] | None = None,
    transition: list[TransitionDensityMatrix] | None = None,
) -> CalculationResult:
    from calcflow.common.results import TddftResults

    tddft = TddftResults(
        unrelaxed_density_matrices=unrelaxed or None,
        transition_density_matrices=transition or None,
    )
    return _make_result(final_geometry=h2o_geometry, atomic_charges=[], final_energy=-76.0)._replace_tddft(tddft)  # type: ignore[attr-defined]


# AnnotatedResult helper that injects tddft/adc directly via dataclass replace.
def _result_with_tddft(base: CalculationResult, tddft: "TddftResults") -> CalculationResult:  # noqa: F821
    import dataclasses

    return dataclasses.replace(base, tddft=tddft)


def _result_with_adc(base: CalculationResult, adc: AdcResults) -> CalculationResult:
    import dataclasses

    return dataclasses.replace(base, adc=adc)


# =============================================================================
# AnnotatedStateView / get_unrelaxed_state
# =============================================================================


@pytest.mark.unit
def test_get_unrelaxed_state_returns_none_when_no_tddft(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.get_unrelaxed_state(1) is None


@pytest.mark.unit
def test_get_unrelaxed_state_returns_none_for_missing_state(
    h2o_geometry: Geometry, unrelaxed_dm_s1: UnrelaxedDensityMatrix
):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.get_unrelaxed_state(99) is None


@pytest.mark.unit
def test_get_unrelaxed_state_returns_view(h2o_geometry: Geometry, unrelaxed_dm_s1: UnrelaxedDensityMatrix):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    view = ag.get_unrelaxed_state(1)
    assert isinstance(view, AnnotatedStateView)
    assert len(view) == 3


@pytest.mark.unit
def test_unrelaxed_state_atom_hole_electron_populations(
    h2o_geometry: Geometry, unrelaxed_dm_s1: UnrelaxedDensityMatrix
):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    view = ag.get_unrelaxed_state(1)

    atom0 = view[0]
    assert isinstance(atom0, AnnotatedStateAtom)
    assert atom0.symbol == "O"
    assert atom0.index == 0
    assert atom0.charge == pytest.approx(-0.1)
    assert atom0.hole_population == pytest.approx(0.8)
    assert atom0.electron_population == pytest.approx(0.2)
    # transition DM fields not populated
    assert atom0.trans_charge is None
    assert atom0.del_q is None


@pytest.mark.unit
def test_unrelaxed_state_atom_uks_alpha_beta(h2o_geometry: Geometry, unrelaxed_dm_s1_uks: UnrelaxedDensityMatrix):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1_uks])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    view = ag.get_unrelaxed_state(1)

    atom0 = view[0]
    assert atom0.hole_population_alpha == pytest.approx(0.6)
    assert atom0.hole_population_beta == pytest.approx(0.3)
    assert atom0.electron_population_alpha == pytest.approx(0.1)
    assert atom0.electron_population_beta == pytest.approx(0.05)
    # scalar hole_population is None (only alpha/beta set)
    assert atom0.hole_population is None


@pytest.mark.unit
def test_unrelaxed_state_view_negative_index(h2o_geometry: Geometry, unrelaxed_dm_s1: UnrelaxedDensityMatrix):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    view = ag.get_unrelaxed_state(1)

    atom = view[-1]
    assert atom.index == 2
    assert atom.hole_population == pytest.approx(0.1)


@pytest.mark.unit
def test_unrelaxed_state_view_out_of_range(h2o_geometry: Geometry, unrelaxed_dm_s1: UnrelaxedDensityMatrix):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    view = ag.get_unrelaxed_state(1)

    with pytest.raises(IndexError):
        _ = view[99]


@pytest.mark.unit
def test_unrelaxed_state_view_iter(h2o_geometry: Geometry, unrelaxed_dm_s1: UnrelaxedDensityMatrix):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(unrelaxed_density_matrices=[unrelaxed_dm_s1])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    view = ag.get_unrelaxed_state(1)

    atoms = list(view)
    assert len(atoms) == 3
    assert [a.index for a in atoms] == [0, 1, 2]


# =============================================================================
# get_transition_state
# =============================================================================


@pytest.mark.unit
def test_get_transition_state_returns_none_when_no_tddft(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.get_transition_state(2) is None


@pytest.mark.unit
def test_get_transition_state_returns_view(h2o_geometry: Geometry, transition_dm_s2: TransitionDensityMatrix):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(transition_density_matrices=[transition_dm_s2])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    view = ag.get_transition_state(2)
    assert isinstance(view, AnnotatedStateView)

    atom0 = view[0]
    assert atom0.trans_charge == pytest.approx(-0.3)
    assert atom0.del_q == pytest.approx(0.1)
    # unrelaxed DM fields not populated
    assert atom0.hole_population is None
    assert atom0.electron_population is None


@pytest.mark.unit
def test_get_transition_state_returns_none_for_missing_state(
    h2o_geometry: Geometry, transition_dm_s2: TransitionDensityMatrix
):
    from calcflow.common.results import TddftResults

    tddft = TddftResults(transition_density_matrices=[transition_dm_s2])
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_tddft(base, tddft)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.get_transition_state(99) is None


# =============================================================================
# get_adc_state
# =============================================================================


@pytest.mark.unit
def test_get_adc_state_returns_none_when_no_adc(h2o_geometry: Geometry):
    result = _make_result(final_geometry=h2o_geometry)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.get_adc_state(3) is None


@pytest.mark.unit
def test_get_adc_state_returns_view(h2o_geometry: Geometry, adc_state_3: AdcExcitedState):
    adc = AdcResults(
        method="adc(2)",
        ground_state=AdcGroundState(hf_energy_au=-75.0, mp2_correlation_energy_au=-0.3, total_energy_au=-75.3),
        excited_states=[adc_state_3],
    )
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_adc(base, adc)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)

    view = ag.get_adc_state(3)
    assert isinstance(view, AnnotatedStateView)

    atom0 = view[0]
    assert atom0.charge == pytest.approx(-0.2)
    assert atom0.hole_population == pytest.approx(0.7)
    assert atom0.electron_population == pytest.approx(0.3)


@pytest.mark.unit
def test_get_adc_state_returns_none_for_missing_state(h2o_geometry: Geometry, adc_state_3: AdcExcitedState):
    adc = AdcResults(
        method="adc(2)",
        ground_state=AdcGroundState(hf_energy_au=-75.0, mp2_correlation_energy_au=-0.3, total_energy_au=-75.3),
        excited_states=[adc_state_3],
    )
    base = _make_result(final_geometry=h2o_geometry)
    result = _result_with_adc(base, adc)
    ag = AnnotatedGeometry(geometry=h2o_geometry, result=result)
    assert ag.get_adc_state(99) is None


# =============================================================================
# AnnotatedStateView with no mulliken data
# =============================================================================


@pytest.mark.unit
def test_state_view_no_mulliken_returns_empty_atom(h2o_geometry: Geometry):
    """TransitionDensityMatrix.mulliken is optional — missing data should yield empty atom."""
    tdm = TransitionDensityMatrix(state_number=1, exciton_total=_EXCITON, mulliken=None)
    view = AnnotatedStateView(geometry=h2o_geometry, state_data=tdm)

    atom = view[0]
    assert isinstance(atom, AnnotatedStateAtom)
    assert atom.symbol == "O"
    assert atom.charge is None
    assert atom.trans_charge is None
