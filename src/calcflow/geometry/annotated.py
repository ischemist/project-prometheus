"""Runtime views that co-locate atomic positions with calculation results.

AnnotatedAtom, AnnotatedGeometry, and AnnotatedTrajectory are read-only
views — no serialization, no magic. They exist purely to make the most
common access pattern ergonomic: "what is this atom's charge and where
is it in space?"

Design notes:
- Not FrozenModel subclasses: no serialization needed. Use
  CalculationResult.to_json() and Geometry.to_xyz_str() separately.
- AnnotatedGeometry holds geometry + result explicitly. from_result() is
  the ergonomic constructor; direct construction is also fine for when
  the geometry comes from an external source (e.g. an XYZ file).
- Only ground-state per-atom scalars appear on AnnotatedAtom.charges/spins.
  Excited-state per-atom data lives on contextual state views obtained via
  ag.get_unrelaxed_state(n), ag.get_transition_state(n), ag.get_adc_state(n).
  The geometry is the invariant spatial scaffolding; properties belong to
  the state, not the atom.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

from calcflow.common.exceptions import ValidationError
from calcflow.common.results import (
    AdcExcitedState,
    AdcResults,
    AtomicCharges,
    CalculationResult,
    TddftResults,
    TransitionDensityMatrix,
    UnrelaxedDensityMatrix,
)
from calcflow.geometry.static import Geometry

# Union of all state-level data types that AnnotatedStateView can wrap.
StateData = UnrelaxedDensityMatrix | TransitionDensityMatrix | AdcExcitedState


def _is_ground_state(ac: AtomicCharges) -> bool:
    """Returns True if this AtomicCharges entry is from a ground-state population analysis.

    Excited-state entries (from TDDFT unrelaxed DM or transition DM analysis) carry
    hole/electron populations or transition charges. Ground-state entries have none of
    these fields set.
    """
    return (
        ac.hole_populations is None
        and ac.electron_populations is None
        and ac.hole_populations_alpha is None
        and ac.hole_populations_beta is None
        and ac.electron_populations_alpha is None
        and ac.electron_populations_beta is None
        and ac.trans_charges is None
        and ac.del_q is None
    )


@dataclass(frozen=True)
class AnnotatedAtom:
    """A single atom with its Cartesian coordinates and ground-state per-atom properties.

    Built on demand by AnnotatedGeometry.__getitem__. Excited-state per-atom data
    is not included here — access it via the parent AnnotatedGeometry.tddft / .adc.

    Attributes:
        symbol: element symbol (capitalized, e.g. "C").
        x: Cartesian x coordinate in Angstrom.
        y: Cartesian y coordinate in Angstrom.
        z: Cartesian z coordinate in Angstrom.
        index: 0-based position of this atom in the geometry.
        charges: ground-state charge per population analysis method, e.g. {"Mulliken": -0.42}.
        spins: ground-state spin density per method (UKS only); empty dict for RKS.
    """

    symbol: str
    x: float
    y: float
    z: float
    index: int
    charges: Mapping[str, float]
    spins: Mapping[str, float]


@dataclass(frozen=True)
class AnnotatedStateAtom:
    """A single atom viewed through the lens of a specific excited state.

    Built on demand by AnnotatedStateView.__getitem__. Fields are populated
    only for the DM type providing the context — all others are None:
    - Unrelaxed DM / ADC difference DM: charge, spin, hole_population,
      electron_population (+ alpha/beta UKS variants).
    - Transition DM: trans_charge, del_q.

    Attributes:
        symbol: element symbol (capitalized, e.g. "C").
        x: Cartesian x coordinate in Angstrom.
        y: Cartesian y coordinate in Angstrom.
        z: Cartesian z coordinate in Angstrom.
        index: 0-based position of this atom in the geometry.
        charge: difference-density Mulliken charge for this state.
        spin: difference-density spin density (UKS only).
        hole_population: hole population h+ (RKS unrelaxed DM / ADC).
        electron_population: electron population e- (RKS unrelaxed DM / ADC).
        hole_population_alpha: alpha h+ (UKS only).
        hole_population_beta: beta h+ (UKS only).
        electron_population_alpha: alpha e- (UKS only).
        electron_population_beta: beta e- (UKS only).
        trans_charge: transition charge (transition DM only).
        del_q: del q value (transition DM only).
    """

    symbol: str
    x: float
    y: float
    z: float
    index: int
    charge: float | None = None
    spin: float | None = None
    hole_population: float | None = None
    electron_population: float | None = None
    hole_population_alpha: float | None = None
    hole_population_beta: float | None = None
    electron_population_alpha: float | None = None
    electron_population_beta: float | None = None
    trans_charge: float | None = None
    del_q: float | None = None


def _state_atom_from_mulliken(
    symbol: str, x: float, y: float, z: float, resolved: int, ac: AtomicCharges | None
) -> AnnotatedStateAtom:
    """Builds an AnnotatedStateAtom from a Mulliken AtomicCharges entry.

    Handles both unrelaxed/ADC difference DM fields and transition DM fields,
    populating only what is present on the given AtomicCharges object.

    Args:
        symbol: element symbol for this atom.
        x: Cartesian x coordinate in Angstrom.
        y: Cartesian y coordinate in Angstrom.
        z: Cartesian z coordinate in Angstrom.
        resolved: normalized 0-based atom index. Negative indices must be converted
            to positive by the caller before passing (e.g. -1 → 2 for a 3-atom
            geometry). Used as the key into all AtomicCharges mappings.
        ac: the AtomicCharges entry to project onto this atom, or None if unavailable.
    """
    if ac is None:
        return AnnotatedStateAtom(symbol=symbol, x=x, y=y, z=z, index=resolved)

    def _get(mapping: Mapping[int, float] | None) -> float | None:
        return mapping.get(resolved) if mapping is not None else None

    return AnnotatedStateAtom(
        symbol=symbol,
        x=x,
        y=y,
        z=z,
        index=resolved,
        charge=_get(ac.charges),
        spin=_get(ac.spins),
        hole_population=_get(ac.hole_populations),
        electron_population=_get(ac.electron_populations),
        hole_population_alpha=_get(ac.hole_populations_alpha),
        hole_population_beta=_get(ac.hole_populations_beta),
        electron_population_alpha=_get(ac.electron_populations_alpha),
        electron_population_beta=_get(ac.electron_populations_beta),
        trans_charge=_get(ac.trans_charges),
        del_q=_get(ac.del_q),
    )


@dataclass(frozen=True)
class AnnotatedStateView:
    """Binds a geometry to a specific excited state's data for atom-centric access.

    The geometry is the invariant spatial scaffolding. State-specific per-atom
    properties (hole/electron populations, transition charges, etc.) are projected
    onto atoms on demand via __getitem__.

    Attributes:
        geometry: the molecular geometry (positions + atom symbols).
        state_data: one of UnrelaxedDensityMatrix, TransitionDensityMatrix,
            or AdcExcitedState — the source of excited-state per-atom data.
    """

    geometry: Geometry
    state_data: StateData

    def __len__(self) -> int:
        return len(self.geometry.atoms)

    def __getitem__(self, index: int) -> AnnotatedStateAtom:
        atom = self.geometry.atoms[index]  # raises IndexError naturally for out-of-range
        resolved = index if index >= 0 else len(self.geometry.atoms) + index
        ac = self.state_data.mulliken  # all three StateData types have .mulliken
        return _state_atom_from_mulliken(atom.symbol, atom.x, atom.y, atom.z, resolved, ac)

    def __iter__(self) -> Iterator[AnnotatedStateAtom]:
        for i in range(len(self)):
            yield self[i]


@dataclass(frozen=True)
class AnnotatedGeometry:
    """A geometry paired with its calculation result for atom-centric data access.

    The primary constructor is from_result(), which selects the appropriate geometry
    from the result. Direct construction is also supported for cases where the geometry
    comes from an external source.

    Attributes:
        geometry: the molecular geometry (positions + atom symbols).
        result: the full calculation result, source of all property data.
    """

    geometry: Geometry
    result: CalculationResult

    @classmethod
    def from_result(cls, result: CalculationResult, *, use_input_geometry: bool = False) -> "AnnotatedGeometry":
        """Constructs an AnnotatedGeometry from a CalculationResult.

        Selects geometry in this order:
        - use_input_geometry=False (default): final_geometry, falling back to input_geometry.
        - use_input_geometry=True: input_geometry only.

        Args:
            result: the calculation result containing geometry and property data.
            use_input_geometry: if True, use input_geometry even when final_geometry exists.

        Returns:
            A new AnnotatedGeometry instance.

        Raises:
            ValidationError: if the requested geometry is not present on the result.
        """
        if use_input_geometry:
            geom = result.input_geometry
            if geom is None:
                raise ValidationError("AnnotatedGeometry.from_result(): result has no input_geometry.")
        else:
            geom = result.final_geometry or result.input_geometry
            if geom is None:
                raise ValidationError(
                    "AnnotatedGeometry.from_result(): result has neither final_geometry nor input_geometry."
                )
        return cls(geometry=geom, result=result)

    def __len__(self) -> int:
        return len(self.geometry.atoms)

    def __getitem__(self, index: int) -> AnnotatedAtom:
        atom = self.geometry.atoms[index]  # raises IndexError naturally for out-of-range
        # Resolve index to 0-based (handles negative indexing from atoms[index])
        resolved = index if index >= 0 else len(self.geometry.atoms) + index
        charges: dict[str, float] = {}
        spins: dict[str, float] = {}
        for ac in self.result.atomic_charges:
            if not _is_ground_state(ac):
                continue
            if resolved in ac.charges:
                charges[ac.method] = ac.charges[resolved]
            if ac.spins is not None and resolved in ac.spins:
                spins[ac.method] = ac.spins[resolved]
        return AnnotatedAtom(
            symbol=atom.symbol,
            x=atom.x,
            y=atom.y,
            z=atom.z,
            index=resolved,
            charges=charges,
            spins=spins,
        )

    def __iter__(self) -> Iterator[AnnotatedAtom]:
        for i in range(len(self)):
            yield self[i]

    def get_charges(self, method: str) -> AtomicCharges | None:
        """Returns the AtomicCharges for the given method, or None.

        Delegates to result.get_charges(). Method names are case-sensitive.
        """
        return self.result.get_charges(method)

    @cached_property
    def energy(self) -> float | None:
        """The final energy from the calculation result, in Hartree."""
        return self.result.final_energy

    @property
    def tddft(self) -> TddftResults | None:
        """TDDFT results, including excited-state per-atom data."""
        return self.result.tddft

    @property
    def adc(self) -> AdcResults | None:
        """ADC results, including excited-state per-atom data."""
        return self.result.adc

    def get_unrelaxed_state(self, state_number: int) -> AnnotatedStateView | None:
        """Returns a state view enriched with unrelaxed DM per-atom data, or None.

        Searches result.tddft.unrelaxed_density_matrices by state_number (1-based).
        Returns None if TDDFT results are absent or the requested state is not found.

        Args:
            state_number: 1-based excited state index, as in the output file.
        """
        if not self.result.tddft or not self.result.tddft.unrelaxed_density_matrices:
            return None
        state = next((s for s in self.result.tddft.unrelaxed_density_matrices if s.state_number == state_number), None)
        return AnnotatedStateView(self.geometry, state) if state else None

    def get_transition_state(self, state_number: int) -> AnnotatedStateView | None:
        """Returns a state view enriched with transition DM per-atom data, or None.

        Searches result.tddft.transition_density_matrices by state_number (1-based).
        Returns None if TDDFT results are absent or the requested state is not found.

        Args:
            state_number: 1-based excited state index, as in the output file.
        """
        if not self.result.tddft or not self.result.tddft.transition_density_matrices:
            return None
        state = next((s for s in self.result.tddft.transition_density_matrices if s.state_number == state_number), None)
        return AnnotatedStateView(self.geometry, state) if state else None

    def get_adc_state(self, state_number: int) -> AnnotatedStateView | None:
        """Returns a state view enriched with ADC excited-state per-atom data, or None.

        Searches result.adc.excited_states by state_number (1-based).
        Returns None if ADC results are absent or the requested state is not found.

        Args:
            state_number: 1-based excited state index, as in the output file.
        """
        if not self.result.adc:
            return None
        state = next((s for s in self.result.adc.excited_states if s.state_number == state_number), None)
        return AnnotatedStateView(self.geometry, state) if state else None


@dataclass(frozen=True)
class AnnotatedTrajectory:
    """A sequence of AnnotatedGeometry frames with consistent atom counts.

    Validates that all frames have the same number of atoms on construction.
    The primary constructor is from_results().

    Attributes:
        frames: tuple of AnnotatedGeometry, one per calculation.
    """

    frames: tuple[AnnotatedGeometry, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            return
        first = self.frames[0].geometry.num_atoms
        if not all(f.geometry.num_atoms == first for f in self.frames):
            raise ValidationError("AnnotatedTrajectory: inconsistent number of atoms across frames.")

    @classmethod
    def from_results(
        cls,
        results: Sequence[CalculationResult],
        *,
        use_input_geometry: bool = False,
    ) -> "AnnotatedTrajectory":
        """Constructs an AnnotatedTrajectory from a sequence of CalculationResults.

        Each result is converted to an AnnotatedGeometry via from_result(). All frames
        missing a geometry are collected and reported in a single ValidationError.

        Args:
            results: sequence of CalculationResult, one per trajectory frame.
            use_input_geometry: if True, use input_geometry on every frame.

        Returns:
            A new AnnotatedTrajectory instance.

        Raises:
            ValidationError: if results is empty, if any frames lack geometry, or if
                atom counts are inconsistent across frames.
        """
        if not results:
            raise ValidationError("AnnotatedTrajectory.from_results(): results sequence is empty.")

        missing: list[int] = []
        frames: list[AnnotatedGeometry] = []
        for i, result in enumerate(results):
            try:
                frames.append(AnnotatedGeometry.from_result(result, use_input_geometry=use_input_geometry))
            except ValidationError:
                missing.append(i)

        if missing:
            raise ValidationError(f"AnnotatedTrajectory.from_results(): frames at indices {missing} have no geometry.")

        return cls(frames=tuple(frames))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> AnnotatedGeometry:
        return self.frames[index]

    def __iter__(self) -> Iterator[AnnotatedGeometry]:
        return iter(self.frames)
