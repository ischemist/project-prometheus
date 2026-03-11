"""
unit tests for detect_molecules(), classify_bond(), and the enriched ELEMENT_DATA.

all tests are pure-logic unit tests — no file I/O, inline atom fixtures only.
coordinates are in Angstrom, consistent with calcflow's Atom convention.
"""

import pytest

from calcflow.common.exceptions import ConfigurationError
from calcflow.common.results import Atom
from calcflow.constants.ptable import ELEMENT_DATA
from calcflow.geometry.topology import classify_bond, detect_molecules

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def a(symbol: str, x: float, y: float, z: float) -> Atom:
    """shorthand for building Atom fixtures."""
    return Atom(symbol=symbol, x=x, y=y, z=z)


# ---------------------------------------------------------------------------
# ELEMENT_DATA sanity checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestElementData:
    def test_all_118_elements_present(self):
        assert len(ELEMENT_DATA) == 118

    def test_hydrogen(self):
        h = ELEMENT_DATA["H"]
        assert h.atomic_number == 1
        assert h.symbol == "H"
        assert h.covalent_radius_single_pm == 32
        assert h.covalent_radius_double_pm is None
        assert h.covalent_radius_triple_pm is None
        assert h.atomic_mass == pytest.approx(1.008, rel=1e-3)
        assert h.atomic_radius_vdw_pm == 120

    def test_carbon(self):
        c = ELEMENT_DATA["C"]
        assert c.atomic_number == 6
        assert c.covalent_radius_single_pm == 75
        assert c.covalent_radius_double_pm == 67
        assert c.covalent_radius_triple_pm == 60

    def test_helium_no_double_triple(self):
        he = ELEMENT_DATA["HE"]
        assert he.covalent_radius_single_pm == 46
        assert he.covalent_radius_double_pm is None
        assert he.covalent_radius_triple_pm is None

    def test_oxygen(self):
        o = ELEMENT_DATA["O"]
        assert o.covalent_radius_single_pm == 63
        assert o.covalent_radius_double_pm == 57
        assert o.covalent_radius_triple_pm == 53

    def test_keys_are_uppercase(self):
        for key in ELEMENT_DATA:
            assert key == key.upper(), f"key {key!r} is not uppercase"

    def test_all_elements_have_atomic_number(self):
        for key, elem in ELEMENT_DATA.items():
            assert isinstance(elem.atomic_number, int), f"{key} missing atomic_number"

    def test_all_elements_have_r1(self):
        # every element in Pyykkö 2009 has at least r1
        for key, elem in ELEMENT_DATA.items():
            assert elem.covalent_radius_single_pm is not None, f"{key} missing r1"

    def test_element_is_frozen(self):
        h = ELEMENT_DATA["H"]
        with pytest.raises((AttributeError, TypeError)):
            h.atomic_number = 99  # type: ignore[misc]

    @pytest.mark.parametrize("symbol,z", [("H", 1), ("C", 6), ("N", 7), ("O", 8), ("AU", 79), ("OG", 118)])
    def test_atomic_numbers_correct(self, symbol: str, z: int):
        assert ELEMENT_DATA[symbol].atomic_number == z


# ---------------------------------------------------------------------------
# detect_molecules — basic cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectMolecules:
    def test_empty_returns_empty(self):
        assert detect_molecules([]) == []

    def test_single_atom_returns_one_fragment(self):
        result = detect_molecules([a("He", 0.0, 0.0, 0.0)])
        assert result == [frozenset({0})]

    def test_single_water_molecule(self):
        # O-H bond ~0.96 Å, well within threshold
        atoms = [
            a("O", 0.000, 0.000, 0.117),
            a("H", 0.000, 0.757, -0.469),
            a("H", 0.000, -0.757, -0.469),
        ]
        result = detect_molecules(atoms)
        assert result == [frozenset({0, 1, 2})]

    def test_water_dimer_gives_two_fragments(self):
        # two H2O molecules separated by ~3 Å (non-bonding distance)
        atoms = [
            a("O", 0.000, 0.000, 0.117),  # mol 1
            a("H", 0.000, 0.757, -0.469),
            a("H", 0.000, -0.757, -0.469),
            a("O", 5.000, 0.000, 0.117),  # mol 2
            a("H", 5.000, 0.757, -0.469),
            a("H", 5.000, -0.757, -0.469),
        ]
        result = detect_molecules(atoms)
        assert len(result) == 2
        assert frozenset({0, 1, 2}) in result
        assert frozenset({3, 4, 5}) in result

    def test_co2_is_one_molecule(self):
        # C=O bond ~1.16 Å
        atoms = [
            a("C", 0.000, 0.000, 0.000),
            a("O", 1.160, 0.000, 0.000),
            a("O", -1.160, 0.000, 0.000),
        ]
        result = detect_molecules(atoms)
        assert result == [frozenset({0, 1, 2})]

    def test_two_isolated_hydrogen_atoms(self):
        # 10 Å apart — clearly non-bonding
        atoms = [a("H", 0.0, 0.0, 0.0), a("H", 10.0, 0.0, 0.0)]
        result = detect_molecules(atoms)
        assert len(result) == 2
        assert frozenset({0}) in result
        assert frozenset({1}) in result

    def test_two_bonded_hydrogen_atoms(self):
        # H2 bond ~0.74 Å
        atoms = [a("H", 0.0, 0.0, 0.0), a("H", 0.74, 0.0, 0.0)]
        result = detect_molecules(atoms)
        assert result == [frozenset({0, 1})]

    def test_ethanol_is_one_molecule(self):
        # C-C ~1.54, C-O ~1.43, C-H ~1.09, O-H ~0.96
        atoms = [
            a("C", 0.000, 0.000, 0.000),
            a("C", 1.540, 0.000, 0.000),
            a("O", 2.100, 1.200, 0.000),
            a("H", -0.390, 1.030, 0.000),
            a("H", -0.390, -0.515, 0.891),
            a("H", -0.390, -0.515, -0.891),
            a("H", 1.930, -1.030, 0.000),
            a("H", 1.930, 0.515, -0.891),
            a("H", 2.490, 1.200, 0.890),
        ]
        result = detect_molecules(atoms)
        assert len(result) == 1
        assert result[0] == frozenset(range(9))

    def test_sorted_by_smallest_index(self):
        # three isolated atoms — result must be in index order
        atoms = [
            a("He", 0.0, 0.0, 0.0),
            a("Ne", 100.0, 0.0, 0.0),
            a("Ar", 200.0, 0.0, 0.0),
        ]
        result = detect_molecules(atoms)
        assert result == [frozenset({0}), frozenset({1}), frozenset({2})]

    def test_returns_list_of_frozensets(self):
        atoms = [a("O", 0.0, 0.0, 0.0), a("H", 0.96, 0.0, 0.0)]
        result = detect_molecules(atoms)
        assert isinstance(result, list)
        for fragment in result:
            assert isinstance(fragment, frozenset)

    def test_all_atom_indices_covered(self):
        # every atom index must appear in exactly one fragment
        atoms = [
            a("O", 0.0, 0.0, 0.0),
            a("H", 0.96, 0.0, 0.0),
            a("H", -0.96, 0.0, 0.0),
            a("N", 10.0, 0.0, 0.0),
            a("H", 10.96, 0.0, 0.0),
        ]
        result = detect_molecules(atoms)
        all_indices = set().union(*result)
        assert all_indices == set(range(len(atoms)))

    def test_no_index_appears_twice(self):
        atoms = [
            a("C", 0.0, 0.0, 0.0),
            a("C", 1.54, 0.0, 0.0),
            a("C", 20.0, 0.0, 0.0),
        ]
        result = detect_molecules(atoms)
        all_indices = [idx for fragment in result for idx in fragment]
        assert len(all_indices) == len(set(all_indices))


# ---------------------------------------------------------------------------
# detect_molecules — tolerance parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectMoleculesTolerance:
    def test_zero_tolerance_misses_stretched_bond(self):
        # O-H stretched to 1.5 Å: r1_O=0.63, r1_H=0.32, sum=0.95
        # with tolerance=0 threshold is 0.95 Å — below 1.5, so not bonded
        atoms = [a("O", 0.0, 0.0, 0.0), a("H", 1.5, 0.0, 0.0)]
        result = detect_molecules(atoms, tolerance=0.0)
        assert len(result) == 2

    def test_large_tolerance_connects_distant_atoms(self):
        # same atoms but tolerance=2.0 → threshold = 0.95 * 3.0 = 2.85 Å > 1.5
        atoms = [a("O", 0.0, 0.0, 0.0), a("H", 1.5, 0.0, 0.0)]
        result = detect_molecules(atoms, tolerance=2.0)
        assert result == [frozenset({0, 1})]


# ---------------------------------------------------------------------------
# detect_molecules — error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectMoleculesErrors:
    def test_unknown_element_raises_configuration_error(self):
        # Force an element with no entry: patch ELEMENT_DATA temporarily
        # by using a symbol that parses as valid Atom but has no radius.
        # We test this by monkeypatching ELEMENT_DATA.
        import calcflow.geometry.topology as topo

        original = topo.ELEMENT_DATA.copy()
        try:
            # remove H from the lookup so detect_molecules can't find its radius
            patched = {k: v for k, v in original.items() if k != "H"}
            topo.ELEMENT_DATA = patched  # type: ignore[assignment]
            with pytest.raises(ConfigurationError, match="no single-bond covalent radius"):
                detect_molecules([a("H", 0.0, 0.0, 0.0)])
        finally:
            topo.ELEMENT_DATA = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# classify_bond
# ---------------------------------------------------------------------------

# Reference radii (Pyykkö 2009, pm → Å):
#   C: r1=0.75, r2=0.67, r3=0.60
#   N: r1=0.71, r2=0.60, r3=0.54
#   O: r1=0.63, r2=0.57, r3=0.53
#   H: r1=0.32  (no r2/r3)

# typical bond lengths (Å):
#   C-C single ~1.54, C=C double ~1.34, C≡C triple ~1.20
#   C-N single ~1.47, C=N double ~1.27, C≡N triple ~1.16
#   C-H ~1.09, O-H ~0.96


@pytest.mark.unit
class TestClassifyBond:
    def test_cc_single(self):
        # C-C thresholds (tol=0.2): triple=1.44, double=1.608, single=1.80
        # 1.65 Å: above triple (1.44) and double (1.608) thresholds → single
        assert classify_bond(a("C", 0, 0, 0), a("C", 1.65, 0, 0), 1.65) == "single"

    def test_cc_double(self):
        # C-C thresholds (tol=0.2): triple=1.44, double=1.608, single=1.80
        # 1.50 Å: above triple threshold (1.44) but below double (1.608) → double
        assert classify_bond(a("C", 0, 0, 0), a("C", 1.50, 0, 0), 1.50) == "double"

    def test_cc_triple(self):
        # 1.20 Å: triple threshold = (0.60+0.60)*1.2 = 1.44 > 1.20 → triple
        assert classify_bond(a("C", 0, 0, 0), a("C", 1.20, 0, 0), 1.20) == "triple"

    def test_ch_single(self):
        # C-H: r1 sum = 0.75+0.32 = 1.07, threshold = 1.07*1.2 = 1.284 > 1.09 → single
        # H has no r2/r3, so only single candidate exists
        assert classify_bond(a("C", 0, 0, 0), a("H", 1.09, 0, 0), 1.09) == "single"

    def test_oh_single(self):
        # O-H: r1 sum = 0.63+0.32 = 0.95, threshold = 1.14 > 0.96 → single
        assert classify_bond(a("O", 0, 0, 0), a("H", 0.96, 0, 0), 0.96) == "single"

    def test_cn_triple(self):
        # C≡N: r3 sum = 0.60+0.54 = 1.14, threshold = 1.368 > 1.16 → triple
        assert classify_bond(a("C", 0, 0, 0), a("N", 1.16, 0, 0), 1.16) == "triple"

    def test_returns_none_for_nonbonding_distance(self):
        # 5.0 Å C-C: way beyond any threshold
        assert classify_bond(a("C", 0, 0, 0), a("C", 5.0, 0, 0), 5.0) == "none"

    def test_returns_none_just_above_single_threshold(self):
        # C-C r1 threshold = 1.50 * 1.2 = 1.80; use 1.85
        assert classify_bond(a("C", 0, 0, 0), a("C", 1.85, 0, 0), 1.85) == "none"

    def test_prefers_higher_order_when_distance_fits_multiple(self):
        # at exactly r3 threshold, triple should win over double and single
        c = a("C", 0, 0, 0)
        # r3_C + r3_C = 1.20 Å; any dist <= 1.20*1.2=1.44 should be triple
        assert classify_bond(c, a("C", 1.20, 0, 0), 1.20) == "triple"

    def test_element_with_no_r2_r3_only_returns_single_or_none(self):
        # H has no r2 or r3 — result must be "single" or "none", never "double"/"triple"
        result = classify_bond(a("H", 0, 0, 0), a("H", 0.74, 0, 0), 0.74)
        assert result in ("single", "none")
        assert result == "single"  # H2 bond length 0.74 is well within threshold

    def test_symmetric(self):
        # classify_bond(i, j, d) == classify_bond(j, i, d)
        c = a("C", 0, 0, 0)
        n = a("N", 1.16, 0, 0)
        assert classify_bond(c, n, 1.16) == classify_bond(n, c, 1.16)

    def test_custom_tolerance(self):
        # C-C at 1.54 with tolerance=0.0: r1 threshold = 1.50, so 1.54 > 1.50 → none
        assert classify_bond(a("C", 0, 0, 0), a("C", 1.54, 0, 0), 1.54, tolerance=0.0) == "none"
        # same distance with tolerance=0.1: threshold = 1.65 > 1.54 → single
        assert classify_bond(a("C", 0, 0, 0), a("C", 1.54, 0, 0), 1.54, tolerance=0.1) == "single"

    def test_raises_configuration_error_for_unknown_element(self):
        import calcflow.geometry.topology as topo

        original = topo.ELEMENT_DATA.copy()
        try:
            patched = {k: v for k, v in original.items() if k != "C"}
            topo.ELEMENT_DATA = patched  # type: ignore[assignment]
            with pytest.raises(ConfigurationError, match="no single-bond covalent radius"):
                classify_bond(a("C", 0, 0, 0), a("H", 1.09, 0, 0), 1.09)
        finally:
            topo.ELEMENT_DATA = original  # type: ignore[assignment]
