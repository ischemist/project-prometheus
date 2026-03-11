"""Molecular topology utilities: bond detection and fragment identification."""

import math
from collections.abc import Sequence

from calcflow.common.exceptions import ConfigurationError
from calcflow.common.results import Atom
from calcflow.constants.ptable import ELEMENT_DATA


def _find(parent: list[int], i: int) -> int:
    """path-compressing union-find root lookup."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # path compression
        i = parent[i]
    return i


def _union(parent: list[int], i: int, j: int) -> None:
    parent[_find(parent, i)] = _find(parent, j)


def detect_molecules(
    atoms: Sequence[Atom],
    tolerance: float = 0.4,
) -> list[frozenset[int]]:
    """Partition atoms into molecules using covalent bond detection.

    Two atoms are considered bonded when their distance is less than
    (r1_i + r1_j) * (1 + tolerance), where r1 is the Pyykkö 2009
    single-bond covalent radius. tolerance=0.4 (40% slack) is the
    conventional value used by ASE, RDKit, etc.

    Args:
        atoms: sequence of Atom objects (0-based indexed).
        tolerance: fractional slack added to the sum of covalent radii.

    Returns:
        list of frozenset[int], one per molecule, sorted by smallest atom index.
        Each frozenset contains the 0-based indices of atoms in that molecule.

    Raises:
        ConfigurationError: if an element has no single-bond covalent radius.
    """
    n = len(atoms)
    if n == 0:
        return []

    # pre-fetch radii in Å for each atom
    radii_ang: list[float] = []
    for atom in atoms:
        key = atom.symbol.upper()
        element = ELEMENT_DATA.get(key)
        if element is None or element.covalent_radius_single_pm is None:
            raise ConfigurationError(
                f"no single-bond covalent radius for element '{atom.symbol}' — "
                "cannot detect bonds. check ELEMENT_DATA in calcflow.constants.ptable."
            )
        radii_ang.append(element.covalent_radius_single_pm / 100.0)

    # union-find: each atom starts as its own component
    parent = list(range(n))

    for i in range(n):
        ai = atoms[i]
        for j in range(i + 1, n):
            aj = atoms[j]
            dx = ai.x - aj.x
            dy = ai.y - aj.y
            dz = ai.z - aj.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            threshold = (radii_ang[i] + radii_ang[j]) * (1.0 + tolerance)
            if dist < threshold:
                _union(parent, i, j)

    # collect components
    components: dict[int, set[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        components.setdefault(root, set()).add(i)

    # return sorted by smallest atom index in each fragment
    return sorted([frozenset(s) for s in components.values()], key=min)


BondOrder = str  # "single" | "double" | "triple" | "none"


def classify_bond(atom_i: Atom, atom_j: Atom, dist_ang: float, tolerance: float = 0.2) -> BondOrder:
    """Classify the bond order between two atoms given their distance.

    Compares dist_ang against the Pyykkö 2009 reference radii (r1/r2/r3) for
    each atom. The order whose combined radius is closest to dist_ang and within
    tolerance wins. Returns "none" if the distance exceeds all thresholds.

    tolerance here is tighter than detect_molecules (0.2 vs 0.4) because we are
    distinguishing between bond orders, not just detecting existence.

    Args:
        atom_i: first atom.
        atom_j: second atom.
        dist_ang: distance between the two atoms in Angstrom.
        tolerance: fractional slack around each reference radius sum.

    Returns:
        "triple", "double", "single", or "none".

    Raises:
        ConfigurationError: if either element has no single-bond covalent radius.
    """
    ei = ELEMENT_DATA.get(atom_i.symbol.upper())
    ej = ELEMENT_DATA.get(atom_j.symbol.upper())

    for label, elem in (("atom_i", ei), ("atom_j", ej)):
        if elem is None or elem.covalent_radius_single_pm is None:
            sym = atom_i.symbol if label == "atom_i" else atom_j.symbol
            raise ConfigurationError(
                f"no single-bond covalent radius for element '{sym}' — "
                "cannot classify bond. check ELEMENT_DATA in calcflow.constants.ptable."
            )

    # build (order_label, combined_radius_ang) pairs, highest order first
    # so the first match wins (prefer triple over double over single)
    candidates: list[tuple[BondOrder, float]] = []
    for order, r_attr in (("triple", "covalent_radius_triple_pm"), ("double", "covalent_radius_double_pm"), ("single", "covalent_radius_single_pm")):  # fmt: skip
        ri = getattr(ei, r_attr)
        rj = getattr(ej, r_attr)
        if ri is not None and rj is not None:
            candidates.append((order, (ri + rj) / 100.0))

    for order, ref in candidates:
        if dist_ang <= ref * (1.0 + tolerance):
            return order

    return "none"
