"""Molecular topology utilities: bond detection, fragment identification, and bond classification."""

import math
from collections.abc import Sequence
from typing import Literal, cast

from calcflow.common.exceptions import ConfigurationError
from calcflow.common.models import Atom
from calcflow.constants.ptable import ELEMENT_DATA

# Elements that can participate in aromatic rings
_AROMATIC_ELEMENTS = frozenset({"C", "N", "O", "S"})

BondOrder = Literal["single", "double", "triple", "aromatic"]


# =============================================================================
# internal helpers
# =============================================================================


def _find(parent: list[int], i: int) -> int:
    """Return the path-compressed union-find root for index ``i``."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: list[int], i: int, j: int) -> None:
    """Union the components containing indices ``i`` and ``j``."""
    parent[_find(parent, i)] = _find(parent, j)


def _covalent_r1_ang(atom: Atom) -> float:
    """Returns single-bond covalent radius in Å, raises ConfigurationError if missing."""
    key = atom.symbol.upper()
    element = ELEMENT_DATA.get(key)
    if element is None or element.covalent_radius_single_pm is None:
        raise ConfigurationError(
            f"no single-bond covalent radius for element '{atom.symbol}' — "
            "cannot detect bonds. check ELEMENT_DATA in calcflow.constants.ptable."
        )
    return element.covalent_radius_single_pm / 100.0


def _dist(ai: Atom, aj: Atom) -> float:
    """Return Euclidean distance between two atoms in Angstrom."""
    dx, dy, dz = ai.x - aj.x, ai.y - aj.y, ai.z - aj.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _find_small_cycles(graph: dict[int, list[int]], max_size: int = 7) -> list[frozenset[int]]:
    """Find all simple cycles up to ``max_size`` using DFS.

    Returns:
        A deduplicated list of cycle atom sets.
    """
    cycles: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()

    for start in graph:
        # DFS stack: (current_node, path_so_far)
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in graph.get(node, []):
                if nb == start and len(path) >= 3:
                    key = frozenset(path)
                    if key not in seen:
                        seen.add(key)
                        cycles.append(key)
                elif nb not in path and len(path) < max_size:
                    stack.append((nb, path + [nb]))

    return cycles


def _aromatic_rings(atoms: Sequence[Atom], bond_graph: dict[int, list[int]]) -> list[frozenset[int]]:
    """Return aromatic rings that pass all structural criteria.

    Criteria include allowed elements, ring size (5/6), planarity, and shortened
    bond lengths consistent with aromatic delocalization.
    """
    aromatic_graph: dict[int, list[int]] = {}
    for i, atom in enumerate(atoms):
        if atom.symbol.upper() in _AROMATIC_ELEMENTS:
            aromatic_graph[i] = [j for j in bond_graph.get(i, []) if atoms[j].symbol.upper() in _AROMATIC_ELEMENTS]

    all_cycles = _find_small_cycles(aromatic_graph, max_size=6)
    small_cycles = [c for c in all_cycles if len(c) in (5, 6)]
    if not small_cycles:
        return []

    rings = _sssr(small_cycles)
    aromatic_rings: list[frozenset[int]] = []
    for ring in rings:
        indices = list(ring)
        ring_atoms = [atoms[i] for i in indices]

        if not _is_planar(ring_atoms):
            continue

        ordered = _order_ring(indices, aromatic_graph)
        if ordered is None:
            continue

        if _bonds_shortened(ordered, atoms, bond_graph):
            aromatic_rings.append(ring)

    return aromatic_rings


def _sssr(cycles: list[frozenset[int]]) -> list[frozenset[int]]:
    """Keep the smallest set of smallest rings from cycle candidates.

    A candidate is discarded when its atom set is exactly the union of two smaller
    already-kept rings.
    """
    sorted_cycles = cast(list[frozenset[int]], sorted(cycles, key=len))
    kept: list[frozenset[int]] = []
    for candidate in sorted_cycles:
        # check if candidate can be formed by merging any two smaller kept rings
        redundant = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if kept[i] | kept[j] == candidate:
                    redundant = True
                    break
            if redundant:
                break
        if not redundant:
            kept.append(candidate)
    return kept


def _is_planar(ring_atoms: list[Atom], threshold_ang: float = 0.1) -> bool:
    """Returns True if all atoms are within threshold_ang of their least-squares plane.

    Uses Newell's method to compute the plane normal from the polygon vertices,
    then checks the max deviation of any atom from that plane.
    """
    n = len(ring_atoms)
    if n < 3:
        return True

    # centroid
    cx = sum(a.x for a in ring_atoms) / n
    cy = sum(a.y for a in ring_atoms) / n
    cz = sum(a.z for a in ring_atoms) / n

    # Newell normal: sum of cross products of consecutive edge pairs
    nx = ny = nz = 0.0
    for i in range(n):
        curr = ring_atoms[i]
        nxt = ring_atoms[(i + 1) % n]
        nx += (curr.y - cy) * (nxt.z - cz) - (curr.z - cz) * (nxt.y - cy)
        ny += (curr.z - cz) * (nxt.x - cx) - (curr.x - cx) * (nxt.z - cz)
        nz += (curr.x - cx) * (nxt.y - cy) - (curr.y - cy) * (nxt.x - cx)

    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-10:
        return True  # degenerate — treat as planar
    nx /= norm
    ny /= norm
    nz /= norm

    # max deviation from plane
    for a in ring_atoms:
        dev = abs((a.x - cx) * nx + (a.y - cy) * ny + (a.z - cz) * nz)
        if dev > threshold_ang:
            return False
    return True


def _bonds_shortened(ring_indices: list[int], atoms: Sequence[Atom], graph: dict[int, list[int]]) -> bool:
    """Returns True if every bond in the ring is shorter than the single-bond threshold
    for that element pair (i.e., all bonds show at least partial multiple-bond character).

    This distinguishes aromatic rings (bonds ~1.36–1.44 Å) from saturated rings
    like cyclohexane (bonds ~1.54 Å).
    """
    n = len(ring_indices)
    for k in range(n):
        i = ring_indices[k]
        j = ring_indices[(k + 1) % n]
        if j not in graph.get(i, []):
            return False  # ring bond doesn't exist in graph
        ai, aj = atoms[i], atoms[j]
        d = _dist(ai, aj)
        # threshold: sum of r1 radii (no tolerance — we want genuinely shortened bonds)
        r1_sum = _covalent_r1_ang(ai) + _covalent_r1_ang(aj)
        if d >= r1_sum:
            return False
    return True


# =============================================================================
# public API
# =============================================================================


def build_bond_graph(
    atoms: Sequence[Atom],
    tolerance: float = 0.4,
) -> dict[int, list[int]]:
    """Build an adjacency list of covalent bonds.

    Two atoms are bonded when their distance is less than
    (r1_i + r1_j) * (1 + tolerance).

    Args:
        atoms: sequence of Atom objects (0-based indexed).
        tolerance: fractional slack on the sum of single-bond covalent radii.

    Returns:
        dict mapping each atom index to a list of bonded atom indices.

    Raises:
        ConfigurationError: if an element has no single-bond covalent radius.
    """
    n = len(atoms)
    radii = [_covalent_r1_ang(a) for a in atoms]
    graph: dict[int, list[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(atoms[i], atoms[j])
            if d < (radii[i] + radii[j]) * (1.0 + tolerance):
                graph[i].append(j)
                graph[j].append(i)

    return graph


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

    graph = build_bond_graph(atoms, tolerance)

    parent = list(range(n))
    for i in range(n):
        for j in graph[i]:
            if j > i:
                _union(parent, i, j)

    components: dict[int, set[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        components.setdefault(root, set()).add(i)

    return sorted([frozenset(s) for s in components.values()], key=min)


def find_aromatic_atoms(
    atoms: Sequence[Atom],
    bond_graph: dict[int, list[int]],
) -> frozenset[int]:
    """Identify atoms that are members of aromatic rings.

    A ring is considered aromatic if:
    - Size is 5 or 6
    - All atoms are C, N, O, or S
    - The ring is planar (all atoms within 0.1 Å of their least-squares plane)
    - All ring bonds are shorter than the sum of single-bond covalent radii
      (i.e., every bond has partial multiple-bond character)

    Uses SSSR (smallest set of smallest rings) to avoid counting fused
    macrocycles as aromatic.

    Args:
        atoms: sequence of Atom objects.
        bond_graph: adjacency list from build_bond_graph().

    Returns:
        frozenset of 0-based atom indices that belong to at least one aromatic ring.
    """
    aromatic_indices: set[int] = set()
    for ring in _aromatic_rings(atoms, bond_graph):
        aromatic_indices.update(ring)
    return frozenset(aromatic_indices)


def _order_ring(indices: list[int], graph: dict[int, list[int]]) -> list[int] | None:
    """Return ring atom indices in walk order, or ``None`` when not walkable."""
    index_set = set(indices)
    start = indices[0]
    ordered = [start]
    visited = {start}

    current = start
    for _ in range(len(indices) - 1):
        neighbors_in_ring = [nb for nb in graph.get(current, []) if nb in index_set and nb not in visited]
        if not neighbors_in_ring:
            return None
        current = neighbors_in_ring[0]
        ordered.append(current)
        visited.add(current)

    return ordered


def classify_bond(
    atom_i: Atom,
    atom_j: Atom,
    dist_ang: float,
    tolerance: float = 0.2,
) -> BondOrder | None:
    """Classify the bond order between two atoms given their distance.

    Compares dist_ang against the Pyykkö 2009 reference radii (r1/r2/r3) for
    each atom. The order whose combined radius is closest to dist_ang and within
    tolerance wins. Returns None if the distance exceeds all thresholds.

    Note: this function is purely geometric and has no knowledge of aromaticity.
    Aromatic bonds (e.g. benzene C-C at ~1.40 Å) will be reported as "double"
    or "triple" since they fall within those distance thresholds. Use
    classify_all_bonds() for aromaticity-aware classification.

    tolerance here is tighter than detect_molecules (0.2 vs 0.4) because we are
    distinguishing between bond orders, not just detecting existence.

    Args:
        atom_i: first atom.
        atom_j: second atom.
        dist_ang: distance between the two atoms in Angstrom.
        tolerance: fractional slack around each reference radius sum.

    Returns:
        "triple", "double", "single", or None if distance exceeds all thresholds.

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

    candidates: list[tuple[str, float]] = []
    for order, r_attr in (
        ("triple", "covalent_radius_triple_pm"),
        ("double", "covalent_radius_double_pm"),
        ("single", "covalent_radius_single_pm"),
    ):
        ri = getattr(ei, r_attr)
        rj = getattr(ej, r_attr)
        if ri is not None and rj is not None:
            candidates.append((order, (ri + rj) / 100.0))

    for order, ref in candidates:
        if dist_ang <= ref * (1.0 + tolerance):
            return order  # type: ignore[return-value]

    return None


def classify_all_bonds(
    atoms: Sequence[Atom],
    tolerance: float = 0.4,
) -> dict[tuple[int, int], BondOrder]:
    """Classify all covalent bonds in a set of atoms, with aromaticity detection.

    Full pipeline: build bond graph → detect aromatic rings via SSSR + planarity
    → classify each bond, substituting "aromatic" for bonds inside aromatic rings.

    Args:
        atoms: sequence of Atom objects (0-based indexed).
        tolerance: fractional slack for bond detection (passed to build_bond_graph).

    Returns:
        dict mapping (i, j) pairs (i < j) to bond order strings:
        "single", "double", "triple", "aromatic". Non-bonded pairs are absent.

    Raises:
        ConfigurationError: if an element has no single-bond covalent radius.
    """
    graph = build_bond_graph(atoms, tolerance)
    aromatic_rings = _aromatic_rings(atoms, graph)
    aromatic = frozenset(i for ring in aromatic_rings for i in ring)

    result: dict[tuple[int, int], BondOrder] = {}
    for i in range(len(atoms)):
        for j in graph[i]:
            if j <= i:
                continue
            if i in aromatic and j in aromatic:
                # verify they're in the same ring, not just both aromatic atoms
                # connected by a non-aromatic bond (e.g. biphenyl C-C bridge)
                order = _aromatic_or_geometric(atoms, i, j, aromatic_rings, tolerance)
            else:
                order = classify_bond(atoms[i], atoms[j], _dist(atoms[i], atoms[j]), tolerance=0.2)
            if order is not None:
                result[(i, j)] = order

    return result


def _aromatic_or_geometric(
    atoms: Sequence[Atom],
    i: int,
    j: int,
    aromatic_rings: list[frozenset[int]],
    tolerance: float,
) -> BondOrder | None:
    """For a bond between two aromatic atoms, return 'aromatic' only if they
    share a common aromatic ring. Otherwise fall back to geometric classification.

    This handles inter-ring bonds in biaryl systems (e.g. biphenyl C1-C1' bond
    connects two aromatic rings but is itself a single bond).
    """
    for ring in aromatic_rings:
        if i in ring and j in ring:
            return "aromatic"

    # both atoms are aromatic but this bond bridges two separate rings
    return classify_bond(atoms[i], atoms[j], _dist(atoms[i], atoms[j]), tolerance=0.2)
