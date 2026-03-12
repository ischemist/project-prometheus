---
icon: lucide/atom
---

# Geometry and Topology

CalcFlow provides a layered geometry API: a simple `Geometry` for XYZ data, a `Trajectory` for multi-frame sequences, and an `AnnotatedGeometry` that co-locates atom coordinates with calculation results.

## Geometry

`Geometry` is an immutable collection of atoms with a comment line — the canonical representation of a molecular structure.

### Loading from file

```python
from calcflow import Geometry

geom = Geometry.from_xyz_file("molecule.xyz")

print(geom.num_atoms)         # int
print(geom.unique_elements)   # set[str] — e.g. {"C", "H", "O"}
```

### Loading from strings

```python
lines = [
    "O    0.000000    0.000000    0.119748",
    "H    0.000000    0.756950   -0.478993",
    "H    0.000000   -0.756950   -0.478993",
]
geom = Geometry.from_lines(lines, source="water")
```

### Accessing atoms

```python
for atom in geom.atoms:
    print(f"{atom.symbol:2s}  {atom.x:10.6f}  {atom.y:10.6f}  {atom.z:10.6f}")
```

`Atom` fields: `symbol` (str), `x`, `y`, `z` (float).

### Exporting

```python
xyz_str = geom.to_xyz_str()
geom.to_xyz_file("output.xyz")

# Just the atom coordinate block (no count line or comment)
block = geom.get_coordinate_block()
```

### Energy from comment line

ORCA writes the energy into the comment line of optimized geometry XYZ files. CalcFlow parses this automatically:

```python
geom = Geometry.from_xyz_file("orca_opt.xyz")
print(geom.energy)   # float | None
```

## Trajectory

`Trajectory` wraps a sequence of `Geometry` frames from a multi-frame XYZ file (e.g. an optimization trajectory).

```python
from calcflow import Trajectory

traj = Trajectory.from_xyz_file("optimization.xyz")

print(len(traj))          # number of frames
first = traj[0]           # Geometry
last  = traj[-1]          # Geometry

for frame in traj:
    print(frame.energy)   # energy per frame if present in comment line
```

All frames in a `Trajectory` must have the same number and order of atoms — this is validated on construction.

## AnnotatedGeometry

`AnnotatedGeometry` is a read-only view that pairs a `Geometry` with a `CalculationResult`. It gives atom-centric access to charges, spin densities, and excited-state populations without duplicating data.

```python
from pathlib import Path
from calcflow import parse_qchem_output, AnnotatedGeometry

result = parse_qchem_output(Path("calculation.out").read_text())
ag = AnnotatedGeometry.from_result(result)
```

### Per-atom ground-state properties

```python
for i, atom in enumerate(ag):
    print(f"Atom {i} ({atom.symbol}):")
    print(f"  Mulliken charge: {atom.charges.get('Mulliken'):+.4f}")
    print(f"  Hirshfeld charge: {atom.charges.get('Hirshfeld'):+.4f}")
    if atom.spins:
        print(f"  Mulliken spin: {atom.spins.get('Mulliken'):+.4f}")
```

`atom.charges` and `atom.spins` are dicts keyed by method name (`"Mulliken"`, `"Hirshfeld"`, `"CM5"`, `"Loewdin"`).

### Using the input geometry

By default `AnnotatedGeometry.from_result()` uses the final (optimized) geometry. To use the input geometry instead:

```python
ag = AnnotatedGeometry.from_result(result, use_input_geometry=True)
```

### Excited-state views

For TDDFT results, you can get a per-state view that exposes hole/electron populations and transition charges per atom:

```python
sv = ag.get_transition_state(state_number=1)   # AnnotatedStateView | None
if sv:
    for i, atom in enumerate(sv):
        print(f"Atom {i} ({atom.symbol}):")
        print(f"  hole population:     {atom.hole_population:.4f}")
        print(f"  electron population: {atom.electron_population:.4f}")
        print(f"  transition charge:   {atom.transition_charge:.4f}")
```

For unrelaxed difference density matrix data:

```python
sv = ag.get_unrelaxed_state(state_number=1)
if sv:
    for i, atom in enumerate(sv):
        print(f"Atom {i}: Δq = {atom.delta_charge:.4f}")
```

For ADC states:

```python
sv = ag.get_adc_state(state_number=1)
```

## AnnotatedTrajectory

`AnnotatedTrajectory` is a sequence of `AnnotatedGeometry` frames — useful for tracking how charges or populations evolve along a geometry optimization when you have a separate `CalculationResult` per step.

```python
from calcflow import AnnotatedTrajectory

# results: list[CalculationResult], one per optimization step
at = AnnotatedTrajectory.from_results(results)

for frame in at:
    print(frame[0].charges.get("Mulliken"))   # Mulliken charge on atom 0 at this step
```

## Topology

CalcFlow includes bond detection, molecule identification, and aromaticity analysis based on covalent radii from Pyykko (2009).

```python
from calcflow import build_bond_graph, detect_molecules, classify_all_bonds, find_aromatic_atoms

atoms = geom.atoms
```

### Bond graph

```python
bond_graph = build_bond_graph(atoms, tolerance=0.4)
# bond_graph[i] = list of atom indices bonded to atom i
```

### Molecule detection

Useful for multi-component systems (e.g. solute + counter-ions):

```python
molecules = detect_molecules(atoms, tolerance=0.4)
# molecules = list of frozensets, each containing the atom indices of one molecule

for i, mol in enumerate(molecules):
    symbols = [atoms[j].symbol for j in sorted(mol)]
    print(f"Fragment {i}: {''.join(symbols)}")
```

### Bond classification

```python
bonds = classify_all_bonds(atoms, tolerance=0.4)
# bonds = dict mapping (i, j) tuple -> "single" | "double" | "triple" | "aromatic"

for (i, j), order in bonds.items():
    print(f"Atoms {i}-{j}: {order}")
```

### Aromatic atoms

```python
aromatic = find_aromatic_atoms(atoms, bond_graph)
# aromatic = set of atom indices participating in aromatic rings
```

## Element Data

CalcFlow bundles a periodic table (all 118 elements) with atomic masses, van der Waals radii, and Pyykko covalent radii. The topology functions use this internally, but you can access it directly:

```python
from calcflow.constants.ptable import ELEMENT_DATA

carbon = ELEMENT_DATA["C"]
carbon.atomic_number              # 6
carbon.atomic_mass               # 12.011
carbon.atomic_radius_vdw_pm      # van der Waals radius in pm
carbon.covalent_radius_single_pm # single-bond covalent radius (Pyykko 2009)
carbon.covalent_radius_double_pm
carbon.covalent_radius_triple_pm
```
