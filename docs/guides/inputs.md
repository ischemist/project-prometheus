---
icon: lucide/file-pen
---

# Building Calculation Inputs

`CalculationInput` describes *what* you want — method, basis, task, solvation, excited states — and the program-specific builders translate that into valid input syntax for Q-Chem or ORCA.

## Construction

`CalculationInput` requires four fields:

```python
from calcflow import CalculationInput

calc = CalculationInput(
    charge=0,
    spin_multiplicity=1,
    task="energy",            # "energy" | "geometry" | "frequency"
    level_of_theory="wB97X-D3",
    basis_set="def2-TZVP",
)
```

Every other field is optional and defaults to a sensible value. Use the fluent setter methods to build up the spec incrementally.

!!! tip "Fluent setters return new instances"
    Every `.set_*()` method returns a **new** `CalculationInput`. The original is never modified. This means you can safely branch a base spec into multiple variants.

## Level of Theory and Basis Set

```python
calc = calc.set_level_of_theory("PBE0")
calc = calc.set_basis_set("def2-TZVP")
```

### Element-specific basis sets

Pass a dict to assign different basis sets to different elements (Q-Chem only):

```python
calc = calc.set_basis({"C": "def2-TZVP", "H": "def2-SVP", "O": "def2-TZVP"})
```

## Task Types

| Task | Description |
| :--- | :--- |
| `"energy"` | Single-point energy |
| `"geometry"` | Geometry optimization |
| `"frequency"` | Harmonic frequency analysis |

```python
calc = calc.set_task("geometry")
```

### Geometry optimization settings

```python
calc = calc.set_optimization(
    calc_hess_initial=True,     # compute Hessian at start
    recalc_hess_freq=5,         # recalculate every N steps
)
```

### Frequency after optimization

To append a frequency calculation immediately after a geometry optimization (a common workflow), use:

```python
calc = calc.run_frequency_after_opt()
```

## Spin and Charge

```python
calc = CalculationInput(
    charge=-1,
    spin_multiplicity=2,   # 2 = doublet (one unpaired electron)
    task="energy",
    level_of_theory="B3LYP",
    basis_set="def2-SVP",
    unrestricted=True,     # UKS/UHF — required for open-shell
)
```

or via setter:

```python
calc = calc.set_unrestricted(True)
```

## Compute Resources

```python
calc = (
    calc
    .set_cores(16)
    .set_memory_per_core(4000)    # MB per core
)
```

For Q-Chem's `MEM_TOTAL` keyword (total memory rather than per-core):

```python
calc = calc.set_total_memory(32000)  # 32 GB total
```

## TDDFT

```python
calc = calc.set_tddft(
    nroots=15,          # number of excited states to compute
    singlets=True,
    triplets=False,
    use_tda=False,      # True for TDA (Tamm-Dancoff approximation)
)
```

### Excited-state geometry optimization

To optimize the geometry on a specific excited state:

```python
calc = calc.set_tddft(nroots=5, singlets=True, triplets=False, state_to_optimize=1)
calc = calc.set_task("geometry")
```

## Solvation

### Named solvent (SMD, CPCM)

```python
calc = calc.set_solvation(model="smd", solvent="water")
calc = calc.set_solvation(model="cpcm", solvent="acetonitrile")
```

Supported models: `"smd"`, `"pcm"`, `"cpcm"`, `"isosvp"`, `"cosmo"`.

Common solvent names: `"water"`, `"acetonitrile"`, `"methanol"`, `"ethanol"`, `"dichloromethane"`, `"thf"`, `"toluene"`, `"acetone"`, `"dmso"`.

### Custom dielectric (PCM)

```python
calc = calc.set_custom_solvation(
    model="pcm",
    dielectric=78.39,          # static dielectric
    optical_dielectric=1.78,   # optical (high-frequency) dielectric
)
```

## Atomic Charge Analysis

```python
calc = calc.set_charges(
    mulliken=True,
    hirshfeld=True,
    cm5=True,            # implies hirshfeld=True
    hirshiter=False,     # Hirshfeld-I (iterative)
)
```

!!! info "CM5 implies Hirshfeld"
    Setting `cm5=True` automatically enables Hirshfeld charges, since CM5 is derived from the Hirshfeld population analysis.

## SCF Convergence

```python
calc = calc.set_scf(
    algorithm="diis",    # "diis" | "gdm" | "rca" | "rca_diis"
    max_cycles=200,
    convergence=8,       # convergence threshold: 10^-N
)
```

## Maximum Overlap Method (MOM)

MOM is used in Q-Chem to target specific electronic configurations — most commonly for ΔSCF excited-state calculations.

```python
calc = CalculationInput(
    charge=0,
    spin_multiplicity=1,
    task="energy",
    level_of_theory="wB97X-D3",
    basis_set="def2-TZVP",
    unrestricted=True,
)

calc = calc.set_mom(
    transition="HOMO->LUMO",   # symbolic transition
    method="imom",             # "mom" or "imom"
)
```

### Transition syntax

| Syntax | Meaning |
| :--- | :--- |
| `"HOMO->LUMO"` | Promote from HOMO to LUMO |
| `"HOMO-1->LUMO+2"` | Relative orbital indices |
| `"HOMO->vac"` | Ionization (remove from HOMO) |
| `"HOMO_A->LUMO_A"` | Spin-specific (alpha) |
| `"HOMO_B->LUMO_B"` | Spin-specific (beta) |
| `"HOMO->LUMO; HOMO-1->LUMO+1"` | Multiple transitions (semicolon-separated) |

### Two-job MOM structure

MOM in Q-Chem uses a two-job structure: the first job computes the ground-state reference, and the second job runs with MOM constraints. CalcFlow generates both jobs from a single spec:

```python
calc = calc.set_mom(
    transition="HOMO->LUMO",
    method="imom",
    job2_charge=0,
    job2_spin_multiplicity=1,
)
input_text = calc.export("qchem", geom)  # contains @@@-separated jobs
```

## RI Approximations (ORCA)

```python
calc = calc.enable_ri_for_orca(
    approx="rijcosx",            # "ri" | "rijcosx" | "rijonx"
    aux_basis="def2/J",
)
```

## Program-Specific Options

For options not covered by the standard API, use `set_options()`. Keys are stored in `program_options` and translated by the builder:

```python
calc = calc.set_options(
    parallelism="mpi",     # Q-Chem: MPI instead of OpenMP (default: "openmp")
)
```

Note: `enable_ri_for_orca()` and some other setters are thin wrappers around `set_options()` internally.

## Exporting

```python
from pathlib import Path
from calcflow import Geometry

geom = Geometry.from_xyz_file("molecule.xyz")

qchem_input = calc.export("qchem", geom)
orca_input  = calc.export("orca", geom)

Path("molecule.inp").write_text(qchem_input)
```

## Serialization

Save a spec to JSON for reproducibility and reload it later:

```python
from pathlib import Path
from calcflow.common.input import CalculationInput

# Save
Path("calc_spec.json").write_text(calc.to_json())

# Reload
calc2 = CalculationInput.from_json(Path("calc_spec.json").read_text())
```

## Runtime API Reference

```python
# Full method catalogue with signatures and docstrings
print(CalculationInput.get_api_docs())

# Compact quick-reference
print(CalculationInput.get_quick_ref())

# Documentation for a specific method
print(CalculationInput.get_method_docs("set_tddft"))
```
