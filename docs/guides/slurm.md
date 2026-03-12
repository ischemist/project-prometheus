---
icon: lucide/server
---

# SLURM Job Scripts

`SlurmJob` generates complete SLURM submission scripts for running Q-Chem or ORCA jobs on HPC clusters. It uses the same `CalculationInput` spec to derive program-specific directives — no need to manually translate core counts or memory limits into `#SBATCH` lines.

## The SlurmJob Object

`SlurmJob` is a frozen dataclass with a fluent setter API, just like `CalculationInput`.

```python
from calcflow.slurm import SlurmJob

job = SlurmJob(
    job_name="h2o_tddft",
    time="04:00:00",          # wall time — HH:MM:SS
    n_cores=16,
    memory_mb=64000,          # total node memory in MB
    partition="cpu",
    account="mygroup",
)
```

Required fields: `job_name`, `time`, `n_cores`, `memory_mb`, `partition`.
Optional fields: `constraint`, `account`, `queue`, `modules`.

## Adding Environment Modules

Most HPC clusters use environment modules to load software. Specify them as a list:

```python
job = job.add_modules(["qchem/6.2", "intel/2023"])
```

`.add_modules()` appends to the existing module list — useful for building up from a base job:

```python
base_job = SlurmJob(
    job_name="base",
    time="02:00:00",
    n_cores=8,
    memory_mb=32000,
    partition="cpu",
    modules=["intel/2023", "openmpi/4.1"],
)

qchem_job = base_job.add_modules(["qchem/6.2"])
orca_job  = base_job.add_modules(["orca/5.0"])
```

## Generating a Script

Pass a `CalculationInput` and the target program to `.export()`:

```python
from calcflow import CalculationInput, Geometry

calc = CalculationInput(
    charge=0,
    spin_multiplicity=1,
    task="energy",
    level_of_theory="wB97X-D3",
    basis_set="def2-TZVP",
).set_tddft(nroots=10, singlets=True, triplets=False).set_cores(16)

geom = Geometry.from_xyz_file("molecule.xyz")

script = job.export(
    calculation=calc,
    program="qchem",
    input_filename="molecule.inp",
    output_filename="molecule.out",
)

print(script)
```

The generated script includes:

- `#SBATCH` directives derived from the `SlurmJob` fields
- `module load` commands for each entry in `modules`
- The program-specific launch command (e.g. `qchem -np 16 molecule.inp molecule.out`)
- Any program-specific directives the builder exposes (e.g. OpenMP thread settings)

## Full Example: Q-Chem TDDFT Workflow

```python
from calcflow import CalculationInput, Geometry
from calcflow.slurm import SlurmJob

geom = Geometry.from_xyz_file("meoq.xyz")

# Calculation spec
calc = (
    CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wB97X-D3",
        basis_set="def2-TZVP",
    )
    .set_tddft(nroots=15, singlets=True, triplets=False)
    .set_solvation(model="smd", solvent="water")
    .set_charges(mulliken=True, hirshfeld=True, cm5=True)
    .set_cores(32)
    .set_memory_per_core(3000)
)

# SLURM job
job = SlurmJob(
    job_name="meoq_tddft",
    time="08:00:00",
    n_cores=32,
    memory_mb=128000,
    partition="cpu",
    account="chemistry",
    modules=["qchem/6.2"],
)

# Generate input and submission script
input_text = calc.export("qchem", geom)
script     = job.export(calc, "qchem", "meoq.inp", "meoq.out")

with open("meoq.inp", "w") as f:
    f.write(input_text)

with open("submit.sh", "w") as f:
    f.write(script)
```

Submit with `sbatch submit.sh`.

## ORCA Example

```python
orca_calc = (
    CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wB97X-D3",
        basis_set="def2-TZVP",
    )
    .set_tddft(nroots=10, singlets=True, triplets=False)
    .set_cores(8)
)

orca_job = SlurmJob(
    job_name="meoq_orca",
    time="04:00:00",
    n_cores=8,
    memory_mb=32000,
    partition="cpu",
    modules=["orca/5.0.4", "openmpi/4.1"],
)

input_text = orca_calc.export("orca", geom)
script     = orca_job.export(orca_calc, "orca", "meoq.inp", "meoq.out")
```

## Saving the Spec for Reproducibility

```python
# Save the calculation spec alongside the job script
with open("calc_spec.json", "w") as f:
    f.write(calc.to_json())
```

Reload later with `CalculationInput.from_json(open("calc_spec.json").read())` — so you always know exactly what was run.

## Partition and Constraint

For clusters with heterogeneous partitions:

```python
job = SlurmJob(
    job_name="gpu_tddft",
    time="02:00:00",
    n_cores=8,
    memory_mb=32000,
    partition="gpu",
    constraint="a100",     # node feature constraint
)
```
