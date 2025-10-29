# CalcFlow: Quantum Chemistry Calculation I/O Done Right

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![coverage](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/ischemist/project-prometheus/master/coverage.json&query=$.totals.percent_covered_display&label=coverage&color=brightgreen&suffix=%25)

CalcFlow provides a robust, Pythonic interface for preparing inputs and parsing outputs for quantum chemistry software like Q-Chem and ORCA. It has **zero external dependencies** and is built for clarity and reliability. Get your calculations set up and results processed without the usual boilerplate.

> [!WARNING]
> Package is in pre-release alpha stage. May introduce backwards-incompatible changes. Contributions & suggestions & advice are welcome.

## Key Features

what you care about:

- **Zero Dependencies**: is true now, will be true forever. Integrate into any project without worrying about whether it matches your numpy or rdkit version.
- **Intuitive Pythonic Interface**: specify calculation parameters how you think about them and calcflow will handle the translation into what QChem or ORCA expects.
- **Proactive Input Validation**: Minimizes runtime errors by rigorously validating all calculation parameters against both general and program-specific constraints.
- **Comprehensive Parsing**: stop wasting time on extracting data you need from a txt out file and start doing meaningful work (analysis).

what's nice under the hood:

- **Immutable & Fluent API**: ensures data integrity and predictable state transitions via `frozen` dataclasses and an expressive, chainable API.
- **Fully Type Annotated**: code is more robust and easier to understand.
- **Extensible Core Architecture**: designed with clear abstract base classes, simplifying the integration of support for additional quantum chemistry programs.
- **Assured Reliability via Comprehensive Testing**: so you don't have to worry if the parser is working correctly.

## Quick Start: Q-Chem TDDFT Example

Set up a Q-Chem tddft calculation for a water molecule with SMD solvation:

```python
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput

# 1. Define Molecular Geometry
atoms = [
    ("O", (0.000000, 0.000000, 0.117300)),
    ("H", (0.000000, 0.757200, -0.469200)),
    ("H", (0.000000, -0.757200, -0.469200)),
]
water_molecule = Geometry(atoms=atoms, comment="Water molecule for Q-Chem SP")
# or load from xyz file
water_molecule = Geometry.from_xyz_file(data_path / "1h2o.xyz")

# 2. Configure Q-Chem Calculation Input
base_job = QchemInput(
    charge=0, spin_multiplicity=1, task="energy",
    level_of_theory="wB97X-D3", basis_set="def2-tzvp", n_cores=16,
    run_tddft=True, tddft_nroots=10, tddft_singlets=True, tddft_triplets=False
)

# Uses a fluent API for modifications.
qchem_job = base_job.set_solvation(model="smd", solvent="water")

# 3. Export the Input File Content
with open("h2o_tddft.in", "w") as f:
    f.write(qchem_job.export_input_file(water_molecule))
```

Want to create another input file for a tddft with triplets or with state analysis? ezy.

```python
job2 = qchem_job.set_tddft(nroots=10, singlets=True, triplets=True, state_analysis=True)
with open("h2o_tddft.in", "w") as f:
    f.write(job2.export_input_file(water_molecule))
```

what about getting an O K-Edge XAS spectrum with an element-specific basis?

```python
job3 = (
    qchem_job.set_tddft(nroots=10, singlets=True, triplets=False, state_analysis=True)
    .set_basis({"H": "pc-2", "O": "pcX-2"})
    .set_reduced_excitation_space(initial_orbitals=[1])
)
with open("h2o_xas.in", "w") as f:
    f.write(job3.export_input_file(water_molecule))
```

By the way, this pythonic (arguably intuitive) method

```py
.set_reduced_excitation_space(initial_orbitals=[1])
```

translates into, take a guess

```
$rem
    TRNSS = TRUE
    TRTYPE = 3
    N_SOL = 1
$end

$solute
    1
$end
```

that was obvious, wasn't it. Honestly, a major reason why this package exists. Btw, if you want to get a XAS spectrum from S1 state, you can do that too:

```python
job4 = (
    qchem_job.set_tddft(nroots=10, singlets=True, triplets=False, state_analysis=True)
    .set_basis({"H": "pc-2", "O": "pcX-2"})
    .set_reduced_excitation_space(initial_orbitals=[1])
    .set_unrestricted()
    .enable_mom()
    .set_mom_transition("HOMO->LUMO")
)
with open("h2o_s1_xas.in", "w") as f:
    f.write(job4.export_input_file(water_molecule))
```

this will create a 2-job input file, initial SCF calculation followed by MOM with HOMO electron moved to LUMO and TDDFT from orbital 1 on top of that.

fun fact: because water_molecule is a Geometry instance, it has `.total_nuclear_charge` property, which is used to calculate indexes of HOMO and LUMO orbitals. You can specify same transition numerically `5->6` or even specify occupation numbers manually:

```py
.set_mom_occupation(alpha_occ="1 2 3 4 6", beta_occ="1 2 3 4 5")
# or
.set_mom_occupation(alpha_occ="1:4 6", beta_occ="1:4 5")
```

## Parsing Q-Chem Output Files

Once your Q-Chem calculation is complete, use `CalcFlow` parsers. Example for the last calculation

```python
from calcflow.parsers.qchem import parse_qchem_mom_output

# Replace with your actual file path
out_path = "h2o_qchem_sp.out"
mom_pc2 = parse_qchem_mom_output((clc_folder / "mom-smd-xas.out").read_text())
```

And just like that you have access to all relevant results.

```py
> mom_pc2.job2
CalculationData(method='src1-r1', basis='gen', status='NORMAL')
> print(mom_pc2.job2.scf)
ScfResults(status='Converged', energy=-76.53682225, n_iterations=9)
> print(mom_pc2.job2.tddft)
TddftResults(tda_states=10 states, tddft_states=None, excited_state_analyses=10 analyses, transition_dm_analyses=10 analyses, nto_analyses=10 analyses)
```

Say you want to get excitation energies and oscillator strenghts? Be my guest:

```py
eVs = [state.excitation_energy_ev for state in mom_pc2.job2.tddft.tda_states]
intens = [state.oscillator_strength for state in mom_pc2.job2.tddft.tda_states]
```

Mulliken populations for 3rd state?

```py
> mom_pc2.job2.tddft.transition_dm_analyses[2].mulliken
TransitionDMMulliken(
    populations=[
        TransitionDMAtomPopulation(atom_index=0, symbol='H', transition_charge_e=-0.000558, hole_charge_rks=None, electron_charge_rks=None, delta_charge_rks=None, hole_charge_alpha_uks=7.2e-05, hole_charge_beta_uks=7.2e-05, electron_charge_alpha_uks=-0.241416, electron_charge_beta_uks=-0.241421),
        TransitionDMAtomPopulation(atom_index=1, symbol='O', transition_charge_e=0.001113, hole_charge_rks=None, electron_charge_rks=None, delta_charge_rks=None, hole_charge_alpha_uks=0.499858, hole_charge_beta_uks=0.499855, electron_charge_alpha_uks=-0.021788, electron_charge_beta_uks=-0.021788),
        TransitionDMAtomPopulation(atom_index=2, symbol='H', transition_charge_e=-0.000555, hole_charge_rks=None, electron_charge_rks=None, delta_charge_rks=None, hole_charge_alpha_uks=7.2e-05, hole_charge_beta_uks=7.2e-05, electron_charge_alpha_uks=-0.236798, electron_charge_beta_uks=-0.23679)
        ],
    sum_abs_trans_charges_qta=0.002226, sum_sq_trans_charges_qt2=2e-06)
```

See [scripts/create-parse-qchem.py](scripts/create-parse-qchem.py) for more examples or to play with outputs used for tests (stored in [data/calculations/examples/qchem/](data/calculations/examples/qchem/))

### Which versions of QChem do you support?

It's tested on 6.2 and 5.4. Good news is that I'm intending to make it easy to adjust for version-specific printing. For example,

```txt
QChem 5.4. prints final SCF energy as
SCF   energy in the final basis set =      -75.3184602363
while 6.2. prints it as
SCF   energy =   -75.32080770

p.s. nevermind the difference in energy, 5.4. mistakenly prints SCF energy same as Total energy, which includes solvation terms
```

which is why the [calcflow/parsers/qchem/blocks/scf.py](src/calcflow/parsers/qchem/blocks/scf.py) block defines different patterns for different versions:

```py
PatternDefinition(field_name="scf_energy", required=True, description="Final SCF energy value",
versioned_patterns=[
        (re.compile(r"^\s*SCF\s+energy\s*=\s*(-?\d+\.\d+)"), "6.2", lambda m: float(m.group(1))),
        (re.compile(r"^\s*SCF\s+energy in the final basis set\s*=\s*(-?\d+\.\d+)"), "5.4", lambda m: float(m.group(1))),
    ],
),
```

## Slurm submission scripts

As a bonus, there's also a pythonic way of preparing slurm submission files. Most likely you have system-specific variations in which modules should be loaded or which env variables should be defined, so you could inherit from a general SlurmArgs (in [calcflow/inputs/slurm.py](src/calcflow/inputs/slurm.py)):

```py
from dataclasses import dataclass

from calcflow.inputs.slurm import SlurmArgs


@dataclass(frozen=True)
class NerscSlurmArgs(SlurmArgs):
    def get_modules(self) -> str:
        if self.software == "qchem":
            return """
module load qchem
            """
        elif self.software == "orca":
            return """
module load openmpi
            """
        return ""

    def get_temp_variables(self) -> str:
        if self.software == "qchem":
            return f"""
export QCSCRATCH=$PSCRATCH
export QCLOCALSCR=$QCSCRATCH
export OMP_NUM_THREADS={self.n_cores}
export QC_THREADS={self.n_cores}
        """
        return ""
```

and then in a very similar manner you can create custom configs:

```py
nersc_args = NerscSlurmArgs(
    exec_fname="mom-sp", time="01:00:00", n_cores=16,
    constraint="cpu", account="mxxxx", queue="regular")
nersc_args = nersc_args.set_parallelism('openmp')
with (clc_folder / "submit.sh").open("w") as f:
    f.write(nersc_args.set_software("qchem").create_submit_script(f"{name}-{state}-{mom_type}"))
```

where `.set_software` dictates which modules/temp variables will be printed, and arg to `create_submit_script` is the job name.

## Contributing

Direct, effective contributions are welcome. Fork, modify, test, and pull request. Adhere to existing quality standards.

## Related Works

> How is it different from [QCEngine](https://github.com/MolSSI/QCEngine)?

- QCEngine requires [QCElemental](https://github.com/MolSSI/QCElemental), a dep that is supposed to give you constants and periodic table, but for some reason requires numpy
- afaict, it also tries to solve a much more difficult and much less relevant problem - automating execution of QC calculations. I don't see how one could anticipate all possible variations in configs of internal clusters. Most importantly, I don't see it as a main problem.

> How is it different from qc suite ([qcio](https://github.com/coltonbh/qcio), [qccodec](https://github.com/coltonbh/qccodec), [qcop](https://github.com/coltonbh/qcop)) by coltonbh?

honestly, qc suite is great and as of now has greater variety of supported QC software and tasks. unfortunately, as is, qcio depends on numpy and qcio/qccodec (which is a direct alternative to calcflow) seem to be too geared for automation with BigChem, which might be a plus for some, but might be too much for people who don't want/need to change their routine and just need easy input prep and output parsing

## License

MIT. Pure and simple.
