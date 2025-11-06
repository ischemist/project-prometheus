# CalcFlow: Quantum Chemistry Calculation I/O Done Right

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![coverage](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/ischemist/project-prometheus/master/coverage.json&query=$.totals.percent_covered_display&label=coverage&color=brightgreen&suffix=%25)

CalcFlow provides a robust, Pythonic interface for preparing inputs and parsing outputs for quantum chemistry software like Q-Chem and ORCA. It has **zero external dependencies** and is built for clarity and reliability. Get your calculations set up and results processed without the usual boilerplate.

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

## Quick Start

### ORCA: Geometry Optimization

Set up an ORCA geometry optimization with frequency calculation for a water molecule:

```python
from calcflow import CalculationInput, Geometry

# 1. Load molecular geometry
water = Geometry.from_xyz_file("tests/testing_data/geometries/1h2o.xyz")

# 2. Configure ORCA calculation - uses fluent API
calc = (
    CalculationInput(
        charge=0, spin_multiplicity=1, task="geometry",
        level_of_theory="wb97x-d3", basis_set="def2-svp", n_cores=16
    )
    .enable_ri_for_orca("RIJCOSX", "def2/j")
    .run_frequency_after_opt()
)

# 3. Export ORCA input file
with open("h2o.inp", "w") as f:
    f.write(calc.export("orca", water))

# 4. Save calculation spec as JSON for reproducibility
with open("h2o_calc_spec.json", "w") as f:
    f.write(calc.to_json())

# 5. Later, load the spec back from JSON
loaded_calc = CalculationInput.from_json(open("h2o_calc_spec.json").read())
assert loaded_calc == calc  # perfect roundtrip!
```

### Q-Chem: TDDFT with SMD Solvation

Set up a Q-Chem TDDFT calculation with implicit solvation:

```python
from calcflow import CalculationInput, Geometry

# 1. Load molecular geometry from XYZ file (recommended)
water = Geometry.from_xyz_file("tests/testing_data/geometries/1h2o.xyz")

# Or define manually:
# from calcflow.common.results import Atom
# atoms = (
#     Atom("O", 0.000000, 0.000000, 0.117300),
#     Atom("H", 0.000000, 0.757200, -0.469200),
#     Atom("H", 0.000000, -0.757200, -0.469200),
# )
# water = Geometry(atoms=atoms, comment="Water molecule")
# 2. Configure Q-Chem calculation
calc = (
    CalculationInput(
        charge=0, spin_multiplicity=1, task="energy",
        level_of_theory="wB97X-D3", basis_set="def2-tzvp", n_cores=16
    )
    .set_tddft(nroots=10, singlets=True, triplets=False)
    .set_solvation(model="smd", solvent="water")
)

# 3. Export Q-Chem input file
with open("h2o_tddft.in", "w") as f:
    f.write(calc.export("qchem", water))
```

Want to include triplets? Easy modification:

```python
calc2 = calc.set_tddft(nroots=10, singlets=True, triplets=True)
```

Need an O K-Edge XAS spectrum with element-specific basis?

```python
calc3 = (
    calc.set_tddft(nroots=10, singlets=True, triplets=False)
    .set_basis({"H": "pc-2", "O": "pcX-2"})
    .set_reduced_excitation_space(initial_orbitals=[1])
)
```

By the way, this pythonic method:

```py
.set_reduced_excitation_space(initial_orbitals=[1])
```

translates into Q-Chem's cryptic syntax:

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

Honestly, a major reason why this package exists.

### Advanced: XAS from Excited State (MOM)

Want XAS spectrum from S1 state? Chain the methods:

```python
calc4 = (
    calc.set_tddft(nroots=10, singlets=True, triplets=False)
    .set_basis({"H": "pc-2", "O": "pcX-2"})
    .set_reduced_excitation_space(initial_orbitals=[1])
    .set_unrestricted()
    .set_mom(transition="HOMO->LUMO")
)
```

This creates a 2-job input file: initial SCF followed by MOM with HOMO→LUMO transition and TDDFT.

Fun fact: `Geometry` instances have a `.total_nuclear_charge` property used to calculate HOMO/LUMO indices. You can also specify transitions numerically (e.g., `"5->6"`, `"3->LUMO"`) or use `"->vac"` for ionization (e.g., `"HOMO->vac"`).

## Parsing Output Files

### ORCA

```python
from pathlib import Path
from calcflow import parse_orca_output

# Parse ORCA output
result = parse_orca_output(Path("h2o.out").read_text())

# Access results
print(result.final_energy)  # -76.1234567
print(result.scf.n_iterations)  # 12

# Save to JSON (excludes raw_output automatically)
with open("orca_result.json", "w") as f:
    f.write(result.to_json())

# Later, load from JSON (much faster than re-parsing)
from calcflow.common.results import CalculationResult
loaded = CalculationResult.from_json(Path("orca_result.json").read_text())
```

### Q-Chem

```python
from pathlib import Path
from calcflow import parse_qchem_output, parse_qchem_multi_job_output

# Single-job output
result = parse_qchem_output(Path("h2o_sp.out").read_text())

# Multi-job output (e.g., MOM, XAS)
jobs = parse_qchem_multi_job_output(Path("h2o_mom_xas.out").read_text())
job1, job2 = jobs

# Access TDDFT results
print(job2.scf.energy)  # -76.53682225
print(job2.tddft.tda_states[0].excitation_energy_ev)  # 7.42

# Get excitation energies and oscillator strengths
energies = [s.excitation_energy_ev for s in job2.tddft.tda_states]
intensities = [s.oscillator_strength for s in job2.tddft.tda_states]

# Mulliken populations for state 3
mulliken = job2.tddft.transition_dm_analyses[2].mulliken
for pop in mulliken.populations:
    print(f"{pop.symbol}: {pop.transition_charge_e}")
```

## LLM & Programmatic Usage

CalcFlow provides built-in API documentation for easy exploration and LLM-assisted workflows:

```python
# Get comprehensive API documentation for input generation
from calcflow.common.input import CalculationInput
print(CalculationInput.get_api_docs())

# Get comprehensive API documentation for results parsing
from calcflow.common.results import CalculationResult
print(CalculationResult.get_api_docs())
```

These methods return complete documentation of all available fields, methods, and usage examples - perfect for LLMs or programmatic exploration without needing to read source code.

### Quick Testing with `uv`

Test CalcFlow without installation using one-liners:

```bash
# Get input API docs
uv run --with calcflow python -c "from calcflow.common.input import CalculationInput; print(CalculationInput.get_api_docs())"

# Get results API docs
uv run --with calcflow python -c "from calcflow.common.results import CalculationResult; print(CalculationResult.get_api_docs())"

# Quick parse and inspect
uv run --with calcflow python -c "from calcflow.io.qchem import parse_qchem_output; from pathlib import Path; result = parse_qchem_output(Path('calc.out').read_text()); print(f'Energy: {result.final_energy}, Status: {result.termination_status}')"

# Save API docs to file
uv run --with calcflow python -c "from calcflow.common.input import CalculationInput; open('api.txt', 'w').write(CalculationInput.get_api_docs())"
```


## More Examples

For more examples, see [scripts/input-orca.py](scripts/input-orca.py) and [scripts/parse-orca.py](scripts/parse-orca.py). Test data is in [tests/testing_data/](tests/testing_data/).

### Which versions of QChem do you support?

It's tested on 6.2 and 5.4. Good news is that I'm intending to make it easy to adjust for version-specific printing. For example,

```txt
QChem 5.4. prints final SCF energy as
SCF   energy in the final basis set =      -75.3184602363
while 6.2. prints it as
SCF   energy =   -75.32080770

p.s. nevermind the difference in energy, 5.4. mistakenly prints SCF energy same as Total energy, which includes solvation terms
```

which is why the [calcflow/io/qchem/blocks/scf.py](src/calcflow/io/qchem/blocks/scf.py) block defines different patterns for different versions:

```py
PatternDefinition(field_name="scf_energy", required=True, description="Final SCF energy value",
versioned_patterns=[
        (re.compile(r"^\s*SCF\s+energy\s*=\s*(-?\d+\.\d+)"), "6.2", lambda m: float(m.group(1))),
        (re.compile(r"^\s*SCF\s+energy in the final basis set\s*=\s*(-?\d+\.\d+)"), "5.4", lambda m: float(m.group(1))),
    ],
),
```

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
