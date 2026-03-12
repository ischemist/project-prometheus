---
icon: lucide/rocket
---

# Quick Start

## 1. Install

=== "uv (recommended)"

    ```bash
    uv add calcflow
    ```

=== "pip"

    ```bash
    pip install calcflow
    ```

## 2. Parse an Output File

CalcFlow exposes one parse function per program. Pass it the raw output text — it returns an immutable `CalculationResult`.

=== "Q-Chem"

    ```python
    from pathlib import Path
    from calcflow import parse_qchem_output

    result = parse_qchem_output(Path("h2o_sp.out").read_text())
    ```

=== "ORCA"

    ```python
    from pathlib import Path
    from calcflow import parse_orca_output

    result = parse_orca_output(Path("h2o_sp.out").read_text())
    ```

Check termination before trusting the results:

```python
if result.termination_status != "NORMAL":
    raise RuntimeError(f"Calculation failed: {result.termination_status}")
```

Extract the final energy:

```python
print(result.final_energy)  # -76.4234... Hartree
```

Extract charges:

```python
mulliken = result.get_charges("Mulliken")
if mulliken:
    for atom_idx, charge in mulliken.charges.items():
        print(f"Atom {atom_idx}: {charge:+.4f}")
```

Extract excited states:

```python
if result.tddft:
    for state in result.tddft.tddft_states:
        ev = state.excitation_energy_ev
        nm = 1239.8 / ev
        f  = state.oscillator_strength
        print(f"S{state.state_number}: {ev:.2f} eV ({nm:.0f} nm)  f={f:.4f}")
```

Energies are in **Hartree** unless the field name says otherwise (`_ev`, `_kcal_mol`).

## 3. Serialize and Reload

```python
from pathlib import Path
from calcflow.common.results import CalculationResult

# Save — raw_output is excluded to keep the file compact
Path("result.json").write_text(result.to_json())

# Reload
result2 = CalculationResult.from_json(Path("result.json").read_text())
print(result2.final_energy)
```

## 4. Build a Calculation Input

`CalculationInput` is a frozen dataclass. Every setter returns a new instance — the original is never mutated.

```python
from calcflow import CalculationInput, Geometry

geom = Geometry.from_xyz_file("h2o.xyz")

calc = (
    CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wB97X-D3",
        basis_set="def2-TZVP",
    )
    .set_tddft(nroots=10, singlets=True, triplets=False)
    .set_solvation(model="smd", solvent="water")
    .set_cores(8)
    .set_memory_per_core(4000)
)
```

Export to Q-Chem or ORCA:

```python
qchem_input = calc.export("qchem", geom)
orca_input  = calc.export("orca", geom)

print(qchem_input)
```

Save the spec for reproducibility:

```python
Path("calc_spec.json").write_text(calc.to_json())
```

## 5. Discover the API at Runtime

```python
# Full method catalogue for CalculationInput
print(CalculationInput.get_api_docs())

# Field map with types and units for CalculationResult
from calcflow.common.results import CalculationResult
print(CalculationResult.get_api_docs())
```

## Next Steps

**[Parsing Guide](guides/parsing.md)** — every result field explained, with multi-job parsing, ADC, and spectrum broadening.

**[Inputs Guide](guides/inputs.md)** — the full fluent API: TDDFT, solvation, MOM, geometry optimization, charge analysis.

**[Geometry Guide](guides/geometry.md)** — `Geometry`, `Trajectory`, `AnnotatedGeometry`, and topology tools.

**[Concepts](concepts.md)** — the design philosophy behind CalcFlow's immutable models and parser architecture.
