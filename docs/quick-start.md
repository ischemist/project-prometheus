---
icon: lucide/rocket
---

# Quick Start

Install CalcFlow, parse your first output file, and build your first input — in about five minutes.

!!! tip "What you'll learn"
    - Installing CalcFlow with `uv` or `pip`
    - Parsing a Q-Chem or ORCA output file into a structured result object
    - Extracting energies, charges, and excited states
    - Building a calculation input with the fluent API
    - Exporting inputs to Q-Chem and ORCA format

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

CalcFlow exposes a single parse function per program. Pass it the raw output text — it returns an immutable `CalculationResult`.

=== "Q-Chem"

    ```python
    from calcflow import parse_qchem_output

    text = open("h2o_sp.out").read()
    result = parse_qchem_output(text)
    ```

=== "ORCA"

    ```python
    from calcflow import parse_orca_output

    text = open("h2o_sp.out").read()
    result = parse_orca_output(text)
    ```

### Check termination

Always check whether the calculation finished normally before reading results:

```python
if result.termination_status != "NORMAL":
    raise RuntimeError(f"Calculation did not converge: {result.termination_status}")
```

### Extract the final energy

```python
print(result.final_energy)  # -76.4234... Hartree
```

### Extract charges

```python
mulliken = result.get_charges("Mulliken")
if mulliken:
    for atom_idx, charge in mulliken.charges.items():
        print(f"Atom {atom_idx}: {charge:+.4f}")
```

### Extract excited states (TDDFT)

```python
if result.tddft:
    for state in result.tddft.tddft_states:
        ev = state.excitation_energy_ev
        nm = 1239.8 / ev  # convert to wavelength
        f  = state.oscillator_strength
        print(f"S{state.state_number}: {ev:.2f} eV ({nm:.0f} nm)  f={f:.4f}")
```

!!! info "Units"
    All energies are in **Hartree** unless otherwise noted. Excitation energies on `ExcitedState` are also available in eV via `excitation_energy_ev`.

## 3. Serialize and Reload

Parsed results can be saved to JSON and reloaded without re-parsing:

```python
# Save
json_str = result.to_json()
with open("result.json", "w") as f:
    f.write(json_str)

# Reload (note: raw_output is excluded from serialization)
from calcflow.common.results import CalculationResult

result2 = CalculationResult.from_json(open("result.json").read())
print(result2.final_energy)
```

## 4. Build a Calculation Input

`CalculationInput` is a frozen dataclass with a fluent setter API. Every setter returns a new instance — the original is never mutated.

```python
from calcflow import CalculationInput, Geometry

# Load geometry from an XYZ file
geom = Geometry.from_xyz_file("h2o.xyz")

# Build the calculation spec
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

### Export to Q-Chem or ORCA

```python
qchem_input = calc.export("qchem", geom)
orca_input  = calc.export("orca", geom)

print(qchem_input)
```

The same `CalculationInput` object produces valid input for both programs.

### Save the spec for reproducibility

```python
with open("calc_spec.json", "w") as f:
    f.write(calc.to_json())
```

## 5. Discover the API at Runtime

Both `CalculationInput` and `CalculationResult` are self-documenting:

```python
# Full method catalogue for CalculationInput
print(CalculationInput.get_api_docs())

# Structural field map with types and units for CalculationResult
from calcflow.common.results import CalculationResult
print(CalculationResult.get_api_docs())
```

!!! success "You're all set"
    You've parsed an output, extracted results, built an input, and exported it — the full CalcFlow workflow. Explore the guides to go deeper.

## Next Steps

**[Parsing Guide](guides/parsing.md)** — every result field explained, with multi-job parsing, ADC, and spectrum broadening.

**[Inputs Guide](guides/inputs.md)** — the full fluent API: TDDFT, solvation, MOM, geometry optimization, charge analysis.

**[Geometry Guide](guides/geometry.md)** — `Geometry`, `Trajectory`, `AnnotatedGeometry`, and topology tools.

**[Concepts](concepts.md)** — the design philosophy behind CalcFlow's immutable models and parser architecture.
