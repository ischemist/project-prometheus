---
icon: lucide/file-search
---

# Parsing Output Files

CalcFlow parses quantum chemistry output files into structured, immutable `CalculationResult` objects. This guide covers every result field, from basic energies to excited-state density matrices.

## The Parsers

=== "Q-Chem"

    ```python
    from calcflow import parse_qchem_output

    result = parse_qchem_output(open("calculation.out").read())
    ```

=== "ORCA"

    ```python
    from calcflow import parse_orca_output

    result = parse_orca_output(open("calculation.out").read())
    ```

Both functions return a `CalculationResult`. The raw output text is stored on `result.raw_output` but is excluded from JSON serialization.

### Multi-job Q-Chem outputs

Q-Chem can concatenate multiple jobs into a single output file (e.g. a two-job MOM calculation). Use the multi-job parser to get a list of results:

```python
from calcflow import parse_qchem_multi_job_output

results = parse_qchem_multi_job_output(open("mom_calc.out").read())
gs_result  = results[0]  # ground state job
es_result  = results[1]  # excited state job
```

## Checking Termination

Always check `termination_status` before trusting the results:

```python
if result.termination_status != "NORMAL":
    raise RuntimeError(f"Calculation failed: {result.termination_status}")
```

| Status | Meaning |
| :--- | :--- |
| `"NORMAL"` | Calculation completed successfully |
| `"ERROR"` | Program reported an error |
| `"UNKNOWN"` | Termination could not be determined from the output |

## Energetics

```python
result.final_energy              # float | None — total energy in Hartree
result.nuclear_repulsion_energy  # float | None — nuclear repulsion in Hartree
result.dispersion                # DispersionCorrection | None
result.smd                       # SmdResults | None
```

### Dispersion correction

```python
if result.dispersion:
    print(result.dispersion.energy_hartree)   # DFT-D correction in Hartree
    print(result.dispersion.method)           # e.g. "D3BJ"
```

### SMD solvation

```python
if result.smd:
    print(result.smd.total_energy_kcal_mol)   # total solvation free energy
    print(result.smd.electrostatic_kcal_mol)  # electrostatic component
    print(result.smd.non_electrostatic_kcal_mol)
```

## SCF Results

```python
scf = result.scf
if scf:
    print(scf.converged)       # bool
    print(scf.energy)          # float — final SCF energy in Hartree
    print(scf.n_iterations)    # int

    # Per-iteration history
    for i, it in enumerate(scf.iterations):
        print(f"  iter {i+1}: {it.energy:.10f} Hartree  delta={it.delta_energy:.2e}")
```

### SCF energy components

Q-Chem outputs include a breakdown of the SCF energy:

```python
if scf.components:
    c = scf.components
    print(c.nuclear_repulsion)    # Hartree
    print(c.one_electron)         # Hartree
    print(c.coulomb)              # Hartree
    print(c.exchange_correlation) # Hartree
```

## Molecular Orbitals

```python
orbs = result.orbitals
if orbs:
    # All alpha (or closed-shell) orbitals
    for orb in orbs.alpha_orbitals:
        occ = "occ" if orb.occupied else "virt"
        print(f"  MO {orb.index}: {orb.energy:.4f} Hartree  [{occ}]")

    # HOMO and LUMO
    homo = next(o for o in reversed(orbs.alpha_orbitals) if o.occupied)
    lumo = next(o for o in orbs.alpha_orbitals if not o.occupied)
    gap  = lumo.energy - homo.energy
    print(f"HOMO-LUMO gap: {gap * 27.211:.2f} eV")

    # Unrestricted: beta orbitals
    if orbs.beta_orbitals:
        for orb in orbs.beta_orbitals:
            print(f"  beta MO {orb.index}: {orb.energy:.4f} Hartree")
```

## Atomic Charges

Multiple charge methods can be present in a single result. Use `get_charges()` to retrieve by method name:

```python
mulliken  = result.get_charges("Mulliken")
hirshfeld = result.get_charges("Hirshfeld")
cm5       = result.get_charges("CM5")
loewdin   = result.get_charges("Loewdin")   # ORCA

if mulliken:
    # charges is a dict: atom_index (0-based) -> charge value
    for idx, charge in mulliken.charges.items():
        print(f"  atom {idx}: {charge:+.4f}")

    # spin densities (unrestricted calculations)
    if mulliken.spins:
        for idx, spin in mulliken.spins.items():
            print(f"  atom {idx} spin: {spin:+.4f}")
```

!!! info "Atom indexing"
    Charge and spin dicts use **0-based** integer keys, consistent with Python conventions. Q-Chem output uses 1-based numbering — CalcFlow converts on ingestion.

## Multipole Moments

```python
mp = result.multipole
if mp:
    dx, dy, dz = mp.dipole.x, mp.dipole.y, mp.dipole.z
    total = (dx**2 + dy**2 + dz**2) ** 0.5
    print(f"Dipole moment: {total:.4f} Debye")
```

Available moments: `dipole`, `quadrupole`, `octopole`, `hexadecapole`.

## TDDFT / TDA Excited States

```python
tddft = result.tddft
if tddft:
    # TDA states (Tamm-Dancoff approximation)
    for state in tddft.tda_states:
        print(f"TDA S{state.state_number}: {state.excitation_energy_ev:.3f} eV"
              f"  f={state.oscillator_strength:.4f}")

    # Full TDDFT states
    for state in tddft.tddft_states:
        ev = state.excitation_energy_ev
        nm = 1239.8 / ev
        print(f"S{state.state_number}: {ev:.3f} eV ({nm:.0f} nm)"
              f"  f={state.oscillator_strength:.4f}")

    # Orbital transitions for a specific state
    state = tddft.tddft_states[0]
    for trans in state.transitions:
        print(f"  {trans.from_orbital} -> {trans.to_orbital}  coeff={trans.coefficient:.4f}")
```

### Excited-state geometry optimization

When a TDDFT geometry optimization is run, `state.total_energy_au` gives the total energy of the excited state on the optimized geometry.

### Natural Transition Orbitals

```python
if tddft.nto_analyses:
    for nto in tddft.nto_analyses:
        print(f"State {nto.state_number}: NTO analysis available")
```

### Transition density matrix analysis

```python
if tddft.transition_density_matrices:
    for tdm in tddft.transition_density_matrices:
        print(f"State {tdm.state_number}:")
        print(f"  hole-electron separation: {tdm.exciton.rhe:.3f} Å")
        print(f"  CT number: {tdm.exciton.ct_number:.4f}")
```

### Unrelaxed difference density matrix analysis

```python
if tddft.unrelaxed_density_matrices:
    for udm in tddft.unrelaxed_density_matrices:
        print(f"State {udm.state_number}:")
        if udm.mulliken:
            print(f"  hole/electron Mulliken charges available")
```

## ADC Excited States

CalcFlow parses ADC(2) outputs from Q-Chem:

```python
adc = result.adc
if adc:
    print(f"Method: {adc.method}")  # e.g. "ADC(2)"

    # Ground state (HF + MP2)
    gs = adc.ground_state
    print(f"HF energy:  {gs.hf_energy:.6f} Hartree")
    print(f"MP2 energy: {gs.mp2_energy:.6f} Hartree")

    # Excited states
    for state in adc.excited_states:
        print(f"State {state.state_number} ({state.multiplicity}):")
        print(f"  {state.excitation_energy_ev:.3f} eV"
              f"  OPA f={state.opa_oscillator_strength:.4f}")
        if state.tpa_cross_section is not None:
            print(f"  TPA cross section: {state.tpa_cross_section:.2f} GM")
```

## Geometry

```python
# Geometry used as input for this calculation
if result.input_geometry:
    print(result.input_geometry.to_xyz_str())

# Final optimized geometry (geometry optimization jobs)
if result.final_geometry:
    result.final_geometry.to_xyz_file("optimized.xyz")
    print(f"Optimized energy: {result.final_geometry.energy}")
```

## Timing

```python
if result.timing:
    for module, duration in result.timing.modules.items():
        print(f"{module}: {duration:.1f} s")
```

## Serialization

```python
# Serialize — raw_output is excluded, schema_version is included
json_str = result.to_json()

# Reload — works across calcflow versions (with migrations)
from calcflow.common.results import CalculationResult
result2 = CalculationResult.from_json(json_str)

# Dict form
data = result.to_dict()
result3 = CalculationResult.from_dict(data)
```

!!! warning "raw_output is excluded"
    The full output text (`result.raw_output`) is intentionally excluded from JSON serialization to keep files compact. If you need the raw output, save it separately alongside the JSON.

## Spectrum Broadening

For UV/Vis spectrum simulation, install the numpy extra and use the `postprocess` module:

```python
from calcflow import parse_qchem_output, spectrum_from_excited_states, make_energy_grid
import numpy as np

result = parse_qchem_output(open("tddft.out").read())
states = result.tddft.tddft_states

grid     = make_energy_grid(
    energies=[s.excitation_energy_ev for s in states],
    padding=1.0, n_points=2000
)
spectrum = spectrum_from_excited_states(states, grid, fwhm=0.3)  # eV FWHM

np.savetxt("spectrum.dat", np.column_stack([grid, spectrum]))
```

ADC OPA and TPA spectra are also available via `opa_spectrum_from_adc_states` and `tpa_spectrum_from_adc_states`.
