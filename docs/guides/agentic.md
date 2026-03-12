---
icon: lucide/bot
---

# Agentic & LLM Workflows

## The CalcFlow OpenCode Agent

The repo ships a purpose-built [opencode](https://opencode.ai) agent at [`calcflow.md`](https://github.com/ischemist/project-prometheus/blob/master/calcflow.md). Install it with:

```bash
curl -fsSL https://raw.githubusercontent.com/ischemist/project-prometheus/master/calcflow.md \
  -o ~/.config/opencode/agents/calcflow.md
```

Open opencode in any directory containing `.out` files or `.xyz` geometries and the CalcFlow agent is available.

### What the agent does

- **Never reads files directly** — all file inspection goes through `calcflow` via the shell tool. This keeps context tight and avoids raw-text hallucinations.
- **Uses `uv run --with calcflow`** for every operation — no assumptions about the local environment.
- **Navigates the API incrementally** — quick ref first, full method docs only when needed.
- **Works in short, focused scripts** — one task per shell call.

## `uv run --with calcflow`

No installation, no virtual environment management. Works anywhere `uv` is installed:

```bash
uv run --with calcflow python -c "
from calcflow.io.qchem import parse_qchem_output
from pathlib import Path
r = parse_qchem_output(Path('calc.out').read_text())
print('status:', r.termination_status)
print('energy:', r.final_energy, 'Hartree')
if r.tddft and r.tddft.tda_states:
    for s in r.tddft.tda_states[:5]:
        print(f'S{s.state_number}: {s.excitation_energy_ev:.3f} eV  f={s.oscillator_strength}')
"
```

If CalcFlow is already installed in the project virtualenv, drop `--with calcflow`:

```bash
uv run python -c "..."
```

## API Navigation

Explore the API at runtime — no source code reading needed.

**Quick reference (usually enough):**
```bash
uv run --with calcflow python -c "
from calcflow.common.input import CalculationInput
print(CalculationInput.get_quick_ref())
"
```

**Full docs for one specific method:**
```bash
uv run --with calcflow python -c "
from calcflow.common.input import CalculationInput
print(CalculationInput.get_method_docs('set_tddft'))
"
```

**Result field schema:**
```bash
uv run --with calcflow python -c "
from calcflow.common.results import CalculationResult
print(CalculationResult.get_schema())
"
```

Prefer `get_quick_ref()` or `get_method_docs('set_tddft')` over `get_api_docs()` unless you need the full catalogue. A focused 30-line reference beats a 600-line blob.

## Common One-Liners

**Parse ORCA:**
```bash
uv run --with calcflow python -c "
from calcflow.io.orca import parse_orca_output
from pathlib import Path
r = parse_orca_output(Path('calc.out').read_text())
print('status:', r.termination_status, '| energy:', r.final_energy)
"
```

**Parse and save to JSON:**
```bash
uv run --with calcflow python -c "
from calcflow.io.qchem import parse_qchem_output
from pathlib import Path
r = parse_qchem_output(Path('calc.out').read_text())
Path('result.json').write_text(r.to_json())
print('saved result.json')
"
```

**Load a saved JSON result:**
```bash
uv run --with calcflow python -c "
from calcflow.common.results import CalculationResult
from pathlib import Path
r = CalculationResult.from_json(Path('result.json').read_text())
print('energy:', r.final_energy)
"
```

**Load a gzip-compressed result (`.json.gz`):**
```bash
uv run --with calcflow python -c "
import gzip, json
from calcflow.common.results import CalculationResult
from pathlib import Path
raw = json.loads(gzip.decompress(Path('result.json.gz').read_bytes()))
r = CalculationResult.from_dict(raw)
print('status:', r.termination_status, '| energy:', r.final_energy)
"
```

**Multi-job output (MOM, XAS):**
```bash
uv run --with calcflow python -c "
from calcflow.io.qchem import parse_qchem_multi_job_output
from pathlib import Path
jobs = parse_qchem_multi_job_output(Path('calc.out').read_text())
for i, r in enumerate(jobs):
    print(f'job {i+1}: {r.termination_status}  energy={r.final_energy}')
"
```

**Build and export a Q-Chem input:**
```bash
uv run --with calcflow python -c "
from calcflow import CalculationInput, Geometry
from pathlib import Path
geom = Geometry.from_xyz_file('molecule.xyz')
calc = (
    CalculationInput(charge=0, spin_multiplicity=1, task='energy',
        level_of_theory='wB97X-D3', basis_set='def2-tzvp', n_cores=16)
    .set_tddft(nroots=10)
    .set_solvation('smd', 'water')
)
Path('calc.in').write_text(calc.export('qchem', geom))
Path('calc_spec.json').write_text(calc.to_json())
print('wrote calc.in and calc_spec.json')
"
```

**Excited-state hole localization (unrelaxed DM):**
```bash
uv run --with calcflow python -c "
from calcflow import AnnotatedGeometry
from calcflow.io.qchem import parse_qchem_output
from pathlib import Path
r = parse_qchem_output(Path('calc.out').read_text())
ag = AnnotatedGeometry.from_result(r)
s1 = ag.get_unrelaxed_state(1)
if s1:
    for atom in s1:
        if atom.hole_population and atom.hole_population > 0.1:
            print(f'S1 hole on {atom.symbol}{atom.index}: {atom.hole_population:.3f}')
"
```

## Multi-Step Workflow Example

CalcFlow's JSON roundtrip lets agent steps pick up where the last one left off — no re-parsing, no re-running.

A typical absorption → excited-state geometry → emission workflow:

**Step 1 — Parse the ground-state single point, save to JSON:**
```bash
uv run --with calcflow python -c "
from calcflow.io.qchem import parse_qchem_output
from pathlib import Path
r = parse_qchem_output(Path('gs_sp.out').read_text())
assert r.termination_status == 'NORMAL'
Path('gs_sp.json').write_text(r.to_json())
# print S1 for reference
s1 = r.tddft.tddft_states[0]
print(f'S1 absorption: {s1.excitation_energy_ev:.3f} eV  f={s1.oscillator_strength:.4f}')
"
```

**Step 2 — Build the S1 geometry optimization input:**
```bash
uv run --with calcflow python -c "
from calcflow import CalculationInput, Geometry
from calcflow.common.results import CalculationResult
from pathlib import Path

gs = CalculationResult.from_json(Path('gs_sp.json').read_text())
geom = gs.input_geometry   # Geometry object from the parsed result

calc = (
    CalculationInput(charge=0, spin_multiplicity=1, task='geometry',
        level_of_theory='wB97X-D3', basis_set='def2-TZVP', n_cores=32)
    .set_tddft(nroots=5, singlets=True, triplets=False, state_to_optimize=1)
    .set_solvation('smd', 'water')
)
Path('s1_opt.in').write_text(calc.export('qchem', geom))
Path('s1_opt_spec.json').write_text(calc.to_json())
print('wrote s1_opt.in')
"
```

**Step 3 — Parse the S1 optimized geometry, compute emission energy:**
```bash
uv run --with calcflow python -c "
from calcflow.io.qchem import parse_qchem_output
from calcflow.common.results import CalculationResult
from pathlib import Path

gs  = CalculationResult.from_json(Path('gs_sp.json').read_text())
s1  = parse_qchem_output(Path('s1_opt.out').read_text())
assert s1.termination_status == 'NORMAL'
Path('s1_opt.json').write_text(s1.to_json())

gs_energy = gs.final_energy
s1_energy = s1.final_energy
emission_ev = (s1_energy - gs_energy) * 27.211   # Hartree to eV
print(f'S1 emission energy: {emission_ev:.3f} eV  ({1239.8 / emission_ev:.0f} nm)')
"
```

Each step is self-contained. If a job fails or you want to branch — try a different functional, check a different state — reload from the JSON and go.

## Units and Conventions

| Quantity | Unit |
| :--- | :--- |
| Energies | **Hartree** (fields ending `_ev` are eV; `_kcal_mol` are kcal/mol) |
| Distances, exciton sizes | **Ångström** |
| Dipole / transition moments | **Debye** |
| Time | **seconds** |
| Atom indices | **0-based** |
| TDDFT state numbers | **1-based** |
