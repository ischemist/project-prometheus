---
icon: lucide/bot
---

# Agentic & LLM Workflows

CalcFlow is designed to drop into agentic workflows. Zero dependencies, a self-documenting API, and `uv`-based instant execution mean an LLM agent can parse and analyze quantum chemistry output files without any pre-existing environment setup.

## Why It Works in Agentic Contexts

Three properties make CalcFlow unusually well-suited for tool use:

1. **Zero dependencies** — `uv run --with calcflow` installs and runs in seconds, in any environment, without version conflicts.
2. **Self-documenting API** — call `CalculationResult.get_schema()` or `CalculationInput.get_quick_ref()` at runtime to get a complete, always-current field reference. The agent doesn't need to read source code.
3. **JSON-first results** — every parsed result serializes to JSON via `.to_json()`, making it trivial to pass structured data between agent steps.

## The CalcFlow OpenCode Agent

The repo ships a purpose-built [opencode](https://opencode.ai) agent at [`calcflow.md`](https://github.com/ischemist/project-prometheus/blob/master/calcflow.md) in the project root. Install it with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/ischemist/project-prometheus/master/calcflow.md \
  -o ~/.config/opencode/agents/calcflow.md
```

Once installed, open opencode in any directory containing `.out` files or `.xyz` geometries and the CalcFlow agent is available.

### What the agent does

The agent is configured to:

- **Never read files directly** — all file inspection happens through `calcflow` via the shell tool (`cat`, `grep`, `head` are disabled). This keeps context tight and avoids raw-text hallucinations.
- **Use `uv run --with calcflow`** for every operation — no assumptions about the local environment.
- **Navigate the API incrementally** — quick ref first, full method docs only when needed, never dumping the entire schema upfront.
- **Speak in one-liners** — short, focused scripts that do one thing.

## `uv run --with calcflow`

The universal entry point — no installation, no virtual environment management:

```bash
# Parse an output file and print key results
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

If CalcFlow is already installed in the project's virtualenv, drop the `--with calcflow`:

```bash
uv run python -c "..."
```

## API Navigation

The agent (and you) can explore the API at runtime — without reading source code.

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

**Result field schema (for parsing questions):**
```bash
uv run --with calcflow python -c "
from calcflow.common.results import CalculationResult
print(CalculationResult.get_schema())
"
```

!!! tip "One targeted call beats one 600-line blob"
    Prefer `get_quick_ref()` or `get_method_docs('set_tddft')` over `get_api_docs()` unless you need the full catalogue. Agents work better with focused context.

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

**Spatial analysis — excited-state hole localization:**
```bash
uv run --with calcflow python -c "
from calcflow import AnnotatedGeometry, build_bond_graph, find_aromatic_atoms
from calcflow.io.qchem import parse_qchem_output
from pathlib import Path
r = parse_qchem_output(Path('calc.out').read_text())
ag = AnnotatedGeometry.from_result(r)
graph = build_bond_graph(ag.geometry.atoms)
aromatic = find_aromatic_atoms(ag.geometry.atoms, graph)
if s1 := ag.get_unrelaxed_state(1):
    for atom in s1:
        if atom.hole_population and atom.hole_population > 0.5:
            print(f'S1 hole localized on {atom.symbol}{atom.index}')
"
```

## Multi-Step Workflows

CalcFlow's JSON roundtrip makes it natural to chain agent steps without re-parsing:

```
parse gs_sp.out  →  result.json
      ↓
agent reads result.json, decides on excited-state geometry optimization
      ↓
generate s1_opt.inp + s1_opt_spec.json
      ↓
submit to cluster, wait
      ↓
parse s1_opt.out  →  s1_result.json
      ↓
agent analyzes S1 geometry, builds emission calculation
      ↓
...
```

Each step stores its output as JSON. The agent can pick up at any point — even across sessions — by loading a previous `result.json` instead of re-parsing the raw output.

## Units and Conventions

| Quantity | Unit |
| :--- | :--- |
| Energies | **Hartree** (fields ending `_ev` are eV; `_kcal_mol` are kcal/mol) |
| Distances, exciton sizes | **Ångström** |
| Dipole / transition moments | **Debye** |
| Time | **seconds** |
| Atom indices | **0-based** |
| TDDFT state numbers | **1-based** |
