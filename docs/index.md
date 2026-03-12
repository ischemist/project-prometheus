---
icon: lucide/home
---

# CalcFlow

**A zero-dependency Python library for quantum chemistry calculation I/O.**

[![PyPI](https://img.shields.io/pypi/v/calcflow)](https://pypi.org/project/calcflow/)
[![Python](https://img.shields.io/pypi/pyversions/calcflow)](https://pypi.org/project/calcflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**CalcFlow** parses Q-Chem and ORCA output files into structured, immutable Python objects, and generates valid input files for both programs from a single shared spec. No external dependencies in the core package.

## Features

- **Immutable results** — every parsed result is a frozen dataclass. Safe to pass around, serialize, use as a dict key.
- **Fluent input API** — build a `CalculationInput` with method chaining, then `.export("qchem", geom)` or `.export("orca", geom)` — same spec, two programs.
- **Versioned parsing** — Q-Chem 5.4, 6.2, and 6.3 output format differences handled transparently.
- **JSON roundtrip** — serialize any result or input to JSON; schema migrations keep old dumps readable across versions.
- **Annotated geometry** — co-locate atom coordinates with charges, spin densities, and excited-state populations via `AnnotatedGeometry`.
- **Self-documenting** — call `CalculationInput.get_api_docs()` or `CalculationResult.get_api_docs()` at runtime for an always-current reference.
- **Agent-ready** — works with `uv run --with calcflow` in any environment; an [opencode](https://opencode.ai) agent ships with the repo.

## Installation

=== "uv (recommended)"

    ```bash
    uv add calcflow
    ```

=== "pip"

    ```bash
    pip install calcflow
    ```

For spectrum broadening (`postprocess` module), install the optional numpy extra:

=== "uv"

    ```bash
    uv add "calcflow[numpy]"
    ```

=== "pip"

    ```bash
    pip install "calcflow[numpy]"
    ```

## Supported Programs

| Program | Input generation | Output parsing |
| :--- | :---: | :---: |
| Q-Chem 5.4 | ✓ | ✓ |
| Q-Chem 6.2 | ✓ | ✓ |
| Q-Chem 6.3 | — | ✓ |
| ORCA | ✓ | ✓ |

## Supported Properties

| Category | Properties |
| :--- | :--- |
| **Energetics** | Final energy, SCF energy, nuclear repulsion, dispersion correction, SMD solvation energy |
| **Wavefunction** | SCF iterations & convergence, energy components, molecular orbitals (alpha/beta) |
| **Geometry** | Input geometry, optimized geometry, trajectory |
| **Excited states** | TDDFT (TDA/RPA), ADC(2) — energies, oscillator strengths, orbital transitions |
| **Density analysis** | NTOs, transition density matrices, unrelaxed difference density matrices, exciton descriptors |
| **Atomic charges** | Mulliken, Hirshfeld, CM5, Loewdin, Hirshfeld-I |
| **Multipole moments** | Dipole through hexadecapole |
| **Timing** | Per-module wall and CPU times |

## Quick Example

```python
from calcflow import parse_qchem_output

result = parse_qchem_output(open("calculation.out").read())

print(result.termination_status)   # "NORMAL"
print(result.final_energy)         # -76.4234... (Hartree)
print(result.scf.converged)        # True

for state in result.tddft.tddft_states:
    print(f"S{state.state_number}: {state.excitation_energy_ev:.2f} eV  f={state.oscillator_strength:.4f}")
```

## Getting Started

[**Quick Start**](quick-start.md) — parse your first output and build your first input in five minutes.

[**Concepts**](concepts.md) — the design philosophy: immutable models, the parser architecture, the fluent API.

[**Guides**](guides/parsing.md) — detailed walkthroughs of every feature.

[**Agentic Workflows**](guides/agentic.md) — `uv run --with calcflow`, the opencode agent, multi-step workflow examples.
