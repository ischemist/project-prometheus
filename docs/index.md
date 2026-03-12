---
icon: lucide/home
---

# CalcFlow

**A zero-dependency Python library for quantum chemistry calculation I/O.**

[![PyPI](https://img.shields.io/pypi/v/calcflow)](https://pypi.org/project/calcflow/)
[![Python](https://img.shields.io/pypi/pyversions/calcflow)](https://pypi.org/project/calcflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**CalcFlow** bridges the gap between quantum chemistry programs and Python workflows. It parses output files from Q-Chem and ORCA into structured, immutable data models, and generates program-ready input files through a fluent, composable API — with zero external dependencies in the core package.

## The Problem

Quantum chemistry workflows share three recurring frustrations:

1. **Fragile parsing**: output formats vary across program versions, leading to brittle `grep`-and-`awk` pipelines that break silently.
2. **Scattered input generation**: each program has its own keyword syntax; constructing inputs programmatically means writing program-specific string templates.
3. **No structured data**: results live in text files, not in objects with types, units, and field names — making downstream analysis error-prone.

**CalcFlow solves this.**

## Key Features

- **Immutable data models**: every parsed result is a frozen dataclass — no mutation, no surprises, safe to pass around and serialize.
- **Fluent input API**: build a `CalculationInput` with method chaining, then `.export("qchem", geom)` or `.export("orca", geom)` — same spec, two programs.
- **Strategy-pattern parsers**: a registry of `BlockParser` objects handles each output section independently; adding support for new blocks is surgical, not global.
- **Version-aware parsing**: Q-Chem 5.4, 6.2, and 6.3 output format differences are handled transparently through versioned regex patterns.
- **Self-documenting API**: call `CalculationInput.get_api_docs()` or `CalculationResult.get_api_docs()` at runtime to get a full, always-current reference.
- **JSON roundtrip**: serialize any result or input spec to JSON and reload it later — with schema migration so old dumps stay readable.
- **Annotated geometry**: co-locate atom coordinates with charges, spin densities, and excited-state populations through a read-only `AnnotatedGeometry` view.
- **Agent-ready**: drop into LLM workflows with `uv run --with calcflow` — no setup, no conflicts; generate a `calcflow.md` rules file for plug-and-play tool use.

## Installation

=== "uv (recommended)"

    ```bash
    uv add calcflow
    ```

=== "pip"

    ```bash
    pip install calcflow
    ```

For spectrum broadening utilities (Lorentzian broadening, `postprocess` module), install the optional numpy extra:

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

# Excited states
for state in result.tddft.tddft_states:
    print(f"S{state.state_number}: {state.excitation_energy_ev:.2f} eV  f={state.oscillator_strength:.4f}")
```

## In an Agent or LLM Workflow?

No setup needed. Parse any output file with a single one-liner:

```bash
uv run --with calcflow python -c "
from calcflow.io.qchem import parse_qchem_output
from pathlib import Path
r = parse_qchem_output(Path('calc.out').read_text())
print(r.termination_status, r.final_energy)
"
```

A purpose-built [opencode](https://opencode.ai) agent ships with the repo. Install it with:

```bash
curl -fsSL https://raw.githubusercontent.com/ischemist/project-prometheus/master/calcflow.md \
  -o ~/.config/opencode/agents/calcflow.md
```

[**Agentic & LLM Workflows**](guides/agentic.md) — `uv run --with calcflow`, the opencode agent, and multi-step workflow examples.

## Getting Started

[**Quick Start**](quick-start.md) — parse your first output and build your first input in five minutes.

[**Concepts**](concepts.md) — understand the design philosophy behind the immutable model, the parser architecture, and the fluent API.

[**Guides**](guides/parsing.md) — detailed walkthroughs of every feature.
