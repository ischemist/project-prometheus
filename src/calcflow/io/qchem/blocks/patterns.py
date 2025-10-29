"""
A registry of version-aware PatternDefinitions for parsing QChem output blocks.
This decouples the regex patterns and version logic from the parser's control flow.
"""

import re

from calcflow.common.patterns import PatternDefinition, VersionSpec

# Helper for floating point numbers
FLOAT = r"(-?\d+\.\d+)"

# Create VersionSpec instances for clarity
V5_4 = VersionSpec.from_str("5.4.0")
V6_0 = VersionSpec.from_str("6.0.0")

# A list of all versioned patterns to be used by various parsers.
# The ScfParser will import and use these.
QCHEM_PATTERNS = [
    # --- Final/Total Energy Patterns ---
    PatternDefinition(
        field_name="scf_energy",
        description="Final converged SCF energy value",
        versioned_patterns=[
            (re.compile(rf"^\s*SCF energy in the final basis set\s*=\s*{FLOAT}"), V5_4, lambda m: float(m.group(1))),
            (re.compile(rf"^\s*SCF\s+energy\s*=\s*{FLOAT}"), V6_0, lambda m: float(m.group(1))),
        ],
    ),
    PatternDefinition(
        field_name="final_energy",
        description="Total energy including corrections (often same as SCF for SP)",
        versioned_patterns=[
            (re.compile(rf"^\s*Total energy in the final basis set\s*=\s*{FLOAT}"), V5_4, lambda m: float(m.group(1))),
            (re.compile(rf"^\s*Total energy\s*=\s*{FLOAT}"), V6_0, lambda m: float(m.group(1))),
        ],
    ),
    # --- SMD Summary Patterns ---
    PatternDefinition(
        field_name="g_pcm_kcal_mol",
        block_type="smd_summary",
        description="SMD polarization energy component",
        versioned_patterns=[
            (re.compile(rf"^\s*G_PCM\s*=\s*{FLOAT}\s*kcal/mol"), None, lambda m: float(m.group(1))),
        ],
    ),
    PatternDefinition(
        field_name="g_cds_kcal_mol",
        block_type="smd_summary",
        description="SMD non-electrostatic energy component",
        versioned_patterns=[
            (re.compile(rf"^\s*free energy\s+{FLOAT}\s*kcal/mol"), V5_4, lambda m: float(m.group(1))),
            (re.compile(rf"^\s*G_CDS\s*=\s*{FLOAT}\s*kcal/mol"), V6_0, lambda m: float(m.group(1))),
        ],
    ),
    PatternDefinition(
        field_name="g_enp_au",
        block_type="smd_summary",
        description="SCF energy in solvent (E_SCF + G_PCM)",
        versioned_patterns=[
            (re.compile(rf"^\s*\(3\)\s+G-ENP\(liq\).*?\s*{FLOAT}\s*a\.u\."), V5_4, lambda m: float(m.group(1))),
            (re.compile(rf"^\s*G_ENP\s*=\s*{FLOAT}\s*a\.u\."), V6_0, lambda m: float(m.group(1))),
        ],
    ),
    PatternDefinition(
        field_name="g_tot_au",
        block_type="smd_summary",
        description="Total free energy in solution (G_ENP + G_CDS)",
        versioned_patterns=[
            (re.compile(rf"^\s*\(6\)\s+G-S\(liq\).*?\s*{FLOAT}\s*a\.u\."), V5_4, lambda m: float(m.group(1))),
            (re.compile(rf"^\s*G\(tot\)\s*=\s*{FLOAT}\s*a\.u\."), V6_0, lambda m: float(m.group(1))),
        ],
    ),
]
