"""Shared fixtures and helpers for ORCA builder tests."""

from __future__ import annotations

import re
from typing import NamedTuple


class OrcaInputComponents(NamedTuple):
    """Parsed components of an ORCA input file."""

    keyword_line: str  # The "! SP RKS b3lyp ..." line
    xyz_block: str  # Everything between "* xyz ..." and closing "*"
    blocks: dict[str, str]  # %pal, %geom, %tddft, etc. blocks
    raw_text: str  # Original input for fallback checks


def parse_orca_input(input_text: str) -> OrcaInputComponents:
    """
    Parse ORCA input into structured components.

    More robust than simple substring checks - extracts actual blocks
    and normalizes whitespace for comparison.
    """
    lines = input_text.strip().split("\n")

    # Extract keyword line (first line starting with !)
    keyword_line = ""
    for line in lines:
        if line.strip().startswith("!"):
            keyword_line = line.strip()
            break

    # Extract XYZ block
    xyz_block = ""
    in_xyz = False
    xyz_lines = []
    for line in lines:
        if line.strip().startswith("* xyz"):
            in_xyz = True
            xyz_lines.append(line)
        elif in_xyz:
            xyz_lines.append(line)
            if line.strip() == "*":
                break
    xyz_block = "\n".join(xyz_lines)

    # Extract % blocks (e.g., %pal, %geom, %tddft, %cpcm)
    blocks = {}
    current_block = None
    block_lines = []

    for line in lines:
        stripped = line.strip()
        # Start of block
        if stripped.startswith("%") and not stripped.startswith("%maxcore"):
            # Handle %pal nprocs 4 end (single line)
            if "end" in stripped:
                block_name = stripped.split()[0][1:]  # Remove %
                blocks[block_name] = stripped
                continue
            # Multi-line block
            current_block = stripped.split()[0][1:]  # Remove %
            block_lines = [line]
        elif current_block:
            block_lines.append(line)
            if stripped == "end":
                blocks[current_block] = "\n".join(block_lines)
                current_block = None
                block_lines = []

    return OrcaInputComponents(
        keyword_line=keyword_line,
        xyz_block=xyz_block,
        blocks=blocks,
        raw_text=input_text,
    )


def assert_block_contains(block_text: str, *expected_settings: str) -> None:
    """
    Assert that a block contains expected settings (whitespace-independent).

    Example:
            assert_block_contains(blocks['geom'], 'Calc_Hess true', 'Recalc_Hess 5')
    """
    # Normalize whitespace in the block
    normalized_block = " ".join(block_text.split())

    for setting in expected_settings:
        normalized_setting = " ".join(setting.split())
        assert normalized_setting in normalized_block, f"Expected '{setting}' in block, but got:\n{block_text}"


def assert_xyz_has_atoms(xyz_block: str, *expected_atoms: tuple[str, int]) -> None:
    """
    Assert XYZ block contains specific atoms with correct counts.

    Example:
            assert_xyz_has_atoms(xyz_block, ('O', 1), ('H', 2))

    This is more robust than just checking 'O ' in result.
    """
    lines = [line.strip() for line in xyz_block.split("\n") if line.strip()]

    # Skip header line (* xyz charge mult)
    atom_lines = [line for line in lines if not line.startswith("*")]

    for symbol, expected_count in expected_atoms:
        actual_count = sum(1 for line in atom_lines if line.split()[0] == symbol)
        assert actual_count == expected_count, (
            f"Expected {expected_count} {symbol} atoms, found {actual_count}\nXYZ block:\n{xyz_block}"
        )


def assert_xyz_charge_mult(xyz_block: str, charge: int, multiplicity: int) -> None:
    """
    Assert XYZ block has correct charge and multiplicity.

    Example:
            assert_xyz_charge_mult(xyz_block, 0, 1)
    """
    # First line should be "* xyz charge mult"
    first_line = xyz_block.strip().split("\n")[0]
    match = re.match(r"\*\s+xyz\s+(-?\d+)\s+(\d+)", first_line)

    assert match, f"Invalid XYZ header: {first_line}"

    actual_charge = int(match.group(1))
    actual_mult = int(match.group(2))

    assert actual_charge == charge, f"Expected charge {charge}, got {actual_charge}"
    assert actual_mult == multiplicity, f"Expected multiplicity {multiplicity}, got {actual_mult}"


def assert_keywords_present(keyword_line: str, *expected_keywords: str) -> None:
    """
    Assert that keyword line contains expected keywords (case-insensitive, order-independent).

    Example:
            assert_keywords_present(kw_line, 'SP', 'RKS', 'b3lyp', 'def2-svp')
    """
    normalized_line = keyword_line.lower()

    for keyword in expected_keywords:
        assert keyword.lower() in normalized_line, f"Expected keyword '{keyword}' in: {keyword_line}"


def assert_pal_block(blocks: dict[str, str], nprocs: int) -> None:
    """
    Assert %pal block exists with correct nprocs setting.

    More robust than just checking "nprocs 4" anywhere in the file.
    """
    assert "pal" in blocks, "Expected %pal block but not found"
    assert_block_contains(blocks["pal"], f"nprocs {nprocs}")


def assert_geom_block(blocks: dict[str, str], **expected_settings: bool | int) -> None:
    """
    Assert %geom block exists with expected optimization settings.

    Example:
            assert_geom_block(blocks, calc_hess=True, recalc_hess=5)
    """
    assert "geom" in blocks, "Expected %geom block but not found"

    settings = []
    if expected_settings.get("calc_hess"):
        settings.append("Calc_Hess true")
    if "recalc_hess" in expected_settings:
        settings.append(f"Recalc_Hess {expected_settings['recalc_hess']}")

    assert_block_contains(blocks["geom"], *settings)


def assert_tddft_block(blocks: dict[str, str], nroots: int, **kwargs: bool | int) -> None:
    """
    Assert %tddft block exists with expected settings.

    Example:
            assert_tddft_block(blocks, nroots=10, triplets=True, tda=False)
    """
    assert "tddft" in blocks, "Expected %tddft block but not found"

    settings = [f"NRoots {nroots}"]

    if kwargs.get("triplets"):
        settings.append("Triplets true")
    if kwargs.get("triplets") is False:
        settings.append("Triplets false")
    if kwargs.get("tda"):
        settings.append("TDA true")
    if kwargs.get("tda") is False:
        settings.append("TDA false")
    if "iroot" in kwargs:
        settings.append(f"IRoot {kwargs['iroot']}")

    assert_block_contains(blocks["tddft"], *settings)


def assert_cpcm_block(blocks: dict[str, str], smd: bool = False, solvent: str | None = None) -> None:
    """
    Assert %cpcm block exists with expected solvation settings.

    Example:
            assert_cpcm_block(blocks, smd=True, solvent='water')
    """
    assert "cpcm" in blocks, "Expected %cpcm block but not found"

    settings = []
    if smd:
        settings.append("smd true")
        if solvent:
            settings.append(f'SMDsolvent "{solvent}"')

    assert_block_contains(blocks["cpcm"], *settings)


def assert_maxcore(raw_text: str, memory_mb: int) -> None:
    """
    Assert %maxcore directive is present with correct value.

    %maxcore is special - it's not a block, just a directive.
    """
    assert f"%maxcore {memory_mb}" in raw_text, f"Expected '%maxcore {memory_mb}' in input"
