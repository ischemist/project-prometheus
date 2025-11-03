"""Shared fixtures and helpers for Q-Chem builder tests."""

from __future__ import annotations

from typing import NamedTuple


class QchemInputComponents(NamedTuple):
    """Parsed components of a Q-Chem input file."""

    molecule_block: str  # The "$molecule ... $end" block
    rem_block: str  # The "$rem ... $end" block
    blocks: dict[str, str]  # $basis, $solvent, $smx, $occupied, $solute blocks
    raw_text: str  # Original input for fallback checks


def parse_qchem_input(input_text: str) -> QchemInputComponents:
    """
    Parse Q-Chem input into structured components.

    More robust than simple substring checks - extracts actual blocks
    and normalizes whitespace for comparison.
    """
    lines = input_text.strip().split("\n")

    # Extract $molecule block
    molecule_block = ""
    in_molecule = False
    molecule_lines = []
    for line in lines:
        if line.strip().startswith("$molecule"):
            in_molecule = True
            molecule_lines.append(line)
        elif in_molecule:
            molecule_lines.append(line)
            if line.strip().startswith("$end"):
                break
    molecule_block = "\n".join(molecule_lines)

    # Extract $rem block
    rem_block = ""
    in_rem = False
    rem_lines = []
    for line in lines:
        if line.strip().startswith("$rem"):
            in_rem = True
            rem_lines.append(line)
        elif in_rem:
            rem_lines.append(line)
            if line.strip().startswith("$end"):
                break
    rem_block = "\n".join(rem_lines)

    # Extract other blocks ($basis, $solvent, $smx, $occupied, $solute, etc.)
    blocks = {}
    current_block = None
    block_lines = []

    for line in lines:
        stripped = line.strip()
        # Start of block
        if stripped.startswith("$") and not stripped.startswith("$end"):
            # Save previous block if it exists
            if current_block is not None:
                blocks[current_block] = "\n".join(block_lines)
            current_block = stripped[1:]  # Remove $
            block_lines = [line]
        elif current_block is not None:
            block_lines.append(line)
            if stripped.startswith("$end"):
                blocks[current_block] = "\n".join(block_lines)
                current_block = None
                block_lines = []

    return QchemInputComponents(
        molecule_block=molecule_block,
        rem_block=rem_block,
        blocks=blocks,
        raw_text=input_text,
    )


def assert_block_contains(block_text: str, *expected_settings: str) -> None:
    """
    Assert that a block contains expected settings (whitespace-independent).

    Example:
            assert_block_contains(rem_block, 'METHOD b3lyp', 'BASIS def2-svp')
    """
    # Normalize whitespace in the block
    normalized_block = " ".join(block_text.split())

    for setting in expected_settings:
        normalized_setting = " ".join(setting.split())
        assert normalized_setting in normalized_block, f"Expected '{setting}' in block, but got:\n{block_text}"


def assert_rem_contains_keys(rem_block: str, *expected_keys: str) -> None:
    """
    Assert that $rem block contains specific keys.

    Example:
            assert_rem_contains_keys(rem_block, 'METHOD', 'BASIS', 'JOBTYPE')
    """
    # Extract key-value pairs from rem block
    lines = [line.strip() for line in rem_block.split("\n") if line.strip() and not line.strip().startswith("$")]
    keys_present = set()
    for line in lines:
        parts = line.split()
        if parts:
            keys_present.add(parts[0])

    for key in expected_keys:
        assert key in keys_present, f"Expected key '{key}' not found in $rem block. Keys present: {keys_present}"


def assert_rem_value(rem_block: str, key: str, expected_value: str | bool | int) -> None:
    """
    Assert that $rem block contains a specific key with expected value.

    Example:
            assert_rem_value(rem_block, 'METHOD', 'b3lyp')
            assert_rem_value(rem_block, 'UNRESTRICTED', True)
    """
    lines = [line.strip() for line in rem_block.split("\n") if line.strip() and not line.strip().startswith("$")]
    for line in lines:
        parts = line.split()
        if parts and parts[0] == key:
            actual_value = " ".join(parts[1:])
            # Normalize boolean strings
            expected_str = str(expected_value).lower() if isinstance(expected_value, bool) else str(expected_value)
            assert actual_value.lower() == expected_str.lower(), (
                f"Expected {key} = {expected_value}, got {actual_value}\nFull rem block:\n{rem_block}"
            )
            return
    raise AssertionError(f"Key '{key}' not found in $rem block")


def assert_molecule_has_charge_mult(molecule_block: str, charge: int, multiplicity: int) -> None:
    """
    Assert $molecule block has correct charge and multiplicity.

    Example:
            assert_molecule_has_charge_mult(molecule_block, 0, 1)
    """
    lines = [line.strip() for line in molecule_block.split("\n") if line.strip()]
    # Second line (after $molecule) should be "charge multiplicity"
    if len(lines) >= 2:
        first_content_line = lines[1]
        parts = first_content_line.split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and parts[1].isdigit():
            actual_charge = int(parts[0])
            actual_mult = int(parts[1])
            assert actual_charge == charge, f"Expected charge {charge}, got {actual_charge}"
            assert actual_mult == multiplicity, f"Expected multiplicity {multiplicity}, got {actual_mult}"
            return

    raise AssertionError(f"Invalid $molecule header: {molecule_block}")


def assert_molecule_has_read_flag(molecule_block: str) -> None:
    """
    Assert $molecule block is set to read geometry from previous job.

    Example:
            assert_molecule_has_read_flag(molecule_block)
    """
    assert "read" in molecule_block.lower(), f"Expected 'read' in molecule block, but got:\n{molecule_block}"


def assert_molecule_has_atoms(molecule_block: str, *expected_atoms: tuple[str, int]) -> None:
    """
    Assert $molecule block contains specific atoms with correct counts.

    Example:
            assert_molecule_has_atoms(molecule_block, ('O', 1), ('H', 2))
    """
    lines = [line.strip() for line in molecule_block.split("\n") if line.strip()]
    atom_lines = [line for line in lines if line and not line.startswith("$")]

    # Skip the charge/mult line and header
    for symbol, expected_count in expected_atoms:
        actual_count = sum(1 for line in atom_lines if line.split() and line.split()[0] == symbol)
        assert actual_count == expected_count, (
            f"Expected {expected_count} {symbol} atoms, found {actual_count}\n$molecule block:\n{molecule_block}"
        )


def assert_block_present(blocks: dict[str, str], block_name: str) -> None:
    """
    Assert that a specific block exists.

    Example:
            assert_block_present(blocks, 'basis')
    """
    assert block_name in blocks, f"Expected ${block_name} block but not found. Blocks present: {list(blocks.keys())}"


def assert_block_not_present(blocks: dict[str, str], block_name: str) -> None:
    """
    Assert that a specific block does NOT exist.

    Example:
            assert_block_not_present(blocks, 'occupied')
    """
    assert block_name not in blocks, f"Unexpected ${block_name} block found in input"


def assert_basis_block_has_elements(blocks: dict[str, str], *expected_elements: str) -> None:
    """
    Assert that $basis block contains definitions for expected elements.

    Example:
            assert_basis_block_has_elements(blocks, 'H', 'O')
    """
    assert_block_present(blocks, "basis")
    basis_block = blocks["basis"]
    for element in expected_elements:
        assert element in basis_block, f"Expected {element} in $basis block, but got:\n{basis_block}"


def assert_occupied_block_contains(blocks: dict[str, str], *expected_occupations: str) -> None:
    """
    Assert that $occupied block contains expected occupation strings.

    Example:
            assert_occupied_block_contains(blocks, '1:5', '1:4')
    """
    assert_block_present(blocks, "occupied")
    occupied_block = blocks["occupied"]
    for occupation in expected_occupations:
        assert occupation in occupied_block, f"Expected '{occupation}' in $occupied block, but got:\n{occupied_block}"


def assert_two_job_structure(input_text: str) -> None:
    """
    Assert that input contains two jobs separated by @@@.

    Example:
            assert_two_job_structure(input_text)
    """
    assert "@@@" in input_text, "Expected two-job structure (separated by @@@) but not found"
    jobs = input_text.split("@@@")
    assert len(jobs) == 2, f"Expected exactly 2 jobs, found {len(jobs)}"


def extract_job(input_text: str, job_number: int) -> str:
    """
    Extract a specific job from a multi-job input (1-indexed).

    Example:
            job1 = extract_job(input_text, 1)
            job2 = extract_job(input_text, 2)
    """
    jobs = input_text.split("@@@")
    if job_number < 1 or job_number > len(jobs):
        raise ValueError(f"Job {job_number} not found. Total jobs: {len(jobs)}")
    return jobs[job_number - 1].strip()
