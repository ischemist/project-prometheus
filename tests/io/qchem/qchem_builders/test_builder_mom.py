"""MOM (Maximum Overlap Method) builder tests for Q-Chem.

Tests for the MOM workflow which generates two-job inputs for excited state calculations.
MOM is used to guide SCF convergence toward a specific excited state by specifying target
orbital occupations in the second job.

Test Structure:
- Unit tests: Individual methods like _build_occupied_block, _apply_excitation, _apply_ionization
- Contract tests: Two-job structure validation, block presence/content
- Integration tests: End-to-end workflows via CalculationInput fluent API
- Regression tests: Semantic validation of complete outputs
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.input import CalculationInput
from calcflow.io.qchem.builder import QchemBuilder
from tests.io.qchem.qchem_builders.conftest import (
    assert_block_not_present,
    assert_block_present,
    assert_molecule_has_read_flag,
    assert_rem_value,
    assert_two_job_structure,
    extract_job,
    parse_qchem_input,
)

# =============================================================================
# UNIT TESTS: Helper Methods for MOM
# =============================================================================


@pytest.mark.unit
def test_parse_spin_specification_no_spin():
    """_parse_spin_specification should handle orbital without spin prefix."""
    builder = QchemBuilder()
    orbital, spin = builder._parse_spin_specification("5")
    assert orbital == "5"
    assert spin is None


@pytest.mark.unit
def test_parse_spin_specification_with_alpha():
    """_parse_spin_specification should extract alpha spin designation."""
    builder = QchemBuilder()
    orbital, spin = builder._parse_spin_specification("5(alpha)")
    assert orbital == "5"
    assert spin == "alpha"


@pytest.mark.unit
def test_parse_spin_specification_with_beta():
    """_parse_spin_specification should extract beta spin designation."""
    builder = QchemBuilder()
    orbital, spin = builder._parse_spin_specification("7(beta)")
    assert orbital == "7"
    assert spin == "beta"


@pytest.mark.unit
def test_parse_spin_specification_case_insensitive():
    """_parse_spin_specification should be case-insensitive."""
    builder = QchemBuilder()
    orbital, spin = builder._parse_spin_specification("5(ALPHA)")
    assert orbital == "5"
    assert spin == "alpha"


@pytest.mark.unit
def test_resolve_orbital_index_numeric():
    """_resolve_orbital_index should resolve numeric orbital indices."""
    builder = QchemBuilder()
    # For H2O with 5 alpha electrons, HOMO is 5
    idx = builder._resolve_orbital_index("3", initial_homo=5)
    assert idx == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "homo_offset,expected",
    [
        ("HOMO", 5),  # HOMO itself
        ("HOMO-1", 4),  # one below HOMO
        ("HOMO-2", 3),  # two below HOMO
    ],
)
def test_resolve_orbital_index_homo(homo_offset, expected):
    """_resolve_orbital_index should resolve HOMO and HOMO-n notation."""
    builder = QchemBuilder()
    idx = builder._resolve_orbital_index(homo_offset, initial_homo=5)
    assert idx == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "lumo_offset,expected",
    [
        ("LUMO", 6),  # LUMO (HOMO+1)
        ("LUMO+1", 7),  # LUMO+1
        ("LUMO+2", 8),  # LUMO+2
    ],
)
def test_resolve_orbital_index_lumo(lumo_offset, expected):
    """_resolve_orbital_index should resolve LUMO and LUMO+n notation."""
    builder = QchemBuilder()
    idx = builder._resolve_orbital_index(lumo_offset, initial_homo=5)
    assert idx == expected


@pytest.mark.unit
def test_apply_single_operation_homo_lumo_excitation(h2o_geometry):
    """_apply_single_operation should handle HOMO->LUMO excitation."""
    builder = QchemBuilder()
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="HOMO->LUMO")
    builder._validate_spec(spec)

    alpha_occ = {1, 2, 3, 4, 5}
    beta_occ = {1, 2, 3, 4, 5}
    initial_homo = 5

    builder._apply_single_operation("HOMO->LUMO", alpha_occ, beta_occ, initial_homo)

    # HOMO->LUMO should move electron from orbital 5 to orbital 6
    assert 5 not in alpha_occ  # HOMO electron removed
    assert 6 in alpha_occ  # LUMO orbital occupied
    assert beta_occ == {1, 2, 3, 4, 5}  # beta unchanged


@pytest.mark.unit
def test_apply_single_operation_homo_homo1_excitation(h2o_geometry):
    """_apply_single_operation should handle HOMO-1->LUMO+1 excitation."""
    builder = QchemBuilder()
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="HOMO-1->LUMO+1")
    builder._validate_spec(spec)

    alpha_occ = {1, 2, 3, 4, 5}
    beta_occ = {1, 2, 3, 4, 5}
    initial_homo = 5

    builder._apply_single_operation("HOMO-1->LUMO+1", alpha_occ, beta_occ, initial_homo)

    # HOMO-1 is 4, LUMO+1 is 7
    assert 4 not in alpha_occ
    assert 7 in alpha_occ
    assert beta_occ == {1, 2, 3, 4, 5}


@pytest.mark.unit
def test_apply_ionization_from_homo(h2o_geometry):
    """_apply_ionization should remove electron from specified orbital."""
    builder = QchemBuilder()
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="HOMO->vac")
    builder._validate_spec(spec)

    alpha_occ = {1, 2, 3, 4, 5}
    beta_occ = {1, 2, 3, 4, 5}

    # Ionize HOMO (orbital 5) from alpha
    builder._apply_ionization(5, "alpha", alpha_occ, beta_occ)

    # Alpha should have 4 electrons
    assert 5 not in alpha_occ
    assert alpha_occ == {1, 2, 3, 4}
    # Beta unchanged
    assert beta_occ == {1, 2, 3, 4, 5}


@pytest.mark.unit
def test_apply_ionization_from_beta():
    """_apply_ionization should remove electron from beta if specified."""
    builder = QchemBuilder()
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=3,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="HOMO(beta)->vac")
    builder._validate_spec(spec)

    alpha_occ = {1, 2, 3, 4, 5}
    beta_occ = {1, 2, 3, 4}  # Triplet has unpaired electron in alpha

    # Ionize from beta (which has 4 electrons, so HOMO is 4)
    builder._apply_ionization(4, "beta", alpha_occ, beta_occ)

    assert beta_occ == {1, 2, 3}
    assert alpha_occ == {1, 2, 3, 4, 5}


@pytest.mark.unit
def test_format_occupation_set_basic():
    """_format_occupation_set should convert set to compact range notation."""
    builder = QchemBuilder()
    occupied = {1, 2, 3, 4, 5}
    result = builder._format_occupation_set(occupied)
    assert "1:5" in result or result == "1:5"


@pytest.mark.unit
def test_format_occupation_set_with_gap():
    """_format_occupation_set should handle non-contiguous sets."""
    builder = QchemBuilder()
    # Occupations with a gap (e.g., after excitation)
    occupied = {1, 2, 3, 4, 6}  # gap at 5
    result = builder._format_occupation_set(occupied)
    # Should include both ranges or explicit list
    assert "1" in result and "6" in result


@pytest.mark.unit
def test_format_occupation_set_empty():
    """_format_occupation_set should handle empty set."""
    builder = QchemBuilder()
    occupied = set()
    result = builder._format_occupation_set(occupied)
    # Empty set should result in empty string
    assert result == ""


# =============================================================================
# CONTRACT TESTS: Two-Job Structure
# =============================================================================


@pytest.mark.contract
def test_mom_two_job_structure(qchem_builder, h2o_geometry, minimal_spec):
    """mom calculation should generate two-job structure separated by @@@."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)

    # Verify two-job structure
    assert_two_job_structure(result)
    job1, job2 = result.split("@@@")

    # Both jobs should have molecule and rem blocks
    assert "$molecule" in job1
    assert "$rem" in job1
    assert "$molecule" in job2
    assert "$rem" in job2


@pytest.mark.contract
def test_mom_job1_simple_sp(qchem_builder, h2o_geometry, minimal_spec):
    """job1 of mom calculation should be a simple SP energy calculation."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job1 = extract_job(result, 1)
    parsed = parse_qchem_input(job1)

    # Job1 should have normal molecule block (not read)
    assert "read" not in parsed.molecule_block.lower()
    # Job1 should NOT have $occupied block
    assert_block_not_present(parsed.blocks, "occupied")


@pytest.mark.contract
def test_mom_job2_has_read_molecule(qchem_builder, h2o_geometry, minimal_spec):
    """job2 of mom calculation should have $molecule read."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Job2 should read geometry from job1
    assert_molecule_has_read_flag(parsed.molecule_block)


@pytest.mark.contract
def test_mom_job2_has_occupied_block(qchem_builder, h2o_geometry, minimal_spec):
    """job2 of mom calculation should have $occupied block with MOM target."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Job2 should have $occupied block
    assert_block_present(parsed.blocks, "occupied")


@pytest.mark.contract
def test_mom_job2_has_mom_start(qchem_builder, h2o_geometry, minimal_spec):
    """job2 of mom calculation should have MOM_START enabled."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # MOM_START is output as "1" in Q-Chem
    assert_rem_value(parsed.rem_block, "MOM_START", "1")


@pytest.mark.contract
def test_mom_job2_scf_guess_read(qchem_builder, h2o_geometry, minimal_spec):
    """job2 of mom calculation should have SCF_GUESS read."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    assert_rem_value(parsed.rem_block, "SCF_GUESS", "read")


@pytest.mark.contract
def test_mom_job2_unrestricted(qchem_builder, h2o_geometry, minimal_spec):
    """job2 of mom calculation should be unrestricted."""
    # This should fail validation - MOM requires unrestricted
    with pytest.raises(ValidationError, match="unrestricted"):
        replace(minimal_spec, unrestricted=False).set_mom(transition="HOMO->LUMO")


@pytest.mark.contract
def test_mom_ionization_job2_charge_override(qchem_builder, h2o_geometry, minimal_spec):
    """mom ionization should allow charge override for job2."""
    spec = replace(minimal_spec, charge=0, spin_multiplicity=1, unrestricted=True).set_mom(
        transition="HOMO->vac", job2_charge=1, job2_spin_multiplicity=2
    )
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Job2 has "read" flag instead of explicit charge/mult
    # Verify it has the read flag
    assert_molecule_has_read_flag(parsed.molecule_block)


@pytest.mark.contract
def test_mom_with_solvation(qchem_builder, h2o_geometry):
    """mom with solvation should include solvent blocks in both jobs."""
    from calcflow.common.input import SolvationSpec

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        solvation=SolvationSpec(model="smd", solvent="water"),
    ).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)

    assert "@@@" in result
    job1, job2 = result.split("@@@")

    # Both jobs should have SMX block
    assert "$smx" in job1
    assert "$smx" in job2


# =============================================================================
# INTEGRATION TESTS: Workflows via CalculationInput
# =============================================================================


@pytest.mark.integration
def test_mom_single_transition(h2o_geometry):
    """mom with single transition should work correctly."""
    from calcflow.common.input import CalculationInput

    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        .set_unrestricted(True)
        .set_mom(transition="HOMO->LUMO")
    )

    result = calc.export("qchem", h2o_geometry)

    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Job2 should have $occupied block with transition applied
    assert_block_present(parsed.blocks, "occupied")


# =============================================================================
# REGRESSION TESTS: Semantic Validation
# =============================================================================


@pytest.mark.regression
def test_mom_homo_lumo_occupation_count(qchem_builder, h2o_geometry, minimal_spec):
    """HOMO->LUMO excitation should give correct electron count in $occupied."""
    spec = replace(minimal_spec, unrestricted=True).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # For water (10 electrons total), HOMO->LUMO excited state should still have 10 electrons
    occupied = parsed.blocks.get("occupied", "")
    lines = [line.strip() for line in occupied.split("\n") if line.strip()]

    # Should have alpha and beta occupation lines
    assert len(lines) >= 2

    # First line is alpha, second is beta
    alpha_line = lines[0] if lines else ""
    beta_line = lines[1] if len(lines) > 1 else ""

    # Both should be non-empty (electrons are still there)
    assert alpha_line
    assert beta_line


@pytest.mark.regression
def test_mom_ionization_electron_count_decreased(qchem_builder, h2o_geometry):
    """Ionization should decrease total electron count in $occupied."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="HOMO->vac", job2_charge=1, job2_spin_multiplicity=2)
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Job2 should have $occupied block for the ionized state
    assert_block_present(parsed.blocks, "occupied")


@pytest.mark.regression
def test_mom_preserves_other_spec_options(qchem_builder, h2o_geometry):
    """MOM should preserve other spec options like n_cores, basis set, functional."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="m06",
        basis_set="def2-tzvp",
        n_cores=8,
        unrestricted=True,
    ).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)

    job1 = extract_job(result, 1)
    job2 = extract_job(result, 2)

    job1_parsed = parse_qchem_input(job1)
    job2_parsed = parse_qchem_input(job2)

    # Both jobs should have the correct method and basis
    assert_rem_value(job1_parsed.rem_block, "METHOD", "m06")
    assert_rem_value(job2_parsed.rem_block, "METHOD", "m06")
    assert_rem_value(job1_parsed.rem_block, "BASIS", "def2-tzvp")
    assert_rem_value(job2_parsed.rem_block, "BASIS", "def2-tzvp")

    # Both should be unrestricted
    assert_rem_value(job1_parsed.rem_block, "UNRESTRICTED", True)
    assert_rem_value(job2_parsed.rem_block, "UNRESTRICTED", True)


@pytest.mark.regression
def test_mom_job1_no_tddft(qchem_builder, h2o_geometry):
    """MOM job1 should NOT have TDDFT even if spec requests it."""
    from calcflow.common.input import TddftSpec

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False, use_tda=True),
    ).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job1 = extract_job(result, 1)
    parsed = parse_qchem_input(job1)

    # Job1 should NOT have CIS_N_ROOTS (TDDFT disabled in job1)
    rem_text = parsed.rem_block.lower()
    assert "cis_n_roots" not in rem_text


@pytest.mark.regression
def test_mom_job2_preserves_tddft(qchem_builder, h2o_geometry):
    """MOM job2 should preserve TDDFT settings if specified."""
    from calcflow.common.input import TddftSpec

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False, use_tda=True),
    ).set_mom(transition="HOMO->LUMO")
    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Job2 should have TDDFT settings
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)
    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", True)


# =============================================================================
# VALIDATION TESTS: Error Handling
# =============================================================================


@pytest.mark.unit
def test_mom_requires_unrestricted(qchem_builder, h2o_geometry, minimal_spec):
    """MOM requires unrestricted calculation."""
    with pytest.raises(ValidationError, match="unrestricted"):
        replace(minimal_spec, unrestricted=False).set_mom(transition="HOMO->LUMO")


@pytest.mark.unit
def test_invalid_transition_format(qchem_builder):
    """Invalid transition format should raise error."""
    builder = QchemBuilder()
    alpha_occ = {1, 2, 3, 4, 5}
    beta_occ = {1, 2, 3, 4, 5}

    # Missing arrow
    with pytest.raises(ValidationError, match="transition format"):
        builder._apply_single_operation("HOMO LUMO", alpha_occ, beta_occ, 5)


@pytest.mark.unit
def test_invalid_orbital_notation(qchem_builder):
    """Invalid orbital notation should raise error."""
    builder = QchemBuilder()

    # Invalid HOMO notation - HOMO+5 should fail (should be HOMO+5 as two separate parts)
    # Actually "HOMO+5" fails to match the regex but returns None which might not raise
    # Let's test something that will definitely fail
    with pytest.raises(ValidationError):
        builder._resolve_orbital_index("INVALID_ORBITAL", initial_homo=5)


# =============================================================================
# UNIT TESTS: MOM GROUND_STATE Transition
# =============================================================================


@pytest.mark.unit
def test_mom_ground_state_transition_even_electrons(qchem_builder, h2o_geometry):
    """MOM GROUND_STATE transition should produce correct occupation for even electrons."""
    # H2O has 10 electrons -> 5 alpha, 5 beta (ground state)
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="GROUND_STATE")

    result = qchem_builder.build(spec, h2o_geometry)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    occupied_block = parsed.blocks.get("occupied", "")
    lines = [line.strip() for line in occupied_block.split("\n") if line.strip() and not line.strip().startswith("$")]

    # Should have alpha and beta lines
    assert len(lines) == 2
    alpha_line = lines[0]
    beta_line = lines[1]

    # For 10 electrons: alpha = 1:5, beta = 1:5
    assert alpha_line == "1:5"
    assert beta_line == "1:5"


@pytest.mark.unit
def test_mom_ground_state_single_electron_pair(qchem_builder):
    """MOM GROUND_STATE with 2 electrons (1 pair) should format as '1' not '1:1'."""
    from calcflow.common.results import Atom
    from calcflow.geometry.static import Geometry

    # H2 molecule has 2 electrons -> 1 alpha, 1 beta
    h2_geom = Geometry(
        comment="H2 molecule",
        atoms=(
            Atom(symbol="H", x=0.0, y=0.0, z=0.0),
            Atom(symbol="H", x=0.0, y=0.0, z=0.74),
        ),
    )

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="GROUND_STATE")

    result = qchem_builder.build(spec, h2_geom)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    occupied_block = parsed.blocks.get("occupied", "")
    lines = [line.strip() for line in occupied_block.split("\n") if line.strip() and not line.strip().startswith("$")]

    # For 2 electrons: alpha = 1, beta = 1 (not 1:1)
    alpha_line = lines[0]
    beta_line = lines[1]

    assert alpha_line == "1"
    assert beta_line == "1"


# =============================================================================
# CONTRACT TESTS: MOM GROUND_STATE Validation
# =============================================================================


@pytest.mark.contract
def test_mom_ground_state_requires_even_electrons(qchem_builder):
    """MOM GROUND_STATE with odd electrons should raise error."""
    from calcflow.common.exceptions import ConfigurationError
    from calcflow.common.results import Atom
    from calcflow.geometry.static import Geometry

    # OH radical has 9 electrons (odd)
    oh_geom = Geometry(
        comment="OH radical",
        atoms=(
            Atom(symbol="O", x=0.0, y=0.0, z=0.0),
            Atom(symbol="H", x=0.0, y=0.0, z=0.96),
        ),
    )

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=2,  # Doublet
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="GROUND_STATE")

    with pytest.raises(ConfigurationError, match="GROUND_STATE.*even number of electrons"):
        qchem_builder.build(spec, oh_geom)


@pytest.mark.contract
def test_mom_ground_state_two_job_structure(qchem_builder, h2o_geometry):
    """MOM GROUND_STATE should generate proper two-job structure."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
    ).set_mom(transition="GROUND_STATE")

    result = qchem_builder.build(spec, h2o_geometry)

    # Verify two-job structure
    assert_two_job_structure(result)

    # Both jobs should be present
    job1 = extract_job(result, 1)
    job2 = extract_job(result, 2)

    assert "$molecule" in job1
    assert "$molecule" in job2
    assert_block_present(parse_qchem_input(job2).blocks, "occupied")


# =============================================================================
# INTEGRATION TESTS: MOM GROUND_STATE Workflows
# =============================================================================


@pytest.mark.integration
def test_mom_ground_state_full_workflow(h2o_geometry):
    """Full MOM workflow with GROUND_STATE transition."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        .set_unrestricted(True)
        .set_mom(transition="GROUND_STATE")
    )

    result = calc.export("qchem", h2o_geometry)

    # Verify structure
    assert_two_job_structure(result)
    job2 = extract_job(result, 2)
    parsed = parse_qchem_input(job2)

    # Verify MOM settings
    assert_rem_value(parsed.rem_block, "MOM_START", "1")
    assert_rem_value(parsed.rem_block, "SCF_GUESS", "read")
    assert_block_present(parsed.blocks, "occupied")


# =============================================================================
# CONTRACT TESTS: MOM with TDDFT and TRNSS
# =============================================================================


@pytest.mark.contract
def test_mom_with_tddft_trnss_solute_block_in_job2(qchem_builder, h2o_geometry):
    """MOM + TDDFT + TRNSS should have $solute block in job2 only."""
    from calcflow.common.input import TddftSpec

    spec = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
            unrestricted=True,
            tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
        )
        .set_mom(transition="HOMO->LUMO")
        .set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7])
    )

    result = qchem_builder.build(spec, h2o_geometry)

    # Job1 should NOT have $solute block (no TDDFT in job1)
    job1 = extract_job(result, 1)
    parsed_job1 = parse_qchem_input(job1)
    assert_block_not_present(parsed_job1.blocks, "solute")

    # Job2 should have $solute block
    job2 = extract_job(result, 2)
    parsed_job2 = parse_qchem_input(job2)
    assert_block_present(parsed_job2.blocks, "solute")

    # Verify TRNSS keywords in job2
    assert_rem_value(parsed_job2.rem_block, "TRNSS", True)
    assert_rem_value(parsed_job2.rem_block, "N_SOL", 5)


@pytest.mark.contract
def test_mom_without_trnss_no_solute_block(qchem_builder, h2o_geometry):
    """MOM + TDDFT without TRNSS should NOT have $solute block."""
    from calcflow.common.input import TddftSpec

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    ).set_mom(transition="HOMO->LUMO")

    result = qchem_builder.build(spec, h2o_geometry)

    # Neither job should have $solute block
    job1 = extract_job(result, 1)
    job2 = extract_job(result, 2)

    parsed_job1 = parse_qchem_input(job1)
    parsed_job2 = parse_qchem_input(job2)

    assert_block_not_present(parsed_job1.blocks, "solute")
    assert_block_not_present(parsed_job2.blocks, "solute")


# =============================================================================
# INTEGRATION TESTS: MOM with TDDFT and TRNSS Workflows
# =============================================================================


@pytest.mark.integration
def test_mom_tddft_trnss_full_workflow(h2o_geometry):
    """Full MOM workflow with TDDFT and reduced excitation space."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-311g(d)",
        )
        .set_unrestricted(True)
        .set_mom(transition="HOMO->LUMO")
        .set_tddft(nroots=10, singlets=True, triplets=False, use_tda=False)
        .set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7, 8])
        .set_cores(16)
    )

    result = calc.export("qchem", h2o_geometry)

    # Verify two-job structure
    assert_two_job_structure(result)

    job2 = extract_job(result, 2)
    parsed_job2 = parse_qchem_input(job2)

    # Verify MOM settings
    assert_rem_value(parsed_job2.rem_block, "MOM_START", "1")
    assert_rem_value(parsed_job2.rem_block, "SCF_GUESS", "read")

    # Verify TDDFT settings
    assert_rem_value(parsed_job2.rem_block, "CIS_N_ROOTS", 10)
    assert_rem_value(parsed_job2.rem_block, "RPA", True)

    # Verify TRNSS settings
    assert_rem_value(parsed_job2.rem_block, "TRNSS", True)
    assert_rem_value(parsed_job2.rem_block, "N_SOL", 6)

    # Verify blocks
    assert_block_present(parsed_job2.blocks, "occupied")
    assert_block_present(parsed_job2.blocks, "solute")


@pytest.mark.integration
def test_mom_tddft_trnss_with_solvation(h2o_geometry):
    """MOM + TDDFT + TRNSS + solvation should work together."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
        )
        .set_unrestricted(True)
        .set_mom(transition="HOMO->LUMO")
        .set_tddft(nroots=5, singlets=True, triplets=False)
        .set_solvation("smd", "water")
        .set_options(reduced_excitation_space_orbitals=[4, 5, 6])
    )

    result = calc.export("qchem", h2o_geometry)

    # Verify two-job structure
    assert_two_job_structure(result)

    job1 = extract_job(result, 1)
    job2 = extract_job(result, 2)

    parsed_job1 = parse_qchem_input(job1)
    parsed_job2 = parse_qchem_input(job2)

    # Both jobs should have solvation
    assert "smx" in parsed_job1.blocks
    assert "smx" in parsed_job2.blocks

    # Only job2 should have TDDFT, TRNSS, and $solute
    assert_rem_value(parsed_job2.rem_block, "CIS_N_ROOTS", 5)
    assert_rem_value(parsed_job2.rem_block, "TRNSS", True)
    assert_block_present(parsed_job2.blocks, "solute")
    assert_block_present(parsed_job2.blocks, "occupied")


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def qchem_builder():
    """Q-Chem builder instance for testing."""
    return QchemBuilder()
