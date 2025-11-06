"""TDDFT builder tests for Q-Chem.

Tests for Time-Dependent DFT (TDDFT/CIS) functionality in Q-Chem input generation.
TDDFT allows calculation of excited state properties and is typically combined with
other features like solvation and optimization.

Test Structure:
- Unit tests: TDDFT keyword generation
- Contract tests: Block structure and keyword presence
- Integration tests: Fluent API workflows
- Regression tests: Semantic validation of complete outputs
"""

from __future__ import annotations

import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.input import CalculationInput, TddftSpec
from calcflow.io.qchem.builder import QchemBuilder
from tests.io.qchem.qchem_builders.conftest import (
    assert_block_not_present,
    assert_block_present,
    assert_rem_value,
    parse_qchem_input,
)

# =============================================================================
# UNIT TESTS: TDDFT Keyword Generation
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "nroots,expected_rem_value",
    [
        (1, 1),
        (5, 5),
        (10, 10),
        (20, 20),
    ],
)
def test_tddft_nroots_keyword(qchem_builder, h2o_geometry, nroots, expected_rem_value):
    """TDDFT nroots should set CIS_N_ROOTS correctly."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=nroots, singlets=True, triplets=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", expected_rem_value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "singlets,triplets,expected_sing,expected_trip",
    [
        (True, False, True, False),  # Only singlets
        (False, True, False, True),  # Only triplets
        (True, True, True, True),  # Both
        (False, False, False, False),  # Neither (unusual but should work)
    ],
)
def test_tddft_singlets_triplets(qchem_builder, h2o_geometry, singlets, triplets, expected_sing, expected_trip):
    """TDDFT should set CIS_SINGLETS and CIS_TRIPLETS correctly."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=singlets, triplets=triplets),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", expected_sing)
    assert_rem_value(parsed.rem_block, "CIS_TRIPLETS", expected_trip)


@pytest.mark.unit
@pytest.mark.parametrize(
    "use_tda,expected_rpa",
    [
        (True, False),  # use_tda=True -> RPA=False (TDA)
        (False, True),  # use_tda=False -> RPA=True
    ],
)
def test_tddft_use_tda_rpa_flag(qchem_builder, h2o_geometry, use_tda, expected_rpa):
    """TDDFT use_tda should set RPA flag correctly (inverted)."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False, use_tda=use_tda),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "RPA", expected_rpa)


# =============================================================================
# CONTRACT TESTS: TDDFT Structure and Content
# =============================================================================


@pytest.mark.contract
def test_tddft_keywords_present(qchem_builder, h2o_geometry):
    """TDDFT calculation should have all required CIS keywords."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=10, singlets=True, triplets=True, use_tda=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have all TDDFT keywords
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)
    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", True)
    assert_rem_value(parsed.rem_block, "CIS_TRIPLETS", True)
    assert_rem_value(parsed.rem_block, "RPA", True)  # RPA method


@pytest.mark.contract
def test_tddft_with_sp_task(qchem_builder, h2o_geometry):
    """TDDFT should work with energy (SP) task."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "JOBTYPE", "sp")
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)


@pytest.mark.contract
def test_tddft_without_tda_uses_rpa(qchem_builder, h2o_geometry):
    """TDDFT without TDA should enable RPA (full TDDFT)."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, use_tda=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "RPA", True)


@pytest.mark.contract
def test_tddft_with_tda(qchem_builder, h2o_geometry):
    """TDDFT with TDA approximation should set RPA=False."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, use_tda=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "RPA", False)


@pytest.mark.contract
def test_tddft_unrestricted(qchem_builder, h2o_geometry):
    """Unrestricted TDDFT should work correctly."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=3,  # Triplet
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        tddft=TddftSpec(nroots=5, singlets=True, triplets=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "UNRESTRICTED", True)
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)


@pytest.mark.contract
def test_tddft_with_solvation(qchem_builder, h2o_geometry):
    """TDDFT with solvation should include solvent blocks."""
    from calcflow.common.input import SolvationSpec

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="smd", solvent="water"),
        tddft=TddftSpec(nroots=10, singlets=True, triplets=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have both TDDFT and solvation
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)
    assert "smx" in parsed.blocks


# =============================================================================
# INTEGRATION TESTS: TDDFT Workflows via CalculationInput
# =============================================================================


@pytest.mark.integration
def test_tddft_singlet_excitations_workflow(h2o_geometry):
    """TDDFT singlet excitation workflow should generate correct input."""

    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-311g(d)",
        )
        .set_tddft(nroots=10, singlets=True, triplets=False, use_tda=False)
        .set_cores(16)
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    # Verify TDDFT settings
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)
    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", True)
    assert_rem_value(parsed.rem_block, "CIS_TRIPLETS", False)
    assert_rem_value(parsed.rem_block, "RPA", True)


@pytest.mark.integration
def test_tddft_tda_workflow(h2o_geometry):
    """TDDFT with TDA approximation should work via fluent API."""

    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
    ).set_tddft(nroots=5, singlets=True, triplets=False, use_tda=True)

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)
    assert_rem_value(parsed.rem_block, "RPA", False)  # TDA mode


@pytest.mark.integration
def test_tddft_with_solvation_workflow(h2o_geometry):
    """TDDFT combined with solvation should work correctly."""

    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
        )
        .set_tddft(nroots=10, singlets=True, triplets=False, use_tda=False)
        .set_solvation("smd", "water")
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have both TDDFT and solvation
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)
    assert "smx" in parsed.blocks


@pytest.mark.integration
def test_tddft_unrestricted_workflow(h2o_geometry):
    """TDDFT for excited triplet state should work."""

    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=3,
            task="energy",  # Triplet state
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
        )
        .set_unrestricted(True)
        .set_tddft(nroots=5, singlets=False, triplets=True, use_tda=False)
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "UNRESTRICTED", True)
    assert_rem_value(parsed.rem_block, "CIS_TRIPLETS", True)
    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", False)


@pytest.mark.integration
def test_tddft_with_different_functionals(h2o_geometry):
    """TDDFT should work with different functionals."""

    for functional in ["b3lyp", "cam-b3lyp", "pbe0", "m06"]:
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory=functional,
            basis_set="6-31g",
        ).set_tddft(nroots=5, singlets=True, triplets=False)

        result = calc.export("qchem", h2o_geometry)
        parsed = parse_qchem_input(result)

        assert_rem_value(parsed.rem_block, "METHOD", functional)
        assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)


# =============================================================================
# REGRESSION TESTS: Semantic Validation
# =============================================================================


@pytest.mark.regression
def test_tddft_basic_output_structure(qchem_builder, h2o_geometry):
    """TDDFT output should have complete structure."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False, use_tda=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)

    # Should be single job (not two-job structure)
    assert "@@@" not in result

    parsed = parse_qchem_input(result)

    # All required blocks present
    assert "$molecule" in parsed.molecule_block
    assert "$rem" in parsed.rem_block
    # For standard basis sets, $basis block is not needed
    # (it's only generated for dictionary basis sets)


@pytest.mark.regression
def test_tddft_preserves_other_settings(qchem_builder, h2o_geometry):
    """TDDFT should preserve other spec settings like n_cores, memory."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="def2-tzvp",
        n_cores=8,
        memory_per_core_mb=4000,
        tddft=TddftSpec(nroots=10, singlets=True, triplets=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "METHOD", "cam-b3lyp")
    assert_rem_value(parsed.rem_block, "BASIS", "def2-tzvp")
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)
    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", True)
    assert_rem_value(parsed.rem_block, "CIS_TRIPLETS", True)


@pytest.mark.regression
def test_tddft_no_tddft_when_none(qchem_builder, h2o_geometry, minimal_spec):
    """Non-TDDFT calculation should not have CIS keywords."""
    # minimal_spec has no TDDFT
    result = qchem_builder.build(minimal_spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    rem_text = parsed.rem_block.lower()
    assert "cis_n_roots" not in rem_text
    assert "cis_singlets" not in rem_text
    assert "cis_triplets" not in rem_text


@pytest.mark.regression
@pytest.mark.parametrize("method", ["b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x"])
def test_tddft_with_various_methods(qchem_builder, h2o_geometry, method):
    """TDDFT should work with various DFT functionals."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory=method,
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "METHOD", method)
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.unit
def test_tddft_zero_nroots_validation(minimal_spec):
    """TDDFT with zero nroots should be rejected during spec validation."""
    with pytest.raises(ValidationError, match="nroots"):
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
            tddft=TddftSpec(nroots=0, singlets=True, triplets=False),
        )


@pytest.mark.unit
def test_tddft_negative_nroots_validation():
    """TDDFT with negative nroots should be rejected."""
    with pytest.raises(ValidationError, match="nroots"):
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
            tddft=TddftSpec(nroots=-1, singlets=True, triplets=False),
        )


# =============================================================================
# UNIT TESTS: TDDFT with Reduced Excitation Space (TRNSS)
# =============================================================================


@pytest.mark.unit
def test_tddft_trnss_keywords_set(qchem_builder, h2o_geometry):
    """TDDFT with reduced excitation space should set TRNSS keywords."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    ).set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7])

    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "TRNSS", True)
    assert_rem_value(parsed.rem_block, "TRTYPE", 3)
    assert_rem_value(parsed.rem_block, "N_SOL", 5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "orbitals,expected_n_sol",
    [
        ([3, 4, 5], 3),
        ([1, 2, 3, 4, 5, 6, 7, 8], 8),
        ([5], 1),
        ([2, 3, 4, 5, 6], 5),
    ],
)
def test_tddft_trnss_n_sol_matches_orbital_count(qchem_builder, h2o_geometry, orbitals, expected_n_sol):
    """N_SOL should match the number of orbitals in reduced excitation space."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    ).set_options(reduced_excitation_space_orbitals=orbitals)

    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "N_SOL", expected_n_sol)


# =============================================================================
# CONTRACT TESTS: TDDFT with TRNSS Structure
# =============================================================================


@pytest.mark.contract
def test_tddft_trnss_solute_block_present(qchem_builder, h2o_geometry):
    """TDDFT with reduced excitation space should have $solute block."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    ).set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7])

    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have $solute block
    assert_block_present(parsed.blocks, "solute")

    # Check content includes the orbital list
    solute_block = parsed.blocks.get("solute", "")
    assert "3 4 5 6 7" in solute_block or "3" in solute_block


@pytest.mark.contract
def test_tddft_trnss_requires_tddft(qchem_builder, h2o_geometry):
    """Reduced excitation space without TDDFT should raise validation error."""
    from calcflow.common.exceptions import ConfigurationError

    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        # No TDDFT spec
    ).set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7])

    with pytest.raises(ConfigurationError, match="reduced excitation space requires tddft"):
        qchem_builder.build(spec, h2o_geometry)


@pytest.mark.contract
def test_tddft_trnss_orbital_list_format(qchem_builder, h2o_geometry):
    """$solute block should format orbital list as space-separated integers."""
    orbitals = [2, 3, 4, 5, 6, 7, 8]
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=10, singlets=True, triplets=False),
    ).set_options(reduced_excitation_space_orbitals=orbitals)

    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    solute_block = parsed.blocks.get("solute", "")
    # Check all orbitals are present as strings
    for orb in orbitals:
        assert str(orb) in solute_block


@pytest.mark.contract
def test_tddft_without_trnss_no_solute_block(qchem_builder, h2o_geometry):
    """TDDFT without reduced excitation space should NOT have $solute block."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False),
    )

    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should NOT have $solute block
    assert_block_not_present(parsed.blocks, "solute")


# =============================================================================
# INTEGRATION TESTS: TDDFT with TRNSS Workflows
# =============================================================================


@pytest.mark.integration
def test_tddft_trnss_full_workflow(h2o_geometry):
    """Full TDDFT workflow with reduced excitation space."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-311g(d)",
        )
        .set_tddft(nroots=10, singlets=True, triplets=False, use_tda=False)
        .set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7, 8])
        .set_cores(16)
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    # Verify TDDFT settings
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)
    assert_rem_value(parsed.rem_block, "CIS_SINGLETS", True)
    assert_rem_value(parsed.rem_block, "RPA", True)

    # Verify TRNSS settings
    assert_rem_value(parsed.rem_block, "TRNSS", True)
    assert_rem_value(parsed.rem_block, "TRTYPE", 3)
    assert_rem_value(parsed.rem_block, "N_SOL", 6)

    # Verify $solute block
    assert_block_present(parsed.blocks, "solute")


@pytest.mark.integration
def test_tddft_trnss_with_solvation(h2o_geometry):
    """TDDFT with reduced excitation space and solvation should work together."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
        )
        .set_tddft(nroots=5, singlets=True, triplets=False)
        .set_solvation("smd", "water")
        .set_options(reduced_excitation_space_orbitals=[4, 5, 6])
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have TDDFT, TRNSS, solvation, and $solute
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 5)
    assert_rem_value(parsed.rem_block, "TRNSS", True)
    assert "smx" in parsed.blocks
    assert_block_present(parsed.blocks, "solute")


# =============================================================================
# REGRESSION TESTS: TDDFT with TRNSS Semantic Validation
# =============================================================================


@pytest.mark.regression
def test_tddft_trnss_complete_output_structure(qchem_builder, h2o_geometry):
    """TDDFT with TRNSS should have complete and correct output structure."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g",
        tddft=TddftSpec(nroots=5, singlets=True, triplets=False, use_tda=True),
    ).set_options(reduced_excitation_space_orbitals=[3, 4, 5, 6, 7])

    result = qchem_builder.build(spec, h2o_geometry)

    # Should be single job (not two-job structure)
    assert "@@@" not in result

    parsed = parse_qchem_input(result)

    # All required blocks present
    assert "$molecule" in parsed.molecule_block
    assert "$rem" in parsed.rem_block
    assert_block_present(parsed.blocks, "solute")

    # All TRNSS keywords present
    assert_rem_value(parsed.rem_block, "TRNSS", True)
    assert_rem_value(parsed.rem_block, "TRTYPE", 3)
    assert_rem_value(parsed.rem_block, "N_SOL", 5)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def qchem_builder():
    """Q-Chem builder instance for testing."""
    return QchemBuilder()
