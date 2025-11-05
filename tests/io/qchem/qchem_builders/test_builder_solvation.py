"""Solvation model tests for Q-Chem builder.

Tests for solvation functionality in Q-Chem input generation.
Q-Chem supports multiple implicit solvation models: PCM, SMD, ISOSVP, CPCM.

Test Structure:
- Unit tests: Solvation block generation, solvent mapping
- Contract tests: Block structure and keyword presence for each model
- Integration tests: Fluent API workflows with solvation
- Regression tests: Semantic validation of complete outputs
"""

from __future__ import annotations

import pytest

from calcflow.common.exceptions import NotSupportedError
from calcflow.common.input import CalculationInput, SolvationSpec
from calcflow.io.qchem.builder import QchemBuilder
from tests.io.qchem.qchem_builders.conftest import (
    assert_block_present,
    assert_rem_value,
    parse_qchem_input,
)

# =============================================================================
# UNIT TESTS: Solvation Block Generation
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "model,solvent",
    [
        ("pcm", "water"),
        ("pcm", "acetonitrile"),
        ("smd", "water"),
        ("smd", "acetonitrile"),
        ("isosvp", "water"),
        ("cpcm", "water"),
    ],
)
def test_solvation_model_support(qchem_builder, h2o_geometry, model, solvent):
    """Builder should support PCM, SMD, ISOSVP, and CPCM solvation models."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model=model, solvent=solvent),
    )
    # Should not raise NotSupportedError
    result = qchem_builder.build(spec, h2o_geometry)
    assert result is not None


@pytest.mark.unit
def test_unsupported_solvation_model(qchem_builder, h2o_geometry):
    """Builder should reject unsupported solvation models."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="cosmo", solvent="water"),  # COSMO not supported in Q-Chem builder
    )
    with pytest.raises(NotSupportedError):
        qchem_builder.build(spec, h2o_geometry)


@pytest.mark.unit
def test_solvation_block_generation_smd(qchem_builder, h2o_geometry):
    """SMD solvation should generate $smx block."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="smd", solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)

    # SMD model should create $smx block
    assert "$smx" in result


@pytest.mark.unit
def test_solvation_block_generation_pcm(qchem_builder, h2o_geometry):
    """PCM solvation should include PCM keywords in $rem."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="pcm", solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # PCM should enable SOLVENT_METHOD or similar
    rem_text = parsed.rem_block.lower()
    assert "solvent" in rem_text or "pcm" in rem_text


# =============================================================================
# CONTRACT TESTS: Solvation Structure and Keywords
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize("model", ["pcm", "smd", "isosvp", "cpcm"])
def test_solvation_models_have_solvent_setup(qchem_builder, h2o_geometry, model):
    """Each solvation model should set up solvent information."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model=model, solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)

    # Should generate some solvent configuration
    assert "water" in result.lower() or "solvent" in result.lower()


@pytest.mark.contract
def test_smd_generates_smx_block(qchem_builder, h2o_geometry):
    """SMD model should generate $smx block."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="smd", solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_block_present(parsed.blocks, "smx")


@pytest.mark.contract
def test_solvation_with_unrestricted(qchem_builder, h2o_geometry):
    """Solvation should work with unrestricted calculations."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=3,  # Triplet
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        solvation=SolvationSpec(model="smd", solvent="acetonitrile"),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "UNRESTRICTED", True)
    assert_block_present(parsed.blocks, "smx")


@pytest.mark.contract
def test_solvation_preserves_basis_and_method(qchem_builder, h2o_geometry):
    """Solvation should not affect basis set and method choices."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="def2-tzvp",
        solvation=SolvationSpec(model="smd", solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "METHOD", "cam-b3lyp")
    assert_rem_value(parsed.rem_block, "BASIS", "def2-tzvp")


@pytest.mark.contract
@pytest.mark.parametrize("solvent", ["water", "acetonitrile", "dichloromethane", "dmso"])
def test_various_solvents_supported(qchem_builder, h2o_geometry, solvent):
    """Builder should support common solvents in SMD model."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="smd", solvent=solvent),
    )
    result = qchem_builder.build(spec, h2o_geometry)

    # Should generate valid output
    assert "$molecule" in result
    assert "$rem" in result


# =============================================================================
# INTEGRATION TESTS: Solvation Workflows via CalculationInput
# =============================================================================


@pytest.mark.integration
def test_solvation_smd_workflow(h2o_geometry):
    """SMD solvation workflow via fluent API."""
    from calcflow.common.input import CalculationInput

    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
    ).set_solvation("smd", "water")

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have solvent block
    assert_block_present(parsed.blocks, "smx")


@pytest.mark.integration
def test_solvation_pcm_workflow(h2o_geometry):
    """PCM solvation workflow via fluent API."""
    from calcflow.common.input import CalculationInput

    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="pbe0",
        basis_set="6-311g(d,p)",
    ).set_solvation("pcm", "acetonitrile")

    result = calc.export("qchem", h2o_geometry)

    # Should have solvation setup
    assert "acetonitrile" in result.lower() or "solvent" in result.lower()


@pytest.mark.integration
def test_solvation_with_tddft_workflow(h2o_geometry):
    """Solvation combined with TDDFT should work correctly."""
    from calcflow.common.input import CalculationInput

    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
        )
        .set_solvation("smd", "water")
        .set_tddft(nroots=10, singlets=True, triplets=False, use_tda=False)
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should have both solvation and TDDFT
    assert_block_present(parsed.blocks, "smx")
    assert_rem_value(parsed.rem_block, "CIS_N_ROOTS", 10)


@pytest.mark.integration
def test_solvation_with_optimization_workflow(h2o_geometry):
    """Solvation combined with geometry optimization."""
    from calcflow.common.input import CalculationInput

    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="geometry",
        level_of_theory="b3lyp",
        basis_set="6-31g",
    ).set_solvation("smd", "water")

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "JOBTYPE", "opt")
    assert_block_present(parsed.blocks, "smx")


@pytest.mark.integration
def test_solvation_with_unrestricted_workflow(h2o_geometry):
    """Solvation with unrestricted calculation."""
    from calcflow.common.input import CalculationInput

    calc = (
        CalculationInput(
            charge=1,
            spin_multiplicity=2,
            task="energy",  # Radical
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        .set_unrestricted(True)
        .set_solvation("smd", "acetonitrile")
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)

    assert_rem_value(parsed.rem_block, "UNRESTRICTED", True)
    assert_block_present(parsed.blocks, "smx")


# =============================================================================
# REGRESSION TESTS: Semantic Validation
# =============================================================================


@pytest.mark.regression
def test_solvation_basic_output_structure(qchem_builder, h2o_geometry):
    """Solvation output should have complete structure."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="smd", solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)

    # Should not be two-job structure
    assert "@@@" not in result

    parsed = parse_qchem_input(result)

    # All required blocks present
    assert "$molecule" in parsed.molecule_block
    assert "$rem" in parsed.rem_block
    assert_block_present(parsed.blocks, "smx")


@pytest.mark.regression
@pytest.mark.parametrize("model", ["pcm", "smd", "isosvp", "cpcm"])
def test_solvation_models_produce_valid_input(qchem_builder, h2o_geometry, model):
    """Each solvation model should produce valid Q-Chem input."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model=model, solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)

    # Should parse without errors
    parsed = parse_qchem_input(result)

    # Should have basic structure
    assert parsed.molecule_block
    assert parsed.rem_block


@pytest.mark.regression
def test_solvation_with_different_functionals(qchem_builder, h2o_geometry):
    """Solvation should work with different functionals."""
    for functional in ["b3lyp", "pbe0", "m06", "cam-b3lyp"]:
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory=functional,
            basis_set="6-31g",
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = qchem_builder.build(spec, h2o_geometry)
        parsed = parse_qchem_input(result)

        assert_rem_value(parsed.rem_block, "METHOD", functional)
        assert_block_present(parsed.blocks, "smx")


@pytest.mark.regression
def test_solvation_preserves_charge_multiplicity(qchem_builder, h2o_geometry):
    """Solvation should preserve charge and multiplicity settings."""
    spec = CalculationInput(
        charge=1,
        spin_multiplicity=2,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        solvation=SolvationSpec(model="smd", solvent="water"),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # Check $molecule block has correct charge/mult
    lines = [line.strip() for line in parsed.molecule_block.split("\n") if line.strip()]
    # Second line should be "charge multiplicity"
    if len(lines) >= 2:
        first_content = lines[1]
        parts = first_content.split()
        assert int(parts[0]) == 1
        assert int(parts[1]) == 2


@pytest.mark.regression
def test_solvation_no_solvation_when_none(qchem_builder, h2o_geometry, minimal_spec):
    """Non-solvated calculation should not have solvent blocks."""
    # minimal_spec has no solvation
    result = qchem_builder.build(minimal_spec, h2o_geometry)
    parsed = parse_qchem_input(result)

    # Should NOT have SMX block
    assert "smx" not in parsed.blocks


@pytest.mark.regression
def test_solvation_multiple_solvents_workflow(h2o_geometry):
    """Builder should work with different solvents in sequence."""
    from calcflow.common.input import CalculationInput

    solvents = ["water", "acetonitrile", "dichloromethane"]

    for solvent in solvents:
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        ).set_solvation("smd", solvent)

        result = calc.export("qchem", h2o_geometry)
        parsed = parse_qchem_input(result)

        assert_block_present(parsed.blocks, "smx")


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def qchem_builder():
    """Q-Chem builder instance for testing."""
    return QchemBuilder()
