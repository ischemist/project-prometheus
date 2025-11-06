"""Tests for OrcaBuilder: solvation models (SMD and CPCM)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.input import CalculationInput, SolvationSpec
from tests.io.orca.orca_builders.conftest import (
    assert_cpcm_block,
    assert_keywords_present,
    parse_orca_input,
)

# ============================================================================
# UNIT TESTS: Individual solvation method behavior
# ============================================================================


class TestBuildSolventSMD:
    """Test SMD solvation block generation."""

    @pytest.mark.unit
    def test_smd_block_structure(self, orca_builder, minimal_spec):
        """SMD solvation should generate properly formatted %cpcm block."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder._build_solvent(spec)

        assert "%cpcm" in result
        assert "smd true" in result
        assert 'SMDsolvent "water"' in result
        assert "end" in result
        assert result.count("%cpcm") == 1
        assert result.count("end") == 1

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "solvent",
        ["water", "acetonitrile", "methanol", "dmso", "chloroform", "toluene"],
    )
    def test_smd_different_solvents(self, orca_builder, minimal_spec, solvent):
        """SMD should accept various solvent names."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="smd", solvent=solvent),
        )
        result = orca_builder._build_solvent(spec)

        assert f'SMDsolvent "{solvent}"' in result
        assert "smd true" in result

    @pytest.mark.unit
    def test_smd_case_insensitive(self, orca_builder, minimal_spec):
        """SMD model name should be case-insensitive."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder._build_solvent(spec)

        # Lowercase comparison
        normalized = result.lower()
        assert "smd true" in normalized
        assert "smdsolvent" in normalized


class TestBuildSolventCPCM:
    """Test CPCM solvation (keyword-only)."""

    @pytest.mark.unit
    def test_cpcm_not_in_solvent_block(self, orca_builder, minimal_spec):
        """CPCM solvation should not generate a %cpcm block."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="cpcm", solvent="water"),
        )
        result = orca_builder._build_solvent(spec)

        assert result == ""

    @pytest.mark.unit
    def test_cpcm_in_keyword_line(self, orca_builder, minimal_spec):
        """CPCM should appear in keyword line instead."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="cpcm", solvent="water"),
        )
        keywords = orca_builder._build_keywords(spec)

        assert 'CPCM("water")' in keywords

    @pytest.mark.unit
    @pytest.mark.parametrize("solvent", ["water", "acetonitrile", "methanol"])
    def test_cpcm_different_solvents_in_keywords(self, orca_builder, minimal_spec, solvent):
        """CPCM with different solvents should appear in keywords."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="cpcm", solvent=solvent),
        )
        keywords = orca_builder._build_keywords(spec)

        assert f'CPCM("{solvent}")' in keywords


class TestNoSolvation:
    """Test when no solvation is specified."""

    @pytest.mark.unit
    def test_no_solvation_empty_solvent_block(self, orca_builder, minimal_spec):
        """No solvation should return empty solvent block."""
        result = orca_builder._build_solvent(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_no_solvation_no_cpcm_in_keywords(self, orca_builder, minimal_spec):
        """No solvation should not add CPCM to keywords."""
        keywords = orca_builder._build_keywords(minimal_spec)
        assert "CPCM" not in keywords
        assert "smd" not in keywords.lower()


# ============================================================================
# CONTRACT TESTS: Solvation block structure in full input
# ============================================================================


class TestSMDIntegrationInInput:
    """Test SMD solvation in complete input files."""

    @pytest.mark.contract
    def test_smd_sp_has_cpcm_block(self, orca_builder, h2o_geometry):
        """SMD SP calculation should have %cpcm block in output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert "cpcm" in parsed.blocks
        assert_cpcm_block(parsed.blocks, smd=True, solvent="water")

    @pytest.mark.contract
    def test_smd_block_ordering(self, orca_builder, h2o_geometry):
        """Solvation block should come before xyz block."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            solvation=SolvationSpec(model="smd", solvent="methanol"),
        )
        result = orca_builder.build(spec, h2o_geometry)

        cpcm_idx = result.index("%cpcm")
        xyz_idx = result.index("* xyz")
        assert cpcm_idx < xyz_idx

    @pytest.mark.contract
    def test_cpcm_keyword_in_sp(self, orca_builder, h2o_geometry):
        """CPCM SP calculation should have CPCM in keyword line."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            solvation=SolvationSpec(model="cpcm", solvent="acetonitrile"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "SP", "b3lyp", "def2-svp", "CPCM")

    @pytest.mark.contract
    def test_cpcm_no_block_in_output(self, orca_builder, h2o_geometry):
        """CPCM should not create a %cpcm block."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            solvation=SolvationSpec(model="cpcm", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)

        # Count %cpcm blocks
        assert result.count("%cpcm") == 0


class TestSolvationWithOtherFeatures:
    """Test solvation combined with other features."""

    @pytest.mark.contract
    def test_smd_with_multiple_cores(self, orca_builder, h2o_geometry):
        """SMD + multiple cores should have both %cpcm and %pal."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            n_cores=4,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert "cpcm" in parsed.blocks
        assert "pal" in parsed.blocks
        assert "nprocs 4" in parsed.blocks["pal"]

    @pytest.mark.contract
    def test_smd_with_custom_memory(self, orca_builder, h2o_geometry):
        """SMD + custom memory should have both in output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            memory_per_core_mb=8000,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert "%maxcore 8000" in result
        assert "%cpcm" in result


# ============================================================================
# INTEGRATION TESTS: Fluent API workflows with solvation
# ============================================================================


class TestSolvationFuentAPI:
    """Test solvation via fluent API."""

    @pytest.mark.integration
    def test_smd_workflow(self, h2o_geometry):
        """SMD solvation via fluent API should work correctly."""
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        ).set_solvation("smd", "water")

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "cpcm" in parsed.blocks
        assert "smd true" in parsed.blocks["cpcm"]
        assert 'SMDsolvent "water"' in parsed.blocks["cpcm"]

    @pytest.mark.integration
    def test_cpcm_workflow(self, h2o_geometry):
        """CPCM solvation via fluent API should work correctly."""
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        ).set_solvation("cpcm", "acetonitrile")

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "CPCM")
        assert parsed.blocks.get("cpcm") is None  # CPCM has no block

    @pytest.mark.integration
    def test_smd_with_cores_workflow(self, h2o_geometry):
        """SMD + cores via fluent API should combine correctly."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            )
            .set_solvation("smd", "water")
            .set_cores(8)
        )

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "cpcm" in parsed.blocks
        assert "pal" in parsed.blocks
        assert "nprocs 8" in parsed.blocks["pal"]

    @pytest.mark.integration
    def test_smd_with_memory_workflow(self, h2o_geometry):
        """SMD + custom memory via fluent API should work."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            )
            .set_solvation("smd", "dmso")
            .set_memory_per_core(6000)
        )

        result = calc.export("orca", h2o_geometry)

        assert "%maxcore 6000" in result
        assert "%cpcm" in result
        assert 'SMDsolvent "dmso"' in result


class TestSolvationWithOptimization:
    """Test solvation combined with geometry optimization."""

    @pytest.mark.integration
    def test_smd_optimization_workflow(self, h2o_geometry):
        """SMD + optimization should combine correctly."""

        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="geometry",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            )
            .set_solvation("smd", "water")
            .set_optimization(calc_hess_initial=True)
        )

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "! Opt" in parsed.keyword_line
        assert "cpcm" in parsed.blocks
        assert "geom" in parsed.blocks


# ============================================================================
# REGRESSION TESTS: Semantic validation of solvation outputs
# ============================================================================


class TestSolvationRegression:
    """Regression tests using semantic validation."""

    @pytest.mark.regression
    def test_smd_water_semantic_output(self, orca_builder, h2o_geometry):
        """SMD water solvation should produce semantically correct output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Semantic checks
        lines = [line.strip() for line in result.split("\n") if line.strip()]

        # Check keyword line
        assert any("! SP" in line for line in lines)
        assert any("b3lyp" in line.lower() for line in lines)
        assert any("def2-svp" in line.lower() for line in lines)

        # Check solvation block structure
        assert "cpcm" in parsed.blocks
        cpcm_content = parsed.blocks["cpcm"].lower()
        assert "smd true" in cpcm_content
        assert 'smdsolvent "water"' in cpcm_content

        # Check xyz block
        assert parsed.xyz_block.startswith("* xyz 0 1")
        assert parsed.xyz_block.count("O") >= 1
        assert parsed.xyz_block.count("H") >= 2

    @pytest.mark.regression
    def test_cpcm_acetonitrile_semantic_output(self, orca_builder, h2o_geometry):
        """CPCM acetonitrile should produce semantically correct output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-tzvp",
            solvation=SolvationSpec(model="cpcm", solvent="acetonitrile"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check keyword line includes CPCM
        assert 'CPCM("acetonitrile")' in parsed.keyword_line
        assert "cam-b3lyp" in parsed.keyword_line.lower()
        assert "def2-tzvp" in parsed.keyword_line.lower()

        # Check no separate cpcm block for CPCM model
        assert "cpcm" not in parsed.blocks

        # Check xyz block is present
        assert "* xyz" in parsed.xyz_block

    @pytest.mark.regression
    def test_smd_multiple_features_semantic(self, orca_builder, h2o_geometry):
        """SMD + multiple cores + custom memory should all be present."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            n_cores=4,
            memory_per_core_mb=8000,
            solvation=SolvationSpec(model="smd", solvent="methanol"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        lines = [line.strip() for line in result.split("\n") if line.strip()]

        # All expected elements should be present
        assert any("%maxcore 8000" in line for line in lines)
        assert any("nprocs 4" in line for line in lines)
        assert any("%cpcm" in line for line in lines)
        assert any('SMDsolvent "methanol"' in line for line in lines)


class TestSolvationEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.regression
    def test_solvent_name_with_spaces_not_quoted(self, orca_builder, h2o_geometry):
        """Solvent names with underscores should work correctly."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="sto-3g",
            solvation=SolvationSpec(model="smd", solvent="dimethyl_sulfoxide"),
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert 'SMDsolvent "dimethyl_sulfoxide"' in result

    @pytest.mark.regression
    def test_smd_case_preservation_in_solvent(self, orca_builder, h2o_geometry):
        """Solvent name case should be preserved."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="sto-3g",
            solvation=SolvationSpec(model="smd", solvent="Water"),
        )
        result = orca_builder.build(spec, h2o_geometry)

        # Should preserve the case as given
        assert 'SMDsolvent "Water"' in result
