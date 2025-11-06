"""Tests for OrcaBuilder: RI (resolution of identity) approximations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.input import CalculationInput
from tests.io.orca.orca_builders.conftest import (
    assert_keywords_present,
    parse_orca_input,
)

# ============================================================================
# UNIT TESTS: RI approximation keyword handling
# ============================================================================


class TestRIApproximationKeywords:
    """Test RI approximation in keyword line."""

    @pytest.mark.unit
    def test_rijcosx_in_keywords(self, orca_builder, minimal_spec):
        """RIJCOSX approximation should appear in keywords."""
        spec = replace(
            minimal_spec,
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder._build_keywords(spec)

        assert "RIJCOSX" in result
        assert "def2/j" in result

    @pytest.mark.unit
    def test_ri_in_keywords(self, orca_builder, minimal_spec):
        """RI approximation should appear in keywords."""
        spec = replace(
            minimal_spec,
            program_options={"ri_approx": "RI", "aux_basis": "def2/jk"},
        )
        result = orca_builder._build_keywords(spec)

        assert "RI" in result
        assert "def2/jk" in result

    @pytest.mark.unit
    def test_rij_in_keywords(self, orca_builder, minimal_spec):
        """RIJ approximation should appear in keywords."""
        spec = replace(
            minimal_spec,
            program_options={"ri_approx": "RIJ", "aux_basis": "def2/j"},
        )
        result = orca_builder._build_keywords(spec)

        assert "RIJ" in result
        assert "def2/j" in result

    @pytest.mark.unit
    def test_no_ri_when_not_specified(self, orca_builder, minimal_spec):
        """No RI keywords when not specified."""
        result = orca_builder._build_keywords(minimal_spec)

        assert "RI" not in result
        assert "RIJCOSX" not in result

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ri_approx,aux_basis",
        [
            ("RIJCOSX", "def2/j"),
            ("RI", "def2/jk"),
            ("RIJ", "def2/j"),
        ],
    )
    def test_various_ri_approximations(self, orca_builder, minimal_spec, ri_approx, aux_basis):
        """Different RI approximations should work."""
        spec = replace(
            minimal_spec,
            program_options={"ri_approx": ri_approx, "aux_basis": aux_basis},
        )
        result = orca_builder._build_keywords(spec)

        assert ri_approx in result
        assert aux_basis in result


# ============================================================================
# UNIT TESTS: RI validation
# ============================================================================


class TestRIValidation:
    """Test RI approximation validation."""

    @pytest.mark.unit
    def test_ri_without_aux_basis_fails(self, orca_builder):
        """RI approximation without aux_basis should raise error."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            program_options={"ri_approx": "RIJCOSX"},
        )

        with pytest.raises(ValidationError, match="aux_basis"):
            orca_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_ri_with_aux_basis_passes(self, orca_builder):
        """RI approximation with aux_basis should pass validation."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )

        # Should not raise
        orca_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_no_ri_no_aux_basis_passes(self, orca_builder):
        """No RI without aux_basis should pass validation."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        )

        # Should not raise
        orca_builder._validate_spec(spec)


# ============================================================================
# CONTRACT TESTS: RI in full input structure
# ============================================================================


class TestRIInFullInput:
    """Test RI approximation in complete input files."""

    @pytest.mark.contract
    def test_rijcosx_sp_structure(self, orca_builder, h2o_geometry):
        """RIJCOSX SP should have correct structure."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check keywords
        assert_keywords_present(parsed.keyword_line, "SP", "b3lyp", "def2-svp", "RIJCOSX", "def2/j")

        # Check xyz block
        assert "* xyz 0 1" in parsed.xyz_block

    @pytest.mark.contract
    def test_ri_with_cores(self, orca_builder, h2o_geometry):
        """RI approximation with multiple cores."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            n_cores=8,
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check RI in keywords
        assert "RIJCOSX" in parsed.keyword_line
        assert "def2/j" in parsed.keyword_line

        # Check %pal block
        assert "pal" in parsed.blocks
        assert "nprocs 8" in parsed.blocks["pal"]

    @pytest.mark.contract
    def test_ri_with_memory(self, orca_builder, h2o_geometry):
        """RI approximation with custom memory."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            memory_per_core_mb=8000,
            program_options={"ri_approx": "RI", "aux_basis": "def2/jk"},
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert "RI" in result
        assert "def2/jk" in result
        assert "%maxcore 8000" in result

    @pytest.mark.contract
    def test_rij_in_output(self, orca_builder, h2o_geometry):
        """RIJ approximation in output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="pbe0",
            basis_set="def2-tzvp",
            program_options={"ri_approx": "RIJ", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "RIJ", "def2/j", "pbe0")


class TestRIWithOtherFeatures:
    """Test RI combined with other features."""

    @pytest.mark.contract
    def test_ri_with_optimization(self, orca_builder, h2o_geometry):
        """RI approximation with geometry optimization."""
        from calcflow.common.input import OptimizationSpec

        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            optimization=OptimizationSpec(calc_hess_initial=True),
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check optimization keyword
        assert "Opt" in parsed.keyword_line
        assert "RIJCOSX" in parsed.keyword_line
        assert "geom" in parsed.blocks

    @pytest.mark.contract
    def test_ri_with_unrestricted(self, orca_builder, h2o_geometry):
        """RI approximation with unrestricted calculation."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=3,  # Triplet
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            unrestricted=True,
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check unrestricted keyword
        assert "UKS" in parsed.keyword_line
        assert "RIJCOSX" in parsed.keyword_line


# ============================================================================
# INTEGRATION TESTS: Fluent API workflows with RI
# ============================================================================


class TestRIFluentAPI:
    """Test RI approximations via fluent API."""

    @pytest.mark.integration
    def test_rijcosx_workflow(self, h2o_geometry):
        """RIJCOSX via fluent API."""
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        ).set_options(ri_approx="RIJCOSX", aux_basis="def2/j")

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "RIJCOSX" in parsed.keyword_line
        assert "def2/j" in parsed.keyword_line

    @pytest.mark.integration
    def test_ri_with_cores_workflow(self, h2o_geometry):
        """RI with cores via fluent API."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="pbe0",
                basis_set="def2-tzvp",
            )
            .set_options(ri_approx="RI", aux_basis="def2/jk")
            .set_cores(16)
        )

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "RI" in parsed.keyword_line
        assert "def2/jk" in parsed.keyword_line
        assert "nprocs 16" in parsed.blocks["pal"]

    @pytest.mark.integration
    def test_rij_with_memory_workflow(self, h2o_geometry):
        """RIJ with custom memory via fluent API."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-tzvp",
            )
            .set_options(ri_approx="RIJ", aux_basis="def2/j")
            .set_memory_per_core(4000)
        )

        result = calc.export("orca", h2o_geometry)

        assert "RIJ" in result
        assert "def2/j" in result
        assert "%maxcore 4000" in result


class TestRIWithOptimizationWorkflow:
    """Test RI with optimization via fluent API."""

    @pytest.mark.integration
    def test_rijcosx_optimization_workflow(self, h2o_geometry):
        """RIJCOSX optimization via fluent API."""

        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="geometry",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            )
            .set_options(ri_approx="RIJCOSX", aux_basis="def2/j")
            .set_optimization(calc_hess_initial=True, recalc_hess_freq=5)
        )

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "Opt" in parsed.keyword_line
        assert "RIJCOSX" in parsed.keyword_line
        assert "geom" in parsed.blocks
        assert "Calc_Hess true" in parsed.blocks["geom"]
        assert "Recalc_Hess 5" in parsed.blocks["geom"]


# ============================================================================
# REGRESSION TESTS: Semantic validation of RI outputs
# ============================================================================


class TestRIRegression:
    """Regression tests using semantic validation."""

    @pytest.mark.regression
    def test_rijcosx_semantic_output(self, orca_builder, h2o_geometry):
        """RIJCOSX should produce semantically correct output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Semantic checks
        assert "! SP" in parsed.keyword_line
        assert "b3lyp" in parsed.keyword_line.lower()
        assert "def2-svp" in parsed.keyword_line.lower()
        assert "RIJCOSX" in parsed.keyword_line
        assert "def2/j" in parsed.keyword_line

        # Check xyz block
        assert "* xyz 0 1" in parsed.xyz_block

    @pytest.mark.regression
    def test_ri_with_all_options_semantic(self, orca_builder, h2o_geometry):
        """RI with all options should be semantically correct."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="pbe0",
            basis_set="def2-tzvp",
            n_cores=8,
            memory_per_core_mb=8000,
            program_options={"ri_approx": "RI", "aux_basis": "def2/jk"},
        )
        result = orca_builder.build(spec, h2o_geometry)
        lines = [line.strip() for line in result.split("\n") if line.strip()]

        # All expected elements should be present
        assert any("%maxcore 8000" in line for line in lines)
        assert any("nprocs 8" in line for line in lines)
        keyword_line = next((line for line in lines if line.startswith("!")), "")
        assert "RI" in keyword_line
        assert "def2/jk" in keyword_line
        assert "pbe0" in keyword_line.lower()

    @pytest.mark.regression
    def test_ri_approximations_vary_output(self, orca_builder, h2o_geometry):
        """Different RI approximations should all be present in output."""
        for ri_approx, aux_basis in [
            ("RIJCOSX", "def2/j"),
            ("RI", "def2/jk"),
            ("RIJ", "def2/j"),
        ]:
            spec = CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
                program_options={"ri_approx": ri_approx, "aux_basis": aux_basis},
            )
            result = orca_builder.build(spec, h2o_geometry)

            assert ri_approx in result
            assert aux_basis in result

    @pytest.mark.regression
    def test_no_ri_no_aux_keywords(self, orca_builder, h2o_geometry):
        """Without RI options, aux basis should not appear."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert "def2/j" not in result
        assert "def2/jk" not in result
        assert "RI" not in result or "RI" in result.lower()  # Only in lowercase (if any)

    @pytest.mark.regression
    def test_ri_with_different_functionals(self, orca_builder, h2o_geometry):
        """RI should work with various functionals."""
        for functional in ["b3lyp", "pbe0", "cam-b3lyp", "m06"]:
            spec = CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory=functional,
                basis_set="def2-svp",
                program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
            )
            result = orca_builder.build(spec, h2o_geometry)

            assert functional.lower() in result.lower()
            assert "RIJCOSX" in result
            assert "def2/j" in result


class TestRIAuxBasisOptions:
    """Test various auxiliary basis set options."""

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "aux_basis",
        ["def2/j", "def2/jk", "def2-universal/j", "def2-universal/jk"],
    )
    def test_various_aux_basis_sets(self, orca_builder, h2o_geometry, aux_basis):
        """Various auxiliary basis sets should work."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            program_options={"ri_approx": "RIJCOSX", "aux_basis": aux_basis},
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert aux_basis in result

    @pytest.mark.regression
    def test_rijcosx_with_def2_j(self, orca_builder, h2o_geometry):
        """RIJCOSX with def2/j should be present."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert "RIJCOSX" in result
        assert "def2/j" in result
        # These should be in the keyword line
        assert any("RIJCOSX" in line for line in result.split("\n") if line.strip().startswith("!"))
