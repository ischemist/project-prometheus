"""Tests for OrcaBuilder: basic unit, contract, integration, and regression tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.input import (
    CalculationInput,
    OptimizationSpec,
    SolvationSpec,
    TddftSpec,
)
from calcflow.io.orca.builder import OrcaBuilder
from tests.io.orca.orca_builders.conftest import (
    assert_cpcm_block,
    assert_geom_block,
    assert_keywords_present,
    assert_maxcore,
    assert_pal_block,
    assert_tddft_block,
    assert_xyz_charge_mult,
    assert_xyz_has_atoms,
    parse_orca_input,
)

# ============================================================================
# UNIT TESTS: Individual method behavior
# ============================================================================


class TestBuildKeywords:
    """Test the _build_keywords method."""

    @pytest.mark.unit
    def test_keywords_sp_basic(self, orca_builder, minimal_spec):
        """Basic SP calculation should have SP keyword."""
        result = orca_builder._build_keywords(minimal_spec)
        assert "! SP" in result

    @pytest.mark.unit
    def test_keywords_includes_basis_set(self, orca_builder, minimal_spec):
        """Keywords should include basis set."""
        result = orca_builder._build_keywords(minimal_spec)
        assert "sto-3g" in result.lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "task,expected",
        [
            ("energy", "SP"),
            ("geometry", "Opt"),
            ("frequency", "Freq"),
        ],
    )
    def test_keywords_task_mapping(self, orca_builder, minimal_spec, task, expected):
        """Task types should map correctly."""
        spec = replace(minimal_spec, task=task)
        result = orca_builder._build_keywords(spec)
        assert expected in result

    @pytest.mark.unit
    def test_keywords_freq_after_opt(self, orca_builder, minimal_spec):
        """Freq after optimization should add Freq keyword."""
        spec = replace(
            minimal_spec,
            task="geometry",
            frequency_after_optimization=True,
        )
        result = orca_builder._build_keywords(spec)
        assert "Opt" in result
        assert "Freq" in result

    @pytest.mark.unit
    def test_keywords_ri_approximation(self, orca_builder, minimal_spec):
        """RI approximation should be included in keywords."""
        spec = replace(
            minimal_spec,
            program_options={"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
        )
        result = orca_builder._build_keywords(spec)
        assert "RIJCOSX" in result
        assert "def2/j" in result

    @pytest.mark.unit
    def test_keywords_cpcm_solvation(self, orca_builder, minimal_spec):
        """CPCM solvation should be in keywords."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="cpcm", solvent="water"),
        )
        result = orca_builder._build_keywords(spec)
        assert 'CPCM("water")' in result

    @pytest.mark.unit
    def test_keywords_smd_not_in_keyword_line(self, orca_builder, minimal_spec):
        """SMD solvation should not appear in keyword line (goes in %cpcm block)."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder._build_keywords(spec)
        assert "smd" not in result.lower()


class TestBuildProcs:
    """Test the _build_procs method."""

    @pytest.mark.unit
    def test_build_procs_single_core(self, orca_builder, minimal_spec):
        """Single core should not generate a %pal block."""
        spec = replace(minimal_spec, n_cores=1)
        result = orca_builder._build_procs(spec)
        assert result == ""

    @pytest.mark.unit
    @pytest.mark.parametrize("n_cores", [2, 4, 8, 16])
    def test_build_procs_multiple_cores(self, orca_builder, minimal_spec, n_cores):
        """Multiple cores should generate %pal block."""
        spec = replace(minimal_spec, n_cores=n_cores)
        result = orca_builder._build_procs(spec)
        assert "%pal" in result
        assert f"nprocs {n_cores}" in result
        assert "end" in result


class TestBuildMem:
    """Test the _build_mem method."""

    @pytest.mark.unit
    def test_build_mem_default(self, orca_builder, minimal_spec):
        """Default memory should be 4000 MB."""
        result = orca_builder._build_mem(minimal_spec)
        assert "%maxcore 4000" in result

    @pytest.mark.unit
    @pytest.mark.parametrize("mem_mb", [1000, 2000, 8000, 16000])
    def test_build_mem_custom_values(self, orca_builder, minimal_spec, mem_mb):
        """Custom memory values should be set correctly."""
        spec = replace(minimal_spec, memory_per_core_mb=mem_mb)
        result = orca_builder._build_mem(spec)
        assert f"%maxcore {mem_mb}" in result


class TestBuildSolvent:
    """Test the _build_solvent method."""

    @pytest.mark.unit
    def test_build_solvent_none(self, orca_builder, minimal_spec):
        """No solvation should return empty string."""
        result = orca_builder._build_solvent(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_build_solvent_smd(self, orca_builder, minimal_spec):
        """SMD solvation should generate %cpcm block."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder._build_solvent(spec)
        assert "%cpcm" in result
        assert "smd true" in result
        assert 'SMDsolvent "water"' in result
        assert "end" in result

    @pytest.mark.unit
    def test_build_solvent_cpcm(self, orca_builder, minimal_spec):
        """CPCM solvation should return empty (keyword line only)."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="cpcm", solvent="water"),
        )
        result = orca_builder._build_solvent(spec)
        assert result == ""

    @pytest.mark.unit
    def test_build_solvent_different_solvents(self, orca_builder, minimal_spec):
        """Different solvents should be recognized."""
        for solvent in ["water", "acetonitrile", "methanol", "dmso"]:
            spec = replace(
                minimal_spec,
                solvation=SolvationSpec(model="smd", solvent=solvent),
            )
            result = orca_builder._build_solvent(spec)
            assert f'SMDsolvent "{solvent}"' in result


class TestBuildTddft:
    """Test the _build_tddft method."""

    @pytest.mark.unit
    def test_build_tddft_none(self, orca_builder, minimal_spec):
        """No TDDFT should return empty string."""
        result = orca_builder._build_tddft(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_build_tddft_basic(self, orca_builder, minimal_spec):
        """Basic TDDFT should generate %tddft block."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10),
        )
        result = orca_builder._build_tddft(spec)
        assert "%tddft" in result
        assert "NRoots 10" in result
        assert "end" in result

    @pytest.mark.unit
    def test_build_tddft_triplets(self, orca_builder, minimal_spec):
        """TDDFT with triplets should set flag."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=True),
        )
        result = orca_builder._build_tddft(spec)
        assert "Triplets true" in result

    @pytest.mark.unit
    def test_build_tddft_no_tda(self, orca_builder, minimal_spec):
        """TDDFT with use_tda=False should set TDA false."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)
        assert "TDA false" in result

    @pytest.mark.unit
    def test_build_tddft_iroot(self, orca_builder, minimal_spec):
        """TDDFT with state_to_optimize should include IRoot."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, state_to_optimize=3),
        )
        result = orca_builder._build_tddft(spec)
        assert "IRoot 3" in result


class TestBuildGeom:
    """Test the _build_geom method."""

    @pytest.mark.unit
    def test_build_geom_none(self, orca_builder, minimal_spec):
        """No optimization should return empty string."""
        result = orca_builder._build_geom(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_build_geom_empty_options(self, orca_builder, minimal_spec):
        """Optimization with no options should return %geom block with only end."""
        spec = replace(
            minimal_spec,
            task="geometry",
            optimization=OptimizationSpec(),
        )
        result = orca_builder._build_geom(spec)
        # When optimization is present but has no options, builder creates %geom...end block
        assert "%geom" in result and "end" in result

    @pytest.mark.unit
    def test_build_geom_calc_hess(self, orca_builder, minimal_spec):
        """Optimization with calc_hess should be included."""
        spec = replace(
            minimal_spec,
            task="geometry",
            optimization=OptimizationSpec(calc_hess_initial=True),
        )
        result = orca_builder._build_geom(spec)
        assert "%geom" in result
        assert "Calc_Hess true" in result
        assert "end" in result

    @pytest.mark.unit
    def test_build_geom_recalc_hess(self, orca_builder, minimal_spec):
        """Optimization with recalc_hess should be included."""
        spec = replace(
            minimal_spec,
            task="geometry",
            optimization=OptimizationSpec(recalc_hess_freq=5),
        )
        result = orca_builder._build_geom(spec)
        assert "%geom" in result
        assert "Recalc_Hess 5" in result
        assert "end" in result

    @pytest.mark.unit
    def test_build_geom_both_options(self, orca_builder, minimal_spec):
        """Optimization with both hess options should include both."""
        spec = replace(
            minimal_spec,
            task="geometry",
            optimization=OptimizationSpec(calc_hess_initial=True, recalc_hess_freq=5),
        )
        result = orca_builder._build_geom(spec)
        assert "%geom" in result
        assert "Calc_Hess true" in result
        assert "Recalc_Hess 5" in result


# ============================================================================
# CONTRACT TESTS: Full build() output structure verification
# ============================================================================


class TestSPBasicStructure:
    """Test basic SP calculation structure."""

    @pytest.mark.contract
    def test_sp_basic_structure(self, orca_builder, h2o_geometry, minimal_spec):
        """Basic SP calculation should have correct structure."""
        result = orca_builder.build(minimal_spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check for required components
        assert_keywords_present(parsed.keyword_line, "SP", "HF", "sto-3g")
        assert_xyz_charge_mult(parsed.xyz_block, 0, 1)
        assert_xyz_has_atoms(parsed.xyz_block, ("O", 1), ("H", 2))
        assert result.count("*") == 2  # opening and closing xyz block

    @pytest.mark.contract
    def test_sp_with_multiple_cores(self, orca_builder, h2o_geometry, minimal_spec):
        """SP with multiple cores should have %pal block."""
        spec = replace(minimal_spec, n_cores=4)
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "SP")
        assert_pal_block(parsed.blocks, nprocs=4)

    @pytest.mark.contract
    def test_sp_with_custom_memory(self, orca_builder, h2o_geometry, minimal_spec):
        """SP with custom memory should be set."""
        spec = replace(minimal_spec, memory_per_core_mb=8000)
        result = orca_builder.build(spec, h2o_geometry)

        assert_maxcore(result, 8000)

    @pytest.mark.contract
    def test_sp_with_smd_solvation(self, orca_builder, h2o_geometry, minimal_spec):
        """SP with SMD solvation should include %cpcm block."""
        spec = replace(
            minimal_spec,
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_cpcm_block(parsed.blocks, smd=True, solvent="water")


class TestOptimizationBasicStructure:
    """Test basic optimization calculation structure."""

    @pytest.mark.contract
    def test_opt_basic_structure(self, orca_builder, h2o_geometry, minimal_spec):
        """Optimization should have Opt keyword."""
        spec = replace(minimal_spec, task="geometry")
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "Opt")
        assert_xyz_has_atoms(parsed.xyz_block, ("O", 1), ("H", 2))

    @pytest.mark.contract
    def test_opt_with_hessian_options(self, orca_builder, h2o_geometry, minimal_spec):
        """Optimization with hessian options should generate %geom block."""
        spec = replace(
            minimal_spec,
            task="geometry",
            optimization=OptimizationSpec(calc_hess_initial=True, recalc_hess_freq=5),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "Opt")
        assert_geom_block(parsed.blocks, calc_hess=True, recalc_hess=5)


class TestFrequencyBasicStructure:
    """Test basic frequency calculation structure."""

    @pytest.mark.contract
    def test_freq_basic_structure(self, orca_builder, h2o_geometry, minimal_spec):
        """Frequency calculation should have Freq keyword."""
        spec = replace(minimal_spec, task="frequency")
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "Freq")


class TestTDDFTStructure:
    """Test TDDFT block structure."""

    @pytest.mark.contract
    def test_tddft_block_present(self, orca_builder, h2o_geometry, minimal_spec):
        """TDDFT should generate %tddft block in output."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=5),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_tddft_block(parsed.blocks, nroots=5)


class TestXYZBlockStructure:
    """Test XYZ block formatting."""

    @pytest.mark.contract
    def test_xyz_block_has_correct_charge_multiplicity(self, orca_builder, h2o_geometry):
        """XYZ block should have correct charge and multiplicity."""
        spec = CalculationInput(
            charge=1,
            spin_multiplicity=2,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_xyz_charge_mult(parsed.xyz_block, 1, 2)

    @pytest.mark.contract
    def test_xyz_block_coordinates_formatted(self, orca_builder, h2o_geometry, minimal_spec):
        """XYZ block coordinates should be properly formatted."""
        result = orca_builder.build(minimal_spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Should have correct atoms
        assert_xyz_has_atoms(parsed.xyz_block, ("O", 1), ("H", 2))


# ============================================================================
# INTEGRATION TESTS: Full fluent API workflow
# ============================================================================


class TestFluentAPIWorkflow:
    """Test integration with CalculationInput fluent API."""

    @pytest.mark.integration
    def test_sp_basic_workflow(self, h2o_geometry):
        """Basic SP workflow should work end-to-end."""

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        )

        builder = OrcaBuilder()
        result = builder.build(calc, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "SP", "b3lyp", "def2-svp")
        assert_xyz_charge_mult(parsed.xyz_block, 0, 1)

    @pytest.mark.integration
    def test_solvation_workflow(self, h2o_geometry):
        """Solvation workflow should work with fluent API."""

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        ).set_solvation("smd", "water")

        builder = OrcaBuilder()
        result = builder.build(calc, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "b3lyp")
        assert_cpcm_block(parsed.blocks, smd=True, solvent="water")

    @pytest.mark.integration
    def test_tddft_workflow(self, h2o_geometry):
        """TDDFT workflow should work with fluent API."""

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-tzvp",
        ).set_tddft(nroots=10, singlets=True, triplets=False, use_tda=True)

        builder = OrcaBuilder()
        result = builder.build(calc, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "cam-b3lyp", "def2-tzvp")
        assert_tddft_block(parsed.blocks, nroots=10, triplets=False, tda=True)

    @pytest.mark.integration
    def test_optimization_workflow(self, h2o_geometry):
        """Optimization workflow should work with fluent API."""

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        ).set_cores(4)

        builder = OrcaBuilder()
        result = builder.build(calc, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "Opt")
        assert_pal_block(parsed.blocks, nprocs=4)


# ============================================================================
# REGRESSION TESTS: Semantic verification of output
# ============================================================================


class TestRegressionSPOutput:
    """Regression tests for SP calculation output."""

    @pytest.mark.regression
    def test_sp_semantic_regression(self, orca_builder, h2o_geometry, minimal_spec):
        """SP output should maintain semantic correctness."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            n_cores=4,
            memory_per_core_mb=8000,
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Required keywords present
        assert_keywords_present(parsed.keyword_line, "SP", "RKS", "b3lyp", "def2-svp")

        # Check memory and procs
        assert_maxcore(result, 8000)
        assert_pal_block(parsed.blocks, nprocs=4)

        # Check xyz block structure
        assert_xyz_charge_mult(parsed.xyz_block, 0, 1)
        assert_xyz_has_atoms(parsed.xyz_block, ("O", 1), ("H", 2))

    @pytest.mark.regression
    def test_opt_semantic_regression(self, orca_builder, h2o_geometry):
        """Optimization output should maintain semantic correctness."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            optimization=OptimizationSpec(
                calc_hess_initial=True,
                recalc_hess_freq=5,
            ),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Verify Opt keyword
        assert_keywords_present(parsed.keyword_line, "Opt")

        # Verify %geom block with correct settings
        assert_geom_block(parsed.blocks, calc_hess=True, recalc_hess=5)

    @pytest.mark.regression
    def test_tddft_semantic_regression(self, orca_builder, h2o_geometry):
        """TDDFT output should maintain semantic correctness."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-tzvp",
            tddft=TddftSpec(nroots=5, singlets=True, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Verify TDDFT block
        assert_tddft_block(parsed.blocks, nroots=5, triplets=False, tda=True)
