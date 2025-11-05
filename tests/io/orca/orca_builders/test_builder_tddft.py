"""Tests for OrcaBuilder: TDDFT functionality."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.input import CalculationInput, SolvationSpec, TddftSpec
from tests.io.orca.orca_builders.conftest import (
    assert_keywords_present,
    assert_tddft_block,
    parse_orca_input,
)

# ============================================================================
# UNIT TESTS: Individual TDDFT method behavior
# ============================================================================


class TestBuildTddftBasic:
    """Test basic TDDFT block generation."""

    @pytest.mark.unit
    def test_no_tddft_returns_empty(self, orca_builder, minimal_spec):
        """No TDDFT in spec should return empty string."""
        result = orca_builder._build_tddft(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_tddft_block_structure(self, orca_builder, minimal_spec):
        """TDDFT should generate properly formatted block."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=True, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)

        assert "%tddft" in result
        assert "NRoots 10" in result
        assert "Triplets true" in result
        assert "TDA false" in result
        assert "end" in result
        assert result.count("%tddft") == 1
        assert result.count("end") == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("nroots", [5, 10, 20, 50])
    def test_tddft_nroots_values(self, orca_builder, minimal_spec, nroots):
        """TDDFT should accept various nroots values."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=nroots, triplets=False, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)

        assert f"NRoots {nroots}" in result


class TestTddftTriplets:
    """Test TDDFT triplet handling."""

    @pytest.mark.unit
    def test_tddft_triplets_true(self, orca_builder, minimal_spec):
        """TDDFT with triplets=True should output 'Triplets true'."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=True, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)

        assert "Triplets true" in result

    @pytest.mark.unit
    def test_tddft_triplets_false(self, orca_builder, minimal_spec):
        """TDDFT with triplets=False should output 'Triplets false'."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)

        assert "Triplets false" in result

    @pytest.mark.unit
    def test_tddft_both_singlets_and_triplets(self, orca_builder, minimal_spec):
        """TDDFT with both singlets and triplets."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=True, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)

        # Both should be mentioned (singlets are implicit with NRoots, triplets explicit)
        assert "NRoots 10" in result
        assert "Triplets true" in result


class TestTddftTDA:
    """Test TDDFT TDA (Tamm-Dancoff approximation) handling."""

    @pytest.mark.unit
    def test_tddft_tda_true(self, orca_builder, minimal_spec):
        """TDDFT with use_tda=True should output 'TDA true'."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
        )
        result = orca_builder._build_tddft(spec)

        assert "TDA true" in result

    @pytest.mark.unit
    def test_tddft_tda_false(self, orca_builder, minimal_spec):
        """TDDFT with use_tda=False should output 'TDA false'."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=False),
        )
        result = orca_builder._build_tddft(spec)

        assert "TDA false" in result


class TestTddftIRoot:
    """Test TDDFT state optimization (IRoot)."""

    @pytest.mark.unit
    def test_tddft_iroot_present(self, orca_builder, minimal_spec):
        """TDDFT with state_to_optimize should include IRoot."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=False, state_to_optimize=3),
        )
        result = orca_builder._build_tddft(spec)

        assert "IRoot 3" in result

    @pytest.mark.unit
    def test_tddft_iroot_absent_when_none(self, orca_builder, minimal_spec):
        """TDDFT without state_to_optimize should not include IRoot."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=False, state_to_optimize=None),
        )
        result = orca_builder._build_tddft(spec)

        assert "IRoot" not in result

    @pytest.mark.unit
    @pytest.mark.parametrize("iroot", [1, 2, 5, 10])
    def test_tddft_various_iroot_values(self, orca_builder, minimal_spec, iroot):
        """TDDFT should accept various IRoot values."""
        spec = replace(
            minimal_spec,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=False, state_to_optimize=iroot),
        )
        result = orca_builder._build_tddft(spec)

        assert f"IRoot {iroot}" in result


# ============================================================================
# CONTRACT TESTS: TDDFT block structure in full input
# ============================================================================


class TestTddftInFullInput:
    """Test TDDFT in complete input files."""

    @pytest.mark.contract
    def test_tddft_sp_has_block(self, orca_builder, h2o_geometry):
        """TDDFT SP should have %tddft block in output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert "tddft" in parsed.blocks
        assert_tddft_block(parsed.blocks, nroots=10, triplets=False, tda=True)

    @pytest.mark.contract
    def test_tddft_with_triplets(self, orca_builder, h2o_geometry):
        """TDDFT with triplets should have correct settings."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=5, triplets=True, use_tda=False),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_tddft_block(parsed.blocks, nroots=5, triplets=True, tda=False)

    @pytest.mark.contract
    def test_tddft_block_ordering(self, orca_builder, h2o_geometry):
        """TDDFT block should come before xyz block."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)

        tddft_idx = result.index("%tddft")
        xyz_idx = result.index("* xyz")
        assert tddft_idx < xyz_idx

    @pytest.mark.contract
    def test_tddft_with_iroot(self, orca_builder, h2o_geometry):
        """TDDFT with state optimization should include IRoot in block."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-tzvp",
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True, state_to_optimize=2),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_tddft_block(parsed.blocks, nroots=10, tda=True, iroot=2)


class TestTddftWithOtherFeatures:
    """Test TDDFT combined with other features."""

    @pytest.mark.contract
    def test_tddft_with_solvation(self, orca_builder, h2o_geometry):
        """TDDFT + solvation should have both blocks."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert "tddft" in parsed.blocks
        assert "cpcm" in parsed.blocks

    @pytest.mark.contract
    def test_tddft_with_multiple_cores(self, orca_builder, h2o_geometry):
        """TDDFT + multiple cores should have both."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            n_cores=8,
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert "tddft" in parsed.blocks
        assert "pal" in parsed.blocks
        assert "nprocs 8" in parsed.blocks["pal"]

    @pytest.mark.contract
    def test_tddft_with_cpcm_keyword(self, orca_builder, h2o_geometry):
        """TDDFT + CPCM solvation should have CPCM in keyword line."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
            solvation=SolvationSpec(model="cpcm", solvent="acetonitrile"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        assert_keywords_present(parsed.keyword_line, "CPCM")
        assert "tddft" in parsed.blocks


# ============================================================================
# INTEGRATION TESTS: Fluent API workflows with TDDFT
# ============================================================================


class TestTddftFluentAPI:
    """Test TDDFT via fluent API."""

    @pytest.mark.integration
    def test_basic_tddft_workflow(self, h2o_geometry):
        """Basic TDDFT via fluent API should work."""
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
        ).set_tddft(nroots=10, singlets=True, triplets=False, use_tda=True)

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "tddft" in parsed.blocks
        assert "NRoots 10" in parsed.blocks["tddft"]
        assert "Triplets false" in parsed.blocks["tddft"]
        assert "TDA true" in parsed.blocks["tddft"]

    @pytest.mark.integration
    def test_tddft_with_solvation_workflow(self, h2o_geometry):
        """TDDFT + solvation via fluent API."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-tzvp",
            )
            .set_solvation("smd", "water")
            .set_tddft(nroots=10, singlets=True, triplets=False, use_tda=True)
        )

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "tddft" in parsed.blocks
        assert "cpcm" in parsed.blocks

    @pytest.mark.integration
    def test_tddft_state_optimization_workflow(self, h2o_geometry):
        """TDDFT state optimization via fluent API (for geometry task)."""
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
        ).set_tddft(nroots=10, singlets=True, triplets=False, use_tda=True, state_to_optimize=2)

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        assert "IRoot 2" in parsed.blocks["tddft"]


class TestTddftWithUnrestricted:
    """Test TDDFT with unrestricted calculations."""

    @pytest.mark.integration
    def test_unrestricted_tddft(self, h2o_geometry):
        """Unrestricted TDDFT should work correctly."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=3,  # Triplet
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            )
            .set_unrestricted(True)
            .set_tddft(nroots=5, singlets=True, triplets=True, use_tda=False)
        )

        result = calc.export("orca", h2o_geometry)
        parsed = parse_orca_input(result)

        # Should have UKS (unrestricted) in keywords
        assert "UKS" in parsed.keyword_line
        assert "tddft" in parsed.blocks


# ============================================================================
# REGRESSION TESTS: Semantic validation of TDDFT outputs
# ============================================================================


class TestTddftRegression:
    """Regression tests using semantic validation."""

    @pytest.mark.regression
    def test_tddft_basic_semantic_output(self, orca_builder, h2o_geometry):
        """TDDFT should produce semantically correct output."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=10, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Semantic checks
        assert "! SP" in parsed.keyword_line
        assert "cam-b3lyp" in parsed.keyword_line.lower()
        assert "def2-svp" in parsed.keyword_line.lower()

        # Check TDDFT block structure
        assert "tddft" in parsed.blocks
        tddft_content = parsed.blocks["tddft"]
        assert "NRoots 10" in tddft_content
        assert "TDA true" in tddft_content
        assert "Triplets false" in tddft_content

    @pytest.mark.regression
    def test_tddft_with_all_options_semantic(self, orca_builder, h2o_geometry):
        """TDDFT with all options should be semantically correct."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-tzvp",
            n_cores=8,
            memory_per_core_mb=6000,
            tddft=TddftSpec(nroots=20, triplets=True, use_tda=False, state_to_optimize=3),
            solvation=SolvationSpec(model="smd", solvent="acetonitrile"),
        )
        result = orca_builder.build(spec, h2o_geometry)
        parsed = parse_orca_input(result)

        # Check all elements present
        assert "b3lyp" in parsed.keyword_line.lower()
        assert "def2-tzvp" in parsed.keyword_line.lower()

        # Check TDDFT block
        tddft_content = parsed.blocks.get("tddft", "")
        assert "NRoots 20" in tddft_content
        assert "Triplets true" in tddft_content
        assert "TDA false" in tddft_content
        assert "IRoot 3" in tddft_content

        # Check other blocks
        assert "pal" in parsed.blocks
        assert "cpcm" in parsed.blocks
        assert "%maxcore 6000" in result

    @pytest.mark.regression
    def test_tddft_different_nroots_values(self, orca_builder, h2o_geometry):
        """TDDFT with different nroots should produce correct output."""
        for nroots in [5, 10, 20, 50]:
            spec = CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-svp",
                tddft=TddftSpec(nroots=nroots, triplets=False, use_tda=True),
            )
            result = orca_builder.build(spec, h2o_geometry)
            parsed = parse_orca_input(result)

            assert f"NRoots {nroots}" in parsed.blocks["tddft"]

    @pytest.mark.regression
    def test_tddft_triplet_combinations(self, orca_builder, h2o_geometry):
        """Test various singlet/triplet combinations."""
        test_cases = [
            (False, True),
            (True, True),
        ]

        for use_triplets, _ in test_cases:
            spec = CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-svp",
                tddft=TddftSpec(nroots=10, triplets=use_triplets, use_tda=True),
            )
            result = orca_builder.build(spec, h2o_geometry)
            parsed = parse_orca_input(result)

            expected_triplets = "true" if use_triplets else "false"
            assert f"Triplets {expected_triplets}" in parsed.blocks["tddft"]


class TestTddftNumericalValues:
    """Test numerical boundaries and edge cases."""

    @pytest.mark.regression
    def test_tddft_minimum_nroots(self, orca_builder, h2o_geometry):
        """TDDFT with minimum nroots should work."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=1, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert "NRoots 1" in result

    @pytest.mark.regression
    def test_tddft_large_nroots(self, orca_builder, h2o_geometry):
        """TDDFT with large nroots should work."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
            tddft=TddftSpec(nroots=100, triplets=False, use_tda=True),
        )
        result = orca_builder.build(spec, h2o_geometry)

        assert "NRoots 100" in result
