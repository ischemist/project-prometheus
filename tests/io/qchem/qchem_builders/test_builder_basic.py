"""Unit, contract, integration, and regression tests for Q-Chem basic builder functionality."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.input import CalculationInput
from calcflow.io.qchem.builder import QchemBuilder

from .conftest import (
    assert_block_present,
    assert_molecule_has_atoms,
    assert_molecule_has_charge_mult,
    assert_rem_contains_keys,
    assert_rem_value,
    parse_qchem_input,
)


@pytest.fixture
def qchem_builder() -> QchemBuilder:
    """Q-Chem builder instance."""
    return QchemBuilder()


# ============================================================================
# UNIT TESTS: Test individual methods in isolation
# ============================================================================


class TestBuildMolecule:
    """Unit tests for _build_molecule method."""

    @pytest.mark.unit
    def test_molecule_with_charge_and_multiplicity(self, qchem_builder, h2o_geometry):
        """$molecule block should include charge and multiplicity."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder._build_molecule(spec, h2o_geometry)
        assert "$molecule" in result
        assert "$end" in result
        assert "0 1" in result  # charge 0, multiplicity 1
        assert "O " in result  # geometry present

    @pytest.mark.unit
    def test_molecule_with_charged_species(self, qchem_builder, h2o_geometry):
        """$molecule block should handle charged species."""
        spec = replace(
            CalculationInput(
                charge=1,
                spin_multiplicity=2,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="6-31g",
            ),
        )
        result = qchem_builder._build_molecule(spec, h2o_geometry)
        assert "1 2" in result

    @pytest.mark.unit
    def test_molecule_read_flag(self, qchem_builder, h2o_geometry):
        """$molecule block with read_geom=True should use 'read' keyword."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder._build_molecule(spec, h2o_geometry, read_geom=True)
        assert "$molecule" in result
        assert "read" in result.lower()
        assert "$end" in result

    @pytest.mark.unit
    def test_molecule_charge_override(self, qchem_builder, h2o_geometry):
        """charge_override should take precedence."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder._build_molecule(spec, h2o_geometry, charge_override=1, mult_override=2)
        assert "1 2" in result

    @pytest.mark.unit
    def test_molecule_multiplicity_override(self, qchem_builder, h2o_geometry):
        """mult_override should take precedence."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder._build_molecule(spec, h2o_geometry, charge_override=0, mult_override=3)
        assert "0 3" in result


class TestBuildRem:
    """Unit tests for _build_rem method."""

    @pytest.mark.unit
    def test_sp_job_basic(self, qchem_builder, minimal_spec):
        """Basic SP calculation should have correct JOBTYPE."""
        result = qchem_builder._build_rem(minimal_spec)
        assert "$rem" in result
        assert "$end" in result
        assert "JOBTYPE" in result
        assert "sp" in result.lower()

    @pytest.mark.unit
    def test_opt_job(self, qchem_builder, minimal_spec):
        """Optimization should have JOBTYPE opt."""
        spec = replace(minimal_spec, task="geometry")
        result = qchem_builder._build_rem(spec)
        assert "opt" in result.lower()

    @pytest.mark.unit
    def test_freq_job(self, qchem_builder, minimal_spec):
        """Frequency should have JOBTYPE freq."""
        spec = replace(minimal_spec, task="frequency")
        result = qchem_builder._build_rem(spec)
        assert "freq" in result.lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method",
        ["b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x", "wb97x-d3", "hf"],
    )
    def test_supported_methods(self, qchem_builder, minimal_spec, method):
        """Supported methods should be in output."""
        spec = replace(minimal_spec, level_of_theory=method)
        result = qchem_builder._build_rem(spec)
        assert method in result.lower()

    @pytest.mark.unit
    def test_basis_set_present(self, qchem_builder, minimal_spec):
        """Basis set should be in $rem block."""
        result = qchem_builder._build_rem(minimal_spec)
        assert "BASIS" in result
        assert "sto-3g" in result.lower()  # minimal_spec uses sto-3g

    @pytest.mark.unit
    def test_unrestricted_flag(self, qchem_builder, minimal_spec):
        """UNRESTRICTED flag should be set correctly."""
        spec = replace(minimal_spec, unrestricted=True)
        result = qchem_builder._build_rem(spec)
        assert "UNRESTRICTED" in result
        assert "True" in result

    @pytest.mark.unit
    def test_symmetry_disabled(self, qchem_builder, minimal_spec):
        """Symmetry should be disabled."""
        result = qchem_builder._build_rem(minimal_spec)
        assert "SYMMETRY" in result
        assert "False" in result

    @pytest.mark.unit
    def test_scf_guess_read(self, qchem_builder, minimal_spec):
        """SCF_GUESS can be set to 'read'."""
        result = qchem_builder._build_rem(minimal_spec, scf_guess="read")
        assert "SCF_GUESS" in result
        assert "read" in result.lower()

    @pytest.mark.unit
    def test_mom_start_flag(self, qchem_builder, minimal_spec):
        """MOM_START flag should be set for mom calculations."""
        result = qchem_builder._build_rem(minimal_spec, mom_start=True)
        assert "MOM_START" in result
        assert "1" in result

    @pytest.mark.unit
    def test_mom_method_imom(self, qchem_builder, minimal_spec):
        """MOM_METHOD should default to IMOM."""
        result = qchem_builder._build_rem(minimal_spec, mom_start=True)
        assert "MOM_METHOD" in result
        assert "IMOM" in result


class TestBuildBasis:
    """Unit tests for _build_basis method."""

    @pytest.mark.unit
    def test_no_basis_block_for_standard_basis(self, qchem_builder, minimal_spec):
        """Standard basis set should not generate $basis block."""
        result = qchem_builder._build_basis(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_dict_basis_generates_block(self, qchem_builder):
        """Dictionary basis should generate $basis block."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        result = qchem_builder._build_basis(spec)
        assert "$basis" in result
        assert "$end" in result
        assert "H" in result
        assert "O" in result
        assert "6-31g" in result
        assert "6-311g(d,p)" in result

    @pytest.mark.unit
    def test_dict_basis_format(self, qchem_builder):
        """Dictionary basis should have proper Q-Chem format."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        result = qchem_builder._build_basis(spec)
        lines = [line.strip() for line in result.split("\n")]
        assert lines[0] == "$basis"
        assert lines[-1] == "$end"


class TestBuildSolvationBlocks:
    """Unit tests for _build_solvation_blocks method."""

    @pytest.mark.unit
    def test_no_solvation_block(self, qchem_builder, minimal_spec):
        """No solvation block if solvation not specified."""
        result = qchem_builder._build_solvation_blocks(minimal_spec)
        assert result == ""

    @pytest.mark.unit
    def test_pcm_solvation_block(self, qchem_builder, minimal_spec):
        """PCM solvation should generate $solvent block."""
        from calcflow.common.input import SolvationSpec

        spec = replace(minimal_spec, solvation=SolvationSpec(model="pcm", solvent="water"))
        result = qchem_builder._build_solvation_blocks(spec)
        assert "$solvent" in result
        assert "water" in result.lower()
        assert "$end" in result

    @pytest.mark.unit
    def test_smd_solvation_block(self, qchem_builder, minimal_spec):
        """SMD solvation should generate $smx block."""
        from calcflow.common.input import SolvationSpec

        spec = replace(minimal_spec, solvation=SolvationSpec(model="smd", solvent="water"))
        result = qchem_builder._build_solvation_blocks(spec)
        assert "$smx" in result
        assert "water" in result.lower()
        assert "$end" in result


# ============================================================================
# CONTRACT TESTS: Test build() method with specific specs
# ============================================================================


class TestSpBasicStructure:
    """Contract tests for basic SP calculation structure."""

    @pytest.mark.contract
    def test_sp_has_required_blocks(self, qchem_builder, h2o_geometry):
        """Basic SP should have required blocks."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert components.molecule_block
        assert components.rem_block
        assert "$molecule" in components.molecule_block
        assert "$rem" in components.rem_block

    @pytest.mark.contract
    def test_sp_molecule_structure(self, qchem_builder, h2o_geometry):
        """SP molecule block should have charge/mult and geometry."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_molecule_has_charge_mult(components.molecule_block, 0, 1)
        assert_molecule_has_atoms(components.molecule_block, ("O", 1), ("H", 2))

    @pytest.mark.contract
    def test_sp_rem_has_required_keys(self, qchem_builder, h2o_geometry):
        """SP $rem should have required keys."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_contains_keys(
            components.rem_block,
            "JOBTYPE",
            "METHOD",
            "BASIS",
            "UNRESTRICTED",
            "SYMMETRY",
        )

    @pytest.mark.contract
    def test_sp_rem_correct_values(self, qchem_builder, h2o_geometry):
        """SP $rem should have correct values."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "JOBTYPE", "sp")
        assert_rem_value(components.rem_block, "METHOD", "b3lyp")
        assert_rem_value(components.rem_block, "BASIS", "6-31g")


class TestOptStructure:
    """Contract tests for geometry optimization structure."""

    @pytest.mark.contract
    def test_opt_jobtype(self, qchem_builder, h2o_geometry):
        """Optimization should have JOBTYPE opt."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "JOBTYPE", "opt")

    @pytest.mark.contract
    def test_opt_has_molecule_and_rem(self, qchem_builder, h2o_geometry):
        """Optimization should have molecule and rem blocks."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        assert "$molecule" in result
        assert "$rem" in result


class TestFreqStructure:
    """Contract tests for frequency calculation structure."""

    @pytest.mark.contract
    def test_freq_jobtype(self, qchem_builder, h2o_geometry):
        """Frequency should have JOBTYPE freq."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="frequency",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "JOBTYPE", "freq")


class TestUnrestrictedCalculations:
    """Contract tests for unrestricted calculations."""

    @pytest.mark.contract
    def test_unrestricted_flag_set(self, qchem_builder, h2o_geometry):
        """Unrestricted calculation should set UNRESTRICTED True."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=3,  # Triplet
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            unrestricted=True,
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "UNRESTRICTED", "True")


class TestDictBasisStructure:
    """Contract tests for dictionary basis sets."""

    @pytest.mark.contract
    def test_dict_basis_generates_basis_block(self, qchem_builder, h2o_geometry):
        """Dictionary basis should generate $basis block."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_block_present(components.blocks, "basis")

    @pytest.mark.contract
    def test_dict_basis_rem_has_gen(self, qchem_builder, h2o_geometry):
        """Dictionary basis should set BASIS gen in $rem."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "BASIS", "gen")


class TestSolvationStructure:
    """Contract tests for solvation."""

    @pytest.mark.contract
    def test_pcm_solvation_structure(self, qchem_builder, h2o_geometry):
        """PCM solvation should generate $solvent block."""
        from calcflow.common.input import SolvationSpec

        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            solvation=SolvationSpec(model="pcm", solvent="water"),
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_block_present(components.blocks, "solvent")

    @pytest.mark.contract
    def test_smd_solvation_structure(self, qchem_builder, h2o_geometry):
        """SMD solvation should generate $smx block."""
        from calcflow.common.input import SolvationSpec

        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_block_present(components.blocks, "smx")


# ============================================================================
# INTEGRATION TESTS: Test fluent API workflows
# ============================================================================


class TestFluentApiWorkflows:
    """Integration tests for CalculationInput fluent API."""

    @pytest.mark.integration
    def test_basic_sp_workflow(self, h2o_geometry):
        """Basic SP workflow using fluent API."""
        from calcflow.common.input import CalculationInput

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = calc.export("qchem", h2o_geometry)

        assert "$molecule" in result
        assert "$rem" in result
        assert "b3lyp" in result.lower()
        assert "6-31g" in result.lower()

    @pytest.mark.integration
    def test_unrestricted_workflow(self, h2o_geometry):
        """Unrestricted calculation workflow."""
        from calcflow.common.input import CalculationInput

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=3,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        ).set_unrestricted(True)
        result = calc.export("qchem", h2o_geometry)

        assert "UNRESTRICTED" in result
        assert "True" in result

    @pytest.mark.integration
    def test_dict_basis_workflow(self, h2o_geometry):
        """Dictionary basis workflow."""
        from calcflow.common.input import CalculationInput

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        result = calc.export("qchem", h2o_geometry)

        assert "$basis" in result
        assert "gen" in result.lower()
        assert "H" in result
        assert "O" in result

    @pytest.mark.integration
    def test_solvation_workflow(self, h2o_geometry):
        """Solvation workflow."""
        from calcflow.common.input import CalculationInput

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        ).set_solvation("smd", "water")

        result = calc.export("qchem", h2o_geometry)

        assert "$smx" in result
        assert "water" in result.lower()

    @pytest.mark.integration
    def test_tddft_workflow(self, h2o_geometry):
        """TDDFT workflow."""
        from calcflow.common.input import CalculationInput

        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="6-31g",
        ).set_tddft(nroots=10, singlets=True, triplets=False, use_tda=True)
        result = calc.export("qchem", h2o_geometry)

        assert "CIS_N_ROOTS" in result
        assert "10" in result
        assert "CIS_SINGLETS" in result

    @pytest.mark.integration
    def test_complex_workflow(self, h2o_geometry):
        """Complex workflow combining multiple features."""
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
            .set_tddft(nroots=5, singlets=True, triplets=False, use_tda=False)
            .set_cores(8)
        )
        result = calc.export("qchem", h2o_geometry)

        # Check for all components
        assert "$molecule" in result
        assert "$rem" in result
        assert "$smx" in result
        assert "CIS_N_ROOTS" in result
        assert "cam-b3lyp" in result.lower()


# ============================================================================
# REGRESSION TESTS: Semantic validation of output
# ============================================================================


class TestRegressionSemanticValidation:
    """Regression tests using semantic validation."""

    @pytest.mark.regression
    def test_sp_semantic_regression(self, qchem_builder, h2o_geometry):
        """Basic SP semantic regression test."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        # Semantic assertions
        assert_rem_value(components.rem_block, "JOBTYPE", "sp")
        assert_rem_value(components.rem_block, "METHOD", "b3lyp")
        assert_rem_value(components.rem_block, "BASIS", "6-31g")
        assert_rem_value(components.rem_block, "UNRESTRICTED", "False")
        assert_molecule_has_charge_mult(components.molecule_block, 0, 1)
        assert_molecule_has_atoms(components.molecule_block, ("O", 1), ("H", 2))

    @pytest.mark.regression
    def test_opt_semantic_regression(self, qchem_builder, h2o_geometry):
        """Geometry optimization semantic regression test."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "JOBTYPE", "opt")
        assert_rem_value(components.rem_block, "METHOD", "b3lyp")

    @pytest.mark.regression
    def test_dict_basis_semantic_regression(self, qchem_builder, h2o_geometry):
        """Dictionary basis semantic regression test."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "BASIS", "gen")
        assert_block_present(components.blocks, "basis")

    @pytest.mark.regression
    def test_solvation_semantic_regression(self, qchem_builder, h2o_geometry):
        """Solvation semantic regression test."""
        from calcflow.common.input import SolvationSpec

        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            solvation=SolvationSpec(model="smd", solvent="water"),
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_block_present(components.blocks, "smx")

    @pytest.mark.regression
    def test_charged_species_semantic_regression(self, qchem_builder, h2o_geometry):
        """Charged species semantic regression test."""
        spec = CalculationInput(
            charge=1,
            spin_multiplicity=2,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_molecule_has_charge_mult(components.molecule_block, 1, 2)

    @pytest.mark.regression
    def test_unrestricted_semantic_regression(self, qchem_builder, h2o_geometry):
        """Unrestricted calculation semantic regression test."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=3,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            unrestricted=True,
        )
        result = qchem_builder.build(spec, h2o_geometry)
        components = parse_qchem_input(result)

        assert_rem_value(components.rem_block, "UNRESTRICTED", "True")
