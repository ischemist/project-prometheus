"""Unit tests for Q-Chem builder validation logic."""

from __future__ import annotations

import pytest

from calcflow.common.exceptions import ConfigurationError, NotSupportedError, ValidationError
from calcflow.common.input import CalculationInput, TddftSpec
from calcflow.io.qchem.builder import QchemBuilder


@pytest.fixture
def qchem_builder() -> QchemBuilder:
    """Q-Chem builder instance."""
    return QchemBuilder()


@pytest.fixture
def minimal_spec(minimal_spec: CalculationInput) -> CalculationInput:
    """Minimal spec with qchem-compatible settings."""
    # minimal_spec from conftest.py uses HF which may not be qchem-compatible
    # so we override with a qchem-specific version
    return CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
    )


class TestValidateSpec:
    """Tests for _validate_spec method."""

    @pytest.mark.unit
    def test_valid_minimal_spec(self, qchem_builder, minimal_spec):
        """Minimal valid spec should not raise."""
        qchem_builder._validate_spec(minimal_spec)  # Should not raise

    @pytest.mark.unit
    def test_unsupported_solvation_model(self, qchem_builder, minimal_spec):
        """Unsupported solvation model should raise NotSupportedError."""
        from dataclasses import replace

        from calcflow.common.input import SolvationSpec

        spec = replace(minimal_spec, solvation=SolvationSpec(model="cosmo", solvent="water"))
        with pytest.raises(NotSupportedError, match="solvation model"):
            qchem_builder._validate_spec(spec)

    @pytest.mark.unit
    @pytest.mark.parametrize("model", ["pcm", "smd", "isosvp", "cpcm"])
    def test_supported_solvation_models(self, qchem_builder, minimal_spec, model):
        """All supported solvation models should be accepted."""
        from dataclasses import replace

        from calcflow.common.input import SolvationSpec

        spec = replace(minimal_spec, solvation=SolvationSpec(model=model, solvent="water"))
        qchem_builder._validate_spec(spec)  # Should not raise

    @pytest.mark.unit
    def test_deprecated_run_mom_in_program_options(self, qchem_builder, minimal_spec):
        """Using program_options['run_mom'] should raise deprecation error."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            unrestricted=True,
            program_options={"run_mom": True},
        )
        with pytest.raises(ConfigurationError, match="deprecated"):
            qchem_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_mom_with_unrestricted_valid(self, qchem_builder):
        """MOM with unrestricted=True should be valid."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            unrestricted=True,
        ).set_mom(transition="HOMO->LUMO")
        qchem_builder._validate_spec(spec)  # Should not raise

    @pytest.mark.unit
    def test_trnss_requires_tddft(self, qchem_builder, minimal_spec):
        """TRNSS (reduced excitation space) requires TDDFT."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            program_options={"reduced_excitation_space_orbitals": [1, 2, 3]},
        )
        with pytest.raises(ConfigurationError, match="reduced excitation space"):
            qchem_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_trnss_with_tddft_valid(self, qchem_builder):
        """TRNSS with TDDFT should be valid."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            tddft=TddftSpec(nroots=10, singlets=True, triplets=False, use_tda=False),
            program_options={"reduced_excitation_space_orbitals": [1, 2, 3]},
        )
        qchem_builder._validate_spec(spec)  # Should not raise

    @pytest.mark.unit
    def test_unsupported_level_of_theory(self, qchem_builder, h2o_geometry):
        """Unsupported level of theory should raise when building."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="unknown_method",
            basis_set="6-31g",
        )
        with pytest.raises(NotSupportedError):
            qchem_builder.build(spec, h2o_geometry)

    @pytest.mark.unit
    def test_unsupported_task(self, qchem_builder, h2o_geometry):
        """Unsupported task should raise when building."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="raman",  # Not a valid task
            level_of_theory="b3lyp",
            basis_set="6-31g",
        )
        with pytest.raises(NotSupportedError, match="task"):
            qchem_builder.build(spec, h2o_geometry)


class TestValidateMomOccupations:
    """Tests for _validate_mom_occupations method."""

    @pytest.mark.unit
    def test_valid_occupation_strings(self, qchem_builder):
        """Valid occupation strings matching electron count and multiplicity."""
        # Closed shell: 10 electrons, multiplicity 1, 5 alpha and 5 beta
        qchem_builder._validate_mom_occupations("1:5", "1:5", 10, 1)

    @pytest.mark.unit
    def test_electron_count_mismatch(self, qchem_builder):
        """Occupation total electron count must match."""
        # 10 electrons expected, but occupations only have 9
        with pytest.raises(ValidationError, match="electron"):
            qchem_builder._validate_mom_occupations("1:5", "1:4", 10, 1)

    @pytest.mark.unit
    def test_multiplicity_mismatch_closed_shell(self, qchem_builder):
        """Closed shell multiplicity must have equal alpha/beta."""
        # 10 electrons, multiplicity 1 (singlet) requires 5 alpha and 5 beta
        with pytest.raises(ValidationError, match="inconsistent"):
            qchem_builder._validate_mom_occupations("1:6", "1:4", 10, 1)

    @pytest.mark.unit
    def test_multiplicity_mismatch_doublet(self, qchem_builder):
        """Doublet must have one more alpha than beta."""
        # 9 electrons, multiplicity 2 (doublet) requires 5 alpha and 4 beta
        with pytest.raises(ValidationError, match="inconsistent"):
            qchem_builder._validate_mom_occupations("1:4", "1:5", 9, 2)

    @pytest.mark.unit
    def test_triplet_electron_distribution(self, qchem_builder):
        """Triplet distribution should be 6 alpha, 4 beta for 10 electrons."""
        qchem_builder._validate_mom_occupations("1:6", "1:4", 10, 3)

    @pytest.mark.unit
    def test_mixed_notation_valid(self, qchem_builder):
        """Mixed occupation notation should work."""
        # 9 electrons: 1:4 5 = {1,2,3,4,5} = 5 electrons + 1:4 = {1,2,3,4} = 4 electrons
        qchem_builder._validate_mom_occupations("1:4 5", "1:4", 9, 2)

    @pytest.mark.unit
    def test_single_orbitals(self, qchem_builder):
        """Single orbital notation."""
        # 2 electrons: alpha = {1}, beta = {1}
        qchem_builder._validate_mom_occupations("1", "1", 2, 1)
