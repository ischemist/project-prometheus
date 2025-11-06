"""Unit tests for OrcaBuilder validation logic."""

from __future__ import annotations

import pytest

from calcflow.common.exceptions import NotSupportedError, ValidationError
from calcflow.common.input import CalculationInput, SolvationSpec


class TestValidateRejectsDictBasis:
    """Test that dict basis sets are rejected."""

    @pytest.mark.unit
    def test_validate_rejects_dict_basis(self, orca_builder):
        """Orca does not support per-element basis sets."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"H": "6-31g", "O": "6-311g(d,p)"},
        )
        with pytest.raises(NotSupportedError, match="dictionary basis"):
            orca_builder._validate_spec(spec)


class TestValidateSolvationModel:
    """Test solvation model validation."""

    @pytest.mark.unit
    def test_validate_rejects_unsupported_solvation_model(self, orca_builder):
        """Orca only supports smd and cpcm solvation."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            solvation=SolvationSpec(model="pcm", solvent="water"),
        )
        with pytest.raises(NotSupportedError, match="smd|cpcm"):
            orca_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_validate_accepts_smd_solvation(self, orca_builder, minimal_spec):
        """Orca should accept smd solvation model."""
        spec = minimal_spec.__class__(
            **{
                **minimal_spec.__dict__,
                "solvation": SolvationSpec(model="smd", solvent="water"),
            }
        )
        # Should not raise
        orca_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_validate_accepts_cpcm_solvation(self, orca_builder, minimal_spec):
        """Orca should accept cpcm solvation model."""
        spec = minimal_spec.__class__(
            **{
                **minimal_spec.__dict__,
                "solvation": SolvationSpec(model="cpcm", solvent="water"),
            }
        )
        # Should not raise
        orca_builder._validate_spec(spec)


class TestValidateRIApproximation:
    """Test RI approximation validation."""

    @pytest.mark.unit
    def test_validate_ri_requires_aux_basis(self, orca_builder, minimal_spec):
        """RI approximation requires aux_basis to be set."""
        spec = minimal_spec.__class__(
            **{
                **minimal_spec.__dict__,
                "program_options": {"ri_approx": "RIJCOSX"},
            }
        )
        with pytest.raises(ValidationError, match="aux_basis"):
            orca_builder._validate_spec(spec)

    @pytest.mark.unit
    def test_validate_ri_with_aux_basis_passes(self, orca_builder, minimal_spec):
        """RI approximation with aux_basis should pass validation."""
        spec = minimal_spec.__class__(
            **{
                **minimal_spec.__dict__,
                "program_options": {"ri_approx": "RIJCOSX", "aux_basis": "def2/j"},
            }
        )
        # Should not raise
        orca_builder._validate_spec(spec)


class TestHandleLevelOfTheory:
    """Test level of theory mapping."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method,expected_keywords",
        [
            ("b3lyp", ["RKS", "b3lyp"]),
            ("pbe0", ["RKS", "pbe0"]),
            ("cam-b3lyp", ["RKS", "cam-b3lyp"]),
            ("m06", ["RKS", "m06"]),
            ("wb97x", ["RKS", "wb97x"]),
            ("wb97x-d3", ["RKS", "wb97x-d3"]),
        ],
    )
    def test_handle_dft_functionals_restricted(self, orca_builder, method, expected_keywords):
        """DFT functionals should map to RKS for restricted."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory=method,
            basis_set="sto-3g",
            unrestricted=False,
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        for kw in expected_keywords:
            assert kw in keywords

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method,expected_keywords",
        [
            ("b3lyp", ["UKS", "b3lyp"]),
            ("pbe0", ["UKS", "pbe0"]),
            ("cam-b3lyp", ["UKS", "cam-b3lyp"]),
        ],
    )
    def test_handle_dft_functionals_unrestricted(self, orca_builder, method, expected_keywords):
        """DFT functionals should map to UKS for unrestricted."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory=method,
            basis_set="sto-3g",
            unrestricted=True,
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        for kw in expected_keywords:
            assert kw in keywords

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["hf", "rhf"])
    def test_handle_hf_restricted(self, orca_builder, method):
        """HF method should map to RHF for restricted."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory=method,
            basis_set="sto-3g",
            unrestricted=False,
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        assert "RHF" in keywords

    @pytest.mark.unit
    def test_handle_hf_unrestricted(self, orca_builder):
        """HF method should map to UHF for unrestricted."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            unrestricted=True,
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        assert "UHF" in keywords

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["mp2", "ri-mp2"])
    def test_handle_mp2_restricted(self, orca_builder, method):
        """MP2 method should map to MP2 for restricted."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory=method,
            basis_set="sto-3g",
            unrestricted=False,
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        assert "MP2" in keywords

    @pytest.mark.unit
    def test_handle_mp2_unrestricted(self, orca_builder):
        """MP2 method should map to UMP2 for unrestricted."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="ump2",
            basis_set="sto-3g",
            unrestricted=True,
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        assert "UMP2" in keywords

    @pytest.mark.unit
    def test_handle_ccsd(self, orca_builder):
        """CCSD method should be handled."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="ccsd",
            basis_set="sto-3g",
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        assert "CCSD" in keywords

    @pytest.mark.unit
    def test_handle_ccsd_t(self, orca_builder):
        """CCSD(T) method should be handled."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="ccsd(t)",
            basis_set="sto-3g",
        )
        keywords = orca_builder._handle_level_of_theory(spec)
        assert "CCSD(T)" in keywords

    @pytest.mark.unit
    def test_handle_unsupported_method(self, orca_builder):
        """Unsupported method should raise NotSupportedError."""
        spec = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="dft-sapt",
            basis_set="sto-3g",
        )
        with pytest.raises(NotSupportedError, match="not supported"):
            orca_builder._handle_level_of_theory(spec)

    @pytest.mark.unit
    def test_handle_case_insensitive(self, orca_builder):
        """Method names should be case-insensitive."""
        spec_upper = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="B3LYP",
            basis_set="sto-3g",
        )
        keywords = orca_builder._handle_level_of_theory(spec_upper)
        assert "b3lyp" in keywords
