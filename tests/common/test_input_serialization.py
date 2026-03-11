"""
unit and contract tests for calculation input serialization/deserialization.

tests cover:
- individual spec dataclasses (unit)
- roundtrip fidelity (contract)
- full CalculationInput with nested specs (integration)
- edge cases: None values, dict basis sets, program_options
"""

import json

import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.common.input import (
    INPUT_SCHEMA_VERSION,
    CalculationInput,
    ChargesSpec,
    MomSpec,
    OptimizationSpec,
    ScfSpec,
    SolvationSpec,
    TddftSpec,
)

# --- unit tests: individual spec serialization ---


@pytest.mark.unit
class TestTddftSpecSerialization:
    def test_to_dict_minimal(self):
        spec = TddftSpec(nroots=10)
        data = spec.to_dict()
        assert data == {
            "nroots": 10,
            "singlets": True,
            "triplets": False,
            "use_tda": True,
            "state_to_optimize": None,
        }

    def test_to_dict_full(self):
        spec = TddftSpec(nroots=20, singlets=False, triplets=True, use_tda=False, state_to_optimize=3)
        data = spec.to_dict()
        assert data == {
            "nroots": 20,
            "singlets": False,
            "triplets": True,
            "use_tda": False,
            "state_to_optimize": 3,
        }

    def test_from_dict_minimal(self):
        data = {"nroots": 10}
        spec = TddftSpec.from_dict(data)
        assert spec.nroots == 10
        assert spec.singlets is True
        assert spec.triplets is False

    def test_from_dict_full(self):
        data = {
            "nroots": 20,
            "singlets": False,
            "triplets": True,
            "use_tda": False,
            "state_to_optimize": 3,
        }
        spec = TddftSpec.from_dict(data)
        assert spec.nroots == 20
        assert spec.singlets is False
        assert spec.state_to_optimize == 3


@pytest.mark.unit
class TestSolvationSpecSerialization:
    def test_to_dict(self):
        spec = SolvationSpec(model="smd", solvent="water")
        data = spec.to_dict()
        assert data == {"model": "smd", "solvent": "water", "dielectric": None, "optical_dielectric": None}

    def test_to_dict_custom_dielectric(self):
        spec = SolvationSpec(model="pcm", dielectric=33.0, optical_dielectric=1.33)
        data = spec.to_dict()
        assert data["dielectric"] == 33.0
        assert data["optical_dielectric"] == 1.33
        assert data["solvent"] == ""

    def test_from_dict(self):
        data = {"model": "cpcm", "solvent": "acetonitrile"}
        spec = SolvationSpec.from_dict(data)
        assert spec.model == "cpcm"
        assert spec.solvent == "acetonitrile"

    def test_from_dict_custom_dielectric(self):
        data = {"model": "pcm", "solvent": "", "dielectric": 33.0, "optical_dielectric": 1.33}
        spec = SolvationSpec.from_dict(data)
        assert spec.dielectric == 33.0
        assert spec.optical_dielectric == 1.33


@pytest.mark.unit
class TestChargesSpecSerialization:
    def test_to_dict_defaults(self):
        """charges defaults serialize with hirshiter threshold."""
        spec = ChargesSpec()
        data = spec.to_dict()
        assert data == {
            "mulliken": True,
            "hirshfeld": False,
            "cm5": False,
            "hirshiter": False,
            "hirshiter_thresh": 5,
        }

    def test_to_dict_hirshfeld_cm5(self):
        """cm5 serialization preserves hirshfeld auto-enable."""
        spec = ChargesSpec(hirshfeld=True, cm5=True)
        data = spec.to_dict()
        assert data["hirshfeld"] is True
        assert data["cm5"] is True

    def test_from_dict(self):
        """charges from_dict restores iterative hirshfeld settings."""
        data = {"mulliken": True, "hirshfeld": True, "cm5": False, "hirshiter": True, "hirshiter_thresh": 9}
        spec = ChargesSpec.from_dict(data)
        assert spec.hirshfeld is True
        assert spec.hirshiter is True
        assert spec.hirshiter_thresh == 9

    def test_cm5_auto_enables_hirshfeld(self):
        """cm5 auto-enables hirshfeld."""
        spec = ChargesSpec(cm5=True)
        assert spec.hirshfeld is True

    def test_cm5_auto_enable_preserved_through_roundtrip(self):
        """cm5 auto-enable survives dict roundtrip."""
        spec = ChargesSpec(cm5=True)
        data = spec.to_dict()
        # hirshfeld was auto-enabled, so it should be True in the dict
        assert data["hirshfeld"] is True
        reconstructed = ChargesSpec.from_dict(data)
        assert reconstructed == spec

    def test_hirshiter_thresh_default(self):
        """iterative hirshfeld threshold defaults to 5."""
        spec = ChargesSpec()
        assert spec.hirshiter_thresh == 5

    def test_hirshiter_thresh_validation(self):
        """iterative hirshfeld threshold must be positive."""
        with pytest.raises(ValidationError):
            ChargesSpec(hirshiter_thresh=0)


@pytest.mark.unit
class TestScfSpecSerialization:
    def test_to_dict_defaults(self):
        """scf defaults serialize correctly."""
        spec = ScfSpec()
        data = spec.to_dict()
        assert data == {"algorithm": "diis", "max_cycles": 100, "convergence": 8}

    def test_to_dict_custom(self):
        """custom scf settings serialize correctly."""
        spec = ScfSpec(algorithm="diis_gdm", max_cycles=50, convergence=7)
        data = spec.to_dict()
        assert data == {"algorithm": "diis_gdm", "max_cycles": 50, "convergence": 7}

    def test_from_dict(self):
        """scf settings deserialize correctly."""
        data = {"algorithm": "diis", "max_cycles": 50, "convergence": 7}
        spec = ScfSpec.from_dict(data)
        assert spec.algorithm == "diis"
        assert spec.max_cycles == 50
        assert spec.convergence == 7


@pytest.mark.unit
class TestOptimizationSpecSerialization:
    def test_to_dict_defaults(self):
        spec = OptimizationSpec()
        data = spec.to_dict()
        assert data == {"calc_hess_initial": False, "recalc_hess_freq": None}

    def test_to_dict_custom(self):
        spec = OptimizationSpec(calc_hess_initial=True, recalc_hess_freq=5)
        data = spec.to_dict()
        assert data == {"calc_hess_initial": True, "recalc_hess_freq": 5}

    def test_from_dict(self):
        data = {"calc_hess_initial": True, "recalc_hess_freq": 10}
        spec = OptimizationSpec.from_dict(data)
        assert spec.calc_hess_initial is True
        assert spec.recalc_hess_freq == 10


@pytest.mark.unit
class TestMomSpecSerialization:
    def test_to_dict_minimal(self):
        spec = MomSpec(transition="HOMO->LUMO")
        data = spec.to_dict()
        assert data == {
            "transition": "HOMO->LUMO",
            "method": "IMOM",
            "job2_charge": None,
            "job2_spin_multiplicity": None,
            "alpha_occupation": None,
            "beta_occupation": None,
        }

    def test_to_dict_ionization(self):
        spec = MomSpec(transition="HOMO->vac", method="MOM", job2_charge=1, job2_spin_multiplicity=2)
        data = spec.to_dict()
        assert data["transition"] == "HOMO->vac"
        assert data["method"] == "MOM"
        assert data["job2_charge"] == 1
        assert data["job2_spin_multiplicity"] == 2

    def test_from_dict_minimal(self):
        data = {"transition": "HOMO->LUMO"}
        spec = MomSpec.from_dict(data)
        assert spec.transition == "HOMO->LUMO"
        assert spec.method == "IMOM"

    def test_from_dict_with_occupation(self):
        data = {
            "transition": "HOMO->LUMO",
            "method": "MOM",
            "job2_charge": None,
            "job2_spin_multiplicity": None,
            "alpha_occupation": "1-5",
            "beta_occupation": "1-4",
        }
        spec = MomSpec.from_dict(data)
        assert spec.alpha_occupation == "1-5"
        assert spec.beta_occupation == "1-4"


@pytest.mark.unit
class TestSpecFromDictStrictness:
    @pytest.mark.parametrize(
        ("spec_cls", "valid_data"),
        [
            (TddftSpec, {"nroots": 10}),
            (SolvationSpec, {"model": "smd", "solvent": "water"}),
            (ChargesSpec, {"mulliken": True}),
            (ScfSpec, {"algorithm": "diis"}),
            (OptimizationSpec, {"calc_hess_initial": True}),
            (MomSpec, {"transition": "HOMO->LUMO"}),
        ],
    )
    def test_from_dict_raises_on_unknown_key(self, spec_cls, valid_data):
        """spec from_dict remains strict and rejects unknown keys."""
        data_with_unknown = {**valid_data, "unknown_key": "unexpected"}

        with pytest.raises(TypeError, match="unknown_key"):
            spec_cls.from_dict(data_with_unknown)


# --- contract tests: roundtrip fidelity ---


@pytest.mark.contract
class TestSpecRoundtrip:
    @pytest.mark.parametrize(
        "spec",
        [
            TddftSpec(nroots=10),
            TddftSpec(nroots=20, singlets=False, triplets=True, state_to_optimize=2),
            SolvationSpec(model="smd", solvent="water"),
            SolvationSpec(model="cpcm", solvent="acetonitrile"),
            SolvationSpec(model="pcm", dielectric=33.0, optical_dielectric=1.33),
            SolvationSpec(model="pcm", dielectric=78.4),
            OptimizationSpec(),
            OptimizationSpec(calc_hess_initial=True, recalc_hess_freq=5),
            MomSpec(transition="HOMO->LUMO"),
            MomSpec(transition="HOMO->vac", job2_charge=1, job2_spin_multiplicity=2),
            ChargesSpec(),
            ChargesSpec(hirshfeld=True, cm5=True),
            ChargesSpec(mulliken=False, hirshfeld=True, hirshiter=True),
            ScfSpec(),
            ScfSpec(algorithm="diis_gdm", max_cycles=50, convergence=7),
        ],
    )
    def test_spec_roundtrip(self, spec):
        """ensure spec == from_dict(spec.to_dict()) for all spec types."""
        data = spec.to_dict()
        reconstructed = spec.__class__.from_dict(data)
        assert reconstructed == spec


# --- integration tests: full CalculationInput serialization ---


@pytest.mark.contract
class TestCalculationInputSerialization:
    def test_to_dict_includes_calcflow_version(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        )

        data = calc.to_dict()
        assert "calcflow_version" in data
        assert isinstance(data["calcflow_version"], str)
        assert data["calcflow_version"]

    def test_to_dict_includes_schema_version(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        )

        data = calc.to_dict()
        assert "schema_version" in data
        assert isinstance(data["schema_version"], int)
        assert data["schema_version"] == INPUT_SCHEMA_VERSION

    def test_from_dict_ignores_calcflow_version(self):
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            "unrestricted": False,
            "n_cores": 1,
            "memory_per_core_mb": 4000,
            "tddft": None,
            "solvation": None,
            "optimization": None,
            "mom": None,
            "charges": None,
            "scf": None,
            "frequency_after_optimization": False,
            "program_options": {},
            "calcflow_version": "0.0.0-test",
        }

        calc = CalculationInput.from_dict(data)
        assert calc.charge == 0
        assert calc.task == "energy"

    def test_from_dict_strips_schema_version(self):
        """schema_version is consumed by from_dict, not passed to the constructor."""
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            "schema_version": INPUT_SCHEMA_VERSION,
        }

        calc = CalculationInput.from_dict(data)
        assert calc.charge == 0
        assert not hasattr(calc, "schema_version")

    def test_from_dict_defaults_missing_schema_version_to_1(self):
        """old dumps without schema_version are treated as version 1."""
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            # no schema_version key at all — simulates a pre-versioning dump
        }

        calc = CalculationInput.from_dict(data)
        assert calc.charge == 0
        assert calc.level_of_theory == "b3lyp"

    def test_from_dict_logs_warning_on_old_schema(self, caplog):
        """migration from an older schema version emits a warning."""
        import logging

        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            "schema_version": 0,  # older than current
        }

        with caplog.at_level(logging.WARNING, logger="calcflow.common.input"):
            CalculationInput.from_dict(data)

        assert "Migrating CalculationInput" in caplog.text

    def test_from_dict_logs_warning_on_future_schema(self, caplog):
        """loading a dump from a newer schema version emits a warning."""
        import logging

        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            "schema_version": INPUT_SCHEMA_VERSION + 1,
        }

        with caplog.at_level(logging.WARNING, logger="calcflow.common.input"):
            CalculationInput.from_dict(data)

        assert "only understands version" in caplog.text

    def test_from_dict_raises_on_unknown_top_level_key(self):
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            "unknown_key": "unexpected",
        }

        with pytest.raises(TypeError, match="unknown_key"):
            CalculationInput.from_dict(data)

    def test_to_dict_minimal(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        )
        data = calc.to_dict()

        assert data["charge"] == 0
        assert data["spin_multiplicity"] == 1
        assert data["task"] == "energy"
        assert data["level_of_theory"] == "b3lyp"
        assert data["basis_set"] == "6-31g*"
        assert data["unrestricted"] is False
        assert data["tddft"] is None
        assert data["solvation"] is None
        assert data["program_options"] == {}

    def test_to_dict_with_tddft(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-svp",
        ).set_tddft(nroots=10, singlets=True, triplets=False)

        data = calc.to_dict()
        assert data["tddft"] == {
            "nroots": 10,
            "singlets": True,
            "triplets": False,
            "use_tda": True,
            "state_to_optimize": None,
        }

    def test_to_dict_with_solvation(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        ).set_solvation(model="smd", solvent="water")

        data = calc.to_dict()
        assert data["solvation"]["model"] == "smd"
        assert data["solvation"]["solvent"] == "water"

    def test_to_dict_with_custom_solvation(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        ).set_custom_solvation("pcm", dielectric=33.0, optical_dielectric=1.33)

        data = calc.to_dict()
        assert data["solvation"]["dielectric"] == 33.0
        assert data["solvation"]["optical_dielectric"] == 1.33

    def test_to_dict_with_charges(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        ).set_charges(hirshfeld=True, cm5=True)

        data = calc.to_dict()
        assert data["charges"]["hirshfeld"] is True
        assert data["charges"]["cm5"] is True

    def test_to_dict_with_scf(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        ).set_scf(algorithm="diis", max_cycles=50, convergence=7)

        data = calc.to_dict()
        assert data["scf"] == {"algorithm": "diis", "max_cycles": 50, "convergence": 7}

    def test_to_dict_with_total_memory(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g*",
        ).set_total_memory(64000)

        data = calc.to_dict()
        assert data["total_memory_mb"] == 64000

    def test_to_dict_with_dict_basis_set(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set={"C": "def2-tzvp", "H": "def2-svp"},
        )
        data = calc.to_dict()
        assert data["basis_set"] == {"C": "def2-tzvp", "H": "def2-svp"}

    def test_to_dict_with_program_options(self):
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
        ).set_options(ri_approx="RIJCOSX", aux_basis="def2/j", custom_flag="value")

        data = calc.to_dict()
        assert data["program_options"] == {
            "ri_approx": "RIJCOSX",
            "aux_basis": "def2/j",
            "custom_flag": "value",
        }

    def test_from_dict_minimal(self):
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "b3lyp",
            "basis_set": "6-31g*",
            "unrestricted": False,
            "n_cores": 1,
            "memory_per_core_mb": 4000,
            "tddft": None,
            "solvation": None,
            "optimization": None,
            "mom": None,
            "frequency_after_optimization": False,
            "program_options": {},
        }
        calc = CalculationInput.from_dict(data)
        assert calc.charge == 0
        assert calc.level_of_theory == "b3lyp"
        assert calc.basis_set == "6-31g*"

    def test_from_dict_with_nested_specs(self):
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "cam-b3lyp",
            "basis_set": "def2-svp",
            "unrestricted": False,
            "n_cores": 8,
            "memory_per_core_mb": 4000,
            "total_memory_mb": None,
            "tddft": {
                "nroots": 10,
                "singlets": True,
                "triplets": False,
                "use_tda": True,
                "state_to_optimize": None,
            },
            "solvation": {"model": "smd", "solvent": "water", "dielectric": None, "optical_dielectric": None},
            "optimization": None,
            "mom": None,
            "charges": None,
            "scf": None,
            "frequency_after_optimization": False,
            "program_options": {"ri_approx": "RIJCOSX"},
        }
        calc = CalculationInput.from_dict(data)

        assert isinstance(calc.tddft, TddftSpec)
        assert calc.tddft.nroots == 10
        assert isinstance(calc.solvation, SolvationSpec)
        assert calc.solvation.model == "smd"
        assert calc.program_options == {"ri_approx": "RIJCOSX"}

    def test_from_dict_with_charges_and_scf(self):
        data = {
            "charge": 0,
            "spin_multiplicity": 1,
            "task": "energy",
            "level_of_theory": "wb97x-d3",
            "basis_set": "def2-tzvp",
            "unrestricted": True,
            "n_cores": 1,
            "memory_per_core_mb": 4000,
            "total_memory_mb": 64000,
            "tddft": None,
            "solvation": {"model": "pcm", "solvent": "", "dielectric": 33.0, "optical_dielectric": 1.33},
            "optimization": None,
            "mom": None,
            "charges": {"mulliken": True, "hirshfeld": True, "cm5": True, "hirshiter": False, "hirshiter_thresh": 5},
            "scf": {"algorithm": "diis", "max_cycles": 50, "convergence": 7},
            "frequency_after_optimization": False,
            "program_options": {},
        }
        calc = CalculationInput.from_dict(data)

        assert isinstance(calc.charges, ChargesSpec)
        assert calc.charges.hirshfeld is True
        assert calc.charges.cm5 is True
        assert isinstance(calc.scf, ScfSpec)
        assert calc.scf.algorithm == "diis"
        assert calc.scf.convergence == 7
        assert calc.total_memory_mb == 64000
        assert calc.solvation.dielectric == 33.0


@pytest.mark.contract
class TestCalculationInputRoundtrip:
    @pytest.mark.parametrize(
        "calc",
        [
            # minimal
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="6-31g*",
            ),
            # with tddft
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-svp",
            ).set_tddft(nroots=10),
            # with solvation
            CalculationInput(
                charge=-1,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wb97x-d3",
                basis_set="def2-tzvp",
            ).set_solvation(model="smd", solvent="acetonitrile"),
            # with dict basis
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set={"C": "def2-tzvp", "H": "def2-svp"},
            ),
            # geometry optimization with tddft
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="geometry",
                level_of_theory="cam-b3lyp",
                basis_set="def2-svp",
            )
            .set_tddft(nroots=5, state_to_optimize=1)
            .set_optimization(calc_hess_initial=True),
            # with program options
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
                n_cores=16,
                memory_per_core_mb=8000,
            ).set_options(ri_approx="RIJCOSX", aux_basis="def2/j"),
            # mom calculation
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
                unrestricted=True,
            ).set_mom(transition="HOMO->LUMO"),
            # with charges
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            ).set_charges(hirshfeld=True, cm5=True),
            # with scf + total memory
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
            )
            .set_scf(algorithm="diis", max_cycles=50, convergence=7)
            .set_total_memory(64000),
            # with custom solvation
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wb97x-d3",
                basis_set="def2-tzvp",
            ).set_custom_solvation("pcm", dielectric=33.0, optical_dielectric=1.33),
            # full combination matching target hirshfeld workflow
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wb97x-d3",
                basis_set="def2-tzvp",
                unrestricted=True,
            )
            .set_custom_solvation("pcm", dielectric=33.0, optical_dielectric=1.33)
            .set_charges(hirshfeld=True, cm5=True)
            .set_scf(algorithm="diis", max_cycles=50, convergence=7)
            .set_total_memory(64000),
        ],
    )
    def test_calculation_input_dict_roundtrip(self, calc):
        """ensure calc == from_dict(calc.to_dict())."""
        data = calc.to_dict()
        reconstructed = CalculationInput.from_dict(data)
        assert reconstructed == calc

    @pytest.mark.parametrize(
        "calc",
        [
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="6-31g*",
            ),
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-svp",
            )
            .set_tddft(nroots=10)
            .set_solvation(model="smd", solvent="water")
            .set_options(ri_approx="RIJCOSX"),
        ],
    )
    def test_calculation_input_json_roundtrip(self, calc):
        """ensure calc == from_json(calc.to_json())."""
        json_str = calc.to_json()
        reconstructed = CalculationInput.from_json(json_str)
        assert reconstructed == calc

    def test_json_is_valid_and_readable(self):
        """ensure the json output is valid and human-readable."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="cam-b3lyp",
                basis_set="def2-svp",
                n_cores=16,
            )
            .set_tddft(nroots=10)
            .set_solvation(model="smd", solvent="water")
        )

        json_str = calc.to_json()

        # verify it's valid json
        parsed = json.loads(json_str)
        assert parsed["charge"] == 0
        assert parsed["tddft"]["nroots"] == 10
        assert parsed["solvation"]["model"] == "smd"

        # verify it's formatted with indentation
        assert "\n" in json_str
        assert "  " in json_str


@pytest.mark.integration
class TestComplexCalculationInputSerialization:
    def test_full_featured_calculation(self):
        """test serialization of a calculation with all features enabled."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="geometry",
                level_of_theory="wb97x-d3",
                basis_set={"C": "def2-tzvp", "H": "def2-svp", "O": "def2-tzvp"},
                unrestricted=False,
                n_cores=32,
                memory_per_core_mb=8000,
            )
            .set_tddft(nroots=20, singlets=True, triplets=True, use_tda=False, state_to_optimize=1)
            .set_solvation(model="cpcm", solvent="acetonitrile")
            .set_optimization(calc_hess_initial=True, recalc_hess_freq=10)
            .run_frequency_after_opt()
            .set_options(ri_approx="RIJCOSX", aux_basis="def2/j", grid="5", scf_conv="8")
        )

        # serialize and deserialize
        json_str = calc.to_json()
        reconstructed = CalculationInput.from_json(json_str)

        # verify all fields preserved
        assert reconstructed == calc
        assert reconstructed.charge == 0
        assert reconstructed.n_cores == 32
        assert reconstructed.basis_set == {"C": "def2-tzvp", "H": "def2-svp", "O": "def2-tzvp"}
        assert reconstructed.tddft.nroots == 20
        assert reconstructed.solvation.solvent == "acetonitrile"
        assert reconstructed.optimization.calc_hess_initial is True
        assert reconstructed.frequency_after_optimization is True
        assert reconstructed.program_options["grid"] == "5"

    def test_mom_ionization_calculation(self):
        """test serialization of a mom ionization calculation."""
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="b3lyp",
                basis_set="def2-svp",
                unrestricted=True,
            )
            .set_mom(transition="HOMO->vac", method="IMOM", job2_charge=1, job2_spin_multiplicity=2)
            .set_solvation(model="smd", solvent="water")
        )

        json_str = calc.to_json()
        reconstructed = CalculationInput.from_json(json_str)

        assert reconstructed == calc
        assert reconstructed.mom.transition == "HOMO->vac"
        assert reconstructed.mom.job2_charge == 1
        assert reconstructed.mom.job2_spin_multiplicity == 2
        assert reconstructed.requires_multiple_jobs is True
