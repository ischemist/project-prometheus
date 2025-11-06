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

from calcflow.common.input import (
    CalculationInput,
    MomSpec,
    OptimizationSpec,
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
        assert data == {"model": "smd", "solvent": "water"}

    def test_from_dict(self):
        data = {"model": "cpcm", "solvent": "acetonitrile"}
        spec = SolvationSpec.from_dict(data)
        assert spec.model == "cpcm"
        assert spec.solvent == "acetonitrile"


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
            OptimizationSpec(),
            OptimizationSpec(calc_hess_initial=True, recalc_hess_freq=5),
            MomSpec(transition="HOMO->LUMO"),
            MomSpec(transition="HOMO->vac", job2_charge=1, job2_spin_multiplicity=2),
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
        assert data["solvation"] == {"model": "smd", "solvent": "water"}

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
            "tddft": {
                "nroots": 10,
                "singlets": True,
                "triplets": False,
                "use_tda": True,
                "state_to_optimize": None,
            },
            "solvation": {"model": "smd", "solvent": "water"},
            "optimization": None,
            "mom": None,
            "frequency_after_optimization": False,
            "program_options": {"ri_approx": "RIJCOSX"},
        }
        calc = CalculationInput.from_dict(data)

        assert isinstance(calc.tddft, TddftSpec)
        assert calc.tddft.nroots == 10
        assert isinstance(calc.solvation, SolvationSpec)
        assert calc.solvation.model == "smd"
        assert calc.program_options == {"ri_approx": "RIJCOSX"}


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
