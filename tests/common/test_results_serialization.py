"""
unit and contract tests for calculation result serialization/deserialization.

tests cover:
- individual result dataclasses (unit)
- roundtrip fidelity (contract)
- full CalculationResult with nested models (integration)
- edge cases: None values, raw_output exclusion
- real parsed output from ORCA and QChem (integration)
"""

import json
from pathlib import Path

import pytest

from calcflow.common.results import (
    Atom,
    AtomicCharges,
    CalculationMetadata,
    CalculationResult,
    DipoleMoment,
    DispersionCorrection,
    ExcitedState,
    MultipoleResults,
    NTOContribution,
    NTOStateAnalysis,
    Orbital,
    OrbitalsSet,
    OrbitalTransition,
    QuadrupoleMoment,
    ScfEnergyComponents,
    ScfIteration,
    ScfResults,
    SmdResults,
    TddftResults,
    TimingResults,
)
from calcflow.io.orca import parse_orca_output
from calcflow.io.qchem import parse_qchem_output

# --- unit tests: individual model serialization ---


@pytest.mark.unit
class TestAtomSerialization:
    def test_to_dict(self):
        atom = Atom(symbol="O", x=0.0, y=0.0, z=0.0)
        data = atom.to_dict()
        assert data == {"symbol": "O", "x": 0.0, "y": 0.0, "z": 0.0}

    def test_from_dict(self):
        data = {"symbol": "H", "x": 0.0, "y": 0.0, "z": 0.96}
        atom = Atom.from_dict(data)
        assert atom.symbol == "H"
        assert atom.z == 0.96

    def test_roundtrip(self):
        atom = Atom(symbol="C", x=1.23, y=-0.45, z=0.67)
        reconstructed = Atom.from_dict(atom.to_dict())
        assert reconstructed == atom


@pytest.mark.unit
class TestOrbitalSerialization:
    def test_to_dict_minimal(self):
        orb = Orbital(index=0, energy=-1.23)
        data = orb.to_dict()
        assert data == {"index": 0, "energy": -1.23, "occupation": None, "energy_ev": None}

    def test_to_dict_full(self):
        orb = Orbital(index=5, energy=-0.456, occupation=2.0, energy_ev=-12.4)
        data = orb.to_dict()
        assert data["index"] == 5
        assert data["occupation"] == 2.0

    def test_roundtrip(self):
        orb = Orbital(index=10, energy=-0.234, occupation=1.0, energy_ev=-6.37)
        reconstructed = Orbital.from_dict(orb.to_dict())
        assert reconstructed == orb


@pytest.mark.unit
class TestScfIterationSerialization:
    def test_roundtrip_orca_style(self):
        scf_iter = ScfIteration(iteration=1, energy=-75.0, delta_e_eh=-0.1, rmsdp=0.01, maxdp=0.02)
        reconstructed = ScfIteration.from_dict(scf_iter.to_dict())
        assert reconstructed == scf_iter

    def test_roundtrip_qchem_style(self):
        scf_iter = ScfIteration(iteration=2, energy=-75.1, diis_error=0.001)
        reconstructed = ScfIteration.from_dict(scf_iter.to_dict())
        assert reconstructed == scf_iter


@pytest.mark.unit
class TestScfResultsSerialization:
    def test_roundtrip_full(self):
        scf = ScfResults(
            converged=True,
            energy=-75.313506,
            n_iterations=10,
            iterations=[
                ScfIteration(iteration=1, energy=-75.0, delta_e_eh=-0.1, rmsdp=0.01),
                ScfIteration(iteration=2, energy=-75.2, delta_e_eh=-0.2, rmsdp=0.001),
            ],
            components=ScfEnergyComponents(
                nuclear_repulsion=8.0,
                electronic_eh=-83.0,
                one_electron_eh=-122.0,
                two_electron_eh=39.0,
                xc_eh=-0.5,
            ),
        )
        reconstructed = ScfResults.from_dict(scf.to_dict())
        assert reconstructed == scf
        assert len(reconstructed.iterations) == 2
        assert reconstructed.components.xc_eh == -0.5


@pytest.mark.unit
class TestOrbitalSetSerialization:
    def test_roundtrip_rhf(self):
        orbitals = OrbitalsSet(
            alpha_orbitals=[
                Orbital(index=0, energy=-20.0, occupation=2.0),
                Orbital(index=1, energy=-1.5, occupation=2.0),
                Orbital(index=2, energy=0.5, occupation=0.0),
            ],
            alpha_homo_index=1,
            alpha_lumo_index=2,
        )
        reconstructed = OrbitalsSet.from_dict(orbitals.to_dict())
        assert reconstructed == orbitals
        assert len(reconstructed.alpha_orbitals) == 3
        assert reconstructed.beta_orbitals is None

    def test_roundtrip_uhf(self):
        orbitals = OrbitalsSet(
            alpha_orbitals=[Orbital(index=0, energy=-1.0, occupation=1.0)],
            beta_orbitals=[Orbital(index=0, energy=-1.1, occupation=1.0)],
            alpha_homo_index=0,
            alpha_lumo_index=1,
            beta_homo_index=0,
            beta_lumo_index=1,
        )
        reconstructed = OrbitalsSet.from_dict(orbitals.to_dict())
        assert reconstructed == orbitals
        assert reconstructed.beta_orbitals is not None
        assert len(reconstructed.beta_orbitals) == 1


@pytest.mark.unit
class TestAtomicChargesSerialization:
    def test_roundtrip_minimal(self):
        charges = AtomicCharges(method="Mulliken", charges={0: -0.5, 1: 0.25, 2: 0.25})
        reconstructed = AtomicCharges.from_dict(charges.to_dict())
        assert reconstructed == charges
        assert reconstructed.charges[0] == -0.5

    def test_roundtrip_with_spins(self):
        charges = AtomicCharges(method="Mulliken", charges={0: -0.5, 1: 0.25}, spins={0: 0.1, 1: -0.1})
        reconstructed = AtomicCharges.from_dict(charges.to_dict())
        assert reconstructed == charges
        assert reconstructed.spins[0] == 0.1


@pytest.mark.unit
class TestDipoleMomentSerialization:
    def test_roundtrip(self):
        dipole = DipoleMoment(x=0.1, y=0.2, z=0.3, magnitude=0.374)
        reconstructed = DipoleMoment.from_dict(dipole.to_dict())
        assert reconstructed == dipole


@pytest.mark.unit
class TestDispersionCorrectionSerialization:
    def test_roundtrip_full(self):
        disp = DispersionCorrection(
            method="DFTD3",
            e_disp_au=-0.001234,
            functional="omegaB97X-D3",
            damping="zero damping",
            molecular_c6_au=100.0,
            parameters={"s6": 1.0, "s8": 1.0},
            e_disp_kcal=-0.774,
            e6_kcal=-0.700,
            e8_kcal=-0.074,
            e8_percentage=10.5,
        )
        reconstructed = DispersionCorrection.from_dict(disp.to_dict())
        assert reconstructed == disp
        assert reconstructed.parameters["s6"] == 1.0


@pytest.mark.unit
class TestMultipoleResultsSerialization:
    def test_roundtrip_full(self):
        multipole = MultipoleResults(
            charge=0.0,
            dipole=DipoleMoment(x=0.1, y=0.2, z=0.3, magnitude=0.374),
            quadrupole=QuadrupoleMoment(xx=1.0, xy=0.0, yy=1.0, xz=0.0, yz=0.0, zz=1.0),
        )
        reconstructed = MultipoleResults.from_dict(multipole.to_dict())
        assert reconstructed == multipole
        assert reconstructed.dipole.magnitude == 0.374


@pytest.mark.unit
class TestSmdResultsSerialization:
    def test_roundtrip(self):
        smd = SmdResults(g_pcm_kcal_mol=-5.0, g_cds_kcal_mol=-0.5, g_enp_au=-75.5, g_tot_au=-75.6)
        reconstructed = SmdResults.from_dict(smd.to_dict())
        assert reconstructed == smd


@pytest.mark.unit
class TestTimingResultsSerialization:
    def test_roundtrip(self):
        timing = TimingResults(
            total_cpu_time_seconds=100.0,
            total_wall_time_seconds=110.0,
            module_times={"SCF": 50.0, "TDDFT": 60.0},
        )
        reconstructed = TimingResults.from_dict(timing.to_dict())
        assert reconstructed == timing
        assert reconstructed.module_times["SCF"] == 50.0


@pytest.mark.unit
class TestExcitedStateSerialization:
    def test_roundtrip_minimal(self):
        state = ExcitedState(
            state_number=1,
            multiplicity="Singlet",
            excitation_energy_ev=3.5,
            total_energy_au=-72.0,
        )
        reconstructed = ExcitedState.from_dict(state.to_dict())
        assert reconstructed == state

    def test_roundtrip_with_transitions(self):
        state = ExcitedState(
            state_number=1,
            multiplicity="Singlet",
            excitation_energy_ev=3.5,
            total_energy_au=-72.0,
            oscillator_strength=0.123,
            transitions=[
                OrbitalTransition(from_idx=4, to_idx=5, amplitude=0.9, is_alpha_spin=True),
                OrbitalTransition(from_idx=3, to_idx=5, amplitude=0.1, is_alpha_spin=True),
            ],
        )
        reconstructed = ExcitedState.from_dict(state.to_dict())
        assert reconstructed == state
        assert len(reconstructed.transitions) == 2


@pytest.mark.unit
class TestNTOStateAnalysisSerialization:
    def test_roundtrip(self):
        nto = NTOStateAnalysis(
            state_number=1,
            contributions=[
                NTOContribution(hole_offset=-1, electron_offset=1, weight_percent=95.0, is_alpha_spin=True),
                NTOContribution(hole_offset=-2, electron_offset=1, weight_percent=5.0, is_alpha_spin=True),
            ],
            omega_percent=98.5,
        )
        reconstructed = NTOStateAnalysis.from_dict(nto.to_dict())
        assert reconstructed == nto
        assert len(reconstructed.contributions) == 2


@pytest.mark.unit
class TestTddftResultsSerialization:
    def test_roundtrip_with_states(self):
        tddft = TddftResults(
            tda_states=[
                ExcitedState(state_number=1, multiplicity="Singlet", excitation_energy_ev=3.5, total_energy_au=-72.0),
                ExcitedState(state_number=2, multiplicity="Singlet", excitation_energy_ev=4.5, total_energy_au=-71.5),
            ]
        )
        reconstructed = TddftResults.from_dict(tddft.to_dict())
        assert reconstructed == tddft
        assert len(reconstructed.tda_states) == 2


@pytest.mark.unit
class TestCalculationMetadataSerialization:
    def test_roundtrip(self):
        meta = CalculationMetadata(software_name="ORCA", software_version="5.0.3")
        reconstructed = CalculationMetadata.from_dict(meta.to_dict())
        assert reconstructed == meta


# --- contract tests: CalculationResult serialization ---


@pytest.mark.contract
class TestCalculationResultSerialization:
    def test_to_dict_excludes_raw_output(self):
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="ORCA", software_version="5.0.3"),
            raw_output="This is a very long output that we don't want in JSON",
            final_energy=-75.313506,
        )
        data = result.to_dict()
        assert "raw_output" not in data
        assert data["final_energy"] == -75.313506

    def test_from_dict_adds_empty_raw_output(self):
        data = {
            "termination_status": "NORMAL",
            "metadata": {"software_name": "ORCA", "software_version": "5.0.3"},
            "final_energy": -75.313506,
            "nuclear_repulsion_energy": None,
            "input_geometry": None,
            "final_geometry": None,
            "scf": None,
            "orbitals": None,
            "multipole": None,
            "smd": None,
            "tddft": None,
            "dispersion": None,
            "timing": None,
            "atomic_charges": [],
            "program_specific": {},
        }
        result = CalculationResult.from_dict(data)
        assert result.raw_output == ""
        assert result.final_energy == -75.313506

    def test_json_roundtrip_minimal(self):
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="Q-Chem", software_version="6.2"),
            raw_output="Long output here...",
            final_energy=-75.5,
        )
        json_str = result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        # raw_output won't match, but everything else should
        assert reconstructed.termination_status == result.termination_status
        assert reconstructed.metadata == result.metadata
        assert reconstructed.final_energy == result.final_energy
        assert reconstructed.raw_output == ""  # excluded from serialization

    def test_json_roundtrip_with_geometry(self):
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="ORCA"),
            raw_output="output",
            input_geometry=[
                Atom(symbol="O", x=0.0, y=0.0, z=0.0),
                Atom(symbol="H", x=0.0, y=0.0, z=0.96),
                Atom(symbol="H", x=0.93, y=0.0, z=-0.24),
            ],
            final_energy=-75.313506,
        )
        json_str = result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        assert len(reconstructed.input_geometry) == 3
        assert reconstructed.input_geometry[0].symbol == "O"
        assert reconstructed.final_energy == result.final_energy

    def test_json_roundtrip_with_scf(self):
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="Q-Chem"),
            raw_output="output",
            final_energy=-75.5,
            scf=ScfResults(
                converged=True,
                energy=-75.5,
                n_iterations=10,
                iterations=[ScfIteration(iteration=1, energy=-75.0, diis_error=0.01)],
            ),
        )
        json_str = result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        assert reconstructed.scf.converged is True
        assert reconstructed.scf.n_iterations == 10
        assert len(reconstructed.scf.iterations) == 1

    def test_json_roundtrip_complex(self):
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="Q-Chem", software_version="6.2"),
            raw_output="very long output...",
            input_geometry=[Atom(symbol="O", x=0.0, y=0.0, z=0.0)],
            final_energy=-75.5,
            nuclear_repulsion_energy=8.0,
            scf=ScfResults(converged=True, energy=-75.5, n_iterations=10, iterations=[]),
            orbitals=OrbitalsSet(
                alpha_orbitals=[Orbital(index=0, energy=-1.0)], alpha_homo_index=0, alpha_lumo_index=1
            ),
            multipole=MultipoleResults(charge=0.0, dipole=DipoleMoment(x=0.1, y=0.2, z=0.3, magnitude=0.374)),
            atomic_charges=[AtomicCharges(method="Mulliken", charges={0: -0.5})],
            smd=SmdResults(g_pcm_kcal_mol=-5.0),
            timing=TimingResults(total_wall_time_seconds=100.0),
            program_specific={"some_key": "some_value"},
        )

        json_str = result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        assert reconstructed.final_energy == result.final_energy
        assert reconstructed.scf.converged is True
        assert reconstructed.orbitals.alpha_homo_index == 0
        assert reconstructed.multipole.dipole.magnitude == 0.374
        assert len(reconstructed.atomic_charges) == 1
        assert reconstructed.smd.g_pcm_kcal_mol == -5.0
        assert reconstructed.timing.total_wall_time_seconds == 100.0
        assert reconstructed.program_specific["some_key"] == "some_value"


# --- integration tests: real parsed outputs ---


@pytest.mark.integration
class TestRealParsedOutputSerialization:
    @pytest.fixture
    def test_data_dir(self):
        return Path(__file__).resolve().parents[1] / "testing_data"

    def test_orca_sp_roundtrip(self, test_data_dir):
        """test serialization of real ORCA single-point output."""
        orca_sp_path = test_data_dir / "orca" / "h2o" / "sp.out"
        if not orca_sp_path.exists():
            pytest.skip("ORCA test file not found")

        # parse original
        original_result = parse_orca_output(orca_sp_path.read_text())

        # serialize and deserialize
        json_str = original_result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        # verify key fields
        assert reconstructed.termination_status == original_result.termination_status
        assert reconstructed.final_energy == original_result.final_energy
        assert reconstructed.metadata.software_name == original_result.metadata.software_name

        # verify raw_output was excluded
        assert "raw_output" not in json.loads(json_str)
        assert reconstructed.raw_output == ""

    def test_qchem_sp_roundtrip(self, test_data_dir):
        """test serialization of real Q-Chem single-point output."""
        qchem_sp_path = test_data_dir / "qchem" / "h2o" / "6.2-sp-smd.out"
        if not qchem_sp_path.exists():
            pytest.skip("Q-Chem test file not found")

        # parse original
        original_result = parse_qchem_output(qchem_sp_path.read_text())

        # serialize and deserialize
        json_str = original_result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        # verify key fields
        assert reconstructed.termination_status == original_result.termination_status
        assert reconstructed.final_energy == original_result.final_energy
        assert reconstructed.metadata.software_name == original_result.metadata.software_name

        # verify SCF data preserved
        if original_result.scf:
            assert reconstructed.scf.converged == original_result.scf.converged
            assert reconstructed.scf.energy == original_result.scf.energy

        # verify raw_output was excluded
        assert "raw_output" not in json.loads(json_str)
        assert reconstructed.raw_output == ""

    def test_qchem_tddft_roundtrip(self, test_data_dir):
        """test serialization of Q-Chem TDDFT output with excited states."""
        qchem_tddft_path = test_data_dir / "qchem" / "h2o" / "6.2-rks-tddft.out"
        if not qchem_tddft_path.exists():
            pytest.skip("Q-Chem TDDFT test file not found")

        # parse original
        original_result = parse_qchem_output(qchem_tddft_path.read_text())

        # serialize and deserialize
        json_str = original_result.to_json()
        reconstructed = CalculationResult.from_json(json_str)

        # verify TDDFT data preserved
        if original_result.tddft and original_result.tddft.tddft_states:
            assert reconstructed.tddft is not None
            assert len(reconstructed.tddft.tddft_states) == len(original_result.tddft.tddft_states)

            # check first excited state
            orig_state = original_result.tddft.tddft_states[0]
            recon_state = reconstructed.tddft.tddft_states[0]
            assert recon_state.state_number == orig_state.state_number
            assert recon_state.excitation_energy_ev == orig_state.excitation_energy_ev
            assert recon_state.multiplicity == orig_state.multiplicity

    @pytest.mark.parametrize(
        "output_file,min_reduction_percent",
        [
            ("qchem/h2o/6.2-sp-smd.out", 50),  # simple SP calculation
            ("qchem/h2o/6.2-uks-tddft.out", 40),  # TDDFT has more parsed data
            ("qchem/h2o/6.2-mom-xas-smd.out", 40),  # multi-job with XAS has lots of parsed data
        ],
    )
    def test_json_file_size_reduction(self, test_data_dir, output_file, min_reduction_percent):
        """verify that excluding raw_output significantly reduces file size."""
        file_path = test_data_dir / output_file
        if not file_path.exists():
            pytest.skip(f"Test file {output_file} not found")

        result = parse_qchem_output(file_path.read_text())

        # size with raw_output excluded (normal case)
        json_without_raw = result.to_json()

        # manually include raw_output to compare
        data_with_raw = result.to_dict()
        data_with_raw["raw_output"] = result.raw_output
        json_with_raw = json.dumps(data_with_raw, indent=2)

        # should be smaller (raw_output adds significant size)
        assert len(json_without_raw) < len(json_with_raw)
        # verify raw_output takes at least min_reduction_percent of the total size when included
        size_reduction = len(json_with_raw) - len(json_without_raw)
        reduction_percent = (size_reduction / len(json_with_raw)) * 100
        assert reduction_percent >= min_reduction_percent, (
            f"Expected at least {min_reduction_percent}% reduction, got {reduction_percent:.1f}%"
        )
