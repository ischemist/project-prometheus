"""
Smoke tests that validate the examples in README.md work end-to-end.

These tests exercise the public API to ensure:
1. Re-exports from calcflow work correctly
2. Input generation produces valid output
3. Parsers can process test data
4. JSON serialization round-trips work
"""

from pathlib import Path

import pytest

from calcflow import (
    CalculationInput,
    Geometry,
    parse_orca_output,
    parse_qchem_multi_job_output,
    parse_qchem_output,
)
from calcflow.common.results import Atom, CalculationResult

# Test data paths
BASE_DIR = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = BASE_DIR / "tests" / "testing_data"
GEOM_DIR = TEST_DATA_DIR / "geometries"
ORCA_DIR = TEST_DATA_DIR / "orca" / "h2o"
QCHEM_DIR = TEST_DATA_DIR / "qchem" / "h2o"


@pytest.mark.unit
class TestSmokeReexports:
    """Test that public API re-exports work correctly."""

    def test_can_import_from_calcflow(self):
        """Verify all main classes can be imported from calcflow package."""
        # This test passes if imports at module level succeed
        assert CalculationInput is not None
        assert Geometry is not None
        assert parse_orca_output is not None
        assert parse_qchem_output is not None
        assert parse_qchem_multi_job_output is not None


@pytest.mark.integration
class TestSmokeOrcaInput:
    """Test ORCA input generation matching README examples."""

    def test_orca_geometry_optimization_example(self):
        """Test the ORCA geometry optimization example from README."""
        # 1. Load molecular geometry
        water = Geometry.from_xyz_file(GEOM_DIR / "1h2o.xyz")
        assert water.num_atoms == 3

        # 2. Configure ORCA calculation
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="geometry",
                level_of_theory="wb97x-d3",
                basis_set="def2-svp",
                n_cores=16,
            )
            .enable_ri_for_orca("RIJCOSX", "def2/j")
            .run_frequency_after_opt()
        )

        # 3. Export ORCA input file
        input_content = calc.export("orca", water)

        # Verify input contains expected sections
        assert "wb97x-d3" in input_content.lower()
        assert "def2-svp" in input_content.lower()
        assert "rijcosx" in input_content.lower()
        assert "freq" in input_content.lower()

        # 4. Test JSON serialization round-trip
        calc_json = calc.to_json()
        loaded_calc = CalculationInput.from_json(calc_json)
        assert loaded_calc == calc


@pytest.mark.integration
class TestSmokeQchemInput:
    """Test Q-Chem input generation matching README examples."""

    def test_qchem_tddft_basic_example(self):
        """Test the basic Q-Chem TDDFT example from README."""
        # 1. Define molecular geometry
        atoms = (
            Atom("O", 0.000000, 0.000000, 0.117300),
            Atom("H", 0.000000, 0.757200, -0.469200),
            Atom("H", 0.000000, -0.757200, -0.469200),
        )
        water = Geometry(atoms=atoms, comment="Water molecule")
        assert water.num_atoms == 3

        # 2. Configure Q-Chem calculation
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wB97X-D3",
                basis_set="def2-tzvp",
                n_cores=16,
            )
            .set_tddft(nroots=10, singlets=True, triplets=False)
            .set_solvation(model="smd", solvent="water")
        )

        # 3. Export Q-Chem input file
        input_content = calc.export("qchem", water)

        # Verify input contains expected sections
        assert "wb97x-d3" in input_content.lower()
        assert "def2-tzvp" in input_content.lower()
        assert "cis_n_roots" in input_content.lower() or "10" in input_content
        assert "solvent" in input_content.lower()

    def test_qchem_tddft_with_triplets(self):
        """Test Q-Chem TDDFT with triplets modification."""
        water = Geometry.from_xyz_file(GEOM_DIR / "1h2o.xyz")

        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wB97X-D3",
                basis_set="def2-tzvp",
                n_cores=16,
            )
            .set_tddft(nroots=10, singlets=True, triplets=False)
            .set_solvation(model="smd", solvent="water")
        )

        # Modify to include triplets and state analysis
        calc2 = calc.set_tddft(nroots=10, singlets=True, triplets=True, use_tda=True)

        input_content = calc2.export("qchem", water)
        assert "cis_triplets" in input_content.lower() or "triplets" in input_content.lower()

    def test_qchem_xas_element_specific_basis(self):
        """Test Q-Chem XAS with element-specific basis."""
        water = Geometry.from_xyz_file(GEOM_DIR / "1h2o.xyz")

        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wB97X-D3",
                basis_set="def2-tzvp",
                n_cores=16,
            )
            .set_tddft(nroots=10, singlets=True, triplets=False)
            .set_basis({"H": "pc-2", "O": "pcX-2"})
            .set_reduced_excitation_space(initial_orbitals=[1])
        )

        input_content = calc.export("qchem", water)

        # Verify reduced excitation space
        assert "trnss" in input_content.lower() or "n_sol" in input_content.lower()


@pytest.mark.integration
class TestSmokeOrcaParser:
    """Test ORCA parsing matching README examples."""

    def test_parse_orca_sp_output(self):
        """Test parsing ORCA single-point output."""
        output_text = (ORCA_DIR / "sp.out").read_text()
        result = parse_orca_output(output_text)

        # Verify basic structure
        assert result.final_energy is not None
        assert result.scf is not None
        assert result.scf.energy is not None

        # Test JSON serialization round-trip
        result_json = result.to_json()
        assert "final_energy" in result_json
        assert "raw_output" not in result_json  # excluded automatically

        # Load from JSON
        loaded = CalculationResult.from_json(result_json)
        assert loaded.final_energy == result.final_energy


@pytest.mark.integration
class TestSmokeQchemParser:
    """Test Q-Chem parsing matching README examples."""

    def test_parse_qchem_single_job(self):
        """Test parsing Q-Chem single-job output."""
        output_text = (QCHEM_DIR / "6.2-sp-smd.out").read_text()
        result = parse_qchem_output(output_text)

        # Verify basic structure
        assert result.scf is not None
        assert result.scf.energy is not None
        assert result.termination_status == "NORMAL"

    def test_parse_qchem_multi_job_mom(self):
        """Test parsing Q-Chem multi-job MOM output."""
        output_text = (QCHEM_DIR / "6.2-mom-sp-smd.out").read_text()
        jobs = parse_qchem_multi_job_output(output_text)

        # Should have 2 jobs (initial SCF + MOM)
        assert len(jobs) >= 2
        job1, job2 = jobs[0], jobs[1]

        # Verify both jobs parsed successfully
        assert job1.scf is not None
        assert job2.scf is not None
        assert job2.scf.energy is not None

    def test_parse_qchem_tddft_results(self):
        """Test accessing TDDFT results from parsed output."""
        output_text = (QCHEM_DIR / "6.2-rks-tddft.out").read_text()
        result = parse_qchem_output(output_text)

        # Verify TDDFT results exist
        assert result.tddft is not None
        assert result.tddft.tda_states is not None
        assert len(result.tddft.tda_states) > 0

        # Test extracting excitation energies and oscillator strengths (README example)
        energies = [s.excitation_energy_ev for s in result.tddft.tda_states]
        intensities = [s.oscillator_strength for s in result.tddft.tda_states]

        assert len(energies) == len(result.tddft.tda_states)
        assert len(intensities) == len(result.tddft.tda_states)
        assert all(e > 0 for e in energies)  # All excitation energies should be positive


@pytest.mark.integration
class TestSmokeEndToEnd:
    """End-to-end smoke tests combining input generation and parsing."""

    def test_orca_roundtrip_concept(self):
        """Test that we can generate input and parse corresponding output."""
        # Generate input
        water = Geometry.from_xyz_file(GEOM_DIR / "1h2o.xyz")
        calc = CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="wb97x-d3",
            basis_set="def2-svp",
            n_cores=16,
        )
        input_content = calc.export("orca", water)
        assert len(input_content) > 0

        # Parse existing output (conceptual roundtrip - input was generated differently)
        output_text = (ORCA_DIR / "sp.out").read_text()
        result = parse_orca_output(output_text)
        assert result.final_energy is not None

    def test_qchem_roundtrip_concept(self):
        """Test that we can generate input and parse corresponding output."""
        # Generate input
        water = Geometry.from_xyz_file(GEOM_DIR / "1h2o.xyz")
        calc = (
            CalculationInput(
                charge=0,
                spin_multiplicity=1,
                task="energy",
                level_of_theory="wB97X-D3",
                basis_set="def2-tzvp",
                n_cores=16,
            )
            .set_tddft(nroots=10, singlets=True, triplets=False)
            .set_solvation(model="smd", solvent="water")
        )
        input_content = calc.export("qchem", water)
        assert len(input_content) > 0

        # Parse existing output
        output_text = (QCHEM_DIR / "6.2-rks-tddft.out").read_text()
        result = parse_qchem_output(output_text)
        assert result.tddft is not None
