"""
Tests for QChem TDDFT output parsing - verifying basic blocks parse correctly.

These tests verify that when the QChem parser runs on TDDFT outputs (which contain
geometry, SCF, orbital, and multipole blocks), those blocks are parsed correctly
regardless of TDDFT-specific sections that we may not yet fully support.

Tests use three TDDFT fixtures (parametrized across all contract and integration tests):
- parsed_qchem_54_h2o_uks_tddft_data: UKS TDDFT calculation (QChem 5.4 version)
- parsed_qchem_62_h2o_uks_tddft_data: UKS TDDFT calculation (QChem 6.2 version)
- parsed_qchem_62_h2o_rks_tddft_data: RKS TDDFT calculation (QChem 6.2 version)

Expected values extracted from output files. Values that differ by fixture version
are tested in regression tests. Shared values (like geometry and SCF energy) are
tested via parametrization across all fixtures for broad coverage.

Test organization by blocks:
- Geometry block tests (parametrized across fixtures)
- SCF block tests (parametrized across fixtures)
- Orbitals block tests (parametrized across fixtures)
- Multipole moments tests (parametrized across fixtures)
- Cross-block integration tests (parametrized across fixtures)
"""

import pytest

from calcflow.common.results import CalculationResult, OrbitalsSet, ScfResults
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA from TDDFT extracts
# =============================================================================

# GEOMETRY DATA (identical for both UKS and RKS TDDFT)
EXPECTED_GEOMETRY_ATOMS = 3
EXPECTED_GEOMETRY_SYMBOLS = ["H", "O", "H"]
EXPECTED_GEOMETRY_H1 = (1.36499, 1.69385, -0.19748)
EXPECTED_GEOMETRY_O = (2.32877, 1.56294, -0.04168)
EXPECTED_GEOMETRY_H2 = (2.70244, 1.31157, -0.91665)

# SCF DATA (identical for both UKS and RKS TDDFT)
EXPECTED_SCF_ENERGY = -76.44125314
EXPECTED_CONVERGED = True
EXPECTED_N_ITERATIONS = 10

# ORBITALS DATA (UKS TDDFT from ex-uks-tddft.md)
EXPECTED_UKS_ALPHA_OCCUPIED = [-19.2346, -1.1182, -0.6261, -0.4888, -0.4147]
EXPECTED_UKS_ALPHA_VIRTUAL_FIRST_5 = [0.0878, 0.1389, 0.3614, 0.3715, 0.4265]
EXPECTED_UKS_ALPHA_VIRTUAL_LAST = 14.8185
EXPECTED_UKS_TOTAL_ORBITALS = 58
EXPECTED_UKS_HOMO_INDEX = 4
EXPECTED_UKS_LUMO_INDEX = 5

# MULTIPOLE DATA (identical for both UKS and RKS TDDFT)
# Dipole moment
EXPECTED_DIPOLE_X = -0.9958
EXPECTED_DIPOLE_Y = -0.2035
EXPECTED_DIPOLE_Z = -1.7403
EXPECTED_DIPOLE_TOT = 2.0154

# Quadrupole moments
EXPECTED_QUAD_XX = -9.3797
EXPECTED_QUAD_XY = -2.6489
EXPECTED_QUAD_YY = -8.0621
EXPECTED_QUAD_XZ = -4.5863
EXPECTED_QUAD_YZ = -2.1771
EXPECTED_QUAD_ZZ = -5.4667

# Octopole moments (sample)
EXPECTED_OCT_XXX = -50.1854
EXPECTED_OCT_YYY = -36.1622
EXPECTED_OCT_ZZZ = 0.8499

# Hexadecapole moments (sample)
EXPECTED_HEX_XXXX = -218.5681
EXPECTED_HEX_YYYY = -117.9763
EXPECTED_HEX_ZZZZ = -6.7076

# Tolerances
COORD_TOL = 1e-5
ENERGY_TOL = 1e-8
ORBITAL_ENERGY_TOL = 1e-4
MULTIPOLE_TOL = 1e-4


# =============================================================================
# GEOMETRY BLOCK TESTS
# =============================================================================


class TestTDDFTGeometryBlock:
    """Tests for geometry block parsing in TDDFT outputs."""

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_geometry_parsed_in_tddft(self, fixture_name: str, request):
        """Contract test: verify geometry is parsed from TDDFT output (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.input_geometry is not None
        assert isinstance(data.input_geometry, tuple)
        assert len(data.input_geometry) > 0

    @pytest.mark.regression
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_geometry_has_three_atoms(self, fixture_name: str, request):
        """Regression test: verify exactly 3 atoms in H2O geometry (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert len(data.input_geometry) == EXPECTED_GEOMETRY_ATOMS

    @pytest.mark.regression
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_geometry_atom_symbols(self, fixture_name: str, request):
        """Regression test: verify atom symbols are H, O, H (all variants)."""
        data = request.getfixturevalue(fixture_name)
        symbols = [atom.symbol for atom in data.input_geometry]
        assert symbols == EXPECTED_GEOMETRY_SYMBOLS

    @pytest.mark.regression
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    @pytest.mark.parametrize(
        "atom_idx,expected_x,expected_y,expected_z",
        [
            (0, EXPECTED_GEOMETRY_H1[0], EXPECTED_GEOMETRY_H1[1], EXPECTED_GEOMETRY_H1[2]),
            (1, EXPECTED_GEOMETRY_O[0], EXPECTED_GEOMETRY_O[1], EXPECTED_GEOMETRY_O[2]),
            (2, EXPECTED_GEOMETRY_H2[0], EXPECTED_GEOMETRY_H2[1], EXPECTED_GEOMETRY_H2[2]),
        ],
        ids=["H1", "O", "H2"],
    )
    def test_geometry_coordinates_exact(
        self,
        fixture_name: str,
        request,
        atom_idx: int,
        expected_x: float,
        expected_y: float,
        expected_z: float,
    ):
        """Regression test: verify exact geometry coordinates (all variants)."""
        data = request.getfixturevalue(fixture_name)
        atom = data.input_geometry[atom_idx]
        assert atom.x == pytest.approx(expected_x, abs=COORD_TOL)
        assert atom.y == pytest.approx(expected_y, abs=COORD_TOL)
        assert atom.z == pytest.approx(expected_z, abs=COORD_TOL)

    @pytest.mark.integration
    def test_geometry_present_in_both_tddft_variants(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        parsed_qchem_62_h2o_rks_tddft_data: CalculationResult,
    ):
        """Integration test: verify geometry parsed in both UKS and RKS TDDFT."""
        # Both should have identical geometry
        assert len(parsed_qchem_62_h2o_uks_tddft_data.input_geometry) == len(
            parsed_qchem_62_h2o_rks_tddft_data.input_geometry
        )

        # Check coordinates match
        for uks_atom, rks_atom in zip(
            parsed_qchem_62_h2o_uks_tddft_data.input_geometry,
            parsed_qchem_62_h2o_rks_tddft_data.input_geometry,
            strict=True,
        ):
            assert uks_atom.x == pytest.approx(rks_atom.x, abs=COORD_TOL)
            assert uks_atom.y == pytest.approx(rks_atom.y, abs=COORD_TOL)
            assert uks_atom.z == pytest.approx(rks_atom.z, abs=COORD_TOL)


# =============================================================================
# SCF BLOCK TESTS
# =============================================================================


class TestTDDFTSCFBlock:
    """Tests for SCF block parsing in TDDFT outputs."""

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_scf_parsed_in_tddft(self, fixture_name: str, request):
        """Contract test: verify SCF block is parsed from TDDFT output (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.scf is not None
        assert isinstance(data.scf, ScfResults)

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_scf_has_iterations(self, fixture_name: str, request):
        """Contract test: verify SCF has iterations data (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.scf is not None
        assert isinstance(data.scf.iterations, tuple)
        assert len(data.scf.iterations) > 0

    @pytest.mark.regression
    def test_scf_converged(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify SCF converged."""
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.scf.converged is EXPECTED_CONVERGED

    @pytest.mark.regression
    def test_scf_number_of_iterations(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact number of SCF iterations."""
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.scf.n_iterations == EXPECTED_N_ITERATIONS

    @pytest.mark.regression
    def test_scf_final_energy(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact SCF final energy."""
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.scf.energy == pytest.approx(EXPECTED_SCF_ENERGY, abs=ENERGY_TOL)

    @pytest.mark.regression
    def test_scf_energy_in_both_tddft_variants(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        parsed_qchem_62_h2o_rks_tddft_data: CalculationResult,
    ):
        """Regression test: verify SCF energy identical in UKS and RKS TDDFT."""
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        assert parsed_qchem_62_h2o_rks_tddft_data.scf is not None
        # Both should have the same SCF energy (same calculation, just different spin setup)
        assert parsed_qchem_62_h2o_uks_tddft_data.scf.energy == pytest.approx(
            parsed_qchem_62_h2o_rks_tddft_data.scf.energy, abs=ENERGY_TOL
        )

    @pytest.mark.regression
    def test_final_energy_set_from_scf(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify final_energy is populated from SCF."""
        assert parsed_qchem_62_h2o_uks_tddft_data.final_energy is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.final_energy == pytest.approx(
            parsed_qchem_62_h2o_uks_tddft_data.scf.energy, abs=ENERGY_TOL
        )

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "iter_idx,expected_energy",
        [
            (0, -76.3054390399),
            (4, -76.4412423142),
            (9, -76.4412531352),
        ],
        ids=["first-iteration", "middle-iteration", "last-iteration"],
    )
    def test_scf_iteration_energies_sample(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        iter_idx: int,
        expected_energy: float,
    ):
        """Regression test: verify sample SCF iteration energy values."""
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        assert len(parsed_qchem_62_h2o_uks_tddft_data.scf.iterations) > iter_idx

        iteration = parsed_qchem_62_h2o_uks_tddft_data.scf.iterations[iter_idx]
        assert iteration.energy == pytest.approx(expected_energy, abs=ENERGY_TOL)


# =============================================================================
# ORBITALS BLOCK TESTS
# =============================================================================


class TestTDDFTOrbitalsBlock:
    """Tests for orbitals block parsing in TDDFT outputs."""

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_orbitals_parsed_in_tddft(self, fixture_name: str, request):
        """Contract test: verify orbitals block is parsed from TDDFT output (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.orbitals is not None
        assert isinstance(data.orbitals, OrbitalsSet)

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_alpha_orbitals_present_in_tddft(self, fixture_name: str, request):
        """Contract test: verify alpha orbitals are parsed (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.orbitals is not None
        assert isinstance(data.orbitals.alpha_orbitals, (list, tuple))
        assert len(data.orbitals.alpha_orbitals) > 0

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["beta_orbitals"])
    def test_uks_has_beta_orbitals(self, fixture_name: str, request):
        """Contract test: verify UKS TDDFT has beta orbitals."""
        data = request.getfixturevalue(fixture_name)
        assert data.orbitals is not None
        assert data.orbitals.beta_orbitals is not None
        assert isinstance(data.orbitals.beta_orbitals, (list, tuple))
        assert len(data.orbitals.beta_orbitals) > 0

    @pytest.mark.regression
    def test_alpha_orbital_count(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact number of alpha orbitals."""
        assert parsed_qchem_62_h2o_uks_tddft_data.orbitals is not None
        assert len(parsed_qchem_62_h2o_uks_tddft_data.orbitals.alpha_orbitals) == EXPECTED_UKS_TOTAL_ORBITALS

    @pytest.mark.regression
    def test_beta_orbital_count(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact number of beta orbitals."""
        assert parsed_qchem_62_h2o_uks_tddft_data.orbitals is not None
        assert len(parsed_qchem_62_h2o_uks_tddft_data.orbitals.beta_orbitals) == EXPECTED_UKS_TOTAL_ORBITALS

    @pytest.mark.regression
    def test_homo_lumo_indices(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify HOMO and LUMO indices."""
        orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
        assert orbitals is not None
        assert orbitals.alpha_homo_index == EXPECTED_UKS_HOMO_INDEX
        assert orbitals.alpha_lumo_index == EXPECTED_UKS_LUMO_INDEX
        assert orbitals.beta_homo_index == EXPECTED_UKS_HOMO_INDEX
        assert orbitals.beta_lumo_index == EXPECTED_UKS_LUMO_INDEX

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "orbital_idx,expected_energy",
        [
            (0, -19.2346),
            (1, -1.1182),
            (2, -0.6261),
            (3, -0.4888),
            (4, -0.4147),  # HOMO
        ],
        ids=["alpha-occ-0", "alpha-occ-1", "alpha-occ-2", "alpha-occ-3", "alpha-occ-4(HOMO)"],
    )
    def test_alpha_occupied_orbital_energies(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        orbital_idx: int,
        expected_energy: float,
    ):
        """Regression test: verify exact alpha occupied orbital energies."""
        orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
        assert orbitals is not None
        assert len(orbitals.alpha_orbitals) > orbital_idx

        orbital = orbitals.alpha_orbitals[orbital_idx]
        assert orbital.energy == pytest.approx(expected_energy, abs=ORBITAL_ENERGY_TOL)

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "orbital_idx,expected_energy",
        [
            (5, 0.0878),  # LUMO
            (6, 0.1389),
            (7, 0.3614),
            (8, 0.3715),
            (9, 0.4265),
        ],
        ids=["alpha-virt-5(LUMO)", "alpha-virt-6", "alpha-virt-7", "alpha-virt-8", "alpha-virt-9"],
    )
    def test_alpha_virtual_orbital_energies(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        orbital_idx: int,
        expected_energy: float,
    ):
        """Regression test: verify exact alpha virtual orbital energies."""
        orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
        assert orbitals is not None
        assert len(orbitals.alpha_orbitals) > orbital_idx

        orbital = orbitals.alpha_orbitals[orbital_idx]
        assert orbital.energy == pytest.approx(expected_energy, abs=ORBITAL_ENERGY_TOL)

    @pytest.mark.regression
    def test_alpha_last_virtual_orbital(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact energy of last alpha virtual orbital."""
        orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
        assert orbitals is not None
        assert len(orbitals.alpha_orbitals) > 0

        last_orbital = orbitals.alpha_orbitals[-1]
        assert last_orbital.energy == pytest.approx(EXPECTED_UKS_ALPHA_VIRTUAL_LAST, abs=ORBITAL_ENERGY_TOL)

    @pytest.mark.regression
    def test_alpha_beta_orbitals_match_in_uks(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify alpha and beta orbitals match (closed-shell UKS)."""
        orbitals = parsed_qchem_62_h2o_uks_tddft_data.orbitals
        assert orbitals is not None
        assert len(orbitals.alpha_orbitals) == len(orbitals.beta_orbitals)

        # Check that energies match
        for alpha_orb, beta_orb in zip(orbitals.alpha_orbitals, orbitals.beta_orbitals, strict=True):
            assert alpha_orb.energy == pytest.approx(beta_orb.energy, abs=ORBITAL_ENERGY_TOL)

    @pytest.mark.integration
    def test_orbitals_present_in_both_tddft_variants(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        parsed_qchem_62_h2o_rks_tddft_data: CalculationResult,
    ):
        """Integration test: verify orbitals parsed in both UKS and RKS TDDFT."""
        assert parsed_qchem_62_h2o_uks_tddft_data.orbitals is not None
        assert parsed_qchem_62_h2o_rks_tddft_data.orbitals is not None

        # Both should have alpha orbitals
        assert len(parsed_qchem_62_h2o_uks_tddft_data.orbitals.alpha_orbitals) > 0
        assert len(parsed_qchem_62_h2o_rks_tddft_data.orbitals.alpha_orbitals) > 0


# =============================================================================
# MULTIPOLE MOMENTS BLOCK TESTS
# =============================================================================


class TestTDDFTMultipoleBlock:
    """Tests for multipole moments block parsing in TDDFT outputs."""

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_multipole_parsed_in_tddft(self, fixture_name: str, request):
        """Contract test: verify multipole block is parsed from TDDFT output (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.multipole is not None

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_multipole_has_dipole(self, fixture_name: str, request):
        """Contract test: verify multipole has dipole moment (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.multipole is not None
        assert data.multipole.dipole is not None

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_multipole_has_quadrupole(self, fixture_name: str, request):
        """Contract test: verify multipole has quadrupole moments (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.multipole is not None
        assert data.multipole.quadrupole is not None

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_multipole_has_octopole(self, fixture_name: str, request):
        """Contract test: verify multipole has octopole moments (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.multipole is not None
        assert data.multipole.octopole is not None

    @pytest.mark.contract
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_multipole_has_hexadecapole(self, fixture_name: str, request):
        """Contract test: verify multipole has hexadecapole moments (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.multipole is not None
        assert data.multipole.hexadecapole is not None

    @pytest.mark.regression
    def test_dipole_x_component(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact dipole X component."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole.x == pytest.approx(
            EXPECTED_DIPOLE_X, abs=MULTIPOLE_TOL
        )

    @pytest.mark.regression
    def test_dipole_y_component(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact dipole Y component."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole.y == pytest.approx(
            EXPECTED_DIPOLE_Y, abs=MULTIPOLE_TOL
        )

    @pytest.mark.regression
    def test_dipole_z_component(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact dipole Z component."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole.z == pytest.approx(
            EXPECTED_DIPOLE_Z, abs=MULTIPOLE_TOL
        )

    @pytest.mark.regression
    def test_dipole_magnitude(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Regression test: verify exact dipole magnitude."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.dipole.magnitude == pytest.approx(
            EXPECTED_DIPOLE_TOT, abs=MULTIPOLE_TOL
        )

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "component,expected_value",
        [
            ("xx", EXPECTED_QUAD_XX),
            ("xy", EXPECTED_QUAD_XY),
            ("yy", EXPECTED_QUAD_YY),
            ("xz", EXPECTED_QUAD_XZ),
            ("yz", EXPECTED_QUAD_YZ),
            ("zz", EXPECTED_QUAD_ZZ),
        ],
    )
    def test_quadrupole_moments(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        component: str,
        expected_value: float,
    ):
        """Regression test: verify exact quadrupole moment components."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.quadrupole is not None

        quad = parsed_qchem_62_h2o_uks_tddft_data.multipole.quadrupole
        actual_value = getattr(quad, component)
        assert actual_value == pytest.approx(expected_value, abs=MULTIPOLE_TOL)

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "component,expected_value",
        [
            ("xxx", EXPECTED_OCT_XXX),
            ("yyy", EXPECTED_OCT_YYY),
            ("zzz", EXPECTED_OCT_ZZZ),
        ],
        ids=["octopole-xxx", "octopole-yyy", "octopole-zzz"],
    )
    def test_octopole_moments_sample(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        component: str,
        expected_value: float,
    ):
        """Regression test: verify sample octopole moment components."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.octopole is not None

        oct_moments = parsed_qchem_62_h2o_uks_tddft_data.multipole.octopole
        actual_value = getattr(oct_moments, component)
        assert actual_value == pytest.approx(expected_value, abs=MULTIPOLE_TOL)

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "component,expected_value",
        [
            ("xxxx", EXPECTED_HEX_XXXX),
            ("yyyy", EXPECTED_HEX_YYYY),
            ("zzzz", EXPECTED_HEX_ZZZZ),
        ],
        ids=["hexadecapole-xxxx", "hexadecapole-yyyy", "hexadecapole-zzzz"],
    )
    def test_hexadecapole_moments_sample(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        component: str,
        expected_value: float,
    ):
        """Regression test: verify sample hexadecapole moment components."""
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.multipole.hexadecapole is not None

        hex_moments = parsed_qchem_62_h2o_uks_tddft_data.multipole.hexadecapole
        actual_value = getattr(hex_moments, component)
        assert actual_value == pytest.approx(expected_value, abs=MULTIPOLE_TOL)

    @pytest.mark.integration
    def test_multipole_identical_in_both_tddft_variants(
        self,
        parsed_qchem_62_h2o_uks_tddft_data: CalculationResult,
        parsed_qchem_62_h2o_rks_tddft_data: CalculationResult,
    ):
        """Integration test: verify multipole moments identical in UKS and RKS TDDFT."""
        uks_mult = parsed_qchem_62_h2o_uks_tddft_data.multipole
        rks_mult = parsed_qchem_62_h2o_rks_tddft_data.multipole

        assert uks_mult is not None
        assert rks_mult is not None

        # Check dipole matches
        assert uks_mult.dipole.x == pytest.approx(rks_mult.dipole.x, abs=MULTIPOLE_TOL)
        assert uks_mult.dipole.y == pytest.approx(rks_mult.dipole.y, abs=MULTIPOLE_TOL)
        assert uks_mult.dipole.z == pytest.approx(rks_mult.dipole.z, abs=MULTIPOLE_TOL)
        assert uks_mult.dipole.magnitude == pytest.approx(rks_mult.dipole.magnitude, abs=MULTIPOLE_TOL)


# =============================================================================
# CROSS-BLOCK INTEGRATION TESTS
# =============================================================================


class TestTDDFTBlockIntegration:
    """Integration tests verifying multiple blocks work together in TDDFT output."""

    @pytest.mark.integration
    @pytest.mark.parametrize("fixture_name", FIXTURE_SPECS["tddft_excitations"])
    def test_all_blocks_present_in_tddft(self, fixture_name: str, request):
        """Integration test: verify all major blocks are parsed in TDDFT (all variants)."""
        data = request.getfixturevalue(fixture_name)
        assert data.input_geometry is not None
        assert len(data.input_geometry) > 0

        assert data.scf is not None
        assert len(data.scf.iterations) > 0

        assert data.orbitals is not None
        assert len(data.orbitals.alpha_orbitals) > 0

        assert data.multipole is not None

    @pytest.mark.integration
    def test_final_energy_consistent_across_blocks(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Integration test: verify final_energy is consistent with SCF energy."""
        assert parsed_qchem_62_h2o_uks_tddft_data.final_energy is not None
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None

        assert parsed_qchem_62_h2o_uks_tddft_data.final_energy == pytest.approx(
            parsed_qchem_62_h2o_uks_tddft_data.scf.energy, abs=ENERGY_TOL
        )

    @pytest.mark.integration
    def test_geometry_atoms_count_consistent_with_scf(self, parsed_qchem_62_h2o_uks_tddft_data: CalculationResult):
        """Integration test: verify geometry atom count is consistent."""
        # H2O should have 3 atoms - verify this matches what we expect
        assert len(parsed_qchem_62_h2o_uks_tddft_data.input_geometry) == 3

        # SCF should also be for H2O system
        assert parsed_qchem_62_h2o_uks_tddft_data.scf is not None
        # Just verify SCF data exists; it should be for same molecule
        assert parsed_qchem_62_h2o_uks_tddft_data.scf.energy is not None
