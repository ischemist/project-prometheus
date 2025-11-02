"""
integration and regression tests for the top-level orca single point parser.

these tests operate on a full, real-world orca output file and verify that
the final `calculationresult` object is assembled correctly.
"""

import pytest

from calcflow.common.results import Atom, CalculationResult, ScfEnergyComponents, ScfResults

GEOM_TOL = 1e-6
ENERGY_TOL = 1e-8


@pytest.mark.integration
def test_orca_sp_parsing_completes(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that parsing a standard orca sp output file runs to completion and
    returns the correct top-level object type.
    """
    assert parsed_orca_h2o_sp_data is not None
    assert isinstance(parsed_orca_h2o_sp_data, CalculationResult)


@pytest.mark.integration
def test_orca_sp_termination_status(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that the parser correctly identifies a normally terminated run.
    """
    assert parsed_orca_h2o_sp_data.termination_status == "NORMAL"


@pytest.mark.integration
def test_orca_sp_input_geometry_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests the structural integrity of the parsed input geometry.
    it checks for the correct number of atoms and their types, but not their
    exact coordinates (which is a regression test).
    """
    geom = parsed_orca_h2o_sp_data.input_geometry
    assert geom is not None
    assert len(geom) == 3
    assert all(isinstance(atom, Atom) for atom in geom)

    symbols = [atom.symbol for atom in geom]
    assert symbols == ["H", "O", "H"]


@pytest.mark.regression
def test_orca_sp_input_geometry_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests for the exact numerical values of the input geometry. this is a
    regression test because it's sensitive to small formatting changes.
    """
    expected_coords = {
        "H": (1.364990, 1.693850, -0.197480),
        "O": (2.328770, 1.562940, -0.041680),
        "H_2": (2.702440, 1.311570, -0.916650),  # using a temp unique key
    }

    parsed_geom = parsed_orca_h2o_sp_data.input_geometry

    # this is a bit verbose but avoids ambiguity with the two H atoms
    h1 = parsed_geom[0]
    o = parsed_geom[1]
    h2 = parsed_geom[2]

    assert h1.symbol == "H"
    assert h1.x == pytest.approx(expected_coords["H"][0], abs=GEOM_TOL)
    assert h1.y == pytest.approx(expected_coords["H"][1], abs=GEOM_TOL)
    assert h1.z == pytest.approx(expected_coords["H"][2], abs=GEOM_TOL)

    assert o.symbol == "O"
    assert o.x == pytest.approx(expected_coords["O"][0], abs=GEOM_TOL)
    assert o.y == pytest.approx(expected_coords["O"][1], abs=GEOM_TOL)
    assert o.z == pytest.approx(expected_coords["O"][2], abs=GEOM_TOL)

    assert h2.symbol == "H"
    assert h2.x == pytest.approx(expected_coords["H_2"][0], abs=GEOM_TOL)
    assert h2.y == pytest.approx(expected_coords["H_2"][1], abs=GEOM_TOL)
    assert h2.z == pytest.approx(expected_coords["H_2"][2], abs=GEOM_TOL)


@pytest.mark.integration
def test_scf_results_structure(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that the scf block was parsed into the correct pydantic models and
    that the convergence status is correctly identified.
    """
    scf = parsed_orca_h2o_sp_data.scf
    assert scf is not None
    assert isinstance(scf, ScfResults)

    # from the "SCF CONVERGENCE" block, it's clear the calculation converged.
    assert scf.converged is True

    assert scf.components is not None
    assert isinstance(scf.components, ScfEnergyComponents)


@pytest.mark.regression
def test_final_energy_value(parsed_orca_h2o_sp_data: CalculationResult):
    """
    regression test for the main 'Total Energy' value. this is the most
    critical numerical result of the calculation.
    """
    expected_scf_energy = -75.31350442244285
    expected_final_energy = -75.313506060725

    assert parsed_orca_h2o_sp_data.scf.energy == pytest.approx(expected_scf_energy, abs=ENERGY_TOL)

    assert parsed_orca_h2o_sp_data.final_energy == pytest.approx(expected_final_energy, abs=ENERGY_TOL)


@pytest.mark.regression
def test_scf_energy_components_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    regression test for the detailed components of the scf energy.
    """
    components = parsed_orca_h2o_sp_data.scf.components

    assert components.nuclear_repulsion == pytest.approx(8.93764967318280, abs=ENERGY_TOL)
    assert components.electronic_eh == pytest.approx(-84.25117753824344, abs=ENERGY_TOL)
    assert components.one_electron_eh == pytest.approx(-121.92433818585613, abs=ENERGY_TOL)
    assert components.two_electron_eh == pytest.approx(37.67316064761268, abs=ENERGY_TOL)
    assert components.xc_eh == pytest.approx(-6.561524953854, abs=ENERGY_TOL)


@pytest.mark.integration
def test_scf_all_iterations_captured(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that all 8 iterations (5 DIIS + 3 SOSCF) are captured from the
    SCF convergence procedure, including the transition from DIIS to SOSCF.
    """
    scf = parsed_orca_h2o_sp_data.scf
    assert scf is not None

    # Should have parsed all 8 iterations
    assert len(scf.iterations) == 8
    assert scf.n_iterations == 8

    # First 5 iterations are DIIS (have diis_error field)
    for i in range(5):
        assert scf.iterations[i].iteration == i + 1
        assert scf.iterations[i].diis_error is not None

    # Last 3 iterations are SOSCF (also have numeric field in diis_error position)
    for i in range(5, 8):
        assert scf.iterations[i].iteration == i + 1
        assert scf.iterations[i].diis_error is not None

    # Verify energy progression (should be monotonically decreasing)
    energies = [it.energy for it in scf.iterations]
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i], "SCF energy should decrease or stay the same"


@pytest.mark.regression
def test_scf_first_iteration_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    regression test for the first SCF iteration's values.
    """
    first_iter = parsed_orca_h2o_sp_data.scf.iterations[0]

    assert first_iter.iteration == 1
    assert first_iter.energy == pytest.approx(-75.2570665045657989, abs=ENERGY_TOL)
    assert first_iter.delta_e_eh == pytest.approx(0.0, abs=ENERGY_TOL)
    assert first_iter.rmsdp == pytest.approx(3.14e-02, abs=1e-4)
    assert first_iter.maxdp == pytest.approx(8.08e-02, abs=1e-4)
    assert first_iter.diis_error == pytest.approx(2.12e-01, abs=1e-4)


@pytest.mark.regression
def test_scf_final_iteration_values(parsed_orca_h2o_sp_data: CalculationResult):
    """
    regression test for the final (8th) SCF iteration's values.
    """
    final_iter = parsed_orca_h2o_sp_data.scf.iterations[7]

    assert final_iter.iteration == 8
    assert final_iter.energy == pytest.approx(-75.3135278650606352, abs=ENERGY_TOL)
    assert final_iter.delta_e_eh == pytest.approx(-1.11e-07, abs=1e-9)
    assert final_iter.rmsdp == pytest.approx(8.42e-05, abs=1e-6)
    assert final_iter.maxdp == pytest.approx(2.23e-04, abs=1e-6)


@pytest.mark.regression
def test_scf_diis_to_soscf_transition(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests the transition from DIIS to SOSCF at iteration 6.
    """
    iter5 = parsed_orca_h2o_sp_data.scf.iterations[4]  # Last DIIS iteration
    iter6 = parsed_orca_h2o_sp_data.scf.iterations[5]  # First SOSCF iteration

    # Last DIIS iteration (5)
    assert iter5.iteration == 5
    assert iter5.energy == pytest.approx(-75.3135093441037071, abs=ENERGY_TOL)
    assert iter5.diis_error == pytest.approx(3.62e-03, abs=1e-5)

    # First SOSCF iteration (6) - should have lower energy
    assert iter6.iteration == 6
    assert iter6.energy == pytest.approx(-75.3135270816374742, abs=ENERGY_TOL)
    assert iter6.energy < iter5.energy  # Energy improves from DIIS to SOSCF


@pytest.mark.regression
def test_scf_energy_components_nuclear_repulsion(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that nuclear repulsion energy component is correctly parsed.
    """
    components = parsed_orca_h2o_sp_data.scf.components
    assert components.nuclear_repulsion == pytest.approx(8.93764967318280, abs=ENERGY_TOL)


@pytest.mark.regression
def test_scf_energy_components_electronic(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that electronic energy component is correctly parsed.
    """
    components = parsed_orca_h2o_sp_data.scf.components
    assert components.electronic_eh == pytest.approx(-84.25117753824344, abs=ENERGY_TOL)


@pytest.mark.regression
def test_scf_energy_components_one_electron(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that one-electron energy component is correctly parsed.
    """
    components = parsed_orca_h2o_sp_data.scf.components
    assert components.one_electron_eh == pytest.approx(-121.92433818585613, abs=ENERGY_TOL)


@pytest.mark.regression
def test_scf_energy_components_two_electron(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that two-electron energy component is correctly parsed.
    """
    components = parsed_orca_h2o_sp_data.scf.components
    assert components.two_electron_eh == pytest.approx(37.67316064761268, abs=ENERGY_TOL)


@pytest.mark.regression
def test_scf_xc_energy_component(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that exchange-correlation (XC) energy component is correctly parsed.
    this is critical for DFT calculations.
    """
    components = parsed_orca_h2o_sp_data.scf.components
    assert components.xc_eh is not None
    assert components.xc_eh == pytest.approx(-6.561524953854, abs=ENERGY_TOL)


@pytest.mark.regression
def test_scf_converged_status(parsed_orca_h2o_sp_data: CalculationResult):
    """
    tests that the SCF convergence status is correctly identified as converged.
    """
    scf = parsed_orca_h2o_sp_data.scf
    assert scf.converged is True
    assert scf.n_iterations == 8
