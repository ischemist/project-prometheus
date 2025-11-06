"""
Tests for the QChem Cartesian multipole moments parser.

These tests verify that the multipole parser correctly extracts atomic multipole
moments (dipole, quadrupole, octopole, hexadecapole) from Q-Chem output files
for single-point calculations.

Test hierarchy:
- unit: isolated `matches()` behavior
- contract: parser produces correct data structure (non-None values)
- integration: multiple components working together
- regression: exact numerical values match expected
"""

import pytest

from calcflow.common.results import CalculationResult, MultipoleResults
from calcflow.io.qchem.blocks.multipole import MultipoleParser
from calcflow.io.state import ParseState
from tests.io.qchem.qchem_parsers.conftest import FIXTURE_SPECS

# =============================================================================
# HARDCODED TEST DATA (Q-Chem 5.4 and 6.2 have identical values)
# =============================================================================

EXPECTED_CHARGE = -0.0
EXPECTED_DIPOLE_X = -0.8826
EXPECTED_DIPOLE_Y = -0.1808
EXPECTED_DIPOLE_Z = -1.5445
EXPECTED_DIPOLE_MAG = 1.788

EXPECTED_QUAD_XX = -8.5235
EXPECTED_QUAD_XY = -2.1415
EXPECTED_QUAD_YY = -6.5392
EXPECTED_QUAD_XZ = -3.8091
EXPECTED_QUAD_YZ = -2.0882
EXPECTED_QUAD_ZZ = -4.6864

EXPECTED_OCT_XXX = -45.1424
EXPECTED_OCT_XXY = -15.7146
EXPECTED_OCT_XYY = -18.0848
EXPECTED_OCT_YYY = -29.1344
EXPECTED_OCT_XXZ = -8.8897
EXPECTED_OCT_XYZ = -5.0592
EXPECTED_OCT_YYZ = -1.9524
EXPECTED_OCT_XZZ = -10.2351
EXPECTED_OCT_YZZ = -7.5112
EXPECTED_OCT_ZZZ = 1.6208

EXPECTED_HEX_XXXX = -193.4737
EXPECTED_HEX_XXXY = -76.9501
EXPECTED_HEX_XXYY = -60.7182
EXPECTED_HEX_XYYY = -71.6161
EXPECTED_HEX_YYYY = -92.5752
EXPECTED_HEX_XXXZ = -19.1566
EXPECTED_HEX_XXYZ = -11.6023
EXPECTED_HEX_XYYZ = -4.5683
EXPECTED_HEX_YYYZ = -0.0879
EXPECTED_HEX_XXZZ = -24.2567
EXPECTED_HEX_XYZZ = -16.4191
EXPECTED_HEX_YYZZ = -13.597
EXPECTED_HEX_XZZZ = 3.6825
EXPECTED_HEX_YZZZ = 2.453
EXPECTED_HEX_ZZZZ = -5.3042

# Numerical tolerance
DIPOLE_TOL = 1e-4
QUAD_TOL = 1e-4
HIGHER_TOL = 1e-3


# =============================================================================
# UNIT TESTS: MultipoleParser.matches() behavior
# =============================================================================


@pytest.mark.unit
def test_multipole_parser_matches_start_line():
    """Unit test: verify MultipoleParser.matches() recognizes multipole block start."""
    parser = MultipoleParser()
    state = ParseState(raw_output="")

    start_line = "Cartesian Multipole Moments"
    assert parser.matches(start_line, state) is True


@pytest.mark.unit
def test_multipole_parser_matches_with_leading_whitespace():
    """Unit test: verify MultipoleParser.matches() handles leading whitespace."""
    parser = MultipoleParser()
    state = ParseState(raw_output="")

    start_line = "  Cartesian Multipole Moments"
    assert parser.matches(start_line, state) is True


@pytest.mark.unit
def test_multipole_parser_does_not_match_non_multipole_lines():
    """Unit test: verify MultipoleParser.matches() rejects non-multipole lines."""
    parser = MultipoleParser()
    state = ParseState(raw_output="")

    # Random lines from QChem output
    assert parser.matches("SCF time:   CPU 0.32s  wall 0.00s", state) is False
    assert parser.matches("Atom          Charge", state) is False
    assert parser.matches("Random calculation output", state) is False


@pytest.mark.unit
def test_multipole_parser_skips_if_already_parsed():
    """Unit test: verify MultipoleParser.matches() returns False when already parsed."""
    parser = MultipoleParser()
    state = ParseState(raw_output="")
    state.parsed_multipole = True

    start_line = "Multipole moment tensor (Debye.Ang**n / e.Bohr**n)"
    assert parser.matches(start_line, state) is False


@pytest.mark.unit
def test_multipole_parser_does_not_mutate_state_in_matches():
    """
    Unit test: verify that matches() is read-only and does not mutate state.
    Critical for parser-spec compliance.
    """
    parser = MultipoleParser()
    state = ParseState(raw_output="")

    # Call matches() multiple times
    line = "Cartesian Multipole Moments"
    result1 = parser.matches(line, state)
    result2 = parser.matches(line, state)

    # State should be identical after calling matches()
    assert result1 is True
    assert result2 is True
    assert state.parsed_multipole is False  # Should NOT be set
    assert state.multipole is None  # Should NOT be populated


# =============================================================================
# CONTRACT TESTS: Data structure validation
# =============================================================================


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["multipole"],
    indirect=True,
)
def test_multipole_results_has_correct_type(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify multipole field is MultipoleResults instance."""
    assert parsed_qchem_data.multipole is not None
    assert isinstance(parsed_qchem_data.multipole, MultipoleResults)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["multipole"],
    indirect=True,
)
def test_multipole_has_charge(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify multipole has charge field populated."""
    multipole = parsed_qchem_data.multipole
    assert multipole is not None
    assert multipole.charge is not None
    assert isinstance(multipole.charge, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["multipole"],
    indirect=True,
)
def test_multipole_has_dipole(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify multipole has dipole moment with all components."""
    multipole = parsed_qchem_data.multipole
    assert multipole is not None
    assert multipole.dipole is not None

    # Check all dipole components are floats
    assert isinstance(multipole.dipole.x, float)
    assert isinstance(multipole.dipole.y, float)
    assert isinstance(multipole.dipole.z, float)
    assert isinstance(multipole.dipole.magnitude, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["multipole"],
    indirect=True,
)
def test_multipole_has_quadrupole(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify multipole has quadrupole moments with all components."""
    multipole = parsed_qchem_data.multipole
    assert multipole is not None
    assert multipole.quadrupole is not None

    # Check all quadrupole components are floats
    assert isinstance(multipole.quadrupole.xx, float)
    assert isinstance(multipole.quadrupole.xy, float)
    assert isinstance(multipole.quadrupole.yy, float)
    assert isinstance(multipole.quadrupole.xz, float)
    assert isinstance(multipole.quadrupole.yz, float)
    assert isinstance(multipole.quadrupole.zz, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["multipole"],
    indirect=True,
)
def test_multipole_has_octopole(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify multipole has octopole moments with all components."""
    multipole = parsed_qchem_data.multipole
    assert multipole is not None
    assert multipole.octopole is not None

    # Check all octopole components are floats
    assert isinstance(multipole.octopole.xxx, float)
    assert isinstance(multipole.octopole.xxy, float)
    assert isinstance(multipole.octopole.xyy, float)
    assert isinstance(multipole.octopole.yyy, float)
    assert isinstance(multipole.octopole.xxz, float)
    assert isinstance(multipole.octopole.xyz, float)
    assert isinstance(multipole.octopole.yyz, float)
    assert isinstance(multipole.octopole.xzz, float)
    assert isinstance(multipole.octopole.yzz, float)
    assert isinstance(multipole.octopole.zzz, float)


@pytest.mark.contract
@pytest.mark.parametrize(
    "parsed_qchem_data",
    FIXTURE_SPECS["multipole"],
    indirect=True,
)
def test_multipole_has_hexadecapole(parsed_qchem_data: CalculationResult) -> None:
    """Contract test: verify multipole has hexadecapole moments with all components."""
    multipole = parsed_qchem_data.multipole
    assert multipole is not None
    assert multipole.hexadecapole is not None

    # Check all hexadecapole components are floats
    assert isinstance(multipole.hexadecapole.xxxx, float)
    assert isinstance(multipole.hexadecapole.xxxy, float)
    assert isinstance(multipole.hexadecapole.xxyy, float)
    assert isinstance(multipole.hexadecapole.xyyy, float)
    assert isinstance(multipole.hexadecapole.yyyy, float)
    assert isinstance(multipole.hexadecapole.xxxz, float)
    assert isinstance(multipole.hexadecapole.xxyz, float)
    assert isinstance(multipole.hexadecapole.xyyz, float)
    assert isinstance(multipole.hexadecapole.yyyz, float)
    assert isinstance(multipole.hexadecapole.xxzz, float)
    assert isinstance(multipole.hexadecapole.xyzz, float)
    assert isinstance(multipole.hexadecapole.yyzz, float)
    assert isinstance(multipole.hexadecapole.xzzz, float)
    assert isinstance(multipole.hexadecapole.yzzz, float)
    assert isinstance(multipole.hexadecapole.zzzz, float)


# =============================================================================
# INTEGRATION TESTS: Multiple components working together
# =============================================================================


@pytest.mark.integration
def test_multipole_parsed_alongside_geometry(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify multipole parser works with geometry parser.
    All should be present in final result.
    """
    assert parsed_qchem_62_h2o_sp_data.input_geometry is not None
    assert len(parsed_qchem_62_h2o_sp_data.input_geometry) == 3  # H2O

    assert parsed_qchem_62_h2o_sp_data.multipole is not None


@pytest.mark.integration
def test_multipole_completion_flag_set(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify that the multipole parser sets its completion flag.
    This is critical for the parser-spec contract.
    """
    # The parsed result should have multipole data, indicating the flag was set
    assert parsed_qchem_62_h2o_sp_data.multipole is not None


@pytest.mark.integration
def test_multipole_parsed_alongside_charges(parsed_qchem_62_h2o_sp_data: CalculationResult):
    """
    Integration test: verify both multipole and charges results are present.
    These are complementary results from the same calculation.
    """
    assert parsed_qchem_62_h2o_sp_data.atomic_charges is not None
    assert len(parsed_qchem_62_h2o_sp_data.atomic_charges) > 0

    assert parsed_qchem_62_h2o_sp_data.multipole is not None


# =============================================================================
# REGRESSION TESTS: Exact numerical values
# =============================================================================


@pytest.mark.regression
def test_multipole_charge_value(parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
    """Regression test: verify exact charge value."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.charge == pytest.approx(EXPECTED_CHARGE, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_x_component(parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
    """Regression test: verify exact dipole X component."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.dipole is not None
    assert multipole.dipole.x == pytest.approx(EXPECTED_DIPOLE_X, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_y_component(parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
    """Regression test: verify exact dipole Y component."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.dipole is not None
    assert multipole.dipole.y == pytest.approx(EXPECTED_DIPOLE_Y, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_z_component(parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
    """Regression test: verify exact dipole Z component."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.dipole is not None
    assert multipole.dipole.z == pytest.approx(EXPECTED_DIPOLE_Z, abs=DIPOLE_TOL)


@pytest.mark.regression
def test_dipole_magnitude(parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
    """Regression test: verify exact dipole magnitude."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.dipole is not None
    assert multipole.dipole.magnitude == pytest.approx(EXPECTED_DIPOLE_MAG, abs=DIPOLE_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "component,expected",
    [
        ("xx", EXPECTED_QUAD_XX),
        ("xy", EXPECTED_QUAD_XY),
        ("yy", EXPECTED_QUAD_YY),
        ("xz", EXPECTED_QUAD_XZ),
        ("yz", EXPECTED_QUAD_YZ),
        ("zz", EXPECTED_QUAD_ZZ),
    ],
)
def test_quadrupole_components(
    parsed_qchem_62_h2o_sp_data: CalculationResult,
    component: str,
    expected: float,
) -> None:
    """Regression test: verify exact quadrupole component values."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.quadrupole is not None
    actual = getattr(multipole.quadrupole, component)
    assert actual == pytest.approx(expected, abs=QUAD_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "component,expected",
    [
        ("xxx", EXPECTED_OCT_XXX),
        ("xxy", EXPECTED_OCT_XXY),
        ("xyy", EXPECTED_OCT_XYY),
        ("yyy", EXPECTED_OCT_YYY),
        ("xxz", EXPECTED_OCT_XXZ),
        ("xyz", EXPECTED_OCT_XYZ),
        ("yyz", EXPECTED_OCT_YYZ),
        ("xzz", EXPECTED_OCT_XZZ),
        ("yzz", EXPECTED_OCT_YZZ),
        ("zzz", EXPECTED_OCT_ZZZ),
    ],
)
def test_octopole_components(
    parsed_qchem_62_h2o_sp_data: CalculationResult,
    component: str,
    expected: float,
) -> None:
    """Regression test: verify exact octopole component values."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.octopole is not None
    actual = getattr(multipole.octopole, component)
    assert actual == pytest.approx(expected, abs=HIGHER_TOL)


@pytest.mark.regression
@pytest.mark.parametrize(
    "component,expected",
    [
        ("xxxx", EXPECTED_HEX_XXXX),
        ("xxxy", EXPECTED_HEX_XXXY),
        ("xxyy", EXPECTED_HEX_XXYY),
        ("xyyy", EXPECTED_HEX_XYYY),
        ("yyyy", EXPECTED_HEX_YYYY),
        ("xxxz", EXPECTED_HEX_XXXZ),
        ("xxyz", EXPECTED_HEX_XXYZ),
        ("xyyz", EXPECTED_HEX_XYYZ),
        ("yyyz", EXPECTED_HEX_YYYZ),
        ("xxzz", EXPECTED_HEX_XXZZ),
        ("xyzz", EXPECTED_HEX_XYZZ),
        ("yyzz", EXPECTED_HEX_YYZZ),
        ("xzzz", EXPECTED_HEX_XZZZ),
        ("yzzz", EXPECTED_HEX_YZZZ),
        ("zzzz", EXPECTED_HEX_ZZZZ),
    ],
)
def test_hexadecapole_components(
    parsed_qchem_62_h2o_sp_data: CalculationResult,
    component: str,
    expected: float,
) -> None:
    """Regression test: verify exact hexadecapole component values."""
    multipole = parsed_qchem_62_h2o_sp_data.multipole
    assert multipole is not None
    assert multipole.hexadecapole is not None
    actual = getattr(multipole.hexadecapole, component)
    assert actual == pytest.approx(expected, abs=HIGHER_TOL)
