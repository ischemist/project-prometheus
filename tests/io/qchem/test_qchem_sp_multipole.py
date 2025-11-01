"""Contract tests for QChem Cartesian multipole moments parsing."""

import pytest

from calcflow.common.models import CalculationResult
from tests.io.qchem.conftest import FIXTURES_SP_ONLY


class TestQChemMultipole:
    """Tests for Cartesian multipole moments parsing from QChem output."""

    @pytest.mark.contract
    @pytest.mark.parametrize(
        "parsed_qchem_h2o_sp_data",
        FIXTURES_SP_ONLY,
        indirect=True,
    )
    def test_qchem_multipole_present(self, parsed_qchem_h2o_sp_data: CalculationResult) -> None:
        """
        Verify that Cartesian multipole moments are correctly parsed from QChem H2O SP output.

        Expected values from ex-multipole.md:
        - Charge: -0.0000
        - Dipole: X=-0.8826, Y=-0.1808, Z=-1.5445, Tot=1.7880
        - Quadrupole: XX=-8.5235, XY=-2.1415, YY=-6.5392, XZ=-3.8091, YZ=-2.0882, ZZ=-4.6864
        - Octopole: XXX=-45.1424, XXY=-15.7146, XYY=-18.0848, YYY=-29.1344, XXZ=-8.8897,
                    XYZ=-5.0592, YYZ=-1.9524, XZZ=-10.2351, YZZ=-7.5112, ZZZ=1.6208
        - Hexadecapole: XXXX=-193.4737, XXXY=-76.9501, XXYY=-60.7182, XYYY=-71.6161,
                        YYYY=-92.5752, XXXZ=-19.1566, XXYZ=-11.6023, XYYZ=-4.5683,
                        YYYZ=-0.0879, XXZZ=-24.2567, XYZZ=-16.4191, YYZZ=-13.5970,
                        XZZZ=3.6825, YZZZ=2.4530, ZZZZ=-5.3042
        """
        # Assert that multipole results were parsed
        assert parsed_qchem_h2o_sp_data.multipole is not None, "Multipole results not found"

        multipole = parsed_qchem_h2o_sp_data.multipole

        # Test charge
        assert multipole.charge is not None, "Charge not found"
        assert abs(multipole.charge) < 1e-4, f"Charge mismatch: expected ~0, got {multipole.charge}"

        # Test dipole moment
        assert multipole.dipole is not None, "Dipole moment not found"
        assert abs(multipole.dipole.x - (-0.8826)) < 1e-4, (
            f"Dipole X mismatch: expected -0.8826, got {multipole.dipole.x}"
        )
        assert abs(multipole.dipole.y - (-0.1808)) < 1e-4, (
            f"Dipole Y mismatch: expected -0.1808, got {multipole.dipole.y}"
        )
        assert abs(multipole.dipole.z - (-1.5445)) < 1e-4, (
            f"Dipole Z mismatch: expected -1.5445, got {multipole.dipole.z}"
        )
        assert abs(multipole.dipole.magnitude - 1.7880) < 1e-4, (
            f"Dipole magnitude mismatch: expected 1.7880, got {multipole.dipole.magnitude}"
        )

        # Test quadrupole moments
        assert multipole.quadrupole is not None, "Quadrupole moments not found"
        assert abs(multipole.quadrupole.xx - (-8.5235)) < 1e-4, (
            f"Quadrupole XX mismatch: expected -8.5235, got {multipole.quadrupole.xx}"
        )
        assert abs(multipole.quadrupole.xy - (-2.1415)) < 1e-4, (
            f"Quadrupole XY mismatch: expected -2.1415, got {multipole.quadrupole.xy}"
        )
        assert abs(multipole.quadrupole.yy - (-6.5392)) < 1e-4, (
            f"Quadrupole YY mismatch: expected -6.5392, got {multipole.quadrupole.yy}"
        )
        assert abs(multipole.quadrupole.xz - (-3.8091)) < 1e-4, (
            f"Quadrupole XZ mismatch: expected -3.8091, got {multipole.quadrupole.xz}"
        )
        assert abs(multipole.quadrupole.yz - (-2.0882)) < 1e-4, (
            f"Quadrupole YZ mismatch: expected -2.0882, got {multipole.quadrupole.yz}"
        )
        assert abs(multipole.quadrupole.zz - (-4.6864)) < 1e-4, (
            f"Quadrupole ZZ mismatch: expected -4.6864, got {multipole.quadrupole.zz}"
        )

        # Test octopole moments
        assert multipole.octopole is not None, "Octopole moments not found"
        assert abs(multipole.octopole.xxx - (-45.1424)) < 1e-3, (
            f"Octopole XXX mismatch: expected -45.1424, got {multipole.octopole.xxx}"
        )
        assert abs(multipole.octopole.xxy - (-15.7146)) < 1e-3, (
            f"Octopole XXY mismatch: expected -15.7146, got {multipole.octopole.xxy}"
        )
        assert abs(multipole.octopole.xyy - (-18.0848)) < 1e-3, (
            f"Octopole XYY mismatch: expected -18.0848, got {multipole.octopole.xyy}"
        )
        assert abs(multipole.octopole.yyy - (-29.1344)) < 1e-3, (
            f"Octopole YYY mismatch: expected -29.1344, got {multipole.octopole.yyy}"
        )
        assert abs(multipole.octopole.xxz - (-8.8897)) < 1e-3, (
            f"Octopole XXZ mismatch: expected -8.8897, got {multipole.octopole.xxz}"
        )
        assert abs(multipole.octopole.xyz - (-5.0592)) < 1e-3, (
            f"Octopole XYZ mismatch: expected -5.0592, got {multipole.octopole.xyz}"
        )
        assert abs(multipole.octopole.yyz - (-1.9524)) < 1e-3, (
            f"Octopole YYZ mismatch: expected -1.9524, got {multipole.octopole.yyz}"
        )
        assert abs(multipole.octopole.xzz - (-10.2351)) < 1e-3, (
            f"Octopole XZZ mismatch: expected -10.2351, got {multipole.octopole.xzz}"
        )
        assert abs(multipole.octopole.yzz - (-7.5112)) < 1e-3, (
            f"Octopole YZZ mismatch: expected -7.5112, got {multipole.octopole.yzz}"
        )
        assert abs(multipole.octopole.zzz - 1.6208) < 1e-3, (
            f"Octopole ZZZ mismatch: expected 1.6208, got {multipole.octopole.zzz}"
        )

        # Test hexadecapole moments
        assert multipole.hexadecapole is not None, "Hexadecapole moments not found"
        assert abs(multipole.hexadecapole.xxxx - (-193.4737)) < 1e-3, (
            f"Hexadecapole XXXX mismatch: expected -193.4737, got {multipole.hexadecapole.xxxx}"
        )
        assert abs(multipole.hexadecapole.xxxy - (-76.9501)) < 1e-3, (
            f"Hexadecapole XXXY mismatch: expected -76.9501, got {multipole.hexadecapole.xxxy}"
        )
        assert abs(multipole.hexadecapole.xxyy - (-60.7182)) < 1e-3, (
            f"Hexadecapole XXYY mismatch: expected -60.7182, got {multipole.hexadecapole.xxyy}"
        )
        assert abs(multipole.hexadecapole.xyyy - (-71.6161)) < 1e-3, (
            f"Hexadecapole XYYY mismatch: expected -71.6161, got {multipole.hexadecapole.xyyy}"
        )
        assert abs(multipole.hexadecapole.yyyy - (-92.5752)) < 1e-3, (
            f"Hexadecapole YYYY mismatch: expected -92.5752, got {multipole.hexadecapole.yyyy}"
        )
        assert abs(multipole.hexadecapole.xxxz - (-19.1566)) < 1e-3, (
            f"Hexadecapole XXXZ mismatch: expected -19.1566, got {multipole.hexadecapole.xxxz}"
        )
        assert abs(multipole.hexadecapole.xxyz - (-11.6023)) < 1e-3, (
            f"Hexadecapole XXYZ mismatch: expected -11.6023, got {multipole.hexadecapole.xxyz}"
        )
        assert abs(multipole.hexadecapole.xyyz - (-4.5683)) < 1e-3, (
            f"Hexadecapole XYYZ mismatch: expected -4.5683, got {multipole.hexadecapole.xyyz}"
        )
        assert abs(multipole.hexadecapole.yyyz - (-0.0879)) < 1e-3, (
            f"Hexadecapole YYYZ mismatch: expected -0.0879, got {multipole.hexadecapole.yyyz}"
        )
        assert abs(multipole.hexadecapole.xxzz - (-24.2567)) < 1e-3, (
            f"Hexadecapole XXZZ mismatch: expected -24.2567, got {multipole.hexadecapole.xxzz}"
        )
        assert abs(multipole.hexadecapole.xyzz - (-16.4191)) < 1e-3, (
            f"Hexadecapole XYZZ mismatch: expected -16.4191, got {multipole.hexadecapole.xyzz}"
        )
        assert abs(multipole.hexadecapole.yyzz - (-13.5970)) < 1e-3, (
            f"Hexadecapole YYZZ mismatch: expected -13.5970, got {multipole.hexadecapole.yyzz}"
        )
        assert abs(multipole.hexadecapole.xzzz - 3.6825) < 1e-3, (
            f"Hexadecapole XZZZ mismatch: expected 3.6825, got {multipole.hexadecapole.xzzz}"
        )
        assert abs(multipole.hexadecapole.yzzz - 2.4530) < 1e-3, (
            f"Hexadecapole YZZZ mismatch: expected 2.4530, got {multipole.hexadecapole.yzzz}"
        )
        assert abs(multipole.hexadecapole.zzzz - (-5.3042)) < 1e-3, (
            f"Hexadecapole ZZZZ mismatch: expected -5.3042, got {multipole.hexadecapole.zzzz}"
        )
