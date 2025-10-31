"""Contract tests for QChem Mulliken charges parsing."""

from calcflow.common.models import CalculationResult


class TestQChemMullikenCharges:
    """Tests for Mulliken atomic charges parsing from QChem output."""

    def test_qchem_mulliken_charges_present(self, parsed_qchem_62_h2o_sp_data: CalculationResult) -> None:
        """
        Verify that Mulliken charges are correctly parsed from QChem 6.2 H2O SP output.

        Expected values from ex-charges.md:
        - Atom 0 (H): 0.193937
        - Atom 1 (O): -0.388200
        - Atom 2 (H): 0.194263
        """
        # Assert that charges were parsed
        assert len(parsed_qchem_62_h2o_sp_data.atomic_charges) >= 1, "No atomic charges found"

        # Find the Mulliken charges entry
        mulliken_charges = None
        for charges in parsed_qchem_62_h2o_sp_data.atomic_charges:
            if charges.method == "Mulliken":
                mulliken_charges = charges
                break

        assert mulliken_charges is not None, "Mulliken charges not found in parsed data"
        assert mulliken_charges.method == "Mulliken"

        # Verify the charge values (0-based atom indices)
        assert len(mulliken_charges.charges) == 3, "Expected 3 atoms in H2O"

        assert abs(mulliken_charges.charges[0] - 0.193937) < 1e-5, (
            f"H(0) charge mismatch: {mulliken_charges.charges[0]}"
        )
        assert abs(mulliken_charges.charges[1] - (-0.388200)) < 1e-5, (
            f"O(1) charge mismatch: {mulliken_charges.charges[1]}"
        )
        assert abs(mulliken_charges.charges[2] - 0.194263) < 1e-5, (
            f"H(2) charge mismatch: {mulliken_charges.charges[2]}"
        )

        # Verify sum of charges is approximately zero
        total_charge = sum(mulliken_charges.charges.values())
        assert abs(total_charge) < 1e-5, f"Sum of charges should be ~0, got {total_charge}"
