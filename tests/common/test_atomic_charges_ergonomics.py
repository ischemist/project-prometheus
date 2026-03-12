"""
unit tests for AtomicCharges.to_array() and CalculationResult.get_charges().

these are pure-logic unit tests — no file I/O, no parsers, minimal fixtures.
"""

import pytest

from calcflow.common.results import AtomicCharges, CalculationMetadata, CalculationResult

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mulliken() -> AtomicCharges:
    return AtomicCharges(method="Mulliken", charges={0: -0.5, 1: 0.25, 2: 0.25})


@pytest.fixture
def cm5() -> AtomicCharges:
    return AtomicCharges(method="CM5", charges={0: -0.4, 1: 0.2, 2: 0.2})


@pytest.fixture
def result_with_charges(mulliken: AtomicCharges, cm5: AtomicCharges) -> CalculationResult:
    return CalculationResult(
        termination_status="NORMAL",
        metadata=CalculationMetadata(software_name="ORCA", software_version="5.0"),
        raw_output="",
        atomic_charges=[mulliken, cm5],
    )


@pytest.fixture
def result_no_charges() -> CalculationResult:
    return CalculationResult(
        termination_status="NORMAL",
        metadata=CalculationMetadata(software_name="ORCA", software_version="5.0"),
        raw_output="",
    )


# ---------------------------------------------------------------------------
# AtomicCharges.to_array()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtomicChargesToArray:
    def test_returns_correct_values_in_order(self, mulliken: AtomicCharges):
        arr = mulliken.to_array(3)
        assert arr == [-0.5, 0.25, 0.25]

    def test_missing_indices_default_to_zero(self):
        # sparse charges: only atom 1 is present
        sparse = AtomicCharges(method="Hirshfeld", charges={1: 0.5})
        arr = sparse.to_array(3)
        assert arr == [0.0, 0.5, 0.0]

    def test_n_atoms_larger_than_charges_pads_with_zeros(self, mulliken: AtomicCharges):
        # caller passes a larger n_atoms than indices present
        arr = mulliken.to_array(5)
        assert arr == [-0.5, 0.25, 0.25, 0.0, 0.0]

    def test_n_atoms_zero_returns_empty_list(self, mulliken: AtomicCharges):
        assert mulliken.to_array(0) == []

    def test_single_atom(self):
        ac = AtomicCharges(method="Mulliken", charges={0: 1.0})
        assert ac.to_array(1) == [1.0]

    def test_returns_list_not_other_sequence(self, mulliken: AtomicCharges):
        result = mulliken.to_array(3)
        assert isinstance(result, list)

    def test_negative_and_positive_charges_preserved(self):
        ac = AtomicCharges(method="CM5", charges={0: -0.834, 1: 0.417, 2: 0.417})
        arr = ac.to_array(3)
        assert arr[0] == pytest.approx(-0.834)
        assert arr[1] == pytest.approx(0.417)
        assert arr[2] == pytest.approx(0.417)

    @pytest.mark.parametrize("n_atoms", [1, 3, 10, 100])
    def test_length_always_equals_n_atoms(self, n_atoms: int):
        ac = AtomicCharges(method="Mulliken", charges={0: 0.1})
        assert len(ac.to_array(n_atoms)) == n_atoms


# ---------------------------------------------------------------------------
# CalculationResult.get_charges()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculationResultGetCharges:
    def test_returns_correct_object_for_known_method(
        self, result_with_charges: CalculationResult, mulliken: AtomicCharges
    ):
        found = result_with_charges.get_charges("Mulliken")
        assert found is not None
        assert found == mulliken

    def test_returns_correct_object_for_second_method(self, result_with_charges: CalculationResult, cm5: AtomicCharges):
        found = result_with_charges.get_charges("CM5")
        assert found is not None
        assert found == cm5

    def test_returns_none_for_unknown_method(self, result_with_charges: CalculationResult):
        assert result_with_charges.get_charges("Loewdin") is None

    def test_returns_none_when_no_charges_present(self, result_no_charges: CalculationResult):
        assert result_no_charges.get_charges("Mulliken") is None

    def test_case_sensitive_match(self, result_with_charges: CalculationResult):
        # "mulliken" (lowercase) must not match "Mulliken"
        assert result_with_charges.get_charges("mulliken") is None
        assert result_with_charges.get_charges("MULLIKEN") is None

    def test_returns_first_match_when_duplicates_exist(self):
        ac1 = AtomicCharges(method="Mulliken", charges={0: -0.1})
        ac2 = AtomicCharges(method="Mulliken", charges={0: -0.9})
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="ORCA", software_version="5.0"),
            raw_output="",
            atomic_charges=[ac1, ac2],
        )
        assert result.get_charges("Mulliken") is ac1

    def test_empty_method_string_returns_none(self, result_with_charges: CalculationResult):
        assert result_with_charges.get_charges("") is None

    @pytest.mark.parametrize("method", ["Mulliken", "CM5", "Hirshfeld", "Loewdin", "NPA"])
    def test_get_charges_roundtrip_with_to_array(self, method: str):
        """get_charges + to_array is a natural usage pattern — verify it composes correctly."""
        ac = AtomicCharges(method=method, charges={0: 0.1, 1: -0.1})
        result = CalculationResult(
            termination_status="NORMAL",
            metadata=CalculationMetadata(software_name="ORCA", software_version="5.0"),
            raw_output="",
            atomic_charges=[ac],
        )
        found = result.get_charges(method)
        assert found is not None
        assert found.to_array(2) == pytest.approx([0.1, -0.1])
