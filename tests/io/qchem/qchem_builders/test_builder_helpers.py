"""Unit tests for Q-Chem builder helper functions."""

from __future__ import annotations

import pytest

from calcflow.common.exceptions import ValidationError
from calcflow.io.qchem.builder import (
    _calculate_expected_electron_distribution,
    _count_electrons_in_qchem_occupation,
)


class TestCountElectronsInQchemOccupation:
    """Tests for _count_electrons_in_qchem_occupation."""

    @pytest.mark.unit
    def test_empty_string(self):
        """Empty string should return 0."""
        assert _count_electrons_in_qchem_occupation("") == 0

    @pytest.mark.unit
    def test_single_orbital(self):
        """Single orbital index should count as 1."""
        assert _count_electrons_in_qchem_occupation("1") == 1
        assert _count_electrons_in_qchem_occupation("5") == 1
        assert _count_electrons_in_qchem_occupation("10") == 1

    @pytest.mark.unit
    def test_range_notation(self):
        """Range notation '1:5' should count orbitals in range."""
        assert _count_electrons_in_qchem_occupation("1:5") == 5  # 1,2,3,4,5
        assert _count_electrons_in_qchem_occupation("1:3") == 3  # 1,2,3
        assert _count_electrons_in_qchem_occupation("5:7") == 3  # 5,6,7

    @pytest.mark.unit
    def test_mixed_notation(self):
        """Mixed single orbitals and ranges."""
        assert _count_electrons_in_qchem_occupation("1:5 7") == 6  # 1,2,3,4,5,7
        assert _count_electrons_in_qchem_occupation("1 3 5") == 3  # 1,3,5
        assert _count_electrons_in_qchem_occupation("1:3 5 7:9") == 7  # 1,2,3,5,7,8,9

    @pytest.mark.unit
    def test_whitespace_handling(self):
        """Should handle extra whitespace."""
        assert _count_electrons_in_qchem_occupation("1:5") == 5
        assert _count_electrons_in_qchem_occupation("  1:5  ") == 5
        assert _count_electrons_in_qchem_occupation("1:3   5") == 4

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "occupation,expected",
        [
            ("1:10", 10),
            ("1:5 7:9", 8),
            ("1 2 3 4 5", 5),
            ("1:3 5:7 9", 7),
        ],
    )
    def test_various_patterns(self, occupation, expected):
        """Test various occupation patterns."""
        assert _count_electrons_in_qchem_occupation(occupation) == expected


class TestCalculateExpectedElectronDistribution:
    """Tests for _calculate_expected_electron_distribution."""

    @pytest.mark.unit
    def test_closed_shell_singlet(self):
        """Closed shell singlet should have equal alpha and beta."""
        n_alpha, n_beta = _calculate_expected_electron_distribution(10, 1)
        assert n_alpha == 5
        assert n_beta == 5

    @pytest.mark.unit
    def test_doublet(self):
        """Doublet (2S+1=2) has one more alpha than beta."""
        n_alpha, n_beta = _calculate_expected_electron_distribution(9, 2)
        assert n_alpha == 5
        assert n_beta == 4

    @pytest.mark.unit
    def test_triplet(self):
        """Triplet (2S+1=3) has two more alpha than beta."""
        n_alpha, n_beta = _calculate_expected_electron_distribution(10, 3)
        assert n_alpha == 6
        assert n_beta == 4

    @pytest.mark.unit
    def test_quartet(self):
        """Quartet (2S+1=4) has three more alpha than beta."""
        n_alpha, n_beta = _calculate_expected_electron_distribution(11, 4)
        assert n_alpha == 7
        assert n_beta == 4

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "electrons,multiplicity,expected_alpha,expected_beta",
        [
            (10, 1, 5, 5),  # closed shell singlet (water)
            (9, 2, 5, 4),  # doublet
            (10, 3, 6, 4),  # triplet
            (11, 4, 7, 4),  # quartet
            (2, 1, 1, 1),  # H2 singlet
            (1, 2, 1, 0),  # H atom doublet
        ],
    )
    def test_various_electron_configs(self, electrons, multiplicity, expected_alpha, expected_beta):
        """Test various electron configurations."""
        n_alpha, n_beta = _calculate_expected_electron_distribution(electrons, multiplicity)
        assert n_alpha == expected_alpha
        assert n_beta == expected_beta

    @pytest.mark.unit
    def test_invalid_multiplicity_even_electrons(self):
        """Even number of electrons cannot have even multiplicity."""
        with pytest.raises(ValidationError, match="cannot achieve multiplicity"):
            _calculate_expected_electron_distribution(10, 2)

    @pytest.mark.unit
    def test_invalid_multiplicity_odd_electrons(self):
        """Odd number of electrons cannot have odd multiplicity."""
        with pytest.raises(ValidationError, match="cannot achieve multiplicity"):
            _calculate_expected_electron_distribution(9, 1)

    @pytest.mark.unit
    def test_impossible_multiplicity_too_high(self):
        """Multiplicity higher than possible for given electrons."""
        with pytest.raises(ValidationError):
            # With 3 electrons, max multiplicity is 4 (all parallel)
            # 3 electrons with multiplicity 5 is impossible
            _calculate_expected_electron_distribution(3, 5)

    @pytest.mark.unit
    def test_impossible_multiplicity_way_too_high(self):
        """Multiplicity impossibly high."""
        with pytest.raises(ValidationError, match="impossible"):
            _calculate_expected_electron_distribution(1, 10)
