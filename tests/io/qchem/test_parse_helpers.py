"""
Unit tests for calcflow.io.qchem.blocks.parse_helpers.

These tests document and guard the edge-case behaviour introduced when the
helpers were centralised (PR 1) and hardened against review comments:

* ``parse_vector`` — the regex ``[\\d.-]+`` can match non-numeric strings such
  as ``.``, ``-``, or ``1.2.3``; the try/except ensures ``None`` is returned
  instead of raising ``ValueError``.

* ``parse_key_value`` — the colon is intentionally optional because some
  Q-Chem output lines use ``=`` (e.g. ``QTa = 0.253653``) while others use
  ``:`` (e.g. ``Number of electrons:  10.0000``).  The function is permissive:
  any line containing the key followed eventually by whitespace and a number
  will match, regardless of intervening characters.
"""

import pytest

from calcflow.io.qchem.blocks.parse_helpers import parse_key_value, parse_vector

# =============================================================================
# parse_vector
# =============================================================================


@pytest.mark.unit
class TestParseVector:
    """Edge cases for parse_vector."""

    def test_valid_vector(self) -> None:
        # No space before closing ']' — matches the actual Q-Chem output format.
        assert parse_vector("[ 1.23, -4.56, 7.89]") == pytest.approx((1.23, -4.56, 7.89))

    def test_no_vector_returns_none(self) -> None:
        assert parse_vector("no vector here") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_vector("") is None

    @pytest.mark.parametrize(
        "line",
        [
            # Single dot — regex matches '.', but float('.') raises ValueError.
            "[ ., 1.0, 2.0]",
            # Single dash — regex matches '-', but float('-') raises ValueError.
            "[ -, 1.0, 2.0]",
            # Two dots — '1.2.3' matches [\\d.-]+ but is not a valid float.
            "[ 1.2.3, 1.0, 2.0]",
        ],
    )
    def test_non_numeric_regex_match_returns_none(self, line: str) -> None:
        """Regex [\\d.-]+ can match strings that aren't valid floats.

        The try/except in parse_vector must swallow the resulting ValueError
        and return None rather than crashing the parser.
        """
        assert parse_vector(line) is None


# =============================================================================
# parse_key_value
# =============================================================================


@pytest.mark.unit
class TestParseKeyValue:
    """Edge cases for parse_key_value."""

    def test_colon_separator(self) -> None:
        """Standard Q-Chem key-value line with colon."""
        assert parse_key_value("    Number of electrons:  10.0000", "Number of electrons") == pytest.approx(10.0)

    def test_equals_separator(self) -> None:
        """CT-number lines use '=' not ':'; colon must be optional.

        Real fixture line from 6.2-rks-tddft.out:
            Sum of absolute trans. charges, QTa = 0.253653
        """
        assert parse_key_value("      Sum of absolute trans. charges, QTa = 0.253653", "QTa") == pytest.approx(0.253653)

    def test_negative_value(self) -> None:
        assert parse_key_value("Energy:  -1.234", "Energy") == pytest.approx(-1.234)

    def test_key_absent_returns_none(self) -> None:
        assert parse_key_value("unrelated line with 3.14", "QTa") is None

    def test_key_present_no_trailing_number_returns_none(self) -> None:
        assert parse_key_value("QTa is mentioned but no number follows", "QTa") is None

    def test_key_with_brackets_in_line(self) -> None:
        """Lines like 'Hole size [Ang]:  0.827946' contain brackets before the colon."""
        assert parse_key_value("      Hole size [Ang]:                0.827946", "Hole size") == pytest.approx(0.827946)
