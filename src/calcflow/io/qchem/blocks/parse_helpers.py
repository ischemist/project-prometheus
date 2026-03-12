"""
Shared parsing helper functions for Q-Chem block parsers.

These utilities recur identically across multiple parser files (ADC ground
state, ADC excited states, TDDFT unrelaxed DM, TDDFT transition DM).
Keeping them here avoids copy-paste drift and gives one place to update if
Q-Chem changes its output format.
"""

import re


def to_float(val: str | None) -> float | None:
    """Safely convert a string to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_key_value(line: str, key: str) -> float | None:
    """Extract a scalar float following *key* on a line.

    Matches patterns like::

        Number of electrons:   10.0000
        PR_NO:   3.4512

    The key is matched literally (regex-escaped), followed by optional
    intervening characters, an optional colon or equals sign, whitespace,
    and the value.  The colon is optional because some Q-Chem output lines
    use ``=`` as the separator (e.g. ``QTa = 0.253653``) while others use
    ``:`` (e.g. ``Number of electrons:  10.0000``).  All known callers have
    at least one of these separators, so no false positives are expected.
    """
    m = re.search(rf"{re.escape(key)}.*?:?\s+(-?[\d.]+)", line)
    return to_float(m.group(1)) if m else None


def parse_vector(line: str) -> tuple[float, float, float] | None:
    """Extract a 3-component vector written as ``[ x, y, z ]`` on a line."""
    m = re.search(r"\[\s*([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\]", line)
    if not m:
        return None
    try:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except ValueError:
        return None
