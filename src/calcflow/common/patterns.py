"""
Shared primitives for QC output parsing.

Canonical float patterns
------------------------
Import these instead of redefining ``FLOAT_PAT`` in every parser file.  Using
a shared constant guarantees all parsers handle the same numeric formats and
makes future changes (e.g. adding support for a new notation) apply everywhere
at once.

Index conversion
----------------
QC programs use 1-based indices in their text output; internal models use
0-based indices.  Use ``extract_index`` for every such conversion so the
off-by-one rule lives in exactly one place.

Pattern definition infrastructure
----------------------------------
The ``VersionSpec`` / ``VersionedPattern`` / ``PatternDefinition`` dataclasses
support version-aware regex patterns for QChem output across software versions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from re import Match, Pattern
from typing import Any, TypeVar

# ---------------------------------------------------------------------------
# Canonical float regex strings
# ---------------------------------------------------------------------------
# These are *raw strings* (not compiled patterns) so callers can embed them
# inside larger patterns with rf"…{FLOAT_PAT}…".

# Full float: optional sign, optional integer part, optional decimal point,
# required digits, optional scientific notation.  Matches: -1.5e-10, .5, 42
FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# Strict signed float: requires digits before and after the decimal point,
# no scientific notation.  Matches: -1.5, +0.0, 42.0  but NOT: 1e-5, .5
SIGNED_FIXED_FLOAT_PAT = r"([-+]?\d+\.\d+)"

# Negative-or-unsigned fixed float: allows a leading minus but not plus,
# requires digits on both sides of the decimal point, no scientific notation.
# Matches: -1.5, 0.0, 42.7  but NOT: +1.5, 1e-5, .5
NEG_FIXED_FLOAT_PAT = r"(-?\d+\.\d+)"

# Unsigned fixed float: no sign allowed, requires digits on both sides of the
# decimal point, no scientific notation.  Use for quantities that are always
# non-negative (e.g. oscillator strengths, timing values).
UNSIGNED_FIXED_FLOAT_PAT = r"(\d+\.\d+)"


# ---------------------------------------------------------------------------
# Index conversion
# ---------------------------------------------------------------------------


def extract_index(value: str | int) -> int:
    """Convert a 1-based atom/orbital index from QC output to 0-based.

    Args:
        value: The raw index as parsed from the output file (string or int).
               Must be a positive integer in the QC program's 1-based scheme.

    Returns:
        The corresponding 0-based index for internal storage.

    Raises:
        ValueError: If *value* cannot be converted to a positive integer.
    """
    idx = int(value)
    if idx < 1:
        raise ValueError(f"QC output indices are 1-based and must be >= 1, got {idx!r}")
    return idx - 1


# Type variable for regex pattern
T = TypeVar("T", bound=str)


@dataclass(order=True, frozen=True)
class VersionSpec:
    """Specification for a version string, enabling comparison."""

    major: int
    minor: int
    patch: int = 0

    @classmethod
    def from_str(cls, version_str: str) -> "VersionSpec":
        """Parse a version string like '6.0.0' or '5.4' into a VersionSpec."""
        parts = version_str.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return cls(major=major, minor=minor, patch=patch)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid version string format: '{version_str}'") from e

    @property
    def version(self) -> str:
        """Return the version as a normalized string (omitting patch if zero)."""
        if self.patch == 0:
            return f"{self.major}.{self.minor}"
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class VersionedPattern:
    """A regex pattern with an associated minimum version and a transform function."""

    pattern: Pattern[str]
    min_version: VersionSpec | None
    transform: Callable[[Match[str]], Any] = lambda m: m.group(1)


@dataclass
class PatternDefinition:
    """Defines a field to extract with a list of version-specific regex patterns."""

    field_name: str
    description: str
    patterns: list[VersionedPattern] = field(default_factory=list, init=False)
    block_type: str | None = None

    def __init__(
        self,
        field_name: str,
        description: str,
        versioned_patterns: list[tuple[Pattern[str], VersionSpec | str | None, Callable[[Match[str]], Any] | None]],
        block_type: str | None = None,
    ) -> None:
        """Initialize and add patterns, sorting them by version."""
        self.field_name = field_name
        self.description = description
        self.block_type = block_type
        self.patterns = []
        for p, v, t in versioned_patterns:
            self.add_pattern(p, v, t)

    def add_pattern(
        self,
        pattern: Pattern[str],
        min_version: str | VersionSpec | None,
        transform: Callable[[Match[str]], Any] | None = None,
    ) -> None:
        """Add a new pattern, converting version string to VersionSpec if needed."""
        resolved_version = VersionSpec.from_str(min_version) if isinstance(min_version, str) else min_version
        transform_func = transform or (lambda m: m.group(1))
        self.patterns.append(VersionedPattern(pattern=pattern, min_version=resolved_version, transform=transform_func))
        # Sort patterns from highest version to lowest, with None (all versions) last.
        self.patterns.sort(key=lambda vp: vp.min_version or VersionSpec(-1, -1, -1), reverse=True)

    def get_matching_pattern(self, version: VersionSpec) -> VersionedPattern | None:
        """
        Get the best matching pattern for the given version.

        It iterates from the newest defined version downwards. The first pattern
        whose min_version is less than or equal to the target version is a match.
        A pattern with min_version=None is a fallback that matches all versions.
        """
        for vp in self.patterns:
            if vp.min_version is None or version >= vp.min_version:
                return vp
        return None
