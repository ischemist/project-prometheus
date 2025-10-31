"""
Pattern definition types for version-aware QChem output parsing.

This module provides dataclasses for defining version-specific regex patterns
that can be used to parse QChem output files across different versions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from re import Match, Pattern
from typing import Any, TypeVar

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
