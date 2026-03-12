"""
Fundamental building blocks shared across the calcflow data model.

``FrozenModel`` and ``Atom`` live here — separate from ``results.py`` — so that
``calcflow.geometry.static`` can import ``Atom`` without creating a circular
dependency with ``calcflow.common.results`` (which imports ``Geometry``).

Import paths that callers already use (``from calcflow.common.results import Atom``)
continue to work because ``results.py`` re-exports both names from this module.
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import Any, Self, TypeAliasType, Union, get_args, get_origin

from calcflow.common.exceptions import ValidationError
from calcflow.constants.ptable import ELEMENT_DATA

# =============================================================================
# §0. BASE MODEL FOR SERIALIZATION & DESERIALIZATION
# =============================================================================


@dataclass(frozen=True)
class FrozenModel:
    """Base class providing ``to_dict`` / ``from_dict`` for frozen dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Recursively converts the dataclass instance to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Recursively constructs a dataclass instance from a dictionary.

        Ignores extraneous keys in the input dictionary.

        Args:
            data: flat or nested dictionary matching the dataclass field layout.

        Returns:
            A fully constructed, frozen instance of ``cls``.

        Raises:
            ValidationError: if any required field is absent from ``data``.
        """
        kwargs: dict[str, Any] = {}
        cls_fields = {f.name: f for f in dataclasses.fields(cls)}

        for field_name, field_info in cls_fields.items():
            if field_name in data:
                value = data[field_name]
                kwargs[field_name] = FrozenModel._convert_value(value, field_info.type)

        required_fields = {
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }
        missing = sorted(required_fields - set(kwargs))
        if missing:
            raise ValidationError(f"{cls.__name__}.from_dict() missing required fields: {missing}")

        return cls(**kwargs)

    @staticmethod
    def _convert_value(value: Any, target_type: Any) -> Any:
        """Recursively convert a raw dict/list value into the target Python type.

        Handles ``Optional[T]``, nested dataclasses, ``Sequence``, ``tuple``,
        and ``Mapping``.

        Args:
            value: raw value, typically from ``json.loads``.
            target_type: the annotated type of the field being populated.

        Returns:
            The value coerced to ``target_type``.
        """
        if value is None:
            return None

        if isinstance(target_type, TypeAliasType):
            return FrozenModel._convert_value(value, target_type.__value__)

        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle Optional[T] / T | None
        if (origin is Union or origin is UnionType) and type(None) in args:
            inner_type = next(t for t in args if t is not type(None))
            return FrozenModel._convert_value(value, inner_type)

        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            # target_type is a FrozenModel subclass; delegate to its from_dict.
            return target_type.from_dict(value)

        if origin in (list, Sequence) and isinstance(value, list):
            item_type = args[0]
            return [FrozenModel._convert_value(item, item_type) for item in value]

        if origin is tuple and isinstance(value, (list, tuple)):
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(FrozenModel._convert_value(item, args[0]) for item in value)
            return tuple(
                FrozenModel._convert_value(item, item_type) for item, item_type in zip(value, args, strict=True)
            )

        if origin in (dict, Mapping) and isinstance(value, dict):
            key_type, val_type = args
            return {
                FrozenModel._convert_value(k, key_type): FrozenModel._convert_value(v, val_type)
                for k, v in value.items()
            }

        if target_type in (int, float, str, bool) and not isinstance(value, target_type):
            try:
                return target_type(value)
            except (TypeError, ValueError):
                return value

        return value

    def to_json(self, indent: int = 2) -> str:
        """Serializes the model to a JSON string.

        Args:
            indent: number of spaces used for JSON indentation.

        Returns:
            A pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserializes a model from a JSON string.

        Args:
            json_str: a valid JSON string produced by ``to_json()``.

        Returns:
            A fully constructed, frozen instance of ``cls``.
        """
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# §1. ATOM
# =============================================================================


@dataclass(frozen=True)
class Atom(FrozenModel):
    """Represents a single atom with its Cartesian coordinates in Angstrom."""

    symbol: str
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        """Validates the element symbol on construction.

        Raises:
            ValidationError: if the symbol is unknown or not properly capitalised.
        """
        if self.symbol.upper() not in ELEMENT_DATA:
            raise ValidationError(f"unknown element symbol: '{self.symbol}'")
        if self.symbol != self.symbol.capitalize():
            raise ValidationError(f"element symbol '{self.symbol}' must be capitalized.")
