"""Shared fixtures for builder tests."""

from __future__ import annotations

import pytest

from calcflow.common.input import CalculationInput
from calcflow.common.results import Atom
from calcflow.geometry.static import Geometry
from calcflow.io.orca.builder import OrcaBuilder


@pytest.fixture
def h2o_geometry() -> Geometry:
    """Minimal water molecule for testing."""
    return Geometry(
        comment="water molecule",
        atoms=(
            Atom(symbol="O", x=0.0, y=0.0, z=0.0),
            Atom(symbol="H", x=0.0, y=0.0, z=1.0),
            Atom(symbol="H", x=0.0, y=1.0, z=0.0),
        ),
    )


@pytest.fixture
def minimal_spec() -> CalculationInput:
    """Bare minimum spec for unit testing."""
    return CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
    )


@pytest.fixture
def orca_builder() -> OrcaBuilder:
    """Orca builder instance."""
    return OrcaBuilder()
