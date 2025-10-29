from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def testing_data_path() -> Path:
    """Fixture for the path to the testing_data directory."""
    return Path(__file__).resolve().parent / "testing_data"
