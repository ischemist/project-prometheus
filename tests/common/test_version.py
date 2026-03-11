import importlib
from importlib.metadata import PackageNotFoundError

import pytest


@pytest.mark.unit
def test_public_version_is_non_empty_string() -> None:
    from calcflow import __version__

    assert isinstance(__version__, str)
    assert __version__


@pytest.mark.unit
def test_version_falls_back_without_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as metadata

    import calcflow._version as version_module

    def _raise_package_not_found(_: str) -> str:
        raise PackageNotFoundError

    try:
        with monkeypatch.context() as context:
            context.setattr(metadata, "version", _raise_package_not_found)
            reloaded_module = importlib.reload(version_module)
            assert reloaded_module.__version__ == "0.0.0.dev0+unknown"
    finally:
        importlib.reload(version_module)
