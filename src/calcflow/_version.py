"""Centralized package version lookup with source-tree fallback."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("calcflow")
except PackageNotFoundError:
    # Running from source without installed package metadata.
    __version__ = "0.0.0.dev0+unknown"
