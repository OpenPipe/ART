"""CPU import compatibility with the SDK version selected by ART's lockfile."""

import importlib
from importlib.metadata import version
from pathlib import Path
import tomllib

import pytest


def test_art_tinker_modules_import_with_locked_sdk():
    pytest.importorskip("tinker")
    with (Path(__file__).parents[2] / "uv.lock").open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    locked = next(
        package["version"] for package in packages if package["name"] == "tinker"
    )
    assert version("tinker") == locked
    for name in (
        "art.tinker",
        "art.tinker.backend",
        "art.tinker.service",
        "art.tinker.server",
        "art.tinker_native",
        "art.tinker_native.backend",
        "art.tinker_native.data",
    ):
        assert importlib.import_module(name)
