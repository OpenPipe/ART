from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import os
from typing import Any

_TILELANG_ENV_KEYS = (
    "PYTHONPATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "TL_CUTLASS_PATH",
    "TL_TEMPLATE_PATH",
    "TL_COMPOSABLE_KERNEL_PATH",
)

_TILELANG_PATH_MARKERS = ("/site-packages/tilelang/", "\\site-packages\\tilelang\\")


def _drop_tilelang_paths(value: str | None) -> str | None:
    if value is None:
        return None
    kept = [
        part
        for part in value.split(os.pathsep)
        if not any(marker in part for marker in _TILELANG_PATH_MARKERS)
    ]
    return os.pathsep.join(kept) if kept else None


def sanitize_tilelang_env() -> None:
    """Remove TileLang's vendored TVM paths from env inherited by child processes."""
    for key in _TILELANG_ENV_KEYS:
        value = _drop_tilelang_paths(os.environ.get(key))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    sanitize_tilelang_env()


@contextmanager
def preserve_tilelang_env() -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in _TILELANG_ENV_KEYS}
    try:
        yield
    finally:
        _restore_env(saved)


def import_tilelang() -> tuple[Any, Any]:
    """Import TileLang without leaking its vendored TVM paths to child processes."""
    with preserve_tilelang_env():
        tilelang = importlib.import_module("tilelang")
        language = importlib.import_module("tilelang.language")
    return tilelang, language
