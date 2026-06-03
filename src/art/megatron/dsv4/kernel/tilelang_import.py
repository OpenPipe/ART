from __future__ import annotations

import importlib
import os
from typing import Any

_TILELANG_ENV_KEYS = (
    "PYTHONPATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "TL_CUTLASS_PATH",
    "TL_TEMPLATE_PATH",
)


def import_tilelang() -> tuple[Any, Any]:
    """Import TileLang without leaking its vendored TVM paths to child processes."""
    saved = {key: os.environ.get(key) for key in _TILELANG_ENV_KEYS}
    try:
        tilelang = importlib.import_module("tilelang")
        language = importlib.import_module("tilelang.language")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return tilelang, language
