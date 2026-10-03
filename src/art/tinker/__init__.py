"""Tinker integrations; importing the inference client needs no training extras."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import TinkerBackend
    from .renderers import get_renderer_name
    from .server import OpenAICompatibleTinkerServer

_EXPORTS = {
    "TinkerBackend": ".backend",
    "get_renderer_name": ".renderers",
    "OpenAICompatibleTinkerServer": ".server",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
