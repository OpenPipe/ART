from typing import TYPE_CHECKING, Any

from .renderers import get_renderer_name

if TYPE_CHECKING:
    from .backend import TinkerBackend
    from .server import OpenAICompatibleTinkerServer

__all__ = ["TinkerBackend", "get_renderer_name", "OpenAICompatibleTinkerServer"]


def __getattr__(name: str) -> Any:
    if name == "TinkerBackend":
        from .backend import TinkerBackend

        globals()[name] = TinkerBackend
        return TinkerBackend
    if name == "OpenAICompatibleTinkerServer":
        from .server import OpenAICompatibleTinkerServer

        globals()[name] = OpenAICompatibleTinkerServer
        return OpenAICompatibleTinkerServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
