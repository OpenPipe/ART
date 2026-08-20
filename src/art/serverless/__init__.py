from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import ServerlessBackend

__all__ = ["ServerlessBackend"]


def __getattr__(name: str) -> Any:
    if name != "ServerlessBackend":
        raise AttributeError(name)
    from .backend import ServerlessBackend

    return ServerlessBackend
