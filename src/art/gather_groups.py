from typing import Any, Coroutine, Iterable, Literal, overload
import warnings
from typing_extensions import deprecated

from .gather_trajectories import gather_trajectories
from .types import Trajectory


@overload
async def gather_groups(
    groups: Iterable[
        Iterable[Coroutine[Any, Any, Trajectory | Iterable[Trajectory]]]
        | Coroutine[Any, Any, Iterable[Trajectory]]
    ],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    return_exceptions: Literal[True] = True,
    stream_chat_completions: bool | int | float = False,
    streaming_chat_completions_dir: str = "./streaming-chat-completions",
    clear_streaming_chat_completions_dir: bool = True,
) -> list[list[Trajectory | BaseException]]: ...


@overload
async def gather_groups(
    groups: Iterable[
        Iterable[Coroutine[Any, Any, Trajectory | Iterable[Trajectory]]]
        | Coroutine[Any, Any, Iterable[Trajectory]]
    ],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    return_exceptions: Literal[False],
    stream_chat_completions: bool | int | float = False,
    streaming_chat_completions_dir: str = "./streaming-chat-completions",
    clear_streaming_chat_completions_dir: bool = True,
) -> list[list[Trajectory]]: ...


@deprecated("Use gather_trajectories instead")
async def gather_groups(
    *args: Any,
    **kwargs: Any,
) -> list[list[Trajectory | BaseException]] | list[list[Trajectory]]:
    warnings.warn(
        "gather_groups is deprecated and will be removed in a future version. "
        "Use gather_trajectories instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return await gather_trajectories(*args, **kwargs)
