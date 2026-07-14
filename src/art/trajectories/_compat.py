from __future__ import annotations

import asyncio
from collections.abc import Iterable
import copy
import traceback
from typing import Any, cast
import warnings

from openai.types.chat.chat_completion import Choice
import pydantic

from ..types import Message, Messages, MessagesAndChoices
from . import PydanticException, Trajectory, TrajectoryGroup


def exception_model(
    exception: BaseException | PydanticException | dict[str, Any],
) -> PydanticException:
    if isinstance(exception, PydanticException):
        return exception
    if isinstance(exception, dict):
        return PydanticException.model_validate(exception)
    return PydanticException(
        type=str(type(exception)),
        message=str(exception),
        traceback="".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        ),
    )


async def _legacy_async_group(
    items: list[Any], kwargs: dict[str, Any]
) -> TrajectoryGroup:
    from ..gather import get_gather_context, record_metrics

    context = get_gather_context()
    trajectories: list[Trajectory] = []
    exceptions = list(kwargs.pop("exceptions", ()))
    for future in asyncio.as_completed(items):
        try:
            item = await future
            trajectories.append(item)
            record_metrics(context, item)
            context.update_pbar(n=1)
        except BaseException as exc:
            exceptions.append(exc)
            context.metric_sums["exceptions"] += 1
            context.update_pbar(n=0)
            if context.too_many_exceptions():
                raise
    return TrajectoryGroup(trajectories, exceptions=exceptions, **kwargs)


class _LegacyGroupCoroutine:
    def __init__(self, coroutine: Any, size: int) -> None:
        self.coroutine = coroutine
        self._num_trajectories = size

    def __await__(self) -> Any:
        return self.coroutine.__await__()


def new_trajectory_group(
    cls: type[TrajectoryGroup], trajectories: Iterable[Any], kwargs: dict[str, Any]
) -> Any:
    items = list(trajectories)
    if any(hasattr(item, "__await__") for item in items):
        warnings.warn(
            "Awaiting TrajectoryGroup(...) is deprecated; use art.trajectory_group(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LegacyGroupCoroutine(
            _legacy_async_group(items, dict(kwargs)), len(items)
        )
    group = object.__new__(cls)
    group.__init__(items, **kwargs)
    return group


def init_trajectory_group(
    group: TrajectoryGroup,
    trajectories: Iterable[Trajectory | BaseException],
    *,
    exceptions: Iterable[BaseException | PydanticException],
    metadata: dict[str, Any] | None,
    metrics: dict[str, float | int | bool] | None,
    logs: list[str] | None,
) -> None:
    items = list(trajectories)
    normalized_trajectories = [
        item if isinstance(item, Trajectory) else Trajectory.model_validate(item)
        for item in items
        if not isinstance(item, BaseException)
    ]
    pydantic.BaseModel.__init__(
        group,
        trajectories=normalized_trajectories or getattr(group, "trajectories", []),
        exceptions=[
            exception_model(item)
            for item in [
                *(item for item in items if isinstance(item, BaseException)),
                *exceptions,
            ]
        ]
        or getattr(group, "exceptions", []),
        metadata=metadata if metadata is not None else getattr(group, "metadata", {}),
        metrics=metrics if metrics is not None else getattr(group, "metrics", {}),
        logs=logs if logs is not None else getattr(group, "logs", []),
    )


def copy_trajectory_group(group: TrajectoryGroup) -> TrajectoryGroup:
    copied = TrajectoryGroup(
        group.trajectories[:],
        metadata=group.metadata.copy(),
        metrics=group.metrics.copy(),
        logs=group.logs[:],
    )
    copied.exceptions = group.exceptions[:]
    return copied


def deepcopy_trajectory_group(
    group: TrajectoryGroup, memo: dict[int, Any] | None
) -> TrajectoryGroup:
    memo = {} if memo is None else memo
    if id(group) in memo:
        return memo[id(group)]
    copied = TrajectoryGroup(
        copy.deepcopy(group.trajectories, memo),
        metadata=copy.deepcopy(group.metadata, memo),
        metrics=copy.deepcopy(group.metrics, memo),
        logs=copy.deepcopy(group.logs, memo),
    )
    memo[id(group)] = copied
    copied.exceptions = copy.deepcopy(group.exceptions, memo)
    return copied


def messages_from_legacy_history(messages_and_choices: MessagesAndChoices) -> Messages:
    messages: Messages = []
    for item in messages_and_choices:
        if isinstance(item, Choice):
            content = item.message.content or ""
            tool_calls = item.message.tool_calls or []
            message: Message = cast(
                Message,
                {
                    "role": "assistant",
                    "content": content,
                    **(
                        {
                            "tool_calls": [
                                tool_call.model_dump(mode="json")
                                for tool_call in tool_calls
                            ]
                        }
                        if tool_calls
                        else {}
                    ),
                },
            )
            messages.append(message)
        else:
            message = dict(item)
            if message.get("content") is None:
                message["content"] = ""
            messages.append(message)  # type: ignore[arg-type]
    return messages


def trajectory_for_logging(trajectory: Trajectory) -> dict[str, Any]:
    if trajectory.exchanges:
        return trajectory.model_dump(mode="json", exclude={"start_time"})
    messages = []
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            message = item.message.to_dict()
            trainable = True
        else:
            message = cast(dict[str, Any], item)
            trainable = False
        messages.append({**message, "trainable": trainable})
    return {
        "reward": trajectory.reward,
        "initial_policy_version": trajectory.initial_policy_version,
        "final_policy_version": trajectory.final_policy_version,
        "metrics": trajectory.metrics,
        "metadata": trajectory.metadata,
        "messages": messages,
        "tools": trajectory.tools,
        "logs": trajectory.logs,
    }
