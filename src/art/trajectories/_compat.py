from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine, Generator, Iterable
import copy
import traceback
from typing import Any
import warnings

from openai.types.chat.chat_completion import Choice
import pydantic

from ..types import Message, Messages, MessagesAndChoices
from . import MetadataValue, PydanticException, Trajectory, TrajectoryGroup


def exception_model(
    exception: BaseException | PydanticException,
) -> PydanticException:
    if isinstance(exception, PydanticException):
        return exception
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
    items: list[Awaitable[Trajectory]],
    *,
    exceptions: Iterable[BaseException | PydanticException],
    metadata: dict[str, MetadataValue] | None,
    metrics: dict[str, float | int | bool] | None,
    logs: list[str] | None,
) -> TrajectoryGroup:
    from ..gather import get_gather_context, record_metrics

    context = get_gather_context()
    trajectories: list[Trajectory] = []
    captured_exceptions = list(exceptions)
    for future in asyncio.as_completed(items):
        try:
            item = await future
            trajectories.append(item)
            record_metrics(context, item)
            context.update_pbar(n=1)
        except BaseException as exc:
            captured_exceptions.append(exc)
            context.metric_sums["exceptions"] += 1
            context.update_pbar(n=0)
            if context.too_many_exceptions():
                raise
    return TrajectoryGroup(
        trajectories,
        exceptions=captured_exceptions,
        metadata=metadata,
        metrics=metrics,
        logs=logs,
    )


class _LegacyGroupCoroutine(Awaitable[TrajectoryGroup]):
    def __init__(
        self, coroutine: Coroutine[Any, Any, TrajectoryGroup], size: int
    ) -> None:
        self.coroutine = coroutine
        self._num_trajectories = size

    def __await__(self) -> Generator[Any, None, TrajectoryGroup]:
        return self.coroutine.__await__()


def new_trajectory_group(
    cls: type[TrajectoryGroup],
    trajectories: Iterable[Trajectory | BaseException | Awaitable[Trajectory]],
    *,
    exceptions: Iterable[BaseException | PydanticException],
    metadata: dict[str, MetadataValue] | None,
    metrics: dict[str, float | int | bool] | None,
    logs: list[str] | None,
) -> TrajectoryGroup | Awaitable[TrajectoryGroup]:
    items = list(trajectories)
    awaitables = [
        item for item in items if not isinstance(item, (Trajectory, BaseException))
    ]
    if awaitables:
        if len(awaitables) != len(items):
            raise TypeError("TrajectoryGroup cannot mix trajectories and awaitables")
        warnings.warn(
            "Awaiting TrajectoryGroup(...) is deprecated; use art.trajectory_group(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LegacyGroupCoroutine(
            _legacy_async_group(
                awaitables,
                exceptions=exceptions,
                metadata=metadata,
                metrics=metrics,
                logs=logs,
            ),
            len(items),
        )
    sync_items = [
        item for item in items if isinstance(item, (Trajectory, BaseException))
    ]
    if len(sync_items) != len(items):
        raise TypeError("TrajectoryGroup items must be trajectories or exceptions")
    group = object.__new__(cls)
    group.__init__(
        sync_items,
        exceptions=exceptions,
        metadata=metadata,
        metrics=metrics,
        logs=logs,
    )
    return group


def init_trajectory_group(
    group: TrajectoryGroup,
    trajectories: Iterable[Trajectory | BaseException],
    *,
    exceptions: Iterable[BaseException | PydanticException],
    metadata: dict[str, MetadataValue] | None,
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
    group: TrajectoryGroup, memo: dict[int, object] | None
) -> TrajectoryGroup:
    memo = {} if memo is None else memo
    if existing := memo.get(id(group)):
        if not isinstance(existing, TrajectoryGroup):
            raise TypeError("TrajectoryGroup deepcopy memo contains an invalid value")
        return existing
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
            # Response messages and request messages are parallel OpenAI models,
            # but their generated Python types are unrelated. Validate at the seam.
            message = pydantic.TypeAdapter(Message).validate_python(
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
                }
            )
            messages.append(message)
        else:
            message = copy.copy(item)
            if message.get("content") is None:
                message["content"] = ""
            messages.append(message)
    return messages


def trajectory_for_logging(trajectory: Trajectory) -> dict[str, object]:
    if trajectory.exchanges:
        return trajectory.model_dump(mode="json", exclude={"start_time"})
    messages = []
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            message = item.message.to_dict()
            trainable = True
        else:
            message = dict(item)
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
