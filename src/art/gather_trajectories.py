import asyncio
import contextvars
import contextlib
from collections import Counter
from dataclasses import dataclass, field
from itertools import cycle
from openai.types.chat.chat_completion import Choice
import os
import random
import shutil
from tqdm import auto as tqdm
from typing import Any, Coroutine, Iterable, Iterator, Literal, overload

from .types import Trajectory


@overload
async def gather_trajectories(
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
async def gather_trajectories(
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


async def gather_trajectories(
    groups: Iterable[
        Iterable[Coroutine[Any, Any, Trajectory | Iterable[Trajectory]]]
        | Coroutine[Any, Any, Iterable[Trajectory]]
    ],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    return_exceptions: bool = True,
    stream_chat_completions: bool | int | float = False,
    streaming_chat_completions_dir: str = "./streaming-chat-completions",
    clear_streaming_chat_completions_dir: bool = True,
) -> list[list[Trajectory | BaseException]] | list[list[Trajectory]]:
    groups = [list([g] if isinstance(g, Coroutine) else g) for g in groups]
    total = sum(len(g) for g in groups)
    if stream_chat_completions:
        if clear_streaming_chat_completions_dir:
            shutil.rmtree(streaming_chat_completions_dir, ignore_errors=True)
        os.makedirs(streaming_chat_completions_dir, exist_ok=True)
        if isinstance(stream_chat_completions, bool):
            true_count = total
        elif isinstance(stream_chat_completions, int):
            true_count = min(stream_chat_completions, total)
        elif isinstance(stream_chat_completions, float):
            true_count = min(int(round(total * stream_chat_completions)), total)
        should_stream = [True] * true_count + [False] * (total - true_count)
        random.shuffle(should_stream)
    else:
        should_stream = [False]
    context = GroupsContext(
        pbar=tqdm.tqdm(desc=pbar_desc, total=total),
        pbar_total_completion_tokens=pbar_total_completion_tokens,
        should_stream=iter(cycle(should_stream)),
        streaming_chat_completions_dir=streaming_chat_completions_dir,
    )
    with set_groups_context(context):
        result_groups = await asyncio.gather(
            *[
                asyncio.gather(
                    *[wrap_coroutine(c) for c in g], return_exceptions=return_exceptions
                )
                for g in groups
            ]
        )
    if context.pbar is not None:
        context.pbar.close()
    return [
        [
            item
            for items in group
            for item in (items if isinstance(items, list) else [items])
        ]
        for group in result_groups
    ]


async def wrap_coroutine(
    coro: Coroutine[Any, Any, Trajectory | Iterable[Trajectory]],
) -> Trajectory | list[Trajectory]:
    context = get_groups_context()
    try:
        result = await coro
        context.update_pbar(n=1)
        if not isinstance(result, Trajectory):
            result = list(result)
        for trajectory in [result] if isinstance(result, Trajectory) else result:
            logprobs = [
                message_or_choice.logprobs
                for message_or_choice in trajectory.messages_and_choices
                if isinstance(message_or_choice, Choice)
                if message_or_choice.logprobs
            ]
            if logprobs:
                trajectory.metrics["completion_tokens"] = sum(
                    len(l.content or l.refusal or []) for l in logprobs
                ) / len(logprobs)
            context.metric_sums["reward"] += trajectory.reward  # type: ignore
            context.metric_divisors["reward"] += 1
            context.metric_sums.update(trajectory.metrics)
            context.metric_divisors.update(trajectory.metrics.keys())
        return result
    except BaseException as e:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        raise e


def record_metrics(context: "GroupsContext", trajectory: Trajectory) -> None:
    logprobs = [
        message_or_choice.logprobs
        for message_or_choice in trajectory.messages_and_choices
        if isinstance(message_or_choice, Choice)
        if message_or_choice.logprobs
    ]
    if logprobs:
        trajectory.metrics["completion_tokens"] = sum(
            len(l.content or l.refusal or []) for l in logprobs
        ) / len(logprobs)
    context.metric_sums["reward"] += trajectory.reward  # type: ignore
    context.metric_divisors["reward"] += 1
    context.metric_sums.update(trajectory.metrics)
    context.metric_divisors.update(trajectory.metrics.keys())


@dataclass
class GroupsContext:
    pbar: tqdm.tqdm | None = None
    metric_sums: Counter[str] = field(default_factory=Counter)
    metric_divisors: Counter[str] = field(default_factory=Counter)
    pbar_total_completion_tokens: bool = False
    should_stream: Iterator[bool] = field(default_factory=lambda: iter(cycle([False])))
    streaming_chat_completions_dir: str = "./streaming-chat-completions"

    def update_pbar(self, n: int) -> None:
        if self.pbar is not None:
            self.pbar.update(n)
            postfix = {}
            for metric in self.metric_sums:
                sum = self.metric_sums[metric]
                divisor = max(1, self.metric_divisors[metric])
                if isinstance(sum, int):
                    postfix[metric] = int(sum / divisor)
                else:
                    postfix[metric] = sum / divisor
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_completion_tokens",
            ):
                if key in postfix:
                    postfix[key] = postfix.pop(key)
            self.pbar.set_postfix(postfix)


groups_context_var = contextvars.ContextVar("groups_context", default=GroupsContext())


@contextlib.contextmanager
def set_groups_context(context: GroupsContext) -> Iterator[None]:
    token = groups_context_var.set(context)
    try:
        yield
    finally:
        groups_context_var.reset(token)


def get_groups_context() -> GroupsContext:
    return groups_context_var.get()
