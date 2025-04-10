import asyncio
import contextvars
import contextlib
from collections import Counter
from dataclasses import dataclass, field
from openai.types.chat.chat_completion import Choice
from tqdm import auto as tqdm
from typing import Awaitable, Iterable, Iterator

from .trajectories import Trajectory, TrajectoryGroup


async def gather_trajectories(
    groups: Iterable[Awaitable[TrajectoryGroup]],
    *,
    pbar_desc: str | None = None,
    pbar_total_completion_tokens: bool = True,
    max_exceptions: int | float = 0,
) -> list[TrajectoryGroup]:
    total = sum(getattr(g, "_num_trajectories", 1) for g in groups)
    context = GroupsContext(
        pbar=tqdm.tqdm(desc=pbar_desc, total=total),
        pbar_total_completion_tokens=pbar_total_completion_tokens,
        max_exceptions=max_exceptions,
    )
    with set_groups_context(context):
        result_groups = await asyncio.gather(*[wrap_awaitable(g) for g in groups])
    if context.pbar is not None:
        context.pbar.close()
    return [g for g in result_groups if g is not None]


async def wrap_awaitable(
    awaitable: Awaitable[TrajectoryGroup],
) -> TrajectoryGroup | None:
    if hasattr(awaitable, "_num_trajectories"):
        return await awaitable
    context = get_groups_context()
    try:
        result = await awaitable
        for trajectory in result.trajectories:
            record_metrics(context, trajectory)
        context.update_pbar(n=len(result.trajectories))
        return result
    except BaseException:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        if context.too_many_exceptions():
            raise


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
    max_exceptions: int | float = 0

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

    def too_many_exceptions(self) -> bool:
        if (
            0 < self.max_exceptions < 1
            and self.pbar is not None
            and self.metric_sums["exceptions"] / self.pbar.total <= self.max_exceptions
        ) or self.metric_sums["exceptions"] <= self.max_exceptions:
            return False
        return True


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
