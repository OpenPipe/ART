import asyncio
from collections import Counter
import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Iterator, Literal, overload

from openai.types.chat.chat_completion import Choice
from tqdm import auto as tqdm

from .trajectories import Trajectory, TrajectoryGroup


async def gather_trajectory_groups(
    groups: Iterable[Awaitable[TrajectoryGroup]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = False,
    max_exceptions: int | float = 0,
    max_metrics: int | None = None,
    after_each: (
        Callable[
            [TrajectoryGroup], Awaitable[TrajectoryGroup | None | list[TrajectoryGroup]]
        ]
        | None
    ) = None,
) -> list[TrajectoryGroup]:
    if pbar_total_completion_tokens:
        print(
            "pbar_total_completion_tokens is deprecated and will be removed in a future version."
        )
    groups = list(groups)
    context = GatherContext(
        pbar=None,
        max_exceptions=max_exceptions,
        max_metrics=max_metrics,
    )

    # Fuse the after_each callback into the gather process
    async def group_forward(g: Awaitable[TrajectoryGroup]):
        group = await wrap_group_awaitable(g)
        if group is None or after_each is None:
            return group
        return await after_each(group)

    with set_gather_context(context):
        future = asyncio.gather(*[group_forward(g) for g in groups])
        total = sum(getattr(g, "_num_trajectories", 1) for g in groups)
        context.pbar = tqdm.tqdm(desc=pbar_desc, total=total)
        result_groups = await future

    if context.pbar is not None:
        context.pbar.close()

    # Filter out any None results that may have been returned due to handled exceptions
    processed_groups = []
    for g in result_groups:
        if g is None:
            continue
        if isinstance(g, list):
            processed_groups.extend(g)
        elif isinstance(g, TrajectoryGroup):
            processed_groups.append(g)

    return processed_groups


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Trajectory]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = False,
    max_exceptions: Literal[0] = 0,
) -> list[Trajectory]: ...


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Trajectory]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = False,
    max_exceptions: int | float,
) -> list[Trajectory | BaseException]: ...


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Iterable[Trajectory]]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = False,
    max_exceptions: Literal[0] = 0,
) -> list[list[Trajectory]]: ...


@overload
async def gather_trajectories(
    trajectories: Iterable[Awaitable[Iterable[Trajectory]]],
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = False,
    max_exceptions: int | float,
) -> list[list[Trajectory] | BaseException]: ...


async def gather_trajectories(
    trajectories: (
        Iterable[Awaitable[Trajectory]] | Iterable[Awaitable[Iterable[Trajectory]]]
    ),
    *,
    pbar_desc: str | None = "gather",
    pbar_total_completion_tokens: bool = False,
    max_exceptions: int | float = 0,
) -> (
    list[Trajectory]
    | list[Trajectory | BaseException]
    | list[list[Trajectory]]
    | list[list[Trajectory] | BaseException]
):
    if pbar_total_completion_tokens:
        print(
            "pbar_total_completion_tokens is deprecated and will be removed in a future version."
        )
    trajectories_list = list(trajectories)
    context = GatherContext(
        pbar=tqdm.tqdm(desc=pbar_desc, total=len(trajectories_list)),
        max_exceptions=max_exceptions,
    )
    with set_gather_context(context):
        results = await asyncio.gather(
            *[wrap_trajectories_awaitable(t) for t in trajectories_list]
        )
    if context.pbar is not None:
        context.pbar.close()
    return results


async def wrap_group_awaitable(
    awaitable: Awaitable[TrajectoryGroup],
) -> TrajectoryGroup | None:
    if hasattr(awaitable, "_num_trajectories"):
        return await awaitable
    context = get_gather_context()
    try:
        group = await awaitable
        for trajectory in group:
            record_metrics(context, trajectory)
        context.update_pbar(n=len(group))
        return group
    except BaseException:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        if context.too_many_exceptions():
            raise


async def wrap_trajectories_awaitable(
    awaitable: Awaitable[Trajectory] | Awaitable[Iterable[Trajectory]],
) -> Trajectory | list[Trajectory] | BaseException:
    context = get_gather_context()
    try:
        result = await awaitable
        if isinstance(result, Trajectory):
            record_metrics(context, result)
            context.update_pbar(n=1)
            return result
        result = list(result)
        for trajectory in result:
            record_metrics(context, trajectory)
        context.update_pbar(n=1)
        return result
    except BaseException as e:
        context.metric_sums["exceptions"] += 1
        context.update_pbar(n=0)
        if context.too_many_exceptions():
            raise
        else:
            return e


def record_metrics(context: "GatherContext", trajectory: Trajectory) -> None:
    logprobs = [
        message_or_choice.logprobs
        for message_or_choice in trajectory.messages_and_choices
        if isinstance(message_or_choice, Choice)
        if message_or_choice.logprobs
    ]
    if logprobs:
        # TODO: probably shouldn't average this
        trajectory.metrics["completion_tokens"] = sum(
            len(l.content or l.refusal or [])
            for l in logprobs  # noqa: E741
        ) / len(logprobs)
    context.metric_sums["reward"] += trajectory.reward
    context.metric_divisors["reward"] += 1
    context.metric_sums.update(trajectory.metrics)
    context.metric_divisors.update(trajectory.metrics.keys())


@dataclass
class GatherContext:
    pbar: tqdm.tqdm | None = None
    metric_sums: Counter[str] = field(default_factory=Counter)
    metric_divisors: Counter[str] = field(default_factory=Counter)
    max_metrics: int | None = None
    max_exceptions: int | float = 0
    increment_pbar: bool = True

    def update_pbar(self, n: int) -> None:
        if self.pbar is None:
            return
        if self.increment_pbar:
            self.pbar.update(n)
        postfix = {}
        included_metrics = self.metric_sums.keys()
        if self.max_metrics is not None:
            included_metrics = list(self.metric_sums.keys())[: self.max_metrics]
        for metric in included_metrics:
            sum = self.metric_sums[metric]
            divisor = max(1, self.metric_divisors[metric])
            postfix[metric] = sum / divisor
        if "completion_tokens" in postfix:
            postfix["completion_tokens"] = postfix.pop("completion_tokens")
        self.pbar.set_postfix(postfix)

    def too_many_exceptions(self) -> bool:
        if (
            0 < self.max_exceptions < 1
            and self.pbar is not None
            and self.metric_sums["exceptions"] / self.pbar.total <= self.max_exceptions  # ty:ignore[unsupported-operator]
        ) or self.metric_sums["exceptions"] <= self.max_exceptions:
            return False
        return True

    def reset(self) -> None:
        self.pbar = None
        self.metric_sums = Counter()
        self.metric_divisors = Counter()
        self.max_exceptions = 0


gather_context_var = contextvars.ContextVar("gather_context", default=GatherContext())


@contextlib.contextmanager
def set_gather_context(context: GatherContext) -> Iterator[None]:
    token = gather_context_var.set(context)
    try:
        yield
    finally:
        gather_context_var.reset(token)


def get_gather_context() -> GatherContext:
    return gather_context_var.get()
