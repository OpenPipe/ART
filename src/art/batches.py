import asyncio
from typing import AsyncIterator, Awaitable, Iterable

from tqdm import auto as tqdm

from .gather import GatherContext, set_gather_context, wrap_group_awaitable
from .trajectories import TrajectoryGroup


async def trajectory_group_batches(
    groups: Iterable[Awaitable[TrajectoryGroup]],
    *,
    batch_size: int,
    max_batch_exceptions: int | float = 0,
    max_concurrent_batches: int = 4,
    skip_batches: int = 0,
    pbar_desc: str | None = "batches",
    pbar_total_completion_tokens: bool = False,
) -> AsyncIterator[list[TrajectoryGroup]]:
    if pbar_total_completion_tokens:
        print(
            "pbar_total_completion_tokens is deprecated and will be removed in a future version."
        )
    unstarted = list(groups)[batch_size * skip_batches :]
    pending = set[asyncio.Task[TrajectoryGroup | None]]()
    batch = list[TrajectoryGroup]()
    context = GatherContext(
        max_exceptions=max_batch_exceptions,
        increment_pbar=False,
    )
    with set_gather_context(context):
        while unstarted or pending:
            if context.pbar is None:
                context.pbar = tqdm.tqdm(desc=pbar_desc, total=batch_size)
            while len(pending) < batch_size * max_concurrent_batches and unstarted:
                pending.add(asyncio.create_task(wrap_group_awaitable(unstarted.pop(0))))
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            context.pbar.update(len(done))
            batch.extend(g for task in done if (g := task.result()) is not None)
            if len(batch) >= batch_size:
                if context.pbar is not None:
                    context.pbar.close()
                    context.reset()
                yield batch[:batch_size]
                batch = batch[batch_size:]
        if batch:
            yield batch
