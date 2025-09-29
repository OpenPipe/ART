import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, ParamSpec, TypeVar

executor = ThreadPoolExecutor(max_workers=1024)

P = ParamSpec("P")
R = TypeVar("R")


async def run(
    func: Callable[P, R],
    in_thread: bool,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    if in_thread:
        return await asyncio.get_running_loop().run_in_executor(
            executor, partial(func, *args, **kwargs)
        )
    return func(*args, **kwargs)
