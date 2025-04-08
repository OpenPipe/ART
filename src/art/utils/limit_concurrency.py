import asyncio
from functools import wraps
from typing import Callable, Optional


def limit_concurrency(n: int, derive_key: Optional[Callable[..., str]] = None):
    semaphores = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if derive_key:
                key = derive_key(*args, **kwargs)
            else:
                key = "default"

            if key not in semaphores:
                semaphores[key] = asyncio.Semaphore(n)

            async with semaphores[key]:
                return await func(*args, **kwargs)

        return wrapper

    return decorator
