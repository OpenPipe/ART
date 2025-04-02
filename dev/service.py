import asyncio
from typing import AsyncIterator


class Service:
    async def load_unsloth(self) -> None:
        import unsloth  # type: ignore

    async def greet(self, name: str, sleep: float) -> str:
        await asyncio.sleep(sleep)
        return f"Hello, {name}!"

    def raise_error(self) -> None:
        raise ValueError("This is a test error")

    async def async_iterator(self) -> AsyncIterator[int]:
        for i in range(10):
            await asyncio.sleep(1)
            yield i
