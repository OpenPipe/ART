import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from typing import AsyncIterator

from art.adapter_leases import pin_inference_step, pinned_inference_step


class AdapterLeaseManager:
    def __init__(self) -> None:
        self._counts: Counter[int] = Counter()
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, step: int) -> AsyncIterator[None]:
        async with self._condition:
            self._counts[step] += 1
        try:
            yield
        finally:
            async with self._condition:
                self._counts[step] -= 1
                if self._counts[step] <= 0:
                    del self._counts[step]
                self._condition.notify_all()

    def active_steps(self) -> set[int]:
        return set(self._counts)
