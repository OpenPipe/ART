from __future__ import annotations

import asyncio
from typing import Any


class RolloutWorkerController:
    def __init__(self, trainer: Any, target_workers: int) -> None:
        self.trainer = trainer
        self.target_workers = target_workers
        self._next_worker_id = 0
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._retiring: set[int] = set()

    def set_target(self, target_workers: int) -> None:
        self.target_workers = max(1, int(target_workers))

    def worker_allowed(self, worker_id: int) -> bool:
        return worker_id not in self._retiring

    async def run(self) -> None:
        try:
            while not self.trainer.state.done:
                self._reconcile()
                await self._raise_finished_errors()
                await asyncio.sleep(0.25)
        finally:
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def _reconcile(self) -> None:
        for worker_id, task in list(self._tasks.items()):
            if task.done():
                self._tasks.pop(worker_id)
                self._retiring.discard(worker_id)

        live = [wid for wid, task in self._tasks.items() if not task.done()]
        self._retiring = set(live[self.target_workers :])

        active = [wid for wid in live if wid not in self._retiring]
        while len(active) < self.target_workers and not self.trainer.state.done:
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            task = asyncio.create_task(
                self.trainer._rollout_worker(worker_id),
                name=f"art_rollout_worker_{worker_id}",
            )
            self._tasks[worker_id] = task
            active.append(worker_id)
        # Retiring workers keep their endpoint until their acquired scenario is
        # delivered; worker_allowed() prevents them from acquiring another.
        self.trainer._rollout_executor.set_workers(tuple(live))

    async def _raise_finished_errors(self) -> None:
        for worker_id, task in list(self._tasks.items()):
            if not task.done():
                continue
            self._tasks.pop(worker_id)
            self._retiring.discard(worker_id)
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc
