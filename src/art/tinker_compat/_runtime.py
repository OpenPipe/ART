from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future
import threading
from typing import Any, TypeVar

from tinker import APIFuture

from art.utils.lifecycle import process_shutdown_timeout

T = TypeVar("T")


class ConcurrentAPIFuture(APIFuture[T]):
    def __init__(self, future: Future[T]) -> None:
        self._future = future

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        async with asyncio.timeout(timeout):
            return await asyncio.wrap_future(self._future)

    def future(self) -> Future[T]:
        return self._future


class AsyncRuntime:
    def __init__(self) -> None:
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="art-tinker-compat",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def submit(self, coroutine: Coroutine[Any, Any, T]) -> APIFuture[T]:
        return ConcurrentAPIFuture(self.submit_future(coroutine))

    def submit_future(self, coroutine: Coroutine[Any, Any, T]) -> Future[T]:
        if self._closed:
            coroutine.close()
            raise RuntimeError("Tinker compatibility client is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def stop(self, timeout_s: float = process_shutdown_timeout(2)) -> None:
        if timeout_s < 0:
            raise ValueError("runtime stop timeout cannot be negative")
        if self._closed and not self._thread.is_alive():
            return
        if not self._closed:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("Tinker compatibility runtime did not stop in time")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()
