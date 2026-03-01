"""Readable regression tests for mp_actors proxy behavior."""

import asyncio
import os
from typing import Any

import pytest

from mp_actors import close_proxy, move_to_child_process


class DemoService:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> int:
        self.value += 1
        return self.value

    async def aincrement(self) -> int:
        self.value += 1
        await asyncio.sleep(0)
        return self.value

    async def ping(self) -> str:
        await asyncio.sleep(0)
        return "pong"

    async def ticker(self, n: int = 1_000):
        for i in range(n):
            await asyncio.sleep(0)
            yield i

    async def slow(self, delay: float = 5.0) -> str:
        await asyncio.sleep(delay)
        return "done"

    async def raise_error(self, message: str) -> None:
        await asyncio.sleep(0)
        raise ValueError(message)


class ExitService:
    async def child_exit(self) -> None:
        await asyncio.sleep(0)
        os._exit(17)

    async def ping(self) -> str:
        await asyncio.sleep(0)
        return "pong"


async def _wait_for_count(values: list[int], target: int, timeout: float = 2.0) -> None:
    start = asyncio.get_event_loop().time()
    while len(values) < target:
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError(f"timed out waiting for {target} values")
        await asyncio.sleep(0.001)


async def test_proxy_supports_sync_async_and_attribute_access() -> None:
    proxy: Any = move_to_child_process(
        DemoService(), process_name="test-mp-actors-basic"
    )
    try:
        assert proxy.increment() == 1
        assert await proxy.aincrement() == 2
        assert proxy.value == 2
        assert await proxy.ping() == "pong"
    finally:
        close_proxy(proxy)


async def test_child_exit_error_is_sticky_for_followup_calls() -> None:
    proxy: Any = move_to_child_process(
        ExitService(), process_name="test-mp-actors-exit"
    )
    try:
        with pytest.raises(RuntimeError, match="exited with code 17"):
            await proxy.child_exit()

        # A second request should fail fast with the same process-death error.
        with pytest.raises(RuntimeError, match="exited with code 17"):
            await asyncio.wait_for(proxy.ping(), timeout=2.0)
    finally:
        close_proxy(proxy)


async def test_async_generator_cancellation_does_not_break_future_calls() -> None:
    proxy: Any = move_to_child_process(
        DemoService(), process_name="test-mp-actors-asyncgen-cancel"
    )
    try:
        seen: list[int] = []

        async def consume() -> None:
            async for value in proxy.ticker(10_000):
                seen.append(value)

        task = asyncio.create_task(consume())
        await _wait_for_count(seen, target=20)
        task.cancel()
        cancelled = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(cancelled[0], asyncio.CancelledError)

        assert await asyncio.wait_for(proxy.ping(), timeout=2.0) == "pong"
    finally:
        close_proxy(proxy)


async def test_close_fails_inflight_requests_and_is_idempotent() -> None:
    proxy: Any = move_to_child_process(
        DemoService(), process_name="test-mp-actors-close-race"
    )
    tasks = [asyncio.create_task(proxy.slow(5.0)) for _ in range(64)]
    await asyncio.sleep(0.05)

    close_proxy(proxy)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert results
    assert all(isinstance(r, RuntimeError) for r in results)
    assert {str(r) for r in results} == {"Proxy is closing"}

    # Idempotent close and predictable post-close behavior.
    close_proxy(proxy)
    with pytest.raises(RuntimeError, match="Proxy is closing"):
        await proxy.ping()


async def test_child_exceptions_are_propagated_and_proxy_recovers() -> None:
    proxy: Any = move_to_child_process(
        DemoService(), process_name="test-mp-actors-errors"
    )
    try:
        with pytest.raises(ValueError, match="boom"):
            await proxy.raise_error("boom")

        assert await proxy.ping() == "pong"
    finally:
        close_proxy(proxy)
