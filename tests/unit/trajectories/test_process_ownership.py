from __future__ import annotations

import asyncio
from concurrent.futures import Future
import os
import threading
import time
from typing import Any

import pytest
from test_parallel_tokenize import _exchange_trajectory

import art
from art.trajectories import _parallel


@pytest.fixture(autouse=True)
def clean_pools(monkeypatch: pytest.MonkeyPatch):
    _parallel._shutdown_process_executor(0)
    monkeypatch.setattr(_parallel, "_PROCESS_BACKEND_DISABLED", False)
    yield
    _parallel._shutdown_process_executor(0)


async def test_failed_waiter_preserves_ready_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = threading.Event()
    release = threading.Event()
    old: list[Any] = []
    original_start = _parallel._start_process_executor
    original_finish = _parallel._finish_process_warmup

    def start(capacity: int) -> Any:
        result = original_start(capacity)
        if not old:
            old.append(result)
            created.set()
        return result

    def finish(futures: Any, capacity: int) -> None:
        if old and futures is old[0][2]:
            assert release.wait(45)
            raise TimeoutError("old waiter readiness deadline")
        original_finish(futures, capacity)

    monkeypatch.setattr(_parallel, "_start_process_executor", start)
    monkeypatch.setattr(_parallel, "_finish_process_warmup", finish)
    monkeypatch.setattr(_parallel, "_cpu_capacity", lambda: 2)
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_: True)
    trajectories = [_exchange_trajectory(i) for i in range(4)]
    task = asyncio.create_task(art.tokenize(trajectories, model="test/model"))
    try:
        deadline = time.monotonic() + 45
        while not created.is_set():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        replacement, capacity, futures = original_start(4)
        await asyncio.to_thread(original_finish, futures, capacity)
        _parallel._complete_process_warmup(replacement, futures)
        release.set()
        with pytest.warns(RuntimeWarning, match="process tokenization is unavailable"):
            result = await asyncio.wait_for(task, 45)
        assert [value.trajectory for value in result] == trajectories
        assert _parallel._PROCESS_EXECUTOR is replacement
        assert await asyncio.wrap_future(replacement.submit(abs, -1)) == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_exit_hook_reaps_retired_busy_pool() -> None:
    old = _parallel._process_executor(2)
    old_workers = _parallel._process_executor_workers(old)
    old_manager = getattr(old, "_executor_manager_thread")
    busy = old.submit(time.sleep, 120)
    replacement = _parallel._process_executor(4)
    workers = old_workers + _parallel._process_executor_workers(replacement)
    managers = [old_manager, getattr(replacement, "_executor_manager_thread")]
    assert old in _parallel._PROCESS_EXECUTORS
    assert old.submit(abs, -1).result(timeout=10) == 1
    started = time.monotonic()
    _parallel._shutdown_process_executor(0.1)
    assert time.monotonic() - started < 8
    assert all(not worker.is_alive() for worker in workers)
    assert all(not manager.is_alive() for manager in managers)
    assert busy.done()
    assert not _parallel._PROCESS_EXECUTORS


@pytest.mark.parametrize("mode", ["closed", "cancelled"])
async def test_shared_pool_shutdown_falls_back_without_cancelling_caller(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    class ClosedPool:
        _processes: dict[int, Any] = {}
        _shutdown_thread = True
        closed = False

        def submit(self, *_: Any) -> Future[bytes]:
            if mode == "closed":
                raise RuntimeError("cannot schedule new futures after shutdown")
            result: Future[bytes] = Future()
            result.cancel()
            return result

        def shutdown(self, **_: Any) -> None:
            self.closed = True

    pool = ClosedPool()
    monkeypatch.setattr(_parallel, "_PROCESS_EXECUTORS", {pool: os.getpid()})
    monkeypatch.setattr(_parallel, "_start_process_executor", lambda _: (pool, 2, ()))
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_: True)
    values = [_exchange_trajectory(i) for i in range(4)]
    with pytest.warns(RuntimeWarning, match="process tokenization is unavailable"):
        result = await art.tokenize(values, model="test/model")
    assert [value.trajectory for value in result] == values
    assert pool.closed


async def test_external_cancellation_stays_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending: Future[bytes] = Future()
    submitted = asyncio.Event()

    class Pool:
        _shutdown_thread = True

        def submit(self, *_: Any) -> Future[bytes]:
            submitted.set()
            return pending

    monkeypatch.setattr(_parallel, "_start_process_executor", lambda _: (Pool(), 2, ()))
    task = asyncio.create_task(
        _parallel._ordered_process_map(
            [b"pending"], [_exchange_trajectory(0)], workers=1, capacity=2
        )
    )
    await asyncio.wait_for(submitted.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_terminal_cleanup_does_not_join_foreign_parent_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForeignPool:
        def shutdown(self, **_: Any) -> None:
            pytest.fail("must not shut down inherited parent executor")

    pool = ForeignPool()
    monkeypatch.setattr(_parallel, "_PROCESS_EXECUTORS", {pool: os.getpid() + 1})
    _parallel._shutdown_process_executor(0)
    assert not _parallel._PROCESS_EXECUTORS


async def test_user_cancellation_exception_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = asyncio.CancelledError("user tokenization exception")

    class Pool:
        _shutdown_thread = True

        def submit(self, *_: Any) -> Future[bytes]:
            future: Future[bytes] = Future()
            future.set_exception(error)
            assert not future.cancelled()
            return future

    pool = Pool()
    monkeypatch.setattr(_parallel, "_start_process_executor", lambda _: (pool, 2, ()))
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_: True)
    with pytest.raises(asyncio.CancelledError):
        await art.tokenize(
            [_exchange_trajectory(i) for i in range(4)], model="test/model"
        )
    assert not _parallel._PROCESS_BACKEND_DISABLED


@pytest.mark.parametrize("first_cleanup", ["exact", "all", "unknown"])
def test_foreign_or_unknown_pool_stays_unowned_after_cleanup(
    monkeypatch: pytest.MonkeyPatch, first_cleanup: str
) -> None:
    class ForeignPool:
        def shutdown(self, **_: Any) -> None:
            pytest.fail("must not shut down an executor without local ownership")

    pool: Any = ForeignPool()
    monkeypatch.setattr(
        _parallel,
        "_PROCESS_EXECUTORS",
        {} if first_cleanup == "unknown" else {pool: os.getpid() + 1},
    )
    if first_cleanup == "all":
        _parallel._shutdown_process_executor(0)
    else:
        _parallel._shutdown_process_executor(0, pool)
    _parallel._shutdown_process_executor(0, pool)
    assert not _parallel._PROCESS_EXECUTORS
