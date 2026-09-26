from __future__ import annotations

import asyncio
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from test_parallel_tokenize import _exchange_trajectory

import art
from art.trajectories import _parallel


def _held_initializer(barrier: Any, counter: Any, release: Any) -> None:
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index and not release.wait(15):
        raise RuntimeError("public delayed initializer timed out")
    if barrier is not None:
        _parallel._initialize_process_worker(barrier)


def _old_identity() -> int:
    time.sleep(0.25)
    return os.getpid()


def _failed_initializer() -> None:
    raise RuntimeError("public initializer failure")


def test_short_tasks_are_not_a_worker_census() -> None:
    context = _parallel._process_context()
    release = context.Event()
    counter = context.Value("i", 0)
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=context,
        initializer=_held_initializer,
        initargs=(None, counter, release),
    ) as pool:
        try:
            futures = [pool.submit(_old_identity) for _ in range(2)]
            assert len({f.result(timeout=20) for f in futures}) == 1
        finally:
            release.set()


async def test_skewed_startup_and_cancelled_waiter_keep_true_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parallel._shutdown_process_executor(grace=0)
    context = _parallel._process_context()
    release = context.Event()
    counter = context.Value("i", 0)
    created: list[Any] = []

    def factory(**kwargs: Any) -> ProcessPoolExecutor:
        assert kwargs["initializer"] is _parallel._initialize_process_worker
        kwargs["initializer"] = _held_initializer
        kwargs["initargs"] = (*kwargs["initargs"], counter, release)
        pool = ProcessPoolExecutor(**kwargs)
        created.append(pool)
        return pool

    monkeypatch.setattr(_parallel, "ProcessPoolExecutor", factory)
    try:
        pool, capacity, futures = _parallel._start_process_executor(2)
        task = asyncio.create_task(
            asyncio.to_thread(_parallel._finish_process_warmup, futures, capacity)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 20
        while counter.value < 2:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.6)  # Longer than two old 0.25-second identity tasks.
        assert not any(f.done() for f in futures)
        release.set()
        await asyncio.wait_for(
            asyncio.to_thread(_parallel._finish_process_warmup, futures, capacity), 30
        )
        assert len({f.result() for f in futures}) == capacity == 2
        _parallel._complete_process_warmup(pool, futures)
        assert _parallel._process_executor(2) is pool
        assert list(pool.map(abs, [-2, -1])) == [2, 1]
    finally:
        release.set()
        workers = [p for pool in created for p in (pool._processes or {}).values()]
        _parallel._shutdown_process_executor(grace=2)
        assert all(not p.is_alive() for p in workers)


def test_initializer_failure_propagates_and_pool_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parallel._shutdown_process_executor(grace=0)

    def factory(**kwargs: Any) -> ProcessPoolExecutor:
        kwargs["initializer"] = _failed_initializer
        kwargs["initargs"] = ()
        return ProcessPoolExecutor(**kwargs)

    monkeypatch.setattr(_parallel, "ProcessPoolExecutor", factory)
    try:
        _, capacity, futures = _parallel._start_process_executor(2)
        with pytest.raises(BrokenProcessPool):
            _parallel._finish_process_warmup(futures, capacity)
    finally:
        _parallel._shutdown_process_executor(grace=0)


def test_missing_initializer_and_barrier_timeout_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_PROCESS_READY_BARRIER", None)
    with pytest.raises(RuntimeError, match="barrier is missing"):
        _parallel._process_identity()
    monkeypatch.setattr(
        _parallel, "_PROCESS_READY_BARRIER", threading.Barrier(2, timeout=0.02)
    )
    with pytest.raises(threading.BrokenBarrierError):
        _parallel._process_identity()


def test_duplicate_pid_still_refuses() -> None:
    futures: tuple[Future[int], ...] = (Future(), Future())
    for future in futures:
        future.set_result(123)
    with pytest.raises(RuntimeError, match="started 1 process workers, expected 2"):
        _parallel._finish_process_warmup(futures, 2)


def test_parent_wait_has_one_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []
    clock = iter([100.0, 101.0, 104.0])
    monkeypatch.setattr(
        _parallel, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    monkeypatch.setattr(_parallel, "_PROCESS_STARTUP_TIMEOUT_SECONDS", 5.0)

    class Ready:
        def result(self, *, timeout: float) -> int:
            observed.append(timeout)
            return len(observed)

    _parallel._finish_process_warmup((Ready(), Ready()), 2)  # type: ignore[arg-type]
    assert observed == [4.0, 1.0]


def test_parent_timeout_when_no_worker_reaches_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_PROCESS_STARTUP_TIMEOUT_SECONDS", 0.02)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _parallel._finish_process_warmup((Future(),), 1)
    assert time.monotonic() - start < 1.0


async def test_public_timeout_falls_back_without_changing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_PROCESS_BACKEND_DISABLED", False)
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda *_, **__: True)
    monkeypatch.setattr(_parallel, "_cpu_capacity", lambda: 2)

    def unavailable(_: int) -> Any:
        raise TimeoutError("public bounded startup timeout")

    monkeypatch.setattr(_parallel, "_start_process_executor", unavailable)
    trajectories = [_exchange_trajectory(i) for i in range(4)]
    with pytest.warns(RuntimeWarning, match="using threads"):
        result = await art.tokenize(trajectories, model="test/model")
    assert [value.tokens for value in result] == [[1, 2], [1, 3], [1, 4], [1, 5]]
    assert [value.trajectory for value in result] == trajectories
    assert _parallel._PROCESS_BACKEND_DISABLED


async def test_timeout_stops_exact_spawned_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parallel._shutdown_process_executor(grace=0)
    monkeypatch.setattr(_parallel, "_PROCESS_STARTUP_TIMEOUT_SECONDS", 0.05)
    original = _parallel._submit_process_warmup
    workers: list[Any] = []

    def submit(pool: Any, capacity: int) -> Any:
        result = original(pool, capacity)
        workers.extend(pool._processes.values())
        return result

    monkeypatch.setattr(_parallel, "_submit_process_warmup", submit)
    try:
        with pytest.raises(_parallel._ProcessBackendError, match="TimeoutError"):
            await _parallel._ordered_process_map([], [], workers=2, capacity=2)
        assert len(workers) == 2 and all(not p.is_alive() for p in workers)
    finally:
        _parallel._shutdown_process_executor(grace=0)


def test_partial_submit_failure_stops_unpublished_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parallel._shutdown_process_executor(grace=0)
    original = _parallel._submit_process_warmup
    workers: list[Any] = []
    failure = RuntimeError("public submission failure")

    def submit(pool: Any, capacity: int) -> Any:
        original(pool, capacity)
        workers.extend(pool._processes.values())
        raise failure

    monkeypatch.setattr(_parallel, "_submit_process_warmup", submit)
    with pytest.raises(RuntimeError) as caught:
        _parallel._start_process_executor(2)
    assert caught.value is failure
    assert len(workers) == 2 and all(not p.is_alive() for p in workers)
    assert _parallel._PROCESS_EXECUTOR is None


def test_failed_pool_cleanup_does_not_release_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = object()
    monkeypatch.setattr(_parallel, "_PROCESS_EXECUTOR", replacement)
    closed: list[dict[str, bool]] = []

    class Failed:
        _processes: dict[int, Any] = {}

        def shutdown(self, **kwargs: bool) -> None:
            closed.append(kwargs)

    _parallel._shutdown_process_executor(0, Failed())  # type: ignore[arg-type]
    assert _parallel._PROCESS_EXECUTOR is replacement
    assert closed == [{"wait": False, "cancel_futures": True}]
