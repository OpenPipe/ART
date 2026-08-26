from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event, Lock, RLock
import time
from types import SimpleNamespace
from typing import Any

import pytest

from art.megatron.runtime import monarch
from art.megatron.runtime.executor import MCoreRunSlotExecutor, _GenerationPublisher
from art.megatron.runtime.monarch import MonarchTrainerActor, MonarchTrainerSlot
from art.megatron.runtime.run_residency import RunResidencyManager


class _CloseTarget:
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        closed: bool = True,
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.closed = closed
        self.error = error

    def close(self, *, deadline: float) -> None:
        assert deadline >= 0.0
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error


class _Executor(_CloseTarget):
    def discard_open_gradients(self) -> None:
        self.calls.append("discard_open_gradients")
        raise RuntimeError("discard failed")


class _WeightOffload:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def after_job(self) -> None:
        self.calls.append("finish_weight_offload_job")
        raise RuntimeError("offload failed")


def _rank_actor(calls: list[str]) -> MonarchTrainerActor:
    actor = MonarchTrainerActor.__new__(MonarchTrainerActor)
    actor._teardown_lock = Lock()
    actor._teardown_complete = False
    actor._teardown_poisoned = False
    actor._valid = True
    actor._command_job_open = True
    actor._deferred_response_lock = Lock()
    actor._deferred_response_threads = set()
    actor._residency_prefetch_thread = None
    actor._cp_lookahead_thread = None
    actor._stop_deferred_responses = lambda _deadline: calls.append(
        "deferred_responses"
    )
    actor._stop_residency_prefetch = lambda _deadline: calls.append(
        "residency_prefetch"
    )
    actor._stop_cp_lookahead = lambda _deadline: calls.append("cp_lookahead")
    actor._weight_offload = _WeightOffload(calls)
    actor._run_slot_executor = _CloseTarget(
        "run_slot_executor",
        calls,
        error=RuntimeError("run-slot failed"),
    )
    actor._executor = _Executor(
        "base_executor", calls, error=RuntimeError("executor failed")
    )
    return actor


def _exception_messages(error: BaseException) -> list[str]:
    if isinstance(error, BaseExceptionGroup):
        return [
            message
            for nested in error.exceptions
            for message in _exception_messages(nested)
        ]
    return [str(error)]


def test_rank_teardown_accumulates_failures_and_destroys_process_group_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    actor = _rank_actor(calls)
    monkeypatch.setattr(
        monarch,
        "_destroy_rank_process_group",
        lambda: calls.append("destroy_process_group"),
    )
    monkeypatch.setattr(monarch, "_rank_process_group_is_initialized", lambda: False)

    with pytest.raises(BaseExceptionGroup) as raised:
        actor._rank_teardown(1.0)

    assert calls == [
        "deferred_responses",
        "discard_open_gradients",
        "finish_weight_offload_job",
        "residency_prefetch",
        "run_slot_executor",
        "cp_lookahead",
        "base_executor",
        "destroy_process_group",
    ]
    assert _exception_messages(raised.value) == [
        "discard failed",
        "offload failed",
        "run-slot failed",
        "executor failed",
    ]
    assert actor._teardown_complete
    assert actor._teardown_poisoned
    assert not actor._valid


def test_rank_with_retained_transition_is_poisoned_and_close_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    actor = _rank_actor(calls)
    actor._command_job_open = False
    actor._run_slot_executor = _CloseTarget("run_slot_executor", calls, closed=False)
    actor._executor = _CloseTarget("base_executor", calls)
    monkeypatch.setattr(
        monarch,
        "_destroy_rank_process_group",
        lambda: calls.append("destroy_process_group"),
    )
    monkeypatch.setattr(monarch, "_rank_process_group_is_initialized", lambda: False)

    with pytest.raises(TimeoutError, match="run-slot executor"):
        actor._rank_teardown(1.0)
    assert not actor._teardown_complete
    assert actor._teardown_poisoned

    with pytest.raises(TimeoutError, match="run-slot executor"):
        actor._rank_teardown(1.0)
    assert calls.count("run_slot_executor") == 2
    assert calls.count("destroy_process_group") == 2


def test_partially_initialized_rank_still_destroys_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    actor = MonarchTrainerActor.__new__(MonarchTrainerActor)
    actor._teardown_lock = Lock()
    actor._teardown_complete = False
    actor._teardown_poisoned = False
    actor._valid = False
    actor._executor = None
    actor._run_slot_executor = None
    actor._weight_offload = None
    actor._command_job_open = False
    actor._deferred_response_lock = Lock()
    actor._deferred_response_threads = set()
    actor._deferred_response_stopping = False
    actor._cp_lookahead_port = None
    actor._cp_lookahead_thread = None
    actor._residency_prefetch_port = None
    actor._residency_prefetch_thread = None
    monkeypatch.setattr(
        monarch,
        "_destroy_rank_process_group",
        lambda: calls.append("destroy_process_group"),
    )
    monkeypatch.setattr(monarch, "_rank_process_group_is_initialized", lambda: False)

    actor._rank_teardown(1.0)

    assert calls == ["destroy_process_group"]
    assert actor._teardown_complete
    assert not actor._teardown_poisoned


def _residency_manager() -> RunResidencyManager:
    manager = RunResidencyManager.__new__(RunResidencyManager)
    manager.config = SimpleNamespace(shutdown_timeout_s=0.02)
    manager._lock = RLock()
    manager._transition_slots = Condition(manager._lock)
    manager._pool = ThreadPoolExecutor(max_workers=1)
    manager._futures = set()
    manager._states = {}
    manager._failures = []
    manager._active_transitions = 0
    manager._closing = False
    manager._closed = False
    return manager


def test_residency_close_never_reports_closed_with_a_live_worker() -> None:
    manager = _residency_manager()
    started, release = Event(), Event()

    def block() -> None:
        started.set()
        release.wait()

    future = manager._submit(block)
    assert started.wait(1.0)
    try:
        with pytest.raises(TimeoutError, match="worker futures"):
            manager.close(deadline=time.monotonic() + 0.02)
        assert not manager.closed
        assert manager._closing
    finally:
        release.set()
        future.result(timeout=1.0)
    manager.close(deadline=time.monotonic() + 1.0)
    assert manager.closed
    assert not manager._closing


def _publisher() -> _GenerationPublisher:
    publisher = _GenerationPublisher.__new__(_GenerationPublisher)
    publisher._lock = Lock()
    publisher._resolution_pool = ThreadPoolExecutor(max_workers=1)
    publisher._sampler_pool = ThreadPoolExecutor(max_workers=1)
    publisher._transport_pool = ThreadPoolExecutor(max_workers=1)
    publisher._ordered_transport_pool = ThreadPoolExecutor(max_workers=1)
    publisher._durability_pool = ThreadPoolExecutor(max_workers=1)
    publisher._completion_pool = ThreadPoolExecutor(max_workers=1)
    publisher._work = set()
    publisher._prepared = {}
    publisher._cache = {}
    publisher._residency_retirements = set()
    publisher._failures = []
    publisher._in_flight = 0
    publisher._transport_sender = None
    publisher._object_store = None
    publisher._residency = None
    publisher._closing = False
    publisher._closed = False
    return publisher


def test_publisher_close_never_reports_closed_with_a_live_worker() -> None:
    publisher = _publisher()
    started, release = Event(), Event()

    def block() -> None:
        started.set()
        release.wait()

    future = publisher._submit(publisher._resolution_pool, block)
    assert started.wait(1.0)
    try:
        with pytest.raises(TimeoutError, match="worker futures"):
            publisher.close(deadline=time.monotonic() + 0.02)
        assert not publisher.closed
        assert publisher._closing
    finally:
        release.set()
        future.result(timeout=1.0)
    publisher.close(deadline=time.monotonic() + 1.0)
    assert publisher.closed
    assert not publisher._closing


def _run_slot_executor(calls: list[str]) -> MCoreRunSlotExecutor:
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor._lifecycle_lock = Lock()
    executor._residency_admission_lock = Lock()
    executor._residency_admissions = {}
    executor._transition_futures = set()
    executor._load_pool = ThreadPoolExecutor(max_workers=1)
    executor._cleanup_pool = ThreadPoolExecutor(max_workers=1)
    executor._load_preparations = {}
    executor._registration_preparations = {}
    executor._run_cleanups = {}
    executor._runs = {}
    executor._publisher = _CloseTarget("publisher", calls)
    executor._residency = _CloseTarget("residency", calls)
    executor._residency.config = SimpleNamespace(shutdown_timeout_s=0.02)
    executor._closing = False
    executor._closed = False
    return executor


def test_run_slot_close_tracks_uncancellable_transition_and_closes_children() -> None:
    calls: list[str] = []
    executor = _run_slot_executor(calls)
    started, release = Event(), Event()

    def block() -> None:
        started.set()
        release.wait()

    future = executor._submit_transition(executor._load_pool, block)
    assert started.wait(1.0)
    try:
        with pytest.raises(BaseExceptionGroup):
            executor.close(deadline=time.monotonic() + 0.02)
        assert calls == ["publisher", "residency"]
        assert not executor.closed
        assert executor._closing
    finally:
        release.set()
        future.result(timeout=1.0)
    executor.close(deadline=time.monotonic() + 1.0)
    assert calls == ["publisher", "residency", "publisher", "residency"]
    assert executor.closed


@pytest.mark.asyncio
async def test_slot_deadline_uses_remaining_budget_to_terminate_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_timeout: list[float] = []
    forced_timeout: list[float] = []

    async def hang() -> None:
        await asyncio.Event().wait()

    class _ActorClose:
        def call(self, timeout_s: float) -> Any:
            actor_timeout.append(timeout_s)
            return hang()

    slot = MonarchTrainerSlot.__new__(MonarchTrainerSlot)
    slot._actors = SimpleNamespace(close=_ActorClose())
    slot._publication_authorizations = {}
    slot._registration_tasks = {}
    slot._removal_tasks = {}
    slot._publications = {}
    slot._shutdown_timeout_s = 1.0
    slot._valid = True
    slot._closed = False
    slot._close_task = None

    async def force_stop(timeout_s: float) -> None:
        forced_timeout.append(timeout_s)

    slot._force_stop = force_stop
    monkeypatch.setattr(
        monarch,
        "process_shutdown_timeout",
        lambda level: {1: 0.05, 2: 0.03}[level],
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await slot.close()
    elapsed = time.monotonic() - started

    assert 0.0 < actor_timeout[0] <= 0.03
    assert 0.0 < forced_timeout[0] <= 0.03
    assert elapsed < 0.08
    assert slot._closed
    assert not slot._valid
