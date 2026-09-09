from __future__ import annotations

import asyncio
import atexit
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from functools import lru_cache
import math
import multiprocessing
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from types import ModuleType
from typing import Any, Literal, TypeVar, cast
import warnings

from . import (
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectory,
    TokenizedTrajectoryGroup,
    Tokenizer,
    Trajectory,
    TrajectoryGroup,
)
from ._serialization import _rebind_history_sources, _without_pickle_string_interning

_ResultT = TypeVar("_ResultT")
_ValueT = TypeVar("_ValueT")
_InputKind = Literal["trajectory", "group"]
_Operation = Literal["tokenize", "tensorize"]
_PROCESS_MAX_WORKERS = 4
_PROCESS_MIN_ITEMS = 4
_PROCESS_MIN_THREAD_SECONDS = 1.0
_PROCESS_EXIT_GRACE_SECONDS = 5.0


def _cgroup_cpu_limit() -> int | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    return max(1, math.ceil(quota / period)) if quota > 0 and period > 0 else None


def _cpu_capacity() -> int:
    candidates = [os.cpu_count() or 1]
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if (limit := _cgroup_cpu_limit()) is not None:
        candidates.append(limit)
    return max(1, min(candidates))


_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_PID: int | None = None
_EXECUTOR_CAPACITY = 0


def _executor(capacity: int) -> ThreadPoolExecutor:
    global _EXECUTOR, _EXECUTOR_CAPACITY, _EXECUTOR_PID
    pid = os.getpid()
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None or _EXECUTOR_PID != pid or _EXECUTOR_CAPACITY < capacity:
            previous = _EXECUTOR if _EXECUTOR_PID == pid else None
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=capacity, thread_name_prefix="art-tokenize"
            )
            _EXECUTOR_PID = pid
            _EXECUTOR_CAPACITY = capacity
            if previous is not None:
                previous.shutdown(wait=False)
        return _EXECUTOR


_PROCESS_EXECUTOR_LOCK = threading.Lock()
_PROCESS_EXECUTOR: ProcessPoolExecutor | None = None
_PROCESS_EXECUTOR_PID: int | None = None
_PROCESS_EXECUTOR_CAPACITY = 0
_PROCESS_STARTUP: tuple[Future[int], ...] = ()
_PROCESS_BACKEND_DISABLED = False


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _process_identity() -> int:
    # Keep every submitted warmup occupied until the bounded pool is started.
    time.sleep(0.25)
    return os.getpid()


def _submit_process_warmup(
    executor: ProcessPoolExecutor, capacity: int
) -> tuple[Future[int], ...]:
    original_main = sys.modules.get("__main__")
    sys.modules["__main__"] = ModuleType("__main__")
    try:
        return tuple(executor.submit(_process_identity) for _ in range(capacity))
    finally:
        if original_main is None:
            del sys.modules["__main__"]
        else:
            sys.modules["__main__"] = original_main


def _finish_process_warmup(futures: tuple[Future[int], ...], capacity: int) -> None:
    if not futures:
        return
    worker_pids = {future.result() for future in futures}
    if len(worker_pids) != capacity:
        raise RuntimeError(
            f"started {len(worker_pids)} process workers, expected {capacity}"
        )


def _start_process_executor(
    capacity: int,
) -> tuple[ProcessPoolExecutor, int, tuple[Future[int], ...]]:
    global _PROCESS_EXECUTOR, _PROCESS_EXECUTOR_CAPACITY, _PROCESS_EXECUTOR_PID
    global _PROCESS_STARTUP
    process_capacity = min(_PROCESS_MAX_WORKERS, capacity)
    pid = os.getpid()
    with _PROCESS_EXECUTOR_LOCK:
        if (
            _PROCESS_EXECUTOR is None
            or _PROCESS_EXECUTOR_PID != pid
            or _PROCESS_EXECUTOR_CAPACITY < process_capacity
        ):
            previous = _PROCESS_EXECUTOR if _PROCESS_EXECUTOR_PID == pid else None
            executor = ProcessPoolExecutor(
                max_workers=process_capacity,
                mp_context=_process_context(),
            )
            try:
                startup = _submit_process_warmup(executor, process_capacity)
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            _PROCESS_EXECUTOR = executor
            _PROCESS_EXECUTOR_PID = pid
            _PROCESS_EXECUTOR_CAPACITY = process_capacity
            _PROCESS_STARTUP = startup
            if previous is not None:
                previous.shutdown(wait=False, cancel_futures=True)
        return (
            _PROCESS_EXECUTOR,
            _PROCESS_EXECUTOR_CAPACITY,
            _PROCESS_STARTUP,
        )


def _complete_process_warmup(
    executor: ProcessPoolExecutor, startup: tuple[Future[int], ...]
) -> None:
    global _PROCESS_STARTUP
    with _PROCESS_EXECUTOR_LOCK:
        if _PROCESS_EXECUTOR is executor and _PROCESS_STARTUP is startup:
            _PROCESS_STARTUP = ()


def _process_executor(capacity: int) -> ProcessPoolExecutor:
    executor, process_capacity, startup = _start_process_executor(capacity)
    _finish_process_warmup(startup, process_capacity)
    _complete_process_warmup(executor, startup)
    return executor


def _release_process_executor() -> ProcessPoolExecutor | None:
    global _PROCESS_EXECUTOR, _PROCESS_EXECUTOR_CAPACITY, _PROCESS_EXECUTOR_PID
    global _PROCESS_STARTUP
    with _PROCESS_EXECUTOR_LOCK:
        previous = _PROCESS_EXECUTOR
        owned = _PROCESS_EXECUTOR_PID == os.getpid()
        _PROCESS_EXECUTOR = None
        _PROCESS_EXECUTOR_PID = None
        _PROCESS_EXECUTOR_CAPACITY = 0
        _PROCESS_STARTUP = ()
    return previous if owned else None


def _discard_process_executor() -> None:
    previous = _release_process_executor()
    if previous is not None:
        previous.shutdown(wait=False, cancel_futures=True)


def _process_executor_workers(executor: ProcessPoolExecutor) -> list[BaseProcess]:
    processes = getattr(executor, "_processes", None)
    return list(processes.values()) if processes else []


def _shutdown_process_executor(grace: float | None = None) -> None:
    """Stop the shared process pool within a bounded time.

    Runs before concurrent.futures joins its workers at interpreter exit. Idle
    workers leave as soon as they read the shutdown sentinel; workers still busy
    with tensorization nobody can consume anymore are terminated after ``grace``
    seconds so the interpreter never waits on them indefinitely.
    """
    executor = _release_process_executor()
    if executor is None:
        return
    if grace is None:
        grace = _PROCESS_EXIT_GRACE_SECONDS
    workers = _process_executor_workers(executor)
    executor.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + max(0.0, grace)
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(1.0)
    for worker in workers:
        if worker.is_alive():
            worker.kill()
            worker.join(1.0)


def _register_process_exit_hook() -> None:
    # threading's private atexit list runs before concurrent.futures joins its
    # worker threads and processes, which is the only point early enough to
    # bound that join. Fall back to atexit where the hook is unavailable.
    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        try:
            register(_shutdown_process_executor)
            return
        except RuntimeError:
            return
    atexit.register(_shutdown_process_executor)


_register_process_exit_hook()


@dataclass
class _Measurement:
    units: int = 0
    seconds: float = 0
    samples: int = 0

    @property
    def rate(self) -> float:
        return self.units / self.seconds


@dataclass
class _TuningState:
    next_workers: int
    measurements: dict[int, _Measurement] = field(default_factory=dict)


_TUNING_LOCK = threading.Lock()


@lru_cache(maxsize=128)
def _tuning_state(key: tuple[object, ...]) -> _TuningState:
    return _TuningState(4)


@dataclass
class _ProcessTuningState(_TuningState):
    warmed: bool = False
    candidate: bool = False
    disabled: bool = False


@lru_cache(maxsize=128)
def _process_tuning_state(key: tuple[object, ...]) -> _ProcessTuningState:
    return _ProcessTuningState(_PROCESS_MAX_WORKERS)


def _size_bucket(size: int) -> int:
    return 1 << max(0, (size - 1).bit_length())


def _workload_bucket(values: Sequence[Trajectory]) -> int:
    branches = sum(
        len(value.exchanges.chat_completions)
        + len(value.exchanges.completions)
        + len(value.exchanges.responses)
        + len(value.exchanges.messages)
        + len(value.additional_histories)
        + bool(value.messages_and_choices)
        for value in values
    )
    return _size_bucket(max(1, math.ceil(branches / len(values))))


def _tuning_key(
    *,
    operation: _Operation,
    kind: _InputKind,
    values: Sequence[Trajectory],
    multi_history: bool,
    tokenizer: Tokenizer | None,
    model: str | None,
    base_model: str | None,
    chat_template: str | None,
    capacity: int,
) -> tuple[object, ...]:
    if tokenizer is None:
        tokenizer_key: tuple[object, ...] = ("automatic", base_model, model)
    else:
        tokenizer_type = type(tokenizer)
        tokenizer_key = (
            tokenizer_type.__module__,
            tokenizer_type.__qualname__,
            bool(getattr(tokenizer, "is_fast", False)),
        )
    return (
        operation,
        kind,
        _size_bucket(len(values)),
        _workload_bucket(values),
        multi_history,
        tokenizer_key,
        chat_template is not None,
        capacity,
    )


def _workers(key: tuple[object, ...], *, capacity: int, size: int) -> int:
    with _TUNING_LOCK:
        state = _tuning_state(key)
        return max(1, min(state.next_workers, capacity, size))


def _observe(
    key: tuple[object, ...],
    *,
    workers: int,
    capacity: int,
    size: int,
    units: int,
    elapsed: float,
) -> None:
    if units <= 0 or elapsed <= 0:
        return
    with _TUNING_LOCK:
        state = _tuning_state(key)
        measurement = state.measurements.setdefault(workers, _Measurement())
        measurement.units += units
        measurement.seconds += elapsed
        measurement.samples += 1
        limit = min(capacity, size)

        smaller = [value for value in state.measurements if value < workers]
        if workers == max(state.measurements) and workers < limit:
            previous = max(smaller) if smaller else None
            if (
                previous is None
                or measurement.rate >= state.measurements[previous].rate * 1.05
            ):
                state.next_workers = min(limit, workers * 2)
                return

        peak = max(value.rate for value in state.measurements.values())
        efficient = min(
            workers
            for workers, value in state.measurements.items()
            if value.rate >= peak * 0.95
        )
        lower = max(1, math.ceil(efficient / 2))
        state.next_workers = (
            lower
            if lower < efficient and lower not in state.measurements
            else efficient
        )


def _process_workers(key: tuple[object, ...], *, capacity: int, size: int) -> int:
    limit = min(_PROCESS_MAX_WORKERS, capacity, size)
    with _TUNING_LOCK:
        state = _process_tuning_state(key)
        return max(1, min(state.next_workers, limit))


def _observe_process(
    key: tuple[object, ...],
    *,
    workers: int,
    capacity: int,
    size: int,
    units: int,
    elapsed: float,
) -> None:
    if units <= 0 or elapsed <= 0:
        return
    limit = min(_PROCESS_MAX_WORKERS, capacity, size)
    with _TUNING_LOCK:
        state = _process_tuning_state(key)
        if not state.warmed:
            # Spawning workers and loading their tokenizers is a one-time cost, not
            # evidence about the steady-state worker width.
            state.warmed = True
            return
        measurement = state.measurements.setdefault(workers, _Measurement())
        measurement.units += units
        measurement.seconds += elapsed
        measurement.samples += 1

        if measurement.samples < 2:
            state.next_workers = workers
            return

        thread_state = _tuning_state(key)
        if thread_state.measurements:
            process_peak = max(value.rate for value in state.measurements.values())
            thread_peak = max(
                value.rate for value in thread_state.measurements.values()
            )
            if process_peak < thread_peak * 1.05:
                state.disabled = True
                return

        probe = min(2, limit)
        if workers > probe and probe not in state.measurements:
            state.next_workers = probe
            return

        peak = max(value.rate for value in state.measurements.values())
        efficient = min(
            width
            for width, value in state.measurements.items()
            if value.rate >= peak * 0.95
        )
        state.next_workers = efficient


def _consider_processes(key: tuple[object, ...], *, elapsed: float) -> None:
    with _TUNING_LOCK:
        state = _process_tuning_state(key)
        thread_state = _tuning_state(key)
        thread_samples = sum(
            measurement.samples for measurement in thread_state.measurements.values()
        )
        if elapsed >= _PROCESS_MIN_THREAD_SECONDS and thread_samples >= 2:
            state.candidate = True


def _processes_enabled(key: tuple[object, ...]) -> bool:
    with _TUNING_LOCK:
        state = _process_tuning_state(key)
        return state.candidate and not state.disabled


def _disable_processes(
    key: tuple[object, ...],
    *,
    reason: BaseException,
) -> None:
    with _TUNING_LOCK:
        state = _process_tuning_state(key)
        if state.disabled:
            return
        state.disabled = True
    warnings.warn(
        f"ART process tokenization is unavailable for this workload; using threads: "
        f"{type(reason).__name__}: {reason}",
        RuntimeWarning,
        stacklevel=4,
    )


def _disable_process_backend(reason: BaseException) -> None:
    global _PROCESS_BACKEND_DISABLED
    with _PROCESS_EXECUTOR_LOCK:
        if _PROCESS_BACKEND_DISABLED:
            return
        _PROCESS_BACKEND_DISABLED = True
    warnings.warn(
        f"ART process tokenization is unavailable; using threads: "
        f"{type(reason).__name__}: {reason}",
        RuntimeWarning,
        stacklevel=4,
    )


async def _ordered_map(
    function: Callable[[_ValueT], _ResultT],
    values: Sequence[_ValueT],
    *,
    workers: int,
    capacity: int,
) -> list[_ResultT]:
    loop = asyncio.get_running_loop()
    executor = _executor(capacity)
    semaphore = asyncio.Semaphore(workers)

    async def invoke(value: _ValueT) -> _ResultT:
        async with semaphore:
            return await loop.run_in_executor(executor, function, value)

    return await _gather_cancel_on_error(invoke(value) for value in values)


async def _gather_cancel_on_error(
    awaitables: Iterable[Awaitable[_ResultT]],
) -> list[_ResultT]:
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(frozen=True)
class _ProcessOptions:
    multi_history: bool
    reconcile_text_equivalent_tokenizations: bool
    model: str | None
    base_model: str | None
    chat_template: str | None
    chat_template_kwargs: Mapping[str, object] | None


class _ProcessTransferError(RuntimeError):
    pass


class _ProcessBackendError(RuntimeError):
    pass


def _tokenize_process_payload(payload: bytes) -> bytes:
    try:
        trajectory, options = cast(
            tuple[Trajectory, _ProcessOptions], pickle.loads(payload)
        )
    except Exception as error:
        raise _ProcessTransferError(
            f"could not deserialize process input: {type(error).__name__}: {error}"
        ) from None
    tokenized = trajectory.tokenize(
        multi_history=options.multi_history,
        reconcile_text_equivalent_tokenizations=(
            options.reconcile_text_equivalent_tokenizations
        ),
        model=options.model,
        base_model=options.base_model,
        tokenizer=None,
        chat_template=options.chat_template,
        chat_template_kwargs=options.chat_template_kwargs,
    )
    try:
        return pickle.dumps(tokenized, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise _ProcessTransferError(
            f"could not serialize {type(tokenized).__name__}: "
            f"{type(error).__name__}: {error}"
        ) from None


def _process_payloads(
    values: Sequence[Trajectory], options: _ProcessOptions
) -> list[bytes]:
    try:
        with _without_pickle_string_interning():
            return [
                pickle.dumps((value, options), protocol=pickle.HIGHEST_PROTOCOL)
                for value in values
            ]
    except Exception as error:
        raise _ProcessTransferError(
            f"could not serialize process input: {type(error).__name__}: {error}"
        ) from None


async def _ordered_process_map(
    payloads: Sequence[bytes],
    trajectories: Sequence[Trajectory],
    *,
    workers: int,
    capacity: int,
) -> list[TokenizedTrajectory | TokenizedMultiHistoryTrajectory]:
    loop = asyncio.get_running_loop()
    thread_executor = _executor(capacity)
    try:
        executor, process_capacity, startup = _start_process_executor(capacity)
        await loop.run_in_executor(
            thread_executor,
            _finish_process_warmup,
            startup,
            process_capacity,
        )
        _complete_process_warmup(executor, startup)
    except BrokenProcessPool:
        raise
    except (OSError, RuntimeError) as error:
        raise _ProcessBackendError(
            f"could not start process workers: {type(error).__name__}: {error}"
        ) from None
    semaphore = asyncio.Semaphore(workers)

    async def invoke(
        payload: bytes, trajectory: Trajectory
    ) -> TokenizedTrajectory | TokenizedMultiHistoryTrajectory:
        async with semaphore:
            serialized = await loop.run_in_executor(
                executor, _tokenize_process_payload, payload
            )
            return await loop.run_in_executor(
                thread_executor,
                _deserialize_process_result,
                serialized,
                trajectory,
            )

    return await _gather_cancel_on_error(
        invoke(payload, trajectory)
        for payload, trajectory in zip(payloads, trajectories, strict=True)
    )


def _supports_processes(
    *, capacity: int, size: int, tokenizer: Tokenizer | None
) -> bool:
    if (
        _PROCESS_BACKEND_DISABLED
        or capacity < 2
        or size < _PROCESS_MIN_ITEMS
        or tokenizer is not None
    ):
        return False
    if multiprocessing.current_process().daemon:
        return False
    return "spawn" in multiprocessing.get_all_start_methods()


def _rebind_process_result(
    result: TokenizedTrajectory | TokenizedMultiHistoryTrajectory,
    trajectory: Trajectory,
) -> None:
    source_trajectory = result.trajectory
    if isinstance(result, TokenizedTrajectory):
        _rebind_history_sources(
            result.history, trajectory, source_trajectory=source_trajectory
        )
    else:
        for history in result.histories:
            _rebind_history_sources(
                history.history, trajectory, source_trajectory=source_trajectory
            )
    result.trajectory = trajectory


def _deserialize_process_result(
    payload: bytes, trajectory: Trajectory
) -> TokenizedTrajectory | TokenizedMultiHistoryTrajectory:
    try:
        result = pickle.loads(payload)
        if not isinstance(
            result, (TokenizedTrajectory, TokenizedMultiHistoryTrajectory)
        ):
            raise TypeError(f"unexpected process result {type(result).__name__}")
        _rebind_process_result(result, trajectory)
        return result
    except Exception as error:
        if isinstance(error, _ProcessTransferError):
            raise
        raise _ProcessTransferError(
            f"could not restore process output: {type(error).__name__}: {error}"
        ) from None


def _result_units(value: object) -> int:
    if (tokens := getattr(value, "tokens", None)) is not None:
        return len(tokens)
    histories = getattr(value, "histories", None)
    return sum(_result_units(history) for history in histories) if histories else 1


def _materialize(
    values: Iterable[Trajectory] | Iterable[TrajectoryGroup],
) -> tuple[_InputKind | None, list[Trajectory] | list[TrajectoryGroup]]:
    materialized = list(values)
    if not materialized:
        return None, []
    if all(isinstance(value, Trajectory) for value in materialized):
        return "trajectory", cast(list[Trajectory], materialized)
    if all(isinstance(value, TrajectoryGroup) for value in materialized):
        return "group", cast(list[TrajectoryGroup], materialized)
    raise TypeError("items must contain only trajectories or only trajectory groups")


async def transform(
    values: Iterable[Trajectory] | Iterable[TrajectoryGroup],
    *,
    operation: _Operation,
    multi_history: bool,
    reconcile_text_equivalent_tokenizations: bool,
    model: str | None,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    device: Any = None,
) -> list[object]:
    kind, materialized = _materialize(values)
    if kind is None:
        return []
    groups = cast(list[TrajectoryGroup], materialized) if kind == "group" else None
    leaves = (
        [trajectory for group in groups for trajectory in group.trajectories]
        if groups is not None
        else cast(list[Trajectory], materialized)
    )

    def convert(trajectory: Trajectory) -> object:
        tokenized = trajectory.tokenize(
            multi_history=multi_history,
            reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        return tokenized if operation == "tokenize" else tokenized.tensorize()

    transformed: list[object]
    if leaves:
        capacity = _cpu_capacity()
        key = _tuning_key(
            operation=operation,
            kind=kind,
            values=leaves,
            multi_history=multi_history,
            tokenizer=tokenizer,
            model=model,
            base_model=base_model,
            chat_template=chat_template,
            capacity=capacity,
        )
        use_processes = _supports_processes(
            capacity=capacity, size=len(leaves), tokenizer=tokenizer
        ) and _processes_enabled(key)
        if use_processes:
            options = _ProcessOptions(
                multi_history=multi_history,
                reconcile_text_equivalent_tokenizations=(
                    reconcile_text_equivalent_tokenizations
                ),
                model=model,
                base_model=base_model,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )
            try:
                workers = _process_workers(key, capacity=capacity, size=len(leaves))
                started = time.perf_counter()
                payloads = await asyncio.get_running_loop().run_in_executor(
                    _executor(capacity), _process_payloads, leaves, options
                )
                tokenized = await _ordered_process_map(
                    payloads,
                    leaves,
                    workers=workers,
                    capacity=capacity,
                )
                if operation == "tokenize":
                    transformed = cast(list[object], tokenized)
                else:
                    transformed = cast(
                        list[object],
                        await _ordered_map(
                            lambda result: result.tensorize(),
                            tokenized,
                            workers=workers,
                            capacity=capacity,
                        ),
                    )
                _observe_process(
                    key,
                    workers=workers,
                    capacity=capacity,
                    size=len(leaves),
                    units=sum(_result_units(value) for value in transformed),
                    elapsed=time.perf_counter() - started,
                )
            except (pickle.PickleError, _ProcessTransferError) as error:
                _disable_processes(
                    key,
                    reason=error,
                )
                use_processes = False
            except (BrokenProcessPool, _ProcessBackendError) as error:
                _discard_process_executor()
                _disable_process_backend(error)
                use_processes = False
        if not use_processes:
            workers = _workers(key, capacity=capacity, size=len(leaves))
            started = time.perf_counter()
            transformed = await _ordered_map(
                convert, leaves, workers=workers, capacity=capacity
            )
            elapsed = time.perf_counter() - started
            _observe(
                key,
                workers=workers,
                capacity=capacity,
                size=len(leaves),
                units=sum(_result_units(value) for value in transformed),
                elapsed=elapsed,
            )
            if _supports_processes(
                capacity=capacity, size=len(leaves), tokenizer=tokenizer
            ):
                _consider_processes(
                    key,
                    elapsed=elapsed,
                )
    else:
        transformed = []

    if groups is not None:
        if operation == "tokenize":
            group_class: Any = TokenizedTrajectoryGroup
        else:
            from .tensors import TensorizedTrajectoryGroup

            group_class = TensorizedTrajectoryGroup
        grouped: list[object] = []
        start = 0
        for group in groups:
            end = start + len(group.trajectories)
            grouped.append(
                group_class(trajectory_group=group, trajectories=transformed[start:end])
            )
            start = end
        transformed = grouped

    if device is not None:
        for value in transformed:
            cast(Any, value).to_(device)
    return transformed
