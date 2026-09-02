from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal, TypeVar, cast

from . import (
    TokenizedTrajectoryGroup,
    Tokenizer,
    Trajectory,
    TrajectoryGroup,
)

_ResultT = TypeVar("_ResultT")
_InputKind = Literal["trajectory", "group"]
_Operation = Literal["tokenize", "tensorize"]


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


@dataclass
class _Measurement:
    units: int = 0
    seconds: float = 0

    @property
    def rate(self) -> float:
        return self.units / self.seconds


@dataclass
class _TuningState:
    next_workers: int
    measurements: dict[int, _Measurement] = field(default_factory=dict)


_TUNING_LOCK = threading.Lock()


@lru_cache(maxsize=128)
def _tuning_state(key: tuple[object, ...], initial_workers: int) -> _TuningState:
    return _TuningState(initial_workers)


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
    initial = min(4, capacity, size)
    with _TUNING_LOCK:
        state = _tuning_state(key, initial)
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
    initial = min(4, capacity, size)
    with _TUNING_LOCK:
        state = _tuning_state(key, initial)
        measurement = state.measurements.setdefault(workers, _Measurement())
        measurement.units += units
        measurement.seconds += elapsed
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


async def _ordered_map(
    function: Callable[[Trajectory], _ResultT],
    values: Sequence[Trajectory],
    *,
    workers: int,
    capacity: int,
) -> list[_ResultT]:
    loop = asyncio.get_running_loop()
    executor = _executor(capacity)
    semaphore = asyncio.Semaphore(workers)

    async def invoke(value: Trajectory) -> _ResultT:
        async with semaphore:
            return await loop.run_in_executor(executor, function, value)

    return list(await asyncio.gather(*(invoke(value) for value in values)))


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
        workers = _workers(key, capacity=capacity, size=len(leaves))
        started = time.perf_counter()
        transformed = await _ordered_map(
            convert, leaves, workers=workers, capacity=capacity
        )
        _observe(
            key,
            workers=workers,
            capacity=capacity,
            size=len(leaves),
            units=sum(_result_units(value) for value in transformed),
            elapsed=time.perf_counter() - started,
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
            cast(Any, value).to(device)
    return transformed
