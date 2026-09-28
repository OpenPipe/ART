from __future__ import annotations

import asyncio
import atexit
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import copyreg
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum, IntFlag
from functools import lru_cache
import io
import math
import multiprocessing
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from types import GenericAlias, ModuleType, UnionType
import typing
from typing import Any, Literal, TypeVar, cast
import warnings

from pydantic import BaseModel
from pydantic import main as pydantic_main
from pydantic.fields import FieldInfo

from . import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
    TokenFlag,
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectory,
    TokenizedTrajectoryGroup,
    Tokenizer,
    Trajectory,
    TrajectoryExchanges,
    TrajectoryGroup,
    _HistorySource,
)
from ._serialization import (
    _rebind_history_sources,
    _StringInterningModel,
    _without_pickle_string_interning,
)

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
    source_refs_allowed: bool = False


class _ProcessTransferError(RuntimeError):
    pass


class _ProcessBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ProcessSourceRef:
    index: int


class _ProcessSourceAlias(Exception):
    pass


def _process_sources(trajectory: Trajectory) -> tuple[object, ...]:
    return (
        trajectory,
        *trajectory.exchanges.chat_completions,
        *trajectory.exchanges.completions,
        *trajectory.exchanges.responses,
        *trajectory.exchanges.messages,
    )


def _process_model_fields(cls: type) -> dict[str, FieldInfo] | None:
    """Read only the standard declaration descriptor and plain metadata slots."""
    if type(cls) is not type(BaseModel):
        return None
    attributes: dict[str, object] = {}
    for base in cls.__mro__:
        for name in ("model_fields", "__pydantic_fields__"):
            if name in vars(base):
                attributes.setdefault(name, vars(base)[name])
    fields = attributes.get("__pydantic_fields__")
    if (
        attributes.get("model_fields") is not vars(BaseModel)["model_fields"]
        or type(fields) is not dict
        or any(
            type(name) is not str or type(field) is not FieldInfo
            for name, field in fields.items()
        )
    ):
        return None
    return cast(dict[str, FieldInfo], fields)


@lru_cache(maxsize=1)
def _process_schema_models() -> frozenset[type[BaseModel]]:
    """Exact ART/provider types in the declared result schema, never subclasses."""
    models: set[type[BaseModel]] = set()
    pending: list[object] = [TokenizedTrajectory, TokenizedMultiHistoryTrajectory]
    seen: set[int] = set()
    aliases = tuple(
        vars(typing).get(name)
        for name in (
            "_GenericAlias",
            "_LiteralGenericAlias",
            "_AnnotatedAlias",
            "_UnionGenericAlias",
        )
    )
    while pending:
        annotation = pending.pop()
        if id(annotation) in seen:
            continue
        seen.add(id(annotation))
        if type(annotation) is type(BaseModel):
            fields = _process_model_fields(cast(type, annotation))
            if fields is None:
                raise _ProcessSourceAlias
            models.add(cast(type[BaseModel], annotation))
            pending.extend(field.annotation for field in fields.values())
        else:
            if type(annotation) is GenericAlias or type(annotation) is UnionType:
                args = object.__getattribute__(annotation, "__args__")
            elif any(type(annotation) is alias for alias in aliases):
                # typing aliases have mutable instance state. Only plain storage
                # may be inspected; get_args() can traverse customized containers.
                state = object.__getattribute__(annotation, "__dict__")
                if type(state) is not dict or any(
                    type(key) is not str for key in state
                ):
                    raise _ProcessSourceAlias
                args = state.get("__args__")
            else:
                continue
            if type(args) is not tuple:
                raise _ProcessSourceAlias
            pending.extend(args)
    return frozenset(models)


def _process_plain_models(
    value: object, *, check_schema: bool = False, exchange_ids: set[int] | None = None
) -> set[type] | None:
    """Require passive reconstruction, retirement and field rebinding.

    This assumes the inspected framework/stdlib bases and generated enum member
    state are unmodified; customized enum behavior instead uses ordinary pickle.
    Only exact schema model types qualify. Arbitrary user models/mixins retain
    ordinary pickle semantics even if no currently known pickle hook is present.
    Inspect omitted graphs too: their retirement can affect retained aliases.
    The parent checks its graph and the full schema before dispatch; classes,
    registrations and graphs must not be externally mutated during transfer.
    This is not a lock.
    """

    # Even an unrelated registry key can run metaclass hashing/equality during
    # lookup. Preserve ordinary pickle's own callbacks and ordering in that case.
    if type(copyreg.dispatch_table) is not dict or any(
        type(key) is not type
        and type(key) is not type(BaseModel)
        and type(key) is not type(Enum)
        for key in copyreg.dispatch_table
    ):
        return None
    try:
        schema_models = _process_schema_models()
    except _ProcessSourceAlias:
        return None

    @lru_cache(maxsize=None)
    def attributes(cls: type) -> dict[str, object]:
        result: dict[str, object] = {}
        for base in cls.__mro__:
            for name, item in vars(base).items():
                result.setdefault(name, item)
        return result

    standard = attributes(BaseModel)
    interned = attributes(_StringInterningModel)
    hooks = (
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
    )
    models: set[type] = set()
    # Omitted flags must preserve constructor and reducer callbacks too. Compare
    # behavior with the inspected framework base, not just __reduce_ex__.
    flag_safe = True
    for name, item in attributes(IntFlag).items():
        if callable(item) or hasattr(type(item), "__get__"):
            actual = attributes(TokenFlag).get(name)
            # Enum construction copies staticmethod wrappers for the standard
            # member-value generator; their underlying function must match.
            if type(item) is staticmethod and type(actual) is staticmethod:
                actual, item = actual.__func__, item.__func__
            if actual is not item:
                flag_safe = False
                break

    @lru_cache(maxsize=None)
    def field_is_plain(cls: type, name: str) -> bool:
        descriptor_type = type(attributes(cls).get(name))
        return type(descriptor_type) is type and not any(
            hook in vars(base)
            for base in descriptor_type.__mro__
            for hook in ("__set__", "__delete__")
        )

    def fields_are_plain(cls: type, names: Iterable[object]) -> bool:
        return all(type(name) is str and field_is_plain(cls, name) for name in names)

    def interning_marker(item: object, extra: dict[str, object] | None) -> object:
        try:
            return object.__getattribute__(item, "_art_pickle_strings_interned")
        except AttributeError:
            # BaseModel.__getattr__ can resolve an absent slot through private
            # attributes or extras. Decline those paths without invoking them.
            private = attributes(type(item)).get("__private_attributes__")
            if any(
                type(mapping) is not dict
                or any(type(key) is not str for key in mapping)
                or "_art_pickle_strings_interned" in mapping
                for mapping in (private, extra if extra is not None else {})
            ):
                return None
            return False

    root_interned: object = False

    def model_is_plain(cls: type) -> bool:
        if cls not in models:
            attrs = attributes(cls)
            fields = _process_model_fields(cls)
            if fields is None:
                return False
            if cls in (TokenizedTrajectory, TokenizedMultiHistoryTrajectory):
                setters = attrs.get("__pydantic_setattr_handlers__")
                config = attrs.get("model_config")
                standard_setters = getattr(
                    pydantic_main, "_SIMPLE_SETATTR_HANDLERS", None
                )
                if (
                    type(setters) is not dict
                    or type(config) is not dict
                    or type(standard_setters) is not dict
                    or any(
                        type(key) is not str
                        for mapping in (setters, config, standard_setters)
                        for key in mapping
                    )
                ):
                    return False
                setter = setters.get("trajectory")
                field_setter = standard_setters.get("model_field")
                if (
                    field_setter is None
                    or (setter is not None and setter is not field_setter)
                    or "trajectory" in attrs
                    or any(
                        config.get(name) is not None and config.get(name) is not False
                        for name in ("validate_assignment", "frozen")
                    )
                ):
                    return False
            if (
                any(
                    attrs.get(name) is not standard.get(name)
                    for name in (
                        "__new__",
                        "__getattribute__",
                        "__getattr__",
                        "__setattr__",
                        "_setattr_handler",
                        "__reduce__",
                        "__getstate__",
                        "__setstate__",
                        "__dict__",
                        "__pydantic_fields_set__",
                        "__pydantic_extra__",
                        "__pydantic_private__",
                    )
                )
                or (
                    attrs.get("__reduce_ex__") is not standard.get("__reduce_ex__")
                    and attrs.get("__reduce_ex__") is not interned.get("__reduce_ex__")
                )
                or "__del__" in attrs
                or any(
                    name in attrs for name in ("__getnewargs__", "__getnewargs_ex__")
                )
                or (
                    issubclass(cls, _StringInterningModel)
                    and any(
                        attrs.get(name) is not interned.get(name)
                        for name in (
                            "_mark_pickle_strings_interned",
                            "_art_pickle_strings_interned",
                        )
                    )
                )
                or not fields_are_plain(
                    cls,
                    (
                        *fields,
                        # The parent may not have a history instance yet. Check
                        # every replacement role even if its field is undeclared.
                        "system_source",
                        "instructions_source",
                        "exchange",
                    ),
                )
            ):
                return False
            models.add(cls)
        return True

    if check_schema and (
        not flag_safe
        or any(
            type(cls) is not type(BaseModel)
            or cls in copyreg.dispatch_table
            or not model_is_plain(cls)
            for cls in schema_models
        )
    ):
        return None
    state_owners: dict[int, object] = {}
    seen: set[tuple[int, bool, bool]] = set()
    pending = [(value, False, False)]
    while pending:
        item, hashed, outside_interning = pending.pop()
        cls = type(item)
        if (
            type(cls) is not type
            and type(cls) is not type(BaseModel)
            and type(cls) is not type(Enum)
        ):
            return None
        if cls in copyreg.dispatch_table:
            return None
        if cls in (type(None), bool, int, float, str, bytes, bytearray):
            continue
        if cls is TokenFlag:
            if not flag_safe:
                return None
            continue
        if cls is datetime:
            pending.append((cast(datetime, item).tzinfo, False, outside_interning))
            continue
        if cls is timezone:
            # Even an exact timezone can retain subclass offsets or names.
            pending.extend(
                (arg, False, outside_interning) for arg in timezone.__reduce__(item)[1]
            )
            continue
        if cls is timedelta:
            continue
        if (id(item), hashed, outside_interning) in seen:
            continue
        seen.add((id(item), hashed, outside_interning))
        if cls in (list, tuple, set, frozenset):
            pending.extend(
                (child, hashed or cls in (set, frozenset), outside_interning)
                for child in cast(Iterable[object], item)
            )
        elif cls is dict:
            for key, child in cast(dict[object, object], item).items():
                pending.extend(
                    ((key, True, outside_interning), (child, False, outside_interning))
                )
        elif type(cls) is type(BaseModel):
            # Keys/set members run hashing and equality during reconstruction.
            # Decline all model keys, including models nested in key containers.
            if hashed or cls not in schema_models:
                return None
            if not model_is_plain(cls):
                return None
            try:
                state = object.__getattribute__(item, "__dict__")
                fields_set = object.__getattribute__(item, "__pydantic_fields_set__")
                extra = object.__getattribute__(item, "__pydantic_extra__")
                private_state = object.__getattribute__(item, "__pydantic_private__")
            except AttributeError:
                return None
            # Omitted __getstate__ and interning must accept the same slot shapes.
            # Public construction/mutation need not respect Pydantic annotations.
            if (
                type(state) is not dict
                or type(fields_set) is not set
                or (extra is not None and type(extra) is not dict)
                or (private_state is not None and type(private_state) is not dict)
                or not fields_are_plain(cls, state)
                # Class passivity does not cover instance attributes replacing
                # methods. Preserve ordinary callback order on late fallback.
                or any(name in attributes(cls) or name in hooks for name in state)
            ):
                return None
            if isinstance(item, _StringInterningModel):
                marker = interning_marker(item, extra)
                if item is value:
                    root_interned = marker
                # A fresh root prepares public models before elision, but skips
                # fields-set/private state. A marked root skips preparation entirely.
                # Do not invoke arbitrary truth callbacks in an unexpected marker.
                if (
                    type(marker) is not bool
                    or outside_interning
                    or (root_interned is True and marker is False)
                ):
                    return None
            if exchange_ids is not None and isinstance(
                item,
                (
                    ChatCompletionsExchange,
                    CompletionsExchange,
                    ResponsesExchange,
                    MessagesExchange,
                ),
            ):
                exchange_ids.add(id(item))
            if state_owners.setdefault(id(state), item) is not item:
                # Copying replaced state must not split aliases between models.
                return None
            pending.extend(
                (child, False, outside_interning) for child in (state, extra)
            )
            # Match _intern_value's exact BaseModel traversal: only field state
            # and extras are prepared. These other pickle-state slots are not.
            pending.extend(
                (child, False, True) for child in (fields_set, private_state)
            )
        else:
            return None
    return models


def _serialize_process_result(
    result: TokenizedTrajectory | TokenizedMultiHistoryTrajectory,
) -> bytes:
    if (
        type(result) is not TokenizedTrajectory
        and type(result) is not TokenizedMultiHistoryTrajectory
    ):
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    exchange_ids: set[int] = set()
    plain_models = _process_plain_models(result, exchange_ids=exchange_ids)
    if plain_models is None:
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    # Prove traversal shapes before reducers: public construct/copy/mutation can
    # retain values outside their annotations. The receiver has broader traversal.
    trajectory = result.__dict__.get("trajectory")
    exchanges = (
        trajectory.__dict__.get("exchanges") if type(trajectory) is Trajectory else None
    )
    if type(exchanges) is not TrajectoryExchanges or any(
        type(exchanges.__dict__.get(name)) not in (list, tuple)
        for name in ("chat_completions", "completions", "responses", "messages")
    ):
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    sources = _process_sources(cast(Trajectory, trajectory))
    if any(
        type(value)
        not in (
            ChatCompletionsExchange,
            CompletionsExchange,
            ResponsesExchange,
            MessagesExchange,
        )
        for value in sources[1:]
    ):
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    indices = {id(value): index for index, value in enumerate(sources)}
    if not exchange_ids.issubset(indices):
        # Noninventory exchanges use the receiver's equality search. Injecting
        # shared sources early can change that search's values or exceptions.
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    replacements = {id(result): {"trajectory": _ProcessSourceRef(0)}}
    replaced_states = {id(result.__dict__)}

    def bind(owner: Any, name: str) -> None:
        value = owner.__dict__.get(name)
        # Match the receiver's role predicate even for unvalidated model state.
        if not isinstance(
            value,
            (
                ChatCompletionsExchange,
                CompletionsExchange,
                ResponsesExchange,
                MessagesExchange,
            ),
        ):
            return
        index = indices.get(id(value))
        if index is not None:
            replacements.setdefault(id(owner), {})[name] = _ProcessSourceRef(index)
            replaced_states.add(id(owner.__dict__))

    histories = (
        [result]
        if isinstance(result, TokenizedTrajectory)
        else result.__dict__.get("histories")
    )
    if type(histories) not in (list, tuple):
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    for tokenized in cast(Sequence[object], histories):
        history = (
            tokenized.__dict__.get("history")
            if isinstance(tokenized, BaseModel)
            else None
        )
        if not isinstance(history, BaseModel):
            return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        for name in ("system_source", "instructions_source"):
            bind(history, name)
        for name in ("message_sources", "input_sources", "prompt_sources"):
            values = history.__dict__.get(name, ())
            if type(values) not in (list, tuple):
                return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            for source in values:
                if isinstance(source, _HistorySource):
                    bind(source, "exchange")
                    nested = source.__dict__.get("source")
                    if isinstance(nested, _HistorySource):
                        bind(nested, "exchange")

    stream = io.BytesIO()

    class Writer(pickle.Pickler):
        def persistent_id(self, value: object) -> int | None:
            if id(value) in replaced_states or id(value) in indices:
                # A source or replaced state was reached through another alias.
                # Preserve the receiver's complete old graph semantics.
                raise _ProcessSourceAlias
            return value.index if type(value) is _ProcessSourceRef else None

        def reducer_override(self, value: Any) -> Any:
            cls = type(value)
            if cls in copyreg.dispatch_table:
                raise _ProcessSourceAlias
            if (
                isinstance(value, type)
                or value is getattr(copyreg, "__newobj__")
                or value is getattr(copyreg, "__newobj_ex__")
                or cls is TokenFlag
            ):
                return NotImplemented
            if cls not in plain_models:
                raise _ProcessSourceAlias
            replacement = replacements.get(id(value))
            if replacement is None:
                return NotImplemented
            reduced = value.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
            if (
                type(reduced) is not tuple
                or len(reduced) < 3
                or type(reduced[2]) is not dict
                or type(reduced[2].get("__dict__")) is not dict
            ):
                raise _ProcessSourceAlias
            state = reduced[2].copy()
            state["__dict__"] = {**state["__dict__"], **replacement}
            return (*reduced[:2], state, *reduced[3:])

    # Replace only fields that the existing receiver canonicalizes. Other
    # references to the same source (e.g. arbitrary kwargs) stay detached copies.
    try:
        Writer(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(result)
    except _ProcessSourceAlias:
        return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    # Ordinary highest-protocol pickle starts with PROTO, never this frame byte.
    # Recognize our format before unpickling arbitrary fallback return values.
    return b"\0" + pickle.dumps(
        (b"art-process-sources-v1", tuple(type(x) for x in sources), stream.getvalue()),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _load_process_result(payload: bytes, trajectory: Trajectory) -> object:
    if not payload.startswith(b"\0"):
        return pickle.loads(payload)
    packed = pickle.loads(payload[1:])
    if not (
        type(packed) is tuple
        and len(packed) == 3
        and type(packed[0]) is bytes
        and packed[0] == b"art-process-sources-v1"
    ):
        raise ValueError("Invalid process source envelope")
    sources = _process_sources(trajectory)
    if packed[1] != tuple(type(x) for x in sources):
        raise ValueError("Source trajectory exchange structure has changed")

    def persistent_load(index: object) -> object:
        if type(index) is not int or not 0 <= index < len(sources):
            raise pickle.UnpicklingError("Invalid process source reference")
        return sources[index]

    # A per-call class would keep this source closure alive through its MRO cycle.
    reader = pickle.Unpickler(io.BytesIO(packed[2]))
    reader.persistent_load = cast(Any, persistent_load)
    return reader.load()


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
        if options.source_refs_allowed:
            return _serialize_process_result(tokenized)
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
        schema_plain = _process_plain_models(None, check_schema=True) is not None
        with _without_pickle_string_interning():
            return [
                pickle.dumps(
                    (
                        value,
                        replace(
                            options,
                            source_refs_allowed=schema_plain
                            and _process_plain_models(value) is not None,
                        ),
                    ),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
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
        result = _load_process_result(payload, trajectory)
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
