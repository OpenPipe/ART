from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkflowTrainerTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str = Field(min_length=1)
    tp: int = Field(default=1, ge=1)
    cp: int = Field(default=1, ge=1)
    ep: int = Field(default=1, ge=1)
    etp: int = Field(default=1, ge=1)
    dp: int = Field(default=1, ge=1)
    pp: int = Field(default=1, ge=1)
    vpp: int = Field(default=1, ge=1)
    sp: bool = False


class WorkflowVllmTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str = Field(min_length=1)
    tp: int = Field(default=1, ge=1)
    pp: int = Field(default=1, ge=1)
    dp: int = Field(default=1, ge=1)
    ep: bool = False


class WorkflowRolePlacement(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str = Field(min_length=1)
    trainer_gpu_ids: tuple[int, ...] = ()
    vllm_gpu_ids: tuple[int, ...] = ()
    vllm_external: bool = False

    @model_validator(mode="after")
    def validate_relative_gpu_ids(self) -> "WorkflowRolePlacement":
        for role, gpu_ids in (
            ("trainer", self.trainer_gpu_ids),
            ("vLLM", self.vllm_gpu_ids),
        ):
            if len(set(gpu_ids)) != len(gpu_ids) or any(
                gpu_id < 0 for gpu_id in gpu_ids
            ):
                raise ValueError(
                    f"{role} relative GPU ids must be unique and non-negative"
                )
        return self


class WorkflowRuntimeTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    trainer_variants: tuple[WorkflowTrainerTopology, ...] = ()
    vllm_variants: tuple[WorkflowVllmTopology, ...] = ()
    role_placements: tuple[WorkflowRolePlacement, ...] = ()


class WorkflowRuntimeKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_fingerprint: str
    handler: str
    fixture: str
    kind: Literal["cpu", "megatron", "vllm", "joint"]
    topology: WorkflowRuntimeTopology = Field(default_factory=WorkflowRuntimeTopology)
    mode: str = ""
    static_options: str = ""


class WorkflowResourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_count: int = Field(default=0, ge=0)
    gpu_share: float = Field(default=1.0, gt=0.0, le=1.0)
    same_host: bool = True
    contiguous: bool = True
    host_affinity: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def cpu_requests_do_not_reserve_gpu_capacity(self) -> "WorkflowResourceRequest":
        if self.gpu_count == 0 and self.gpu_share != 1.0:
            raise ValueError("CPU operations cannot reserve fractional GPU capacity")
        return self


class WorkflowOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    stage: str
    runtime: WorkflowRuntimeKey
    resources: WorkflowResourceRequest = Field(default_factory=WorkflowResourceRequest)
    dependencies: tuple[str, ...] = ()
    estimated_duration_s: float = Field(default=0.0, ge=0.0)
    estimated_shared_startup_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def shared_startup_is_part_of_duration(self) -> "WorkflowOperation":
        if self.estimated_shared_startup_s > self.estimated_duration_s:
            raise ValueError("shared startup cannot exceed operation duration")
        return self


class WorkflowSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    runtime: WorkflowRuntimeKey
    operations: tuple[WorkflowOperation, ...]
    resources: WorkflowResourceRequest
    dependencies: tuple[str, ...] = ()
    estimated_duration_s: float = Field(ge=0.0)


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    sessions: tuple[WorkflowSession, ...]


class WorkflowDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str
    gpu: str


class WorkflowPlacement(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str | None = None
    devices: tuple[WorkflowDevice, ...] = ()

    @model_validator(mode="after")
    def devices_are_on_placement_host(self) -> "WorkflowPlacement":
        device_hosts = {device.host for device in self.devices}
        if len(device_hosts) > 1 or (
            self.host is not None and device_hosts and device_hosts != {self.host}
        ):
            raise ValueError("workflow placement devices must share its host")
        return self


class WorkflowSessionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    placement: WorkflowPlacement
    started_monotonic_s: float
    ended_monotonic_s: float
    output: Any = None
    failed_operation_id: str | None = None


class WorkflowExecution(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: dict[str, WorkflowSessionResult]
    blocked_by_failed_operations: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    elapsed_s: float


class WorkflowOperationFailed(RuntimeError):
    def __init__(self, operation_id: str) -> None:
        if not operation_id:
            raise ValueError("failed workflow operation id cannot be empty")
        self.operation_id = operation_id
        super().__init__(f"workflow operation failed: {operation_id}")


def compile_workflow(operations: Sequence[WorkflowOperation]) -> WorkflowPlan:
    """Collapse compatible operations into fixed-runtime sessions."""
    by_id = {operation.id: operation for operation in operations}
    if len(by_id) != len(operations):
        raise ValueError("workflow operation ids must be unique")
    unknown = {
        dependency
        for operation in operations
        for dependency in operation.dependencies
        if dependency not in by_id
    }
    if unknown:
        raise ValueError(f"unknown workflow dependencies: {sorted(unknown)}")
    operation_order = _topological_order(
        by_id, {key: value.dependencies for key, value in by_id.items()}
    )
    operation_rank = {
        operation_id: rank for rank, operation_id in enumerate(operation_order)
    }

    grouped: dict[WorkflowRuntimeKey, list[WorkflowOperation]] = defaultdict(list)
    runtime_order: list[WorkflowRuntimeKey] = []
    for operation in operations:
        if operation.runtime not in grouped:
            runtime_order.append(operation.runtime)
        grouped[operation.runtime].append(operation)

    operation_session = {
        operation.id: f"session_{index:03d}"
        for index, runtime in enumerate(runtime_order)
        for operation in grouped[runtime]
    }
    sessions: list[WorkflowSession] = []
    for index, runtime in enumerate(runtime_order):
        grouped_operations = tuple(
            sorted(grouped[runtime], key=lambda operation: operation_rank[operation.id])
        )
        resources = grouped_operations[0].resources
        if any(
            operation.resources != resources for operation in grouped_operations[1:]
        ):
            raise ValueError(f"runtime {runtime} has inconsistent resource requests")
        session_id = f"session_{index:03d}"
        dependencies = tuple(
            dict.fromkeys(
                operation_session[dependency]
                for operation in grouped_operations
                for dependency in operation.dependencies
                if operation_session[dependency] != session_id
            )
        )
        sessions.append(
            WorkflowSession(
                id=session_id,
                runtime=runtime,
                operations=grouped_operations,
                resources=resources,
                dependencies=dependencies,
                estimated_duration_s=sum(
                    operation.estimated_duration_s
                    - operation.estimated_shared_startup_s
                    for operation in grouped_operations
                )
                + max(
                    (
                        operation.estimated_shared_startup_s
                        for operation in grouped_operations
                    ),
                    default=0.0,
                ),
            )
        )
    session_by_id = {session.id: session for session in sessions}
    _topological_order(
        session_by_id,
        {key: value.dependencies for key, value in session_by_id.items()},
    )
    return WorkflowPlan(sessions=tuple(sessions))


def _topological_order(
    values: dict[str, Any], dependencies: dict[str, Iterable[str]]
) -> list[str]:
    pending = {key: set(dependencies[key]) for key in values}
    order: list[str] = []
    while pending:
        ready = sorted(key for key, deps in pending.items() if not deps)
        if not ready:
            raise ValueError(f"workflow dependency cycle: {sorted(pending)}")
        order.extend(ready)
        for key in ready:
            pending.pop(key)
        for deps in pending.values():
            deps.difference_update(ready)
    return order


class _GpuPool:
    def __init__(self, devices: Sequence[WorkflowDevice]) -> None:
        if len(set(devices)) != len(devices):
            raise ValueError("workflow devices must be unique")
        self._devices = tuple(devices)
        self._hosts = tuple(dict.fromkeys(device.host for device in devices))
        self._available = {device: 1.0 for device in devices}
        self._active_by_host = {host: 0 for host in self._hosts}
        self._affinity_hosts: dict[str, str] = {}
        self._affinity_bindings_by_host = {host: 0 for host in self._hosts}
        self._lock = Lock()

    def acquire(self, request: WorkflowResourceRequest) -> WorkflowPlacement | None:
        with self._lock:
            if request.gpu_count == 0:
                host = self._affinity_hosts.get(request.host_affinity or "")
                if host is None:
                    host = min(
                        self._hosts,
                        key=lambda value: (
                            self._unbound_affinity_load(request, value),
                            self._active_by_host[value],
                            self._hosts.index(value),
                        ),
                        default=None,
                    )
                if host is not None:
                    self._bind_affinity(request, host)
                    self._active_by_host[host] += 1
                return WorkflowPlacement(host=host)
            placements = self._candidate_placements(request)
            affinity_host = self._affinity_hosts.get(request.host_affinity or "")
            if affinity_host is not None:
                placements = [
                    devices
                    for devices in placements
                    if devices[0].host == affinity_host
                ]
            if placements:
                selected = min(
                    placements,
                    key=lambda devices: self._placement_priority(devices, request),
                )
                for device in selected:
                    self._available[device] -= request.gpu_share
                host = selected[0].host
                self._bind_affinity(request, host)
                self._active_by_host[host] += 1
                return WorkflowPlacement(host=host, devices=selected)
        return None

    def _bind_affinity(self, request: WorkflowResourceRequest, host: str) -> None:
        affinity = request.host_affinity
        if affinity is not None and affinity not in self._affinity_hosts:
            self._affinity_hosts[affinity] = host
            self._affinity_bindings_by_host[host] += 1

    def _unbound_affinity_load(
        self, request: WorkflowResourceRequest, host: str
    ) -> int:
        affinity = request.host_affinity
        if affinity is None or affinity in self._affinity_hosts:
            return 0
        return self._affinity_bindings_by_host[host]

    def _placement_priority(
        self,
        devices: tuple[WorkflowDevice, ...],
        request: WorkflowResourceRequest,
    ) -> tuple[Any, ...]:
        host = devices[0].host
        available = tuple(self._available[device] for device in devices)
        indices = tuple(self._devices.index(device) for device in devices)
        if request.gpu_share < 1.0:
            return (
                self._unbound_affinity_load(request, host),
                self._active_by_host[host],
                sum(available),
                indices,
            )
        return (
            self._unbound_affinity_load(request, host),
            self._active_by_host[host],
            -min(available),
            -sum(available),
            indices,
        )

    def _candidate_placements(
        self, request: WorkflowResourceRequest
    ) -> list[tuple[WorkflowDevice, ...]]:
        eligible = [
            device
            for device in self._devices
            if self._available[device] + 1e-9 >= request.gpu_share
        ]
        by_host: dict[str, list[WorkflowDevice]] = defaultdict(list)
        for device in eligible:
            by_host[device.host].append(device)
        if request.contiguous:
            return [
                selected
                for devices in by_host.values()
                for start in range(len(devices) - request.gpu_count + 1)
                if _contiguous(
                    selected := tuple(devices[start : start + request.gpu_count]),
                    self._devices,
                )
            ]
        pools = list(by_host.values()) if request.same_host else [eligible]
        return [
            tuple(
                sorted(
                    devices,
                    key=lambda device: (
                        -self._available[device],
                        self._devices.index(device),
                    ),
                )[: request.gpu_count]
            )
            for devices in pools
            if len(devices) >= request.gpu_count
        ]

    def release(
        self, placement: WorkflowPlacement, request: WorkflowResourceRequest
    ) -> None:
        with self._lock:
            for device in placement.devices:
                self._available[device] += request.gpu_share
                if self._available[device] > 1.0 + 1e-9:
                    raise RuntimeError(f"released unowned workflow GPU {device}")
            if placement.host is not None:
                self._active_by_host[placement.host] -= 1
                if self._active_by_host[placement.host] < 0:
                    raise RuntimeError(
                        f"released unowned workflow host {placement.host}"
                    )


def _contiguous(
    selected: Sequence[WorkflowDevice], inventory: Sequence[WorkflowDevice]
) -> bool:
    if not selected:
        return True
    if len({device.host for device in selected}) != 1:
        return False
    positions = sorted(inventory.index(device) for device in selected)
    return positions == list(range(positions[0], positions[0] + len(positions)))


def execute_workflow(
    plan: WorkflowPlan,
    *,
    devices: Sequence[WorkflowDevice],
    runner: Callable[[WorkflowSession, WorkflowPlacement], Any],
    max_workers: int | None = None,
) -> WorkflowExecution:
    """Execute a session DAG with exact GPU leases and longest-path priority."""
    started = time.monotonic()
    sessions = {session.id: session for session in plan.sessions}
    critical_path = _critical_path_durations(sessions)
    pending = set(sessions)
    completed: set[str] = set()
    failed_dependencies: dict[str, tuple[str, ...]] = {}
    blocked_by_failed_operations: dict[str, tuple[str, ...]] = {}
    running: dict[Future[Any], tuple[WorkflowSession, WorkflowPlacement, float]] = {}
    results: dict[str, WorkflowSessionResult] = {}
    pool = _GpuPool(devices)

    with ThreadPoolExecutor(
        max_workers=max_workers or max(1, len(sessions))
    ) as executor:
        while pending or running:
            while newly_blocked := {
                session_id: tuple(
                    sorted(
                        {
                            failed_operation
                            for dependency in sessions[session_id].dependencies
                            for failed_operation in failed_dependencies.get(
                                dependency, ()
                            )
                        }
                    )
                )
                for session_id in sorted(pending)
                if any(
                    dependency in failed_dependencies
                    for dependency in sessions[session_id].dependencies
                )
            }:
                for session_id, failed_operations in newly_blocked.items():
                    pending.remove(session_id)
                    failed_dependencies[session_id] = failed_operations
                    blocked_by_failed_operations[session_id] = failed_operations
            ready = sorted(
                (
                    sessions[session_id]
                    for session_id in pending
                    if set(sessions[session_id].dependencies) <= completed
                ),
                key=lambda session: (
                    session.resources.gpu_count > 0
                    and session.resources.gpu_share < 1.0,
                    -session.resources.gpu_count,
                    -critical_path[session.id],
                    session.id,
                ),
            )
            launched = False
            for session in ready:
                placement = pool.acquire(session.resources)
                if placement is None:
                    continue
                session_started = time.monotonic()
                future = executor.submit(runner, session, placement)
                running[future] = (session, placement, session_started)
                pending.remove(session.id)
                launched = True
            if not running:
                if pending:
                    blocked = sorted(pending)
                    raise RuntimeError(
                        f"workflow sessions do not fit available GPUs: {blocked}"
                    )
                break
            if launched:
                done = {future for future in running if future.done()}
                if not done:
                    continue
            else:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                session, placement, session_started = running.pop(future)
                failure: WorkflowOperationFailed | None = None
                try:
                    output = future.result()
                except WorkflowOperationFailed as exc:
                    if exc.operation_id not in {
                        operation.id for operation in session.operations
                    }:
                        raise RuntimeError(
                            f"session {session.id} reported failure for unknown operation "
                            f"{exc.operation_id}"
                        ) from exc
                    failure = exc
                    output = None
                finally:
                    pool.release(placement, session.resources)
                results[session.id] = WorkflowSessionResult(
                    session_id=session.id,
                    placement=placement,
                    started_monotonic_s=session_started,
                    ended_monotonic_s=time.monotonic(),
                    output=output,
                    failed_operation_id=(
                        failure.operation_id if failure is not None else None
                    ),
                )
                if failure is None:
                    completed.add(session.id)
                else:
                    failed_dependencies[session.id] = (failure.operation_id,)
    return WorkflowExecution(
        results=results,
        blocked_by_failed_operations=blocked_by_failed_operations,
        elapsed_s=time.monotonic() - started,
    )


def _critical_path_durations(
    sessions: dict[str, WorkflowSession],
) -> dict[str, float]:
    children: dict[str, list[str]] = defaultdict(list)
    for session in sessions.values():
        for dependency in session.dependencies:
            children[dependency].append(session.id)
    memo: dict[str, float] = {}

    def duration(session_id: str) -> float:
        if session_id not in memo:
            memo[session_id] = sessions[session_id].estimated_duration_s + max(
                (duration(child) for child in children[session_id]), default=0.0
            )
        return memo[session_id]

    return {session_id: duration(session_id) for session_id in sessions}
