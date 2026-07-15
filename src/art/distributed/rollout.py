from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
import importlib
import inspect
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InstalledAsyncCallable(BaseModel):
    """Import path for installed user code; functions and closures are never shipped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str = Field(min_length=1)
    qualname: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_import_path(self) -> "InstalledAsyncCallable":
        if self.qualname == "<lambda>" or "<locals>" in self.qualname.split("."):
            raise ValueError(
                "distributed rollout callable must be a top-level function"
            )
        return self

    @classmethod
    def from_callable(
        cls, function: Callable[..., Awaitable[Any]]
    ) -> "InstalledAsyncCallable":
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        if not module or not qualname:
            raise ValueError(
                "distributed rollout callable requires module and qualname"
            )
        reference = cls(module=module, qualname=qualname)
        if not inspect.iscoroutinefunction(function):
            raise TypeError("distributed rollout callable must be async")
        if reference.resolve() is not function:
            raise ValueError(
                "distributed rollout callable must resolve from installed code"
            )
        return reference

    def resolve(self) -> Callable[..., Awaitable[Any]]:
        value: Any = importlib.import_module(self.module)
        for component in self.qualname.split("."):
            value = getattr(value, component)
        if not inspect.iscoroutinefunction(value):
            raise TypeError(f"{self.module}:{self.qualname} is not an async function")
        return value


class RolloutInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    callable: InstalledAsyncCallable
    model: Any
    scenario: Any
    config: Any


class RolloutExecutor(Protocol):
    def set_target(self, target_workers: int) -> None: ...

    def set_workers(self, worker_ids: tuple[int, ...]) -> None: ...

    async def run(
        self,
        worker_id: int,
        rollout_fn: Callable[..., Awaitable[Any]],
        model: Any,
        scenario: Any,
        config: Any,
    ) -> Any: ...


class LocalRolloutExecutor:
    def set_target(self, target_workers: int) -> None:
        if target_workers < 1:
            raise ValueError("target_workers must be >= 1")

    def set_workers(self, worker_ids: tuple[int, ...]) -> None:
        del worker_ids

    async def run(
        self,
        worker_id: int,
        rollout_fn: Callable[..., Awaitable[Any]],
        model: Any,
        scenario: Any,
        config: Any,
    ) -> Any:
        del worker_id
        return await rollout_fn(model, scenario, config)


class RolloutHostEndpoint(Protocol):
    async def run(self, invocation: RolloutInvocation) -> Any: ...


def apportion_rollout_workers(
    target_workers: int, host_slots: Mapping[str, int]
) -> dict[str, int]:
    """Deterministically assign one global exact target without host-local policy."""

    if target_workers < 1:
        raise ValueError("target_workers must be >= 1")
    if not host_slots or any(slots < 1 for slots in host_slots.values()):
        raise ValueError("rollout hosts must each provide at least one CPU slot")
    allocation = dict.fromkeys(host_slots, 0)
    for _ in range(target_workers):
        candidates = [
            host for host, slots in host_slots.items() if allocation[host] < slots
        ]
        if not candidates:
            raise ValueError(
                f"global rollout-worker target {target_workers} exceeds host capacity "
                f"{sum(host_slots.values())}"
            )
        host_id = min(
            candidates, key=lambda host: (allocation[host] / host_slots[host], host)
        )
        allocation[host_id] += 1
    return allocation


class DistributedRolloutExecutor:
    def __init__(
        self,
        *,
        callable: InstalledAsyncCallable,
        hosts: Mapping[str, RolloutHostEndpoint],
        host_slots: Mapping[str, int],
        target_workers: int,
    ) -> None:
        if set(hosts) != set(host_slots):
            raise ValueError("rollout host endpoints and slot declarations must match")
        self.callable = callable
        self.hosts = dict(hosts)
        self.host_slots = dict(host_slots)
        self._worker_hosts: tuple[str, ...] = ()
        self._host_by_worker: dict[int, str] = {}
        self.set_target(target_workers)

    def set_target(self, target_workers: int) -> None:
        allocation = apportion_rollout_workers(target_workers, self.host_slots)
        self._worker_hosts = tuple(
            host_id
            for host_id in sorted(allocation)
            for _ in range(allocation[host_id])
        )

    def set_workers(self, worker_ids: tuple[int, ...]) -> None:
        if len(worker_ids) > len(self._worker_hosts):
            raise ValueError("live rollout workers exceed the global target")
        self._host_by_worker = dict(
            zip(sorted(worker_ids), self._worker_hosts, strict=False)
        )

    async def run(
        self,
        worker_id: int,
        rollout_fn: Callable[..., Awaitable[Any]],
        model: Any,
        scenario: Any,
        config: Any,
    ) -> Any:
        if InstalledAsyncCallable.from_callable(rollout_fn) != self.callable:
            raise ValueError(
                "PipelineTrainer rollout_fn differs from distributed callable"
            )
        try:
            host_id = self._host_by_worker[worker_id]
        except KeyError:
            raise RuntimeError(
                f"rollout worker {worker_id} has no host assignment"
            ) from None
        return await self.hosts[host_id].run(
            RolloutInvocation(
                callable=self.callable,
                model=model,
                scenario=scenario,
                config=config,
            )
        )


class InProcessRolloutHost:
    """One coarse host service used by local collapse and tests."""

    async def run(self, invocation: RolloutInvocation) -> Any:
        function = invocation.callable.resolve()
        return await function(invocation.model, invocation.scenario, invocation.config)
