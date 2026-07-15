from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import hashlib
import importlib
import inspect
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.model import TrainableModel
from art.serving_capabilities import ServingCapabilities


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


class RolloutModelSpec(BaseModel):
    """Serializable inference-only view of a registered trainable model."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    payload: dict[str, Any]
    user_config: Any = None
    internal_config: dict[str, Any] | None = None
    serving_capabilities: ServingCapabilities | None = None
    binary_routes_base_url: str | None = None

    @classmethod
    def from_model(cls, model: TrainableModel) -> "RolloutModelSpec":
        payload = model.model_dump(mode="json")
        payload["config"] = None
        payload["inference_model_name"] = model.get_inference_name()
        return cls(
            payload=payload,
            user_config=model.config,
            internal_config=(
                dict(model._internal_config)
                if model._internal_config is not None
                else None
            ),
            serving_capabilities=model._serving_capabilities,
            binary_routes_base_url=model._art_binary_routes_base_url,
        )

    @property
    def cache_key(self) -> str:
        payload = {
            "model": self.payload,
            "internal_config": self.internal_config,
            "capabilities": (
                self.serving_capabilities.model_dump(mode="json")
                if self.serving_capabilities is not None
                else None
            ),
            "binary_routes_base_url": self.binary_routes_base_url,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def build(self) -> TrainableModel:
        model = TrainableModel.model_validate(self.payload)
        object.__setattr__(model, "config", self.user_config)
        object.__setattr__(model, "_internal_config", self.internal_config)
        object.__setattr__(model, "_serving_capabilities", self.serving_capabilities)
        object.__setattr__(
            model, "_art_binary_routes_base_url", self.binary_routes_base_url
        )
        return model


class RolloutInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    callable: InstalledAsyncCallable
    model: RolloutModelSpec
    scenario: Any
    config: Any


class RolloutResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    value: Any
    metrics: dict[str, float] = Field(default_factory=dict)


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
    async def run(self, invocation: RolloutInvocation) -> RolloutResult: ...

    async def close(self) -> None: ...


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
        hosts: Mapping[str, Sequence[RolloutHostEndpoint]],
        target_workers: int,
    ) -> None:
        if not hosts or any(not endpoints for endpoints in hosts.values()):
            raise ValueError("rollout hosts must each provide at least one endpoint")
        self.callable = callable
        self.hosts = {host: tuple(endpoints) for host, endpoints in hosts.items()}
        self._worker_endpoints: tuple[RolloutHostEndpoint, ...] = ()
        self._endpoint_by_worker: dict[int, RolloutHostEndpoint] = {}
        self.set_target(target_workers)

    def set_target(self, target_workers: int) -> None:
        allocation = apportion_rollout_workers(
            target_workers,
            {host: len(endpoints) for host, endpoints in self.hosts.items()},
        )
        self._worker_endpoints = tuple(
            endpoint
            for host_id in sorted(allocation)
            for endpoint in self.hosts[host_id][: allocation[host_id]]
        )

    def set_workers(self, worker_ids: tuple[int, ...]) -> None:
        workers = tuple(sorted(worker_ids))
        assignments = {
            worker_id: self._endpoint_by_worker[worker_id]
            for worker_id in workers
            if worker_id in self._endpoint_by_worker
        }
        available = [
            endpoint
            for endpoint in self._worker_endpoints
            if endpoint not in assignments.values()
        ]
        unassigned = [
            worker_id for worker_id in workers if worker_id not in assignments
        ]
        if len(unassigned) > len(available):
            raise ValueError("new rollout workers exceed the global target")
        assignments.update(zip(unassigned, available, strict=False))
        self._endpoint_by_worker = assignments

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
            endpoint = self._endpoint_by_worker[worker_id]
        except KeyError:
            raise RuntimeError(
                f"rollout worker {worker_id} has no host assignment"
            ) from None
        result = await endpoint.run(
            RolloutInvocation(
                callable=self.callable,
                model=RolloutModelSpec.from_model(model),
                scenario=scenario,
                config=config,
            )
        )
        if result.metrics:
            from art.metrics import MetricsBuilder

            try:
                builder = MetricsBuilder.get_active()
            except LookupError:
                raise RuntimeError(
                    "distributed rollout produced metrics without an active ART metrics context"
                ) from None
            for key, value in result.metrics.items():
                builder.add_metric(key, value)
        return result.value

    async def close(self) -> None:
        await asyncio.gather(
            *(
                endpoint.close()
                for endpoints in self.hosts.values()
                for endpoint in endpoints
            )
        )


class InProcessRolloutHost:
    """One coarse host service used by local collapse and tests."""

    async def run(self, invocation: RolloutInvocation) -> RolloutResult:
        from art.metrics import MetricsBuilder

        function = invocation.callable.resolve()
        builder = MetricsBuilder(cost_context="train")
        token = builder.activate()
        try:
            value = await function(
                invocation.model.build(), invocation.scenario, invocation.config
            )
        finally:
            token.var.reset(token)
        return RolloutResult(value=value, metrics=await builder.drain_pending())

    async def close(self) -> None:
        return None
