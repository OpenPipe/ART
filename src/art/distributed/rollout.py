from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
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
        result = await self.hosts[host_id].run(
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
        await asyncio.gather(*(endpoint.close() for endpoint in self.hosts.values()))


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
