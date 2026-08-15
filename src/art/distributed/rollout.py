from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time
from typing import Any, Literal, Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.model import TrainableModel
from art.serving_capabilities import ServingCapabilities
from art.trajectories import (
    MetadataValue,
    PydanticException,
    Trajectory,
    TrajectoryGroup,
)

from .trajectory_store import (
    TrajectoryCapacityError,
    TrajectoryEnqueueResult,
    TrajectoryGroupAnnotations,
    TrajectoryGroupRef,
    TrajectoryLeaseError,
    TrajectoryQueueItem,
    TrajectoryQueueLease,
    TrajectoryQueuePacking,
    TrajectoryQueueRelease,
    TrajectoryQueueResize,
    TrajectoryQueueSnapshot,
    TrajectoryQueueStore,
    TrajectoryQueueTake,
    TrajectoryRecordStore,
)


class InstalledAsyncCallable(BaseModel):
    """Import path for installed user code; functions and closures are never shipped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str = Field(min_length=1)
    qualname: str = Field(min_length=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_import_path(self) -> "InstalledAsyncCallable":
        if self.qualname == "<lambda>" or "<locals>" in self.qualname.split("."):
            raise ValueError(
                "distributed rollout callable must be a top-level function"
            )
        if self.source_sha256 is None:
            object.__setattr__(
                self, "source_sha256", _callable_source_sha256(self._resolve())
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
        assert self.source_sha256 is not None
        return _verified_callable(self.module, self.qualname, self.source_sha256)

    def _resolve(self) -> Callable[..., Awaitable[Any]]:
        value: Any = importlib.import_module(self.module)
        for component in self.qualname.split("."):
            value = getattr(value, component)
        if not inspect.iscoroutinefunction(value):
            raise TypeError(f"{self.module}:{self.qualname} is not an async function")
        return value


@lru_cache(maxsize=128)
def _verified_callable(
    module: str, qualname: str, source_sha256: str
) -> Callable[..., Awaitable[Any]]:
    value: Any = importlib.import_module(module)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not inspect.iscoroutinefunction(value):
        raise TypeError(f"{module}:{qualname} is not an async function")
    if _callable_source_sha256(value) != source_sha256:
        raise RuntimeError(f"installed callable source differs for {module}:{qualname}")
    return value


def _callable_source_sha256(function: Callable[..., Awaitable[Any]]) -> str:
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError("distributed callable must come from a source-backed module")
    try:
        payload = Path(source).read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"cannot read distributed callable source {source}: {error}"
        ) from None
    return hashlib.sha256(payload).hexdigest()


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
    store_result: bool = False


class RolloutResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    value: Any
    metrics: dict[str, float] = Field(default_factory=dict)


class RolloutExecutor(Protocol):
    @property
    def max_workers(self) -> int | None: ...

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
    max_workers: int | None = None

    def __init__(
        self,
        *,
        trajectory_capacity_records: int = 16_384,
        trajectory_capacity_bytes: int = 4 << 30,
    ) -> None:
        self._owner = InProcessRolloutWorker(
            capacity_records=trajectory_capacity_records,
            capacity_bytes=trajectory_capacity_bytes,
        )
        self._owner_endpoints: dict[str, RolloutWorkerEndpoint] = {
            self._owner.owner_actor_id: self._owner
        }
        self._trajectory_capacity_records = trajectory_capacity_records
        self._trajectory_capacity_bytes = trajectory_capacity_bytes
        self._result_queue: DistributedTrajectoryQueue | None = None

    def create_result_queue(self, maxsize: int) -> DistributedTrajectoryQueue:
        if self._result_queue is not None:
            raise RuntimeError("local rollout result queue already exists")
        self._result_queue = DistributedTrajectoryQueue(
            endpoint=_InProcessTrajectoryQueueEndpoint(),
            owner_endpoints=self._owner_endpoints,
            maxsize=maxsize,
            capacity_records=self._trajectory_capacity_records,
            capacity_bytes=self._trajectory_capacity_bytes,
        )
        return self._result_queue

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
        result = await rollout_fn(model, scenario, config)
        if self._result_queue is not None and isinstance(result, TrajectoryGroup):
            return self._owner.store(result)
        return result


class RolloutWorkerEndpoint(Protocol):
    async def run(self, invocation: RolloutInvocation) -> RolloutResult: ...

    async def materialize(self, ref: TrajectoryGroupRef) -> TrajectoryGroup: ...

    async def drop(self, ref: TrajectoryGroupRef) -> None: ...

    async def close(self) -> None: ...


class TrajectoryQueueEndpoint(Protocol):
    async def create(
        self,
        queue_id: str,
        max_ready_groups: int,
        capacity_records: int,
        capacity_bytes: int,
    ) -> None: ...

    async def enqueue(
        self, queue_id: str, item: TrajectoryQueueItem
    ) -> TrajectoryEnqueueResult: ...

    async def resize(self, operation: TrajectoryQueueResize) -> None: ...

    async def take(
        self, queue_id: str, consumer_id: str, count: int
    ) -> TrajectoryQueueTake: ...

    async def mark_packed(self, operation: TrajectoryQueuePacking) -> None: ...

    async def release(self, operation: TrajectoryQueueRelease) -> None: ...

    async def finish(self, queue_id: str) -> None: ...

    async def snapshot(self, queue_id: str) -> TrajectoryQueueSnapshot: ...

    async def close(self, queue_id: str) -> tuple[TrajectoryGroupRef, ...]: ...


class _InProcessTrajectoryQueueEndpoint:
    def __init__(self) -> None:
        self._queues: dict[str, TrajectoryQueueStore] = {}

    async def create(
        self,
        queue_id: str,
        max_ready_groups: int,
        capacity_records: int,
        capacity_bytes: int,
    ) -> None:
        if queue_id in self._queues:
            raise ValueError(f"trajectory queue {queue_id!r} already exists")
        self._queues[queue_id] = TrajectoryQueueStore(
            max_ready_groups=max_ready_groups,
            capacity_records=capacity_records,
            capacity_bytes=capacity_bytes,
        )

    async def enqueue(
        self, queue_id: str, item: TrajectoryQueueItem
    ) -> TrajectoryEnqueueResult:
        return self._queue(queue_id).enqueue(item)

    async def resize(self, operation: TrajectoryQueueResize) -> None:
        self._queue(operation.queue_id).resize(
            maxsize=operation.maxsize, generation=operation.generation
        )

    async def take(
        self, queue_id: str, consumer_id: str, count: int
    ) -> TrajectoryQueueTake:
        return self._queue(queue_id).take(consumer_id, count)

    async def mark_packed(self, operation: TrajectoryQueuePacking) -> None:
        self._queue(operation.queue_id).mark_packed(operation)

    async def release(self, operation: TrajectoryQueueRelease) -> None:
        self._queue(operation.queue_id).release(operation)

    async def finish(self, queue_id: str) -> None:
        self._queue(queue_id).finish()

    async def snapshot(self, queue_id: str) -> TrajectoryQueueSnapshot:
        return self._queue(queue_id).snapshot()

    async def close(self, queue_id: str) -> tuple[TrajectoryGroupRef, ...]:
        queue = self._queues.pop(queue_id, None)
        return () if queue is None else queue.close()

    def _queue(self, queue_id: str) -> TrajectoryQueueStore:
        try:
            return self._queues[queue_id]
        except KeyError:
            raise ValueError(f"unknown trajectory queue {queue_id!r}") from None


class DistributedTrajectoryQueue:
    def __init__(
        self,
        *,
        endpoint: TrajectoryQueueEndpoint,
        owner_endpoints: dict[str, RolloutWorkerEndpoint],
        maxsize: int,
        capacity_records: int,
        capacity_bytes: int,
    ) -> None:
        if maxsize < 1:
            raise ValueError("trajectory queue maxsize must be positive")
        self.endpoint = endpoint
        self.owner_endpoints = owner_endpoints
        self.maxsize = maxsize
        self._effective_maxsize = maxsize
        self._minimum_take_size = 0
        self.capacity_records = capacity_records
        self.capacity_bytes = capacity_bytes
        self.queue_id = uuid.uuid4().hex
        self.consumer_id = f"pipeline:{uuid.uuid4().hex}"
        self.put_waiters = 0
        self._started = False
        self._finished = False
        self._closed = False
        self._cleanup_refs: tuple[TrajectoryGroupRef, ...] = ()
        self._resize_generation = 0
        self._resize_tasks: set[asyncio.Task[None]] = set()
        self._space_waiters: set[asyncio.Future[None]] = set()
        self._item_waiters: set[asyncio.Future[None]] = set()
        self._take_lock = asyncio.Lock()
        self._owner_cleanup_refs: dict[str, deque[TrajectoryGroupRef]] = {}
        self._owner_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._owner_cleanup_failure: BaseException | None = None

    async def start(self) -> None:
        if self._started:
            return
        created_maxsize = self._effective_maxsize
        await self.endpoint.create(
            self.queue_id,
            created_maxsize,
            self.capacity_records,
            self.capacity_bytes,
        )
        self._started = True
        if self._required_maxsize() != created_maxsize:
            self._effective_maxsize = created_maxsize
            self._sync_maxsize()

    def set_maxsize(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("trajectory queue maxsize must be positive")
        if maxsize == self.maxsize:
            return
        self.maxsize = maxsize
        self._sync_maxsize()

    async def put(
        self,
        ref: TrajectoryGroupRef,
        *,
        metadata: dict[str, MetadataValue],
        initial_policy_version: int,
        final_policy_version: int,
        rollout_wall_s: float,
        actor_idle_s: float,
    ) -> tuple[bool, float]:
        started = time.monotonic()
        transferred = False
        self.put_waiters += 1
        try:
            while not self._closed:
                wait_s = time.monotonic() - started
                space_available = asyncio.get_running_loop().create_future()
                self._space_waiters.add(space_available)
                request = asyncio.create_task(
                    self.endpoint.enqueue(
                        self.queue_id,
                        TrajectoryQueueItem(
                            ref=ref,
                            annotations=TrajectoryGroupAnnotations(
                                metadata=metadata,
                                initial_policy_version=initial_policy_version,
                                final_policy_version=final_policy_version,
                                rollout_wall_s=rollout_wall_s,
                                actor_idle_s=actor_idle_s + wait_s,
                                queue_wait_s=wait_s,
                            ),
                        ),
                    )
                )
                try:
                    try:
                        result = await asyncio.shield(request)
                    except asyncio.CancelledError:
                        result = await request
                        transferred = result.status == "accepted"
                        if transferred:
                            self._notify_items()
                        raise
                    if result.status == "accepted":
                        transferred = True
                        self._notify_items()
                        return True, time.monotonic() - started
                    if result.status in ("oversize", "minimum_unreachable"):
                        self._notify_items()
                        raise TrajectoryCapacityError(
                            result.reason or "oversize result"
                        )
                    if result.status == "closed":
                        self._notify_items()
                        return False, time.monotonic() - started
                    await space_available
                finally:
                    self._space_waiters.discard(space_available)
                    if not space_available.done():
                        space_available.cancel()
            return False, time.monotonic() - started
        finally:
            self.put_waiters -= 1
            if not transferred:
                await self._owner(ref).drop(ref)

    async def get(self) -> TrajectoryGroup | None:
        groups, _ = await self.get_many(1, wait=True)
        return groups[0] if groups else None

    async def get_nowait(self) -> tuple[bool, TrajectoryGroup | None]:
        groups, closed = await self.get_many(1, wait=False)
        return bool(groups) or closed, groups[0] if groups else None

    async def get_many(
        self, count: int, *, wait: bool
    ) -> tuple[list[TrajectoryGroup], bool]:
        if count < 1:
            raise ValueError("trajectory queue get count must be positive")
        self._raise_owner_cleanup_failure()
        async with self._take_lock:
            minimum_reserved = wait and count <= self.maxsize
            if minimum_reserved:
                self._minimum_take_size = count
                self._sync_maxsize()
            try:
                if wait:
                    await self._flush_resizes()
                closed = self._closed
                while not closed:
                    item_available = asyncio.get_running_loop().create_future()
                    self._item_waiters.add(item_available)
                    try:
                        # Negative counts retain best-effort bulk reads above the minimum.
                        take = await self._take_trajectories(count if wait else -count)
                        if take.leases:
                            return await self._resolve_many(take.leases), take.closed
                        closed = take.closed
                        if closed or not wait:
                            break
                        self._notify_space()
                        try:
                            await item_available
                        except asyncio.CancelledError:
                            if not self._closed:
                                await self.endpoint.take(
                                    self.queue_id, self.consumer_id, 0
                                )
                            raise
                        closed = self._closed
                    finally:
                        self._item_waiters.discard(item_available)
                        if not item_available.done():
                            item_available.cancel()
                return [], closed
            finally:
                if minimum_reserved:
                    self._minimum_take_size = 0
                    self._sync_maxsize()

    async def discard_group(self, group: TrajectoryGroup) -> None:
        selection = group._distributed_lease
        if not isinstance(selection, DistributedTrajectorySelection):
            return
        await self.release_selection(selection, disposition="discarded")

    async def mark_packed(
        self,
        selections: Sequence[DistributedTrajectorySelection],
        generation_id: str,
    ) -> None:
        if any(selection.queue is not self for selection in selections):
            raise TrajectoryLeaseError("trajectory selection belongs to another queue")
        await self.endpoint.mark_packed(
            TrajectoryQueuePacking(
                queue_id=self.queue_id,
                leases=tuple(selection.lease for selection in selections),
                generation_id=generation_id,
            )
        )

    async def release_selection(
        self,
        selection: DistributedTrajectorySelection,
        *,
        disposition: Literal["consumed", "discarded"],
        generation_id: str | None = None,
    ) -> None:
        await self.release_selections(
            (selection,),
            disposition=disposition,
            generation_id=generation_id,
        )

    async def release_selections(
        self,
        selections: Sequence[DistributedTrajectorySelection],
        *,
        disposition: Literal["consumed", "discarded"],
        generation_id: str | None = None,
    ) -> None:
        if any(selection.queue is not self for selection in selections):
            raise TrajectoryLeaseError("trajectory selection belongs to another queue")
        cleanup = await self._release_many(
            tuple(selection.lease for selection in selections),
            tuple(self._owner(selection.lease.item.ref) for selection in selections),
            disposition=disposition,
            generation_id=generation_id,
        )
        if cleanup:
            raise BaseExceptionGroup("trajectory selection release failed", cleanup)
        self._raise_owner_cleanup_failure()

    async def finish(self) -> None:
        if self._started and not self._finished and not self._closed:
            await self.endpoint.finish(self.queue_id)
            self._finished = True
            self._notify_space()
            self._notify_items()

    async def discard(self, ref: TrajectoryGroupRef) -> None:
        await self._owner(ref).drop(ref)

    async def snapshot(self) -> TrajectoryQueueSnapshot:
        if not self._started or self._closed:
            return TrajectoryQueueSnapshot(
                items=(),
                max_ready_groups=self._effective_maxsize,
                generation=self._resize_generation,
                capacity_records=self.capacity_records,
                capacity_bytes=self.capacity_bytes,
                used_records=0,
                used_bytes=0,
                leased_groups=0,
                ready_groups=0,
                packing_groups=0,
                packed_groups=0,
                released_leases=0,
                lease_lifetime_s=0.0,
                max_lease_lifetime_s=0.0,
            )
        while True:
            await self._flush_resizes()
            snapshot = await self.endpoint.snapshot(self.queue_id)
            if snapshot.generation >= self._resize_generation:
                return snapshot

    async def close(self) -> None:
        failures: list[BaseException] = []
        try:
            await self._flush_resizes()
        except BaseException as error:
            failures.append(error)
        if not self._closed:
            self._closed = True
            self._notify_space()
            self._notify_items()
            if self._started:
                try:
                    self._cleanup_refs += await self.endpoint.close(self.queue_id)
                except BaseException as error:
                    failures.append(error)
        if self._owner_cleanup_tasks:
            await asyncio.gather(*tuple(self._owner_cleanup_tasks.values()))
        refs = self._cleanup_refs
        self._owner_cleanup_failure = None
        results = await asyncio.gather(
            *(self._owner(ref).drop(ref) for ref in refs), return_exceptions=True
        )
        self._cleanup_refs = tuple(
            ref
            for ref, result in zip(refs, results, strict=True)
            if isinstance(result, BaseException)
        )
        failures.extend(
            result for result in results if isinstance(result, BaseException)
        )
        if failures:
            raise BaseExceptionGroup("trajectory queue cleanup failed", failures)

    def _required_maxsize(self) -> int:
        return max(self.maxsize, self._minimum_take_size)

    def _sync_maxsize(self) -> None:
        maxsize = self._required_maxsize()
        if maxsize == self._effective_maxsize:
            return
        self._effective_maxsize = maxsize
        if self._started and not self._closed:
            self._schedule_resize(maxsize)

    def _schedule_resize(self, maxsize: int) -> None:
        self._resize_generation += 1
        operation = TrajectoryQueueResize(
            queue_id=self.queue_id,
            maxsize=maxsize,
            generation=self._resize_generation,
        )

        async def resize() -> None:
            await self.endpoint.resize(operation)
            self._notify_space()
            self._notify_items()

        task = asyncio.create_task(resize())
        self._resize_tasks.add(task)

    async def _flush_resizes(self) -> None:
        while self._resize_tasks:
            tasks = tuple(self._resize_tasks)
            self._resize_tasks.difference_update(tasks)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if failures:
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup("trajectory queue resize failed", failures)

    async def _consume(self, lease: TrajectoryQueueLease) -> TrajectoryGroup:
        item = lease.item
        owner = self._owner(item.ref)
        try:
            group = await owner.materialize(item.ref)
        except BaseException as error:
            cleanup = await self._release(lease, owner)
            if cleanup:
                raise BaseExceptionGroup(
                    "trajectory materialization and release failed", [error, *cleanup]
                ) from None
            raise
        cleanup = await self._release(lease, owner)
        if cleanup:
            raise BaseExceptionGroup("trajectory result release failed", cleanup)
        return item.apply_annotations(group)

    async def _resolve_many(
        self, leases: Sequence[TrajectoryQueueLease]
    ) -> list[TrajectoryGroup]:
        return [self._summary_group(lease) for lease in leases]

    async def _take_trajectories(self, count: int) -> TrajectoryQueueTake:
        request = asyncio.create_task(
            self.endpoint.take(self.queue_id, self.consumer_id, count)
        )
        try:
            return await asyncio.shield(request)
        except asyncio.CancelledError as cancelled:
            take = await request
            if take.leases:
                cleanup = await self._release_many(
                    take.leases,
                    tuple(self._owner(lease.item.ref) for lease in take.leases),
                    disposition="discarded",
                    generation_id=None,
                )
                if cleanup:
                    raise BaseExceptionGroup(
                        "trajectory acquisition cancellation cleanup failed",
                        [cancelled, *cleanup],
                    ) from None
            elif count > 0 and not take.closed and not self._closed:
                await self.endpoint.take(self.queue_id, self.consumer_id, 0)
            raise

    async def materialize_selection(
        self, selection: DistributedTrajectorySelection
    ) -> TrajectoryGroup:
        if selection.queue is not self:
            raise TrajectoryLeaseError("trajectory selection belongs to another queue")
        item = selection.lease.item
        return item.apply_annotations(await self._owner(item.ref).materialize(item.ref))

    def _summary_group(self, lease: TrajectoryQueueLease) -> TrajectoryGroup:
        item = lease.item
        descriptor = item.ref.descriptor
        trajectories = []
        for reward, initial, final, counts, metrics, metadata in zip(
            descriptor.rewards,
            descriptor.trajectory_initial_policy_versions,
            descriptor.trajectory_final_policy_versions,
            descriptor.trajectory_policy_token_counts,
            descriptor.trajectory_metrics,
            descriptor.trajectory_metadata,
            strict=True,
        ):
            trajectory = Trajectory(
                reward=reward,
                initial_policy_version=(
                    initial
                    if initial is not None
                    else item.annotations.initial_policy_version
                ),
                final_policy_version=(
                    final
                    if final is not None
                    else item.annotations.final_policy_version
                ),
                metrics=dict(metrics),
                metadata=dict(metadata),
            )
            trajectory._policy_token_counts = dict(counts)
            trajectories.append(trajectory)
        group = TrajectoryGroup(
            trajectories,
            metadata={**descriptor.group_metadata, **item.annotations.metadata},
            metrics=dict(descriptor.group_metrics),
        )
        group.exceptions = [
            PydanticException(type=kind, message=message, traceback="")
            for kind, message in descriptor.exceptions
        ]
        group.metadata["_art_rollout_wall_s"] = item.annotations.rollout_wall_s
        group.metadata["_art_actor_idle_s"] = item.annotations.actor_idle_s
        group.metadata["_art_queue_wait_s"] = item.annotations.queue_wait_s
        group._distributed_lease = DistributedTrajectorySelection(self, lease)
        return group

    async def _release(
        self,
        lease: TrajectoryQueueLease,
        owner: RolloutWorkerEndpoint,
        *,
        disposition: Literal["consumed", "discarded"] = "discarded",
        generation_id: str | None = None,
    ) -> list[BaseException]:
        return await self._release_many(
            (lease,),
            (owner,),
            disposition=disposition,
            generation_id=generation_id,
        )

    async def _release_many(
        self,
        leases: tuple[TrajectoryQueueLease, ...],
        owners: tuple[RolloutWorkerEndpoint, ...],
        *,
        disposition: Literal["consumed", "discarded"],
        generation_id: str | None,
    ) -> list[BaseException]:
        if not leases:
            return []
        try:
            await self.endpoint.release(
                TrajectoryQueueRelease(
                    queue_id=self.queue_id,
                    leases=leases,
                    generation_id=generation_id,
                    disposition=disposition,
                )
            )
        except BaseException as error:
            return [error]
        self._notify_space()
        for lease, owner in zip(leases, owners, strict=True):
            self._schedule_owner_cleanup(owner, lease.item.ref)
        return []

    def _notify_space(self) -> None:
        for waiter in tuple(self._space_waiters):
            if not waiter.done():
                waiter.set_result(None)

    def _notify_items(self) -> None:
        for waiter in tuple(self._item_waiters):
            if not waiter.done():
                waiter.set_result(None)

    def _schedule_owner_cleanup(
        self, owner: RolloutWorkerEndpoint, ref: TrajectoryGroupRef
    ) -> None:
        owner_id = ref.owner_actor_id
        self._owner_cleanup_refs.setdefault(owner_id, deque()).append(ref)
        if owner_id not in self._owner_cleanup_tasks:
            self._owner_cleanup_tasks[owner_id] = asyncio.create_task(
                self._drain_owner_cleanup(owner_id, owner)
            )

    async def _drain_owner_cleanup(
        self, owner_id: str, owner: RolloutWorkerEndpoint
    ) -> None:
        refs = self._owner_cleanup_refs[owner_id]
        while refs:
            ref = refs.popleft()
            try:
                await owner.drop(ref)
            except Exception as error:
                self._cleanup_refs += (ref,)
                self._owner_cleanup_failure = self._owner_cleanup_failure or error
        del self._owner_cleanup_refs[owner_id]
        del self._owner_cleanup_tasks[owner_id]

    def _raise_owner_cleanup_failure(self) -> None:
        error = self._owner_cleanup_failure
        self._owner_cleanup_failure = None
        if error is not None:
            raise error

    def _owner(self, ref: TrajectoryGroupRef) -> RolloutWorkerEndpoint:
        try:
            return self.owner_endpoints[ref.owner_actor_id]
        except KeyError:
            raise RuntimeError(
                f"trajectory owner {ref.owner_actor_id!r} is unavailable"
            ) from None


class DistributedTrajectorySelection:
    __slots__ = ("lease", "queue")

    def __init__(
        self, queue: DistributedTrajectoryQueue, lease: TrajectoryQueueLease
    ) -> None:
        self.queue = queue
        self.lease = lease


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
        hosts: Mapping[str, Sequence[RolloutWorkerEndpoint]],
        target_workers: int,
        queue_endpoint: TrajectoryQueueEndpoint | None = None,
        trajectory_capacity_records: int = 16_384,
        trajectory_capacity_bytes: int = 4 << 30,
    ) -> None:
        if not hosts or any(not endpoints for endpoints in hosts.values()):
            raise ValueError("rollout hosts must each provide at least one endpoint")
        self.callable = callable
        self.hosts = {host: tuple(endpoints) for host, endpoints in hosts.items()}
        self.max_workers = sum(len(endpoints) for endpoints in self.hosts.values())
        self._worker_endpoints: tuple[RolloutWorkerEndpoint, ...] = ()
        self._endpoint_by_worker: dict[int, RolloutWorkerEndpoint] = {}
        self._queue_endpoint = queue_endpoint
        self._trajectory_capacity_records = trajectory_capacity_records
        self._trajectory_capacity_bytes = trajectory_capacity_bytes
        self._endpoint_by_owner: dict[str, RolloutWorkerEndpoint] = {}
        self._result_queue: DistributedTrajectoryQueue | None = None
        self.set_target(target_workers)

    def create_result_queue(self, maxsize: int) -> DistributedTrajectoryQueue:
        if self._result_queue is not None:
            raise RuntimeError("distributed rollout result queue already exists")
        queue_endpoint = self._queue_endpoint
        if queue_endpoint is None:
            endpoints = next(iter(self.hosts.values()))
            if len(self.hosts) != 1 or not all(
                isinstance(endpoint, InProcessRolloutWorker) for endpoint in endpoints
            ):
                raise RuntimeError(
                    "queue_endpoint is required unless one host uses only "
                    "in-process rollout workers"
                )
            queue_endpoint = _InProcessTrajectoryQueueEndpoint()
            self._queue_endpoint = queue_endpoint
        self._result_queue = DistributedTrajectoryQueue(
            endpoint=queue_endpoint,
            owner_endpoints=self._endpoint_by_owner,
            maxsize=maxsize,
            capacity_records=self._trajectory_capacity_records,
            capacity_bytes=self._trajectory_capacity_bytes,
        )
        return self._result_queue

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
        drained = len(workers) <= len(self._worker_endpoints)
        assignments = {
            worker_id: self._endpoint_by_worker[worker_id]
            for worker_id in workers
            if worker_id in self._endpoint_by_worker
            and (
                not drained
                or self._endpoint_by_worker[worker_id] in self._worker_endpoints
            )
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
                store_result=self._result_queue is not None,
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
        if isinstance(result.value, TrajectoryGroupRef):
            existing = self._endpoint_by_owner.setdefault(
                result.value.owner_actor_id, endpoint
            )
            if existing is not endpoint:
                raise RuntimeError(
                    f"trajectory owner {result.value.owner_actor_id!r} changed endpoint"
                )
        return result.value

    async def close(self) -> None:
        failures: list[BaseException] = []
        if self._result_queue is not None:
            try:
                await self._result_queue.close()
            except BaseException as error:
                failures.append(error)
        results = await asyncio.gather(
            *(
                endpoint.close()
                for endpoints in self.hosts.values()
                for endpoint in endpoints
            ),
            return_exceptions=True,
        )
        failures.extend(
            result for result in results if isinstance(result, BaseException)
        )
        if failures:
            raise BaseExceptionGroup("distributed rollout cleanup failed", failures)


class InProcessRolloutWorker:
    """One in-process rollout execution slot used by local collapse and tests."""

    def __init__(
        self, *, capacity_records: int = 16_384, capacity_bytes: int = 4 << 30
    ) -> None:
        self._results = TrajectoryRecordStore(
            owner_actor_id=f"in-process:{uuid.uuid4().hex}",
            capacity_records=capacity_records,
            capacity_bytes=capacity_bytes,
        )

    @property
    def owner_actor_id(self) -> str:
        return self._results.owner_actor_id

    def store(self, group: TrajectoryGroup) -> TrajectoryGroupRef:
        return self._results.put(group)

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
        if invocation.store_result and isinstance(value, TrajectoryGroup):
            value = self._results.put(value)
        return RolloutResult(value=value, metrics=await builder.drain_pending())

    async def materialize(self, ref: TrajectoryGroupRef) -> TrajectoryGroup:
        return self._results.materialize(ref)

    async def drop(self, ref: TrajectoryGroupRef) -> None:
        self._results.drop(ref)

    async def close(self) -> None:
        self._results.close()
