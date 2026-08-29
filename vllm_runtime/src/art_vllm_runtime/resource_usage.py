"""Exact GPU service and physical KV residency for paired vLLM."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
from threading import RLock
import time
from typing import Any

from art_vllm_runtime.runtime_usage import RuntimeRequestContext, runtime_usage_journal

_TRACE_HEADER = "art-runtime-usage-v1"
_CACHE_KEY = "art_runtime_owner_v1"
_GPU_OWNERS = "_art_gpu_owners"
_GPU_ALLOCATIONS = "_art_gpu_allocations"
_GPU_TRACKER = "_art_gpu_tracker"
_GPU_STAGES = "_art_gpu_stages"
_KV_TRACKER = "_art_kv_tracker"
_KV_BYTES_PER_BLOCK = "_art_physical_kv_bytes_per_block"
_OUTPUT_FIELD = "art_runtime_usage_updates"
_MAX_BATCHES = 4_096
_MAX_UPDATES = 4_096


@dataclass(frozen=True, slots=True)
class RuntimeUsageOwner:
    request_id: str
    tenant_id: str
    run_id: str
    service_tier: str
    model: str

    def __post_init__(self) -> None:
        for value, name, maximum in (
            (self.request_id, "request_id", 255),
            (self.tenant_id, "tenant_id", 255),
            (self.run_id, "run_id", 255),
            (self.service_tier, "service_tier", 128),
            (self.model, "model", 128),
        ):
            if not isinstance(value, str) or not value or len(value) > maximum:
                raise ValueError(f"runtime usage {name} is invalid")

    def payload(self) -> dict[str, str]:
        return {
            "model": self.model,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "service_tier": self.service_tier,
            "tenant_id": self.tenant_id,
        }

    def trace_value(self) -> str:
        return json.dumps(
            self.payload(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    @property
    def kv_owner(self) -> KVUsageOwner:
        return KVUsageOwner(
            tenant_id=self.tenant_id,
            run_id=self.run_id,
            service_tier=self.service_tier,
            model=self.model,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RuntimeUsageOwner:
        expected = {"model", "request_id", "run_id", "service_tier", "tenant_id"}
        if set(payload) != expected or not all(
            isinstance(payload[name], str) for name in expected
        ):
            raise ValueError("runtime usage owner payload is invalid")
        return cls(
            request_id=str(payload["request_id"]),
            tenant_id=str(payload["tenant_id"]),
            run_id=str(payload["run_id"]),
            service_tier=str(payload["service_tier"]),
            model=str(payload["model"]),
        )

    @classmethod
    def from_trace_value(cls, value: str) -> RuntimeUsageOwner:
        if not isinstance(value, str) or len(value) > 1_024:
            raise ValueError("runtime usage owner trace is invalid")
        try:
            payload = json.loads(value)
        except (TypeError, ValueError) as error:
            raise ValueError("runtime usage owner trace is invalid") from error
        if not isinstance(payload, Mapping):
            raise ValueError("runtime usage owner trace is invalid")
        owner = cls.from_payload(payload)
        if owner.trace_value() != value:
            raise ValueError("runtime usage owner trace is not canonical")
        return owner


@dataclass(frozen=True, slots=True)
class KVUsageOwner:
    tenant_id: str
    run_id: str
    service_tier: str
    model: str

    def __post_init__(self) -> None:
        for value, name, maximum in (
            (self.tenant_id, "tenant_id", 255),
            (self.run_id, "run_id", 255),
            (self.service_tier, "service_tier", 128),
            (self.model, "model", 128),
        ):
            if not isinstance(value, str) or not value or len(value) > maximum:
                raise ValueError(f"KV usage {name} is invalid")

    def payload(self) -> dict[str, str]:
        return {
            "model": self.model,
            "run_id": self.run_id,
            "service_tier": self.service_tier,
            "tenant_id": self.tenant_id,
        }

    def trace_value(self) -> str:
        return json.dumps(
            self.payload(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> KVUsageOwner:
        expected = {"model", "run_id", "service_tier", "tenant_id"}
        if set(payload) != expected or not all(
            isinstance(payload[name], str) for name in expected
        ):
            raise ValueError("KV usage owner payload is invalid")
        return cls(
            tenant_id=str(payload["tenant_id"]),
            run_id=str(payload["run_id"]),
            service_tier=str(payload["service_tier"]),
            model=str(payload["model"]),
        )


_CURRENT_OWNER: ContextVar[RuntimeUsageOwner | None] = ContextVar(
    "art_runtime_usage_owner", default=None
)
_CURRENT_BLOCK_OWNER: ContextVar[KVUsageOwner | None] = ContextVar(
    "art_runtime_kv_block_owner", default=None
)


@contextmanager
def bind_runtime_usage_context(
    request_id: str, context: RuntimeRequestContext
) -> Iterator[None]:
    owner = RuntimeUsageOwner(
        request_id=request_id,
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        service_tier=context.service_tier,
        model=context.model,
    )
    token = _CURRENT_OWNER.set(owner)
    try:
        yield
    finally:
        _CURRENT_OWNER.reset(token)


def bind_request_usage_owner(request: Any) -> None:
    """Replace any caller-supplied owner with authenticated frontend context."""

    headers = dict(getattr(request, "trace_headers", None) or {})
    headers.pop(_TRACE_HEADER, None)
    if owner := _CURRENT_OWNER.get():
        headers[_TRACE_HEADER] = owner.trace_value()
    request.trace_headers = headers or None


def request_usage_owner(request: Any) -> RuntimeUsageOwner | None:
    headers = getattr(request, "trace_headers", None)
    if not isinstance(headers, Mapping) or _TRACE_HEADER not in headers:
        return None
    try:
        return RuntimeUsageOwner.from_trace_value(headers[_TRACE_HEADER])
    except (TypeError, ValueError) as error:
        raise RuntimeError("runtime usage owner trace is invalid") from error


def usage_cache_extra_key(request: Any) -> tuple[str, str] | None:
    owner = request_usage_owner(request)
    if owner is None:
        return None
    digest = hashlib.sha256(owner.kv_owner.trace_value().encode("ascii")).hexdigest()
    return _CACHE_KEY, digest


@dataclass(slots=True)
class _ActiveBatch:
    weights: dict[str, int]
    allocations: dict[str, int] = field(default_factory=dict)


class GPUServiceTracker:
    """Conserve one engine's GPU time across overlapping executor stages."""

    def __init__(
        self,
        world_size: int,
        *,
        monotonic_ns: Any = time.monotonic_ns,
        max_batches: int = _MAX_BATCHES,
    ) -> None:
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size < 1
        ):
            raise ValueError("GPU service world size must be positive")
        if (
            isinstance(max_batches, bool)
            or not isinstance(max_batches, int)
            or max_batches < 1
        ):
            raise ValueError("GPU service batch capacity must be positive")
        self.world_size = world_size
        self._clock = monotonic_ns
        self._max_batches = max_batches
        self._active: dict[int, _ActiveBatch] = {}
        self._next_sequence = 0
        self._last_ns: int | None = None
        self._lock = RLock()

    def start(self, scheduler_output: Any) -> int | None:
        weights = dict(scheduler_output.num_scheduled_tokens)
        if not weights:
            return None
        if (
            any(
                not isinstance(request_id, str)
                or not request_id
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens <= 0
                for request_id, tokens in weights.items()
            )
            or sum(weights.values()) != scheduler_output.total_num_scheduled_tokens
        ):
            raise RuntimeError("GPU service scheduler weights are invalid")
        with self._lock:
            self._advance(self._now())
            if len(self._active) >= self._max_batches:
                raise RuntimeError("GPU service active-batch capacity exhausted")
            sequence = self._next_sequence
            self._next_sequence += 1
            self._active[sequence] = _ActiveBatch(
                weights=weights,
                allocations={request_id: 0 for request_id in weights},
            )
            return sequence

    def finish(self, sequence: int | None, scheduler_output: Any) -> None:
        if sequence is None:
            return
        with self._lock:
            batch = self._active.get(sequence)
            if batch is None:
                raise RuntimeError("GPU service batch completion is stale")
            self._advance(self._now())
            self._active.pop(sequence)
            prior = getattr(scheduler_output, _GPU_ALLOCATIONS, None)
            if prior is None:
                allocations = batch.allocations
            elif isinstance(prior, Mapping) and set(prior) == set(batch.allocations):
                allocations = {
                    request_id: prior[request_id] + value
                    for request_id, value in batch.allocations.items()
                }
            else:
                raise RuntimeError("GPU service stage allocation is inconsistent")
            setattr(scheduler_output, _GPU_ALLOCATIONS, allocations)

    def _now(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("GPU service monotonic timestamp is invalid")
        return value

    def _advance(self, now: int) -> None:
        previous = self._last_ns
        if previous is not None and now < previous:
            raise RuntimeError("GPU service monotonic time moved backwards")
        self._last_ns = now
        if previous is None or now == previous or not self._active:
            return
        weighted = [
            (sequence, request_id, tokens)
            for sequence, batch in sorted(self._active.items())
            for request_id, tokens in sorted(batch.weights.items())
        ]
        total_weight = sum(tokens for _, _, tokens in weighted)
        service_ns = (now - previous) * self.world_size
        cumulative_weight = allocated = 0
        for sequence, request_id, tokens in weighted:
            cumulative_weight += tokens
            cumulative = service_ns * cumulative_weight // total_weight
            self._active[sequence].allocations[request_id] += cumulative - allocated
            allocated = cumulative
        if allocated != service_ns:
            raise RuntimeError("GPU service allocation did not conserve time")


@dataclass(slots=True)
class _PendingStage:
    scheduler_output: Any
    claimed: bool = False


class _StageCoordinator:
    def __init__(self) -> None:
        self._pending: deque[_PendingStage] = deque()
        self._lock = RLock()

    def register(self, scheduler_output: Any) -> _PendingStage:
        with self._lock:
            if len(self._pending) >= _MAX_BATCHES:
                raise RuntimeError("GPU sample-stage capacity exhausted")
            pending = _PendingStage(scheduler_output)
            self._pending.append(pending)
            return pending

    def resolve(self, pending: _PendingStage, requires_sample: bool) -> None:
        with self._lock:
            if pending.claimed or requires_sample:
                return
            try:
                self._pending.remove(pending)
            except ValueError as error:
                raise RuntimeError("GPU execute stage is stale") from error

    def claim(self) -> Any:
        with self._lock:
            if not self._pending:
                raise RuntimeError("GPU sample stage has no execution")
            pending = self._pending.popleft()
            pending.claimed = True
            return pending.scheduler_output


class PhysicalKVTracker:
    """Track each resident physical block once, including prefix-cache blocks."""

    def __init__(
        self,
        kv_cache_config: Any,
        *,
        monotonic_ns: Any = time.monotonic_ns,
        max_updates: int = _MAX_UPDATES,
    ) -> None:
        self.num_blocks = kv_cache_config.num_blocks
        bytes_per_block = getattr(kv_cache_config, _KV_BYTES_PER_BLOCK, None)
        if (
            isinstance(self.num_blocks, bool)
            or not isinstance(self.num_blocks, int)
            or self.num_blocks < 1
            or isinstance(bytes_per_block, bool)
            or not isinstance(bytes_per_block, int)
            or bytes_per_block < 1
        ):
            raise RuntimeError("physical KV geometry is invalid")
        self.bytes_per_block = bytes_per_block
        self._clock = monotonic_ns
        self._max_updates = max_updates
        self._owners: dict[int, KVUsageOwner] = {}
        self._bytes: dict[KVUsageOwner, int] = {}
        self._emitted: dict[KVUsageOwner, int] = {}
        self._changed: set[KVUsageOwner] = set()
        self._pending: list[dict[str, object]] = []
        self._batch_depth = 0

    @contextmanager
    def batch(self) -> Iterator[None]:
        outer = self._batch_depth == 0
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if outer:
                self._flush()

    def assign(self, blocks: Iterable[Any], owner: KVUsageOwner | None) -> None:
        for block_id in {
            self._block_id(block) for block in blocks if not block.is_null
        }:
            existing = self._owners.get(block_id)
            if owner is None:
                if existing is not None:
                    raise RuntimeError("unattributed request reused owned KV")
                continue
            if existing is not None and existing != owner:
                raise RuntimeError("physical KV crossed request ownership")
            if existing is None:
                self._owners[block_id] = owner
                self._bytes[owner] = self._bytes.get(owner, 0) + self.bytes_per_block
                self._mark(owner)

    def release_unresident(self, blocks: Iterable[Any]) -> None:
        for block in blocks:
            if block.is_null or block.ref_cnt != 0 or block.block_hash is not None:
                continue
            owner = self._owners.pop(self._block_id(block), None)
            if owner is None:
                continue
            remaining = self._bytes[owner] - self.bytes_per_block
            if remaining < 0:
                raise RuntimeError("physical KV accounting underflow")
            if remaining:
                self._bytes[owner] = remaining
            else:
                self._bytes.pop(owner)
            self._mark(owner)

    def take_updates(self) -> list[dict[str, object]]:
        updates = self._pending
        self._pending = []
        return updates

    def _block_id(self, block: Any) -> int:
        value = block.block_id
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < self.num_blocks
        ):
            raise RuntimeError("physical KV block identity is invalid")
        return value

    def _mark(self, owner: KVUsageOwner) -> None:
        self._changed.add(owner)
        if self._batch_depth == 0:
            self._flush()

    def _flush(self) -> None:
        changed = {
            owner: self._bytes.get(owner, 0)
            for owner in self._changed
            if self._emitted.get(owner) != self._bytes.get(owner, 0)
            and (self._bytes.get(owner, 0) or owner in self._emitted)
        }
        if len(self._pending) + len(changed) > self._max_updates:
            raise RuntimeError("physical KV update capacity exhausted")
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, int) or now < 0:
            raise RuntimeError("physical KV timestamp is invalid")
        for owner, byte_count in sorted(
            changed.items(), key=lambda item: item[0].trace_value()
        ):
            self._pending.append(
                {
                    "byte_count": byte_count,
                    "kind": "kv",
                    "monotonic_ns": now,
                    "owner": owner.payload(),
                    "version": 1,
                }
            )
            if byte_count:
                self._emitted[owner] = byte_count
            else:
                self._emitted.pop(owner)
        self._changed.clear()


def _append_updates(target: Any, updates: Sequence[dict[str, object]]) -> None:
    if not updates:
        return
    existing = getattr(target, _OUTPUT_FIELD, None)
    setattr(target, _OUTPUT_FIELD, [*(existing or ()), *updates])


def attach_scheduler_usage(
    scheduler: Any,
    scheduler_output: Any,
    outputs_by_client: MutableMapping[int, Any],
) -> None:
    allocations = getattr(scheduler_output, _GPU_ALLOCATIONS, None)
    owners = getattr(scheduler_output, _GPU_OWNERS, None)
    if scheduler_output.total_num_scheduled_tokens:
        if not isinstance(owners, Mapping):
            raise RuntimeError("GPU service owner transport is unavailable")
        if not isinstance(allocations, Mapping):
            if owners:
                raise RuntimeError("GPU service allocation transport is unavailable")
            allocations = {}
        for request_id, payload in owners.items():
            if request_id not in allocations:
                raise RuntimeError("GPU service request allocation is unavailable")
            client_index, owner_payload = payload
            target = outputs_by_client.get(client_index)
            if target is None:
                from vllm.v1.engine import EngineCoreOutputs

                target = outputs_by_client[client_index] = EngineCoreOutputs()
            owner = RuntimeUsageOwner.from_payload(owner_payload)
            _append_updates(
                target,
                [
                    {
                        "gpu_service_ns": allocations[request_id],
                        "kind": "gpu",
                        "request_id": owner.request_id,
                        "version": 1,
                    }
                ],
            )
    tracker = getattr(scheduler.kv_cache_manager.block_pool, _KV_TRACKER, None)
    if not isinstance(tracker, PhysicalKVTracker):
        raise RuntimeError("physical KV tracker is unavailable")
    updates = tracker.take_updates()
    if updates:
        target = outputs_by_client.get(0)
        if target is None:
            from vllm.v1.engine import EngineCoreOutputs

            target = outputs_by_client[0] = EngineCoreOutputs()
        _append_updates(target, updates)


def consume_runtime_usage_updates(outputs: Any) -> None:
    updates = getattr(outputs, _OUTPUT_FIELD, None)
    if updates is None:
        return
    _consume_updates(outputs.engine_index, updates)


def _consume_updates(engine_index: Any, updates: Any) -> None:
    if not isinstance(updates, list) or not all(
        isinstance(item, Mapping) for item in updates
    ):
        raise RuntimeError("runtime usage core output is invalid")
    if not updates:
        return
    journal = runtime_usage_journal()
    for update in updates:
        if update.get("version") != 1:
            raise RuntimeError("runtime usage core output version is invalid")
        kind = update.get("kind")
        if kind == "gpu" and set(update) == {
            "gpu_service_ns",
            "kind",
            "request_id",
            "version",
        }:
            journal.record_gpu_service(update["request_id"], update["gpu_service_ns"])
        elif kind == "kv" and set(update) == {
            "byte_count",
            "kind",
            "monotonic_ns",
            "owner",
            "version",
        }:
            owner_payload = update["owner"]
            if not isinstance(owner_payload, Mapping):
                raise RuntimeError("physical KV owner payload is invalid")
            journal.record_kv_residency(
                engine_index,
                KVUsageOwner.from_payload(owner_payload),
                byte_count=update["byte_count"],
                monotonic_ns=update["monotonic_ns"],
            )
        else:
            raise RuntimeError("runtime usage core output shape is invalid")


async def drain_runtime_usage(engine_client: Any) -> None:
    result = await engine_client.engine_core.call_utility_async("art_drain_usage")
    if not isinstance(result, Mapping) or set(result) != {"engine_index", "updates"}:
        raise RuntimeError("runtime usage drain response is invalid")
    updates = result["updates"]
    if not isinstance(updates, list):
        raise RuntimeError("runtime usage drain response is invalid")
    _consume_updates(result["engine_index"], updates)


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.schedule
    if getattr(original, "__art_resource_usage_patched__", False):
        return

    def schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original(self, *args, **kwargs)
        owners = {}
        for request_id in output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is None:
                raise RuntimeError("runtime usage lost a scheduled request")
            owner = request_usage_owner(request)
            if owner is not None:
                owners[request_id] = (request.client_index, owner.payload())
        setattr(output, _GPU_OWNERS, owners)
        return output

    setattr(schedule, "__art_resource_usage_patched__", True)
    Scheduler.schedule = schedule  # type: ignore[method-assign]


def _patch_executor(executor_class: type[Any]) -> None:
    original_execute = executor_class.execute_model
    if getattr(original_execute, "__art_resource_usage_patched__", False):
        return
    original_sample = executor_class.sample_tokens

    def execute_model(self: Any, scheduler_output: Any, non_block: bool = False) -> Any:
        tracker, stages = _gpu_tracking(self)
        pending = stages.register(scheduler_output)
        sequence = tracker.start(scheduler_output)

        def finish(value: Any) -> Any:
            stages.resolve(pending, value is None)
            tracker.finish(sequence, scheduler_output)
            return value

        def fail() -> None:
            stages.resolve(pending, False)
            tracker.finish(sequence, scheduler_output)

        try:
            result = original_execute(self, scheduler_output, non_block=non_block)
        except BaseException:
            stages.resolve(pending, False)
            tracker.finish(sequence, scheduler_output)
            raise
        return _account_future(result, finish, fail)

    def sample_tokens(self: Any, grammar_output: Any, non_block: bool = False) -> Any:
        tracker, stages = _gpu_tracking(self)
        scheduler_output = stages.claim()
        sequence = tracker.start(scheduler_output)

        def finish(value: Any) -> Any:
            tracker.finish(sequence, scheduler_output)
            return value

        def fail() -> None:
            tracker.finish(sequence, scheduler_output)

        try:
            result = original_sample(self, grammar_output, non_block=non_block)
        except BaseException:
            tracker.finish(sequence, scheduler_output)
            raise
        return _account_future(result, finish, fail)

    setattr(execute_model, "__art_resource_usage_patched__", True)
    setattr(sample_tokens, "__art_resource_usage_patched__", True)
    executor_class.execute_model = execute_model
    executor_class.sample_tokens = sample_tokens


def _account_future(result: Any, finish: Any, fail: Any) -> Any:
    if not isinstance(result, Future):
        return finish(result)
    return _AccountingFuture(result, finish, fail)


class _AccountingFuture(Future[Any]):
    def __init__(self, source: Future[Any], finish: Any, fail: Any) -> None:
        super().__init__()
        self._source = source
        self._finish = finish
        self._fail = fail
        self._resolve_lock = RLock()

    def result(self, timeout: float | None = None) -> Any:
        with self._resolve_lock:
            if not self.done():
                try:
                    value = self._source.result(timeout)
                except BaseException as execution_error:
                    try:
                        self._fail()
                    except BaseException as accounting_error:
                        self.set_exception(accounting_error)
                    else:
                        self.set_exception(execution_error)
                else:
                    try:
                        self.set_result(self._finish(value))
                    except BaseException as accounting_error:
                        self.set_exception(accounting_error)
        return super().result(timeout)


def _gpu_tracking(executor: Any) -> tuple[GPUServiceTracker, _StageCoordinator]:
    tracker = getattr(executor, _GPU_TRACKER, None)
    stages = getattr(executor, _GPU_STAGES, None)
    if tracker is None and stages is None:
        tracker = GPUServiceTracker(executor.vllm_config.parallel_config.world_size)
        stages = _StageCoordinator()
        setattr(executor, _GPU_TRACKER, tracker)
        setattr(executor, _GPU_STAGES, stages)
    if not isinstance(tracker, GPUServiceTracker) or not isinstance(
        stages, _StageCoordinator
    ):
        raise RuntimeError("GPU service tracker state is invalid")
    return tracker, stages


def _patch_executors() -> None:
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    _patch_executor(MultiprocExecutor)
    _patch_executor(UniProcExecutor)


def _patch_engine_core_drain() -> None:
    from vllm.v1.engine.core import EngineCore

    def art_drain_usage(self: Any) -> dict[str, object]:
        tracker = getattr(self.scheduler.kv_cache_manager.block_pool, _KV_TRACKER)
        return {
            "engine_index": getattr(self, "engine_index", 0),
            "updates": tracker.take_updates(),
        }

    setattr(EngineCore, "art_drain_usage", art_drain_usage)


def _patch_kv() -> None:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    import vllm.v1.engine.core as engine_core

    if getattr(KVCacheManager, "_art_resource_usage_patched", False):
        return

    original_geometry = kv_cache_utils.generate_scheduler_kv_cache_config

    def generate_scheduler_kv_cache_config(configs: list[Any]) -> Any:
        config = original_geometry(configs)
        total = sum(
            tensor.size for worker in configs for tensor in worker.kv_cache_tensors
        )
        if total <= 0 or total % config.num_blocks:
            raise RuntimeError("physical KV bytes cannot be derived")
        setattr(config, _KV_BYTES_PER_BLOCK, total // config.num_blocks)
        return config

    setattr(
        kv_cache_utils,
        "generate_scheduler_kv_cache_config",
        generate_scheduler_kv_cache_config,
    )
    setattr(
        engine_core,
        "generate_scheduler_kv_cache_config",
        generate_scheduler_kv_cache_config,
    )

    original_init = KVCacheManager.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        setattr(self.block_pool, _KV_TRACKER, PhysicalKVTracker(self.kv_cache_config))

    original_allocate = KVCacheManager.allocate_slots

    def allocate_slots(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        tracker: PhysicalKVTracker = getattr(self.block_pool, _KV_TRACKER)
        owner = request_usage_owner(request)
        token = _CURRENT_BLOCK_OWNER.set(None if owner is None else owner.kv_owner)
        try:
            with tracker.batch():
                return original_allocate(self, request, *args, **kwargs)
        finally:
            _CURRENT_BLOCK_OWNER.reset(token)

    original_get = BlockPool.get_new_blocks

    def get_new_blocks(self: Any, num_blocks: int) -> list[Any]:
        blocks = original_get(self, num_blocks)
        getattr(self, _KV_TRACKER).assign(blocks, _CURRENT_BLOCK_OWNER.get())
        return blocks

    original_touch = BlockPool.touch

    def touch(self: Any, blocks: Sequence[Any]) -> None:
        getattr(self, _KV_TRACKER).assign(blocks, _CURRENT_BLOCK_OWNER.get())
        original_touch(self, blocks)

    original_free = BlockPool.free_blocks

    def free_blocks(self: Any, ordered_blocks: Iterable[Any]) -> None:
        blocks = tuple(ordered_blocks)
        tracker: PhysicalKVTracker = getattr(self, _KV_TRACKER)
        with tracker.batch():
            original_free(self, blocks)
            tracker.release_unresident(blocks)

    original_evict = BlockPool._maybe_evict_cached_block

    def _maybe_evict_cached_block(self: Any, block: Any) -> bool:
        tracker: PhysicalKVTracker = getattr(self, _KV_TRACKER)
        with tracker.batch():
            evicted = original_evict(self, block)
            tracker.release_unresident((block,))
            return evicted

    original_reset = BlockPool.reset_prefix_cache

    def reset_prefix_cache(self: Any) -> bool:
        tracker: PhysicalKVTracker = getattr(self, _KV_TRACKER)
        with tracker.batch():
            reset = original_reset(self)
            if reset:
                tracker.release_unresident(self.blocks)
            return reset

    KVCacheManager.__init__ = __init__  # type: ignore[method-assign]
    KVCacheManager.allocate_slots = allocate_slots  # type: ignore[method-assign]
    BlockPool.get_new_blocks = get_new_blocks  # type: ignore[method-assign]
    BlockPool.touch = touch  # type: ignore[method-assign]
    BlockPool.free_blocks = free_blocks  # type: ignore[method-assign]
    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block  # type: ignore[method-assign]
    BlockPool.reset_prefix_cache = reset_prefix_cache  # type: ignore[method-assign]
    setattr(KVCacheManager, "_art_resource_usage_patched", True)


def _patch_output_transport() -> None:
    from vllm.v1.engine.core_client import (
        AsyncMPClient,
        DPAsyncMPClient,
        DPLBAsyncMPClient,
    )

    for client_class in (AsyncMPClient, DPAsyncMPClient, DPLBAsyncMPClient):
        descriptor = client_class.__dict__.get("process_engine_outputs")
        original = (
            getattr(client_class, "process_engine_outputs") if descriptor else None
        )
        if getattr(original, "__art_resource_usage_patched__", False):
            continue

        async def process_engine_outputs(
            self: Any, outputs: Any, _original: Any = original
        ) -> None:
            consume_runtime_usage_updates(outputs)
            if _original is not None:
                await _original(self, outputs)

        setattr(process_engine_outputs, "__art_resource_usage_patched__", True)
        setattr(
            client_class,
            "process_engine_outputs",
            staticmethod(process_engine_outputs),
        )


def patch_resource_usage() -> None:
    _patch_scheduler()
    _patch_executors()
    _patch_engine_core_drain()
    _patch_kv()
    _patch_output_transport()
