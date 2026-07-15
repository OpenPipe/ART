from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
import hashlib
import pickle
import random
import time
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .specs import VLLM_KV_EVENT_SCHEMA_VERSION, VLLM_PREFIX_HASH_SEED, EndpointSpec
from .vllm_replica import ReplicaUpdateReport


class _RoutingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrefixBlockHashes(_RoutingModel):
    version: str = Field(min_length=1)
    block_size: int = Field(gt=0)
    hashes: tuple[str, ...]


class VllmPrefixHashConfig(_RoutingModel):
    version: str = VLLM_KV_EVENT_SCHEMA_VERSION
    block_size: int = Field(gt=0)
    lora_name: str | None = Field(default=None, min_length=1)
    policy_cache_key: str | None = Field(default=None, min_length=1)


class RoutingInput(_RoutingModel):
    policy_version: str
    policy_digest: str = Field(min_length=1)
    stable_key: str | None = None
    prefix: PrefixBlockHashes | None = None
    prompt_token_ids: tuple[StrictInt, ...] | None = Field(
        default=None, max_length=1_000_000
    )
    cache_salt: str | None = None

    @model_validator(mode="after")
    def _validate_prefix_source(self) -> "RoutingInput":
        if self.prefix is not None and self.prompt_token_ids is not None:
            raise ValueError(
                "prefix hashes and prompt token IDs are mutually exclusive"
            )
        if self.prompt_token_ids is not None and any(
            isinstance(token_id, bool) or token_id < 0
            for token_id in self.prompt_token_ids
        ):
            raise ValueError("prompt token IDs must be nonnegative integers")
        return self


class ReplicaTelemetry(_RoutingModel):
    observed_at: float
    in_flight: int = Field(ge=0)
    capacity: int = Field(gt=0)


class RoutableReplica(_RoutingModel):
    replica_id: str = Field(min_length=1)
    endpoint: EndpointSpec
    phase: Literal["ready", "quarantined"]
    generation: int = Field(ge=0)
    generation_digest: str = Field(min_length=1)
    committed_version: str | None
    policy_digest: str | None
    update_identity: str | None
    telemetry: ReplicaTelemetry
    kv_event_publishers: int = Field(default=1, gt=0)
    quarantine_reason: str | None = None


class RoutingTable(_RoutingModel):
    policy_generation: int = Field(ge=0)
    policy_version: str
    policy_digest: str = Field(min_length=1)
    update_identity: str = Field(min_length=1)
    replicas: tuple[RoutableReplica, ...]
    prefix_hash: VllmPrefixHashConfig | None = None

    @model_validator(mode="after")
    def _validate_replicas(self) -> "RoutingTable":
        if not self.replicas:
            raise ValueError("routing table must contain at least one replica")
        if len({replica.replica_id for replica in self.replicas}) != len(self.replicas):
            raise ValueError("routing table replica IDs must be unique")
        return self


class KvCacheEvent(_RoutingModel):
    replica_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    publisher_rank: int = Field(default=0, ge=0)
    sequence: int = Field(ge=0)
    version: str = Field(min_length=1)
    block_size: int | None = Field(default=None, gt=0)
    group_idx: int | None = Field(default=0, ge=0)
    operation: Literal["store", "remove", "reset", "noop"]
    block_hashes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_operation(self) -> "KvCacheEvent":
        if self.operation == "store" and self.block_size is None:
            raise ValueError("store events require block_size")
        if self.operation in {"reset", "noop"} and self.block_hashes:
            raise ValueError(f"{self.operation} events must not contain block hashes")
        return self


class PreparedRoutingTable(_RoutingModel):
    previous_policy_generation: int
    candidate: RoutingTable


class RoutingUnavailableError(RuntimeError):
    pass


class RoutingQueueFullError(RoutingUnavailableError):
    pass


class RoutingDeadlineExceededError(TimeoutError):
    pass


class PolicyGenerationCommitError(RuntimeError):
    pass


def canonical_block_hash(value: int | bytes) -> str:
    """Canonicalize vLLM's version-dependent external block-hash type."""
    if isinstance(value, bool) or (isinstance(value, int) and value < 0):
        raise ValueError("vLLM block hashes must be nonnegative integers or bytes")
    return f"i:{value:x}" if isinstance(value, int) else f"b:{value.hex()}"


def _vllm_sha256(value: object) -> bytes:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).digest()


def _effective_cache_salt(
    config: VllmPrefixHashConfig, cache_salt: str | None
) -> str | None:
    if config.policy_cache_key is None:
        return cache_salt
    if cache_salt:
        return f"{cache_salt}|art_policy={config.policy_cache_key}"
    return f"art_policy_cache_salt={config.policy_cache_key}"


def vllm_request_block_hashes(
    prompt_token_ids: Iterable[int],
    config: VllmPrefixHashConfig,
    *,
    cache_salt: str | None = None,
) -> tuple[bytes, ...]:
    """Compute vLLM 0.23 request hashes for complete logical blocks."""
    token_ids = tuple(prompt_token_ids)
    if any(isinstance(token_id, bool) or token_id < 0 for token_id in token_ids):
        raise ValueError("prompt token IDs must be nonnegative integers")
    parent = _vllm_sha256(VLLM_PREFIX_HASH_SEED)
    salt = _effective_cache_salt(config, cache_salt)
    hashes: list[bytes] = []
    for start in range(0, len(token_ids) - config.block_size + 1, config.block_size):
        extra = []
        if config.lora_name is not None:
            extra.append(config.lora_name)
        if start == 0 and salt:
            extra.append(salt)
        parent = _vllm_sha256(
            (
                parent,
                token_ids[start : start + config.block_size],
                tuple(extra) or None,
            )
        )
        hashes.append(parent)
    return tuple(hashes)


class _KvIndex:
    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.next_sequence: int | None = None
        self.blocks: dict[tuple[str, int, int], set[str]] = {}

    def copy(self) -> "_KvIndex":
        clone = _KvIndex(self.generation)
        clone.next_sequence = self.next_sequence
        clone.blocks = {key: set(blocks) for key, blocks in self.blocks.items()}
        return clone


T = TypeVar("T")


class RouteReservation(Generic[T]):
    def __init__(
        self,
        router: "ReplicaRouter",
        replica: RoutableReplica,
        policy_generation: int,
        token: int,
    ) -> None:
        self.replica = replica
        self.policy_generation = policy_generation
        self._token = token
        self._router = router
        self._released = False

    async def __aenter__(self) -> "RouteReservation[T]":
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._router._release(self.replica, self._token)

    async def stream(self, source: AsyncIterator[T]) -> AsyncIterator[T]:
        try:
            async for item in source:
                yield item
        finally:
            await self.release()


class ReplicaRouter:
    """Prefix-aware bounded-load admission over immutable table snapshots."""

    def __init__(
        self,
        table: RoutingTable,
        *,
        telemetry_max_age_s: float = 5.0,
        max_queued: int = 128,
        random_seed: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if telemetry_max_age_s <= 0 or max_queued < 0:
            raise ValueError(
                "telemetry_max_age_s must be positive and max_queued nonnegative"
            )
        self._table = table
        self._telemetry_max_age_s = telemetry_max_age_s
        self._max_queued = max_queued
        self._clock = clock
        self._random = random.Random(random_seed)
        self._condition = asyncio.Condition()
        self._reservations: dict[tuple[str, int], dict[int, float]] = {}
        self._next_reservation = 0
        self._queued = 0
        self._kv: dict[tuple[str, int], _KvIndex] = {}

    @property
    def table(self) -> RoutingTable:
        return self._table

    @property
    def queued(self) -> int:
        return self._queued

    def reserved(self, replica_id: str, generation: int) -> int:
        return len(self._reservations.get((replica_id, generation), ()))

    async def acquire(
        self, request: RoutingInput, *, timeout_s: float
    ) -> RouteReservation[object]:
        if timeout_s <= 0:
            raise RoutingDeadlineExceededError("routing deadline has elapsed")
        deadline = self._clock() + timeout_s
        queued = False
        async with self._condition:
            try:
                while True:
                    table = self._table
                    compatible = self._compatible(table, request)
                    if not compatible:
                        raise RoutingUnavailableError(
                            "no ready replica has the requested committed policy and fresh telemetry"
                        )
                    eligible = [
                        replica
                        for replica in compatible
                        if self._effective_load(replica) < replica.telemetry.capacity
                    ]
                    if eligible:
                        replica = self._choose(eligible, request)
                        key = (replica.replica_id, replica.generation)
                        self._next_reservation += 1
                        token = self._next_reservation
                        self._reservations.setdefault(key, {})[token] = self._clock()
                        return RouteReservation(
                            self, replica, table.policy_generation, token
                        )
                    if not queued:
                        if self._queued >= self._max_queued:
                            raise RoutingQueueFullError("replica routing queue is full")
                        self._queued += 1
                        queued = True
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise RoutingDeadlineExceededError(
                            "replica capacity was unavailable before the deadline"
                        )
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except TimeoutError as error:
                        raise RoutingDeadlineExceededError(
                            "replica capacity was unavailable before the deadline"
                        ) from error
            finally:
                if queued:
                    self._queued -= 1

    async def update_telemetry(
        self, replica_id: str, generation: int, telemetry: ReplicaTelemetry
    ) -> None:
        async with self._condition:
            replicas = list(self._table.replicas)
            for index, replica in enumerate(replicas):
                if (
                    replica.replica_id == replica_id
                    and replica.generation == generation
                ):
                    replicas[index] = replica.model_copy(
                        update={"telemetry": telemetry}
                    )
                    break
            else:
                return
            self._table = self._table.model_copy(update={"replicas": tuple(replicas)})
            self._condition.notify_all()

    def apply_kv_event(self, event: KvCacheEvent) -> bool:
        return self.apply_kv_events((event,))

    def apply_kv_events(self, events: Iterable[KvCacheEvent]) -> bool:
        """Apply one ordered publisher batch, clearing affinity on a gap."""
        events = tuple(events)
        if not events:
            raise ValueError("KV event batch must not be empty")
        first = events[0]
        identity = (
            first.replica_id,
            first.generation,
            first.publisher_rank,
            first.sequence,
        )
        if any(
            (
                event.replica_id,
                event.generation,
                event.publisher_rank,
                event.sequence,
            )
            != identity
            for event in events[1:]
        ):
            raise ValueError("KV event batch must come from one publisher sequence")
        replica = next(
            (
                item
                for item in self._table.replicas
                if item.replica_id == first.replica_id
            ),
            None,
        )
        if (
            replica is None
            or replica.generation != first.generation
            or first.publisher_rank >= replica.kv_event_publishers
        ):
            return False
        key = (first.replica_id, first.publisher_rank)
        index = self._kv.get(key)
        if index is None or index.generation != first.generation:
            index = _KvIndex(first.generation)
            self._kv[key] = index
        if index.next_sequence is not None and first.sequence < index.next_sequence:
            return True
        contiguous = (
            index.next_sequence is None or first.sequence == index.next_sequence
        )
        if not contiguous:
            index.blocks.clear()
        reset = False
        for event in events:
            if event.operation == "reset":
                index.blocks.clear()
                reset = True
            elif event.operation == "store":
                assert event.block_size is not None
                assert event.group_idx is not None
                index.blocks.setdefault(
                    (event.version, event.block_size, event.group_idx), set()
                ).update(event.block_hashes)
            elif event.operation == "remove":
                for (_, _, group_idx), blocks in index.blocks.items():
                    if event.group_idx is None or event.group_idx == group_idx:
                        blocks.difference_update(event.block_hashes)
        index.next_sequence = first.sequence + 1
        return contiguous and not reset

    def invalidate_kv(self, replica_id: str) -> None:
        for key in tuple(self._kv):
            if key[0] == replica_id:
                self._kv.pop(key)

    def inherit_kv(self, source: "ReplicaRouter") -> None:
        generations = {
            (replica.replica_id, replica.generation) for replica in self._table.replicas
        }
        self._kv = {
            key: index.copy()
            for key, index in source._kv.items()
            if (key[0], index.generation) in generations
        }

    def prepare(self, candidate: RoutingTable) -> PreparedRoutingTable:
        current = self._table
        if candidate.policy_generation <= current.policy_generation:
            raise PolicyGenerationCommitError("policy generation must increase")
        invalid = [
            replica.replica_id
            for replica in candidate.replicas
            if replica.phase != "ready"
            or replica.committed_version != candidate.policy_version
            or replica.policy_digest != candidate.policy_digest
            or replica.update_identity != candidate.update_identity
        ]
        if invalid:
            raise PolicyGenerationCommitError(
                f"candidate replicas are not exactly committed: {invalid}"
            )
        return PreparedRoutingTable(
            previous_policy_generation=current.policy_generation,
            candidate=candidate,
        )

    async def verify(
        self,
        prepared: PreparedRoutingTable,
        reports: Iterable[ReplicaUpdateReport],
    ) -> None:
        by_id = {report.replica_id: report for report in reports}
        target_ids = {replica.replica_id for replica in prepared.candidate.replicas}
        bad: list[str] = []
        for replica in prepared.candidate.replicas:
            report = by_id.get(replica.replica_id)
            if (
                report is None
                or report.ambiguous
                or (
                    report.generation,
                    report.generation_digest,
                    report.policy_version,
                    report.policy_digest,
                    report.update_identity,
                )
                != (
                    replica.generation,
                    replica.generation_digest,
                    prepared.candidate.policy_version,
                    prepared.candidate.policy_digest,
                    prepared.candidate.update_identity,
                )
            ):
                bad.append(replica.replica_id)
        if by_id.keys() != target_ids:
            bad.extend(sorted(by_id.keys() ^ target_ids))
        if bad:
            await self.quarantine(target_ids, "ambiguous or partial policy update")
            raise PolicyGenerationCommitError(
                f"policy generation verification failed: {sorted(set(bad))}"
            )

    async def commit(self, prepared: PreparedRoutingTable) -> RoutingTable:
        async with self._condition:
            if self._table.policy_generation != prepared.previous_policy_generation:
                raise PolicyGenerationCommitError(
                    "routing table changed after policy generation prepare"
                )
            previous = {replica.replica_id: replica for replica in self._table.replicas}
            self._table = prepared.candidate
            for replica_id in previous.keys() - {
                replica.replica_id for replica in prepared.candidate.replicas
            }:
                self.invalidate_kv(replica_id)
            for replica in prepared.candidate.replicas:
                old = previous.get(replica.replica_id)
                if old is None or (
                    old.generation,
                    old.generation_digest,
                    old.committed_version,
                    old.policy_digest,
                    old.update_identity,
                ) != (
                    replica.generation,
                    replica.generation_digest,
                    replica.committed_version,
                    replica.policy_digest,
                    replica.update_identity,
                ):
                    self.invalidate_kv(replica.replica_id)
            self._condition.notify_all()
            return self._table

    async def quarantine(self, replica_ids: Iterable[str], reason: str) -> RoutingTable:
        ids = set(replica_ids)
        async with self._condition:
            self._table = self._table.model_copy(
                update={
                    "replicas": tuple(
                        replica.model_copy(
                            update={"phase": "quarantined", "quarantine_reason": reason}
                        )
                        if replica.replica_id in ids
                        else replica
                        for replica in self._table.replicas
                    )
                }
            )
            for replica_id in ids:
                self.invalidate_kv(replica_id)
            self._condition.notify_all()
            return self._table

    async def _release(self, replica: RoutableReplica, token: int) -> None:
        async with self._condition:
            key = (replica.replica_id, replica.generation)
            reservations = self._reservations.get(key)
            if reservations is None:
                return
            reservations.pop(token, None)
            if not reservations:
                self._reservations.pop(key, None)
            self._condition.notify(1)

    def _compatible(
        self, table: RoutingTable, request: RoutingInput
    ) -> list[RoutableReplica]:
        now = self._clock()
        return [
            replica
            for replica in table.replicas
            if replica.phase == "ready"
            and replica.committed_version == request.policy_version
            and replica.policy_digest == request.policy_digest
            and now - replica.telemetry.observed_at <= self._telemetry_max_age_s
        ]

    def _effective_load(self, replica: RoutableReplica) -> int:
        reservations = self._reservations.get(
            (replica.replica_id, replica.generation), {}
        )
        unobserved = sum(
            started_at > replica.telemetry.observed_at
            for started_at in reservations.values()
        )
        return replica.telemetry.in_flight + unobserved

    def _choose(
        self, replicas: list[RoutableReplica], request: RoutingInput
    ) -> RoutableReplica:
        token_prefixes = self._token_prefixes(request)
        matches = {
            replica.replica_id: self._prefix_match(replica, request, token_prefixes)
            for replica in replicas
        }
        best = max(matches.values(), default=0)
        pool = [replica for replica in replicas if matches[replica.replica_id] == best]
        if len(pool) == 1:
            return pool[0]
        if request.stable_key:
            pool.sort(
                key=lambda replica: hashlib.sha256(
                    f"{request.stable_key}\0{replica.replica_id}".encode()
                ).digest(),
                reverse=True,
            )
            contenders = pool[:2]
        else:
            contenders = self._random.sample(pool, 2)
        loads = [
            self._effective_load(replica) / replica.telemetry.capacity
            for replica in contenders
        ]
        if loads[0] == loads[1]:
            return (
                contenders[0] if request.stable_key else self._random.choice(contenders)
            )
        return contenders[0] if loads[0] < loads[1] else contenders[1]

    def _token_prefixes(self, request: RoutingInput) -> dict[int, tuple[str, ...]]:
        token_ids = request.prompt_token_ids
        config = self._table.prefix_hash
        if token_ids is None or config is None:
            return {}
        base_hashes = vllm_request_block_hashes(
            token_ids, config, cache_salt=request.cache_salt
        )
        block_sizes = {
            block_size
            for index in self._kv.values()
            for version, block_size, _group_idx in index.blocks
            if version == config.version and block_size % config.block_size == 0
        }
        prefixes: dict[int, tuple[str, ...]] = {}
        for block_size in block_sizes:
            scale = block_size // config.block_size
            prefixes[block_size] = tuple(
                canonical_block_hash(b"".join(base_hashes[start : start + scale]))
                for start in range(0, len(base_hashes) - scale + 1, scale)
            )
        return prefixes

    def _prefix_match(
        self,
        replica: RoutableReplica,
        request: RoutingInput,
        token_prefixes: dict[int, tuple[str, ...]],
    ) -> int:
        prefix = request.prefix
        if prefix is None and request.prompt_token_ids is None:
            return 0
        matches = []
        for publisher_rank in range(replica.kv_event_publishers):
            index = self._kv.get((replica.replica_id, publisher_rank))
            if index is None or index.generation != replica.generation:
                return 0
            group_matches = []
            for (version, block_size, _group_idx), blocks in index.blocks.items():
                if prefix is not None:
                    if (version, block_size) != (prefix.version, prefix.block_size):
                        continue
                    hashes = prefix.hashes
                else:
                    config = self._table.prefix_hash
                    if config is None or version != config.version:
                        continue
                    hashes = token_prefixes.get(block_size, ())
                matched = 0
                for block_hash in hashes:
                    if block_hash not in blocks:
                        break
                    matched += block_size
                group_matches.append(matched)
            if not group_matches:
                return 0
            matches.append(min(group_matches))
        return min(matches)
