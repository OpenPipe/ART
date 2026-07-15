from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
import hashlib
import random
import time
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .specs import EndpointSpec
from .vllm_replica import ReplicaUpdateReport


class _RoutingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrefixBlockHashes(_RoutingModel):
    version: str = Field(min_length=1)
    block_size: int = Field(gt=0)
    hashes: tuple[str, ...]


class RoutingInput(_RoutingModel):
    policy_version: str
    policy_digest: str = Field(min_length=1)
    stable_key: str | None = None
    prefix: PrefixBlockHashes | None = None


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
    quarantine_reason: str | None = None


class RoutingTable(_RoutingModel):
    policy_generation: int = Field(ge=0)
    policy_version: str
    policy_digest: str = Field(min_length=1)
    update_identity: str = Field(min_length=1)
    replicas: tuple[RoutableReplica, ...]

    @model_validator(mode="after")
    def _validate_replicas(self) -> "RoutingTable":
        if not self.replicas:
            raise ValueError("routing table must contain at least one replica")
        if len({replica.replica_id for replica in self.replicas}) != len(self.replicas):
            raise ValueError("routing table replica IDs must be unique")
        return self


class KvCacheEvent(_RoutingModel):
    replica_id: str
    generation: int = Field(ge=0)
    sequence: int = Field(ge=0)
    version: str = Field(min_length=1)
    block_size: int = Field(gt=0)
    operation: Literal["store", "remove", "reset"]
    block_hashes: tuple[str, ...] = ()


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


class _KvIndex:
    def __init__(self, generation: int, version: str, block_size: int) -> None:
        self.generation = generation
        self.version = version
        self.block_size = block_size
        self.next_sequence: int | None = None
        self.blocks: set[str] = set()


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
        self._kv: dict[str, _KvIndex] = {}

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
        replica = next(
            (
                item
                for item in self._table.replicas
                if item.replica_id == event.replica_id
            ),
            None,
        )
        if replica is None or replica.generation != event.generation:
            self._kv.pop(event.replica_id, None)
            return False
        index = self._kv.get(event.replica_id)
        compatible = (
            index is not None
            and index.generation == event.generation
            and index.version == event.version
            and index.block_size == event.block_size
        )
        if not compatible:
            index = _KvIndex(event.generation, event.version, event.block_size)
            self._kv[event.replica_id] = index
        contiguous = (
            index.next_sequence is None or event.sequence == index.next_sequence
        )
        if not contiguous or event.operation == "reset":
            index.blocks.clear()
        if event.operation == "store":
            index.blocks.update(event.block_hashes)
        elif event.operation == "remove":
            index.blocks.difference_update(event.block_hashes)
        index.next_sequence = event.sequence + 1
        return contiguous and event.operation != "reset"

    def invalidate_kv(self, replica_id: str) -> None:
        self._kv.pop(replica_id, None)

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
            for replica in prepared.candidate.replicas:
                old = previous.get(replica.replica_id)
                if old is None or (
                    old.generation,
                    old.generation_digest,
                    old.policy_digest,
                ) != (
                    replica.generation,
                    replica.generation_digest,
                    replica.policy_digest,
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
        matches = {
            replica.replica_id: self._prefix_match(replica, request)
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

    def _prefix_match(self, replica: RoutableReplica, request: RoutingInput) -> int:
        prefix = request.prefix
        index = self._kv.get(replica.replica_id)
        if (
            prefix is None
            or index is None
            or index.generation != replica.generation
            or index.version != prefix.version
            or index.block_size != prefix.block_size
        ):
            return 0
        matched = 0
        for block_hash in prefix.hashes:
            if block_hash not in index.blocks:
                break
            matched += 1
        return matched
