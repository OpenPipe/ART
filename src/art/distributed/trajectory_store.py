from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
import secrets
import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.preprocessing.policy_spans import PolicyTokenSpan
from art.trajectories import MetadataValue, Trajectory, TrajectoryGroup

from .data_plane import (
    ByteStreamPublisher,
    ByteStreamServerLoop,
    ByteStreamTransfer,
    receive_byte_stream,
)

if TYPE_CHECKING:
    from .packing import TrajectoryGroupPayload

TRAJECTORY_FORMAT = "art_trajectory_v1"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrajectoryRecordRef(_Contract):
    record_id: str = Field(min_length=1)
    owner_actor_id: str = Field(min_length=1)
    byte_count: int = Field(ge=0)


class TrajectoryGroupDescriptor(_Contract):
    grouping_key: str = Field(min_length=1)
    trajectory_count: int = Field(ge=0)
    exception_count: int = Field(ge=0)
    rewards: tuple[float, ...]
    initial_policy_versions: tuple[int, ...]
    completion_tokens: tuple[float, ...]
    policy_token_counts: dict[int, int]
    trajectory_initial_policy_versions: tuple[int | None, ...]
    trajectory_final_policy_versions: tuple[int | None, ...]
    trajectory_policy_token_counts: tuple[dict[int, int], ...]
    trajectory_metrics: tuple[dict[str, float | int | bool], ...]
    trajectory_metadata: tuple[dict[str, MetadataValue], ...]
    group_metadata: dict[str, MetadataValue]
    group_metrics: dict[str, float | int | bool]
    exceptions: tuple[tuple[str, str], ...]
    byte_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_trajectory_summaries(self) -> "TrajectoryGroupDescriptor":
        aligned = (
            self.rewards,
            self.completion_tokens,
            self.trajectory_initial_policy_versions,
            self.trajectory_final_policy_versions,
            self.trajectory_policy_token_counts,
            self.trajectory_metrics,
            self.trajectory_metadata,
        )
        if any(len(values) != self.trajectory_count for values in aligned):
            raise ValueError("trajectory descriptor summaries are not aligned")
        if len(self.exceptions) != self.exception_count:
            raise ValueError("trajectory descriptor exceptions are not aligned")
        return self


class TrajectoryGroupBundle(_Contract):
    """Binary trajectory records for bulk transport across actor boundaries."""

    header: bytes
    records: tuple[bytes, ...]

    @classmethod
    def from_payload(cls, payload: TrajectoryGroupPayload) -> "TrajectoryGroupBundle":
        from msgspec import msgpack

        return cls(
            header=msgpack.encode(
                payload.model_copy(update={"trajectories": ()}).model_dump(
                    mode="python"
                )
            ),
            records=tuple(
                msgpack.encode(record.model_dump(mode="python"))
                for record in payload.trajectories
            ),
        )

    @classmethod
    def from_group(cls, group: TrajectoryGroup) -> "TrajectoryGroupBundle":
        from .packing import TrajectoryGroupPayload

        return cls.from_payload(TrajectoryGroupPayload.from_group(group))

    def payload(self) -> TrajectoryGroupPayload:
        from msgspec import msgpack

        from .packing import TrajectoryGroupPayload

        header = msgpack.decode(self.header)
        header["trajectories"] = tuple(
            msgpack.decode(record) for record in self.records
        )
        return TrajectoryGroupPayload.model_validate(header)

    def build(self) -> TrajectoryGroup:
        return self.payload().build()


class TrajectoryGroupLayout(_Contract):
    header_byte_count: int = Field(ge=1)
    record_byte_counts: tuple[int, ...]


class TrajectoryBatchTransfer(_Contract):
    stream: ByteStreamTransfer
    groups: tuple[TrajectoryGroupLayout, ...]

    @model_validator(mode="after")
    def _validate_layout(self) -> "TrajectoryBatchTransfer":
        if any(
            byte_count < 0
            for group in self.groups
            for byte_count in group.record_byte_counts
        ):
            raise ValueError("trajectory record byte counts must be non-negative")
        byte_count = sum(
            group.header_byte_count + sum(group.record_byte_counts)
            for group in self.groups
        )
        if not self.groups or byte_count != self.stream.byte_count:
            raise ValueError("trajectory stream layout does not match its payload")
        return self

    async def receive_groups(self, *, timeout_s: float) -> tuple[TrajectoryGroup, ...]:
        payload = await receive_byte_stream(self.stream, timeout_s=timeout_s)
        return await asyncio.to_thread(self._build_groups, payload)

    def _build_groups(self, payload: bytearray) -> tuple[TrajectoryGroup, ...]:
        from msgspec import msgpack

        from .packing import TrajectoryGroupPayload

        view = memoryview(payload)
        offset = 0
        groups = []
        try:
            for layout in self.groups:
                end = offset + layout.header_byte_count
                header = msgpack.decode(view[offset:end])
                offset = end
                records = []
                for byte_count in layout.record_byte_counts:
                    end = offset + byte_count
                    records.append(msgpack.decode(view[offset:end]))
                    offset = end
                header["trajectories"] = tuple(records)
                groups.append(TrajectoryGroupPayload.model_validate(header).build())
        finally:
            view.release()
        return tuple(groups)


class TrajectoryGroupRef(_Contract):
    result_id: str = Field(min_length=1)
    owner_actor_id: str = Field(min_length=1)
    lease_id: str = Field(min_length=1)
    format: Literal["art_trajectory_v1"] = TRAJECTORY_FORMAT
    records: tuple[TrajectoryRecordRef, ...]
    descriptor: TrajectoryGroupDescriptor
    transfer: TrajectoryBatchTransfer | None = None


async def publish_trajectory_bundles(
    bundles: tuple[TrajectoryGroupBundle, ...],
    *,
    stream_id: str,
    advertise_host: str,
    on_sent: Callable[[], None] | None = None,
    server_loop: ByteStreamServerLoop | None = None,
) -> tuple[TrajectoryBatchTransfer, ByteStreamPublisher]:
    publisher = await ByteStreamPublisher.create(
        stream_id,
        tuple(
            chunk for bundle in bundles for chunk in (bundle.header, *bundle.records)
        ),
        advertise_host=advertise_host,
        on_sent=on_sent,
        server_loop=server_loop,
    )
    try:
        transfer = TrajectoryBatchTransfer(
            stream=publisher.transfer,
            groups=tuple(
                TrajectoryGroupLayout(
                    header_byte_count=len(bundle.header),
                    record_byte_counts=tuple(map(len, bundle.records)),
                )
                for bundle in bundles
            ),
        )
    except BaseException:
        await publisher.close()
        raise
    return transfer, publisher


class TrajectoryGroupAnnotations(_Contract):
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    initial_policy_version: int = Field(ge=0)
    final_policy_version: int = Field(ge=0)
    rollout_wall_s: float = Field(default=0.0, ge=0)
    actor_idle_s: float = Field(default=0.0, ge=0)
    queue_wait_s: float = Field(default=0.0, ge=0)


class TrajectoryQueueItem(_Contract):
    ref: TrajectoryGroupRef
    annotations: TrajectoryGroupAnnotations

    async def receive(self, *, timeout_s: float) -> TrajectoryGroup:
        transfer = self.ref.transfer
        if transfer is None:
            raise RuntimeError("remote trajectory has no data-plane transfer")
        if transfer.stream.stream_id != self.ref.result_id:
            raise RuntimeError("trajectory owner returned the wrong result ID")
        if transfer.stream.byte_count != self.ref.descriptor.byte_count:
            raise RuntimeError("trajectory owner returned the wrong byte count")
        groups = await transfer.receive_groups(timeout_s=timeout_s)
        if len(groups) != 1:
            raise RuntimeError("trajectory owner returned the wrong group count")
        return self.apply_annotations(groups[0])

    def apply_annotations(self, group: TrajectoryGroup) -> TrajectoryGroup:
        annotations = self.annotations
        group.metadata.update(annotations.metadata)
        group.metadata["_art_rollout_wall_s"] = annotations.rollout_wall_s
        group.metadata["_art_actor_idle_s"] = annotations.actor_idle_s
        group.metadata["_art_queue_wait_s"] = annotations.queue_wait_s
        for trajectory in group.trajectories:
            if trajectory.initial_policy_version is None:
                trajectory.initial_policy_version = annotations.initial_policy_version
            if trajectory.final_policy_version is None:
                trajectory.final_policy_version = annotations.final_policy_version
        return group


class TrajectoryQueueResize(_Contract):
    queue_id: str = Field(min_length=1)
    maxsize: int = Field(ge=1)
    generation: int = Field(ge=1)


class TrajectoryEnqueueResult(_Contract):
    status: Literal["accepted", "full", "oversize", "minimum_unreachable", "closed"]
    reason: str | None = None


class TrajectoryQueueTake(_Contract):
    leases: tuple[TrajectoryQueueLease, ...] = ()
    closed: bool = False


class TrajectoryQueueLease(_Contract):
    claim_id: str = Field(min_length=1)
    consumer_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    item: TrajectoryQueueItem


class TrajectoryQueuePacking(_Contract):
    queue_id: str = Field(min_length=1)
    leases: tuple[TrajectoryQueueLease, ...]
    generation_id: str = Field(min_length=1)


class TrajectoryQueueRelease(_Contract):
    queue_id: str = Field(min_length=1)
    leases: tuple[TrajectoryQueueLease, ...] = Field(min_length=1)
    generation_id: str | None = None
    disposition: Literal["consumed", "discarded"]


class TrajectoryQueueSnapshot(_Contract):
    items: tuple[TrajectoryQueueItem, ...]
    max_ready_groups: int = Field(ge=1)
    generation: int = Field(ge=0)
    capacity_records: int = Field(ge=1)
    capacity_bytes: int = Field(ge=1)
    used_records: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    leased_groups: int = Field(ge=0)
    ready_groups: int = Field(ge=0)
    packing_groups: int = Field(ge=0)
    packed_groups: int = Field(ge=0)
    released_leases: int = Field(ge=0)
    lease_lifetime_s: float = Field(ge=0)
    max_lease_lifetime_s: float = Field(ge=0)


class TrajectoryCapacityError(RuntimeError):
    pass


class TrajectoryLeaseError(RuntimeError):
    pass


class _StoredGroup:
    def __init__(self, header: bytes, ref: TrajectoryGroupRef) -> None:
        self.header = header
        self.ref = ref


class TrajectoryRecordStore:
    """Own typed trajectory records until their rollout-result lease is released."""

    def __init__(
        self, *, owner_actor_id: str, capacity_records: int, capacity_bytes: int
    ) -> None:
        if capacity_records < 1 or capacity_bytes < 1:
            raise ValueError("trajectory store capacities must be positive")
        self.owner_actor_id = owner_actor_id
        self.capacity_records = capacity_records
        self.capacity_bytes = capacity_bytes
        self._records: dict[str, bytes] = {}
        self._groups: dict[str, _StoredGroup] = {}
        self._used_bytes = 0

    def put(self, group: TrajectoryGroup) -> TrajectoryGroupRef:
        from .packing import TrajectoryGroupPayload

        payload = TrajectoryGroupPayload.from_group(group)
        bundle = TrajectoryGroupBundle.from_payload(payload)
        record_sizes = tuple(len(record) for record in bundle.records)
        byte_count = sum(record_sizes) + len(bundle.header)
        record_count = len(payload.trajectories)
        if record_count > self.capacity_records or byte_count > self.capacity_bytes:
            raise TrajectoryCapacityError(
                f"trajectory group requires {record_count} records/{byte_count} bytes; "
                f"store capacity is {self.capacity_records}/{self.capacity_bytes}"
            )
        if (
            len(self._records) + record_count > self.capacity_records
            or self._used_bytes + byte_count > self.capacity_bytes
        ):
            raise TrajectoryCapacityError("trajectory record store capacity exhausted")

        result_id = secrets.token_hex(16)
        records = tuple(
            TrajectoryRecordRef(
                record_id=secrets.token_hex(16),
                owner_actor_id=self.owner_actor_id,
                byte_count=size,
            )
            for size in record_sizes
        )
        for record_ref, record in zip(records, bundle.records, strict=True):
            self._records[record_ref.record_id] = record
        descriptor = TrajectoryGroupDescriptor(
            grouping_key=_grouping_key(group, result_id),
            trajectory_count=len(group.trajectories),
            exception_count=len(group.exceptions),
            rewards=tuple(trajectory.reward for trajectory in group.trajectories),
            initial_policy_versions=tuple(
                trajectory.initial_policy_version
                for trajectory in group.trajectories
                if trajectory.initial_policy_version is not None
            ),
            completion_tokens=tuple(
                float(value)
                if not isinstance(value, bool) and isinstance(value, int | float)
                else 0.0
                for trajectory in group.trajectories
                for value in (trajectory.metrics.get("completion_tokens"),)
            ),
            policy_token_counts=_policy_token_counts(group.trajectories),
            trajectory_initial_policy_versions=tuple(
                trajectory.initial_policy_version for trajectory in group.trajectories
            ),
            trajectory_final_policy_versions=tuple(
                trajectory.final_policy_version for trajectory in group.trajectories
            ),
            trajectory_policy_token_counts=tuple(
                _trajectory_policy_token_counts(trajectory)
                for trajectory in group.trajectories
            ),
            trajectory_metrics=tuple(
                trajectory.metrics for trajectory in group.trajectories
            ),
            trajectory_metadata=tuple(
                trajectory.metadata for trajectory in group.trajectories
            ),
            group_metadata=group.metadata,
            group_metrics=group.metrics,
            exceptions=tuple(
                (exception.type, exception.message) for exception in group.exceptions
            ),
            byte_count=byte_count,
        )
        ref = TrajectoryGroupRef(
            result_id=result_id,
            owner_actor_id=self.owner_actor_id,
            lease_id=secrets.token_hex(16),
            records=records,
            descriptor=descriptor,
        )
        self._groups[result_id] = _StoredGroup(bundle.header, ref)
        self._used_bytes += byte_count
        return ref

    def bundle(self, ref: TrajectoryGroupRef) -> TrajectoryGroupBundle:
        stored = self._entry(ref)
        return TrajectoryGroupBundle(
            header=stored.header,
            records=tuple(
                self._records[record.record_id] for record in stored.ref.records
            ),
        )

    def payload(self, ref: TrajectoryGroupRef) -> TrajectoryGroupPayload:
        return self.bundle(ref).payload()

    def materialize(self, ref: TrajectoryGroupRef) -> TrajectoryGroup:
        return self.payload(ref).build()

    def drop(self, ref: TrajectoryGroupRef) -> None:
        stored = self._groups.get(ref.result_id)
        if stored is None:
            return
        self._require_same_lease(stored.ref, ref)
        self._groups.pop(ref.result_id)
        for record in stored.ref.records:
            self._records.pop(record.record_id)
        self._used_bytes -= stored.ref.descriptor.byte_count

    def close(self) -> None:
        self._groups.clear()
        self._records.clear()
        self._used_bytes = 0

    def _entry(self, ref: TrajectoryGroupRef) -> _StoredGroup:
        try:
            stored = self._groups[ref.result_id]
        except KeyError:
            raise TrajectoryLeaseError(
                f"unknown trajectory result {ref.result_id!r}"
            ) from None
        self._require_same_lease(stored.ref, ref)
        return stored

    @staticmethod
    def _require_same_lease(
        expected: TrajectoryGroupRef, received: TrajectoryGroupRef
    ) -> None:
        if (
            expected.owner_actor_id != received.owner_actor_id
            or expected.lease_id != received.lease_id
            or expected.records != received.records
        ):
            raise TrajectoryLeaseError("trajectory result lease does not match storage")


class _QueueEntry:
    def __init__(self, item: TrajectoryQueueItem) -> None:
        self.item = item
        self.phase: Literal["ready", "packing", "packed"] = "ready"
        self.consumer_id: str | None = None
        self.claim_id: str | None = None
        self.claim_generation: int | None = None
        self.packing_generation_id: str | None = None
        self.acquired_at: float | None = None


class TrajectoryQueueStore:
    """Bounded FIFO and consumer-lease owner for trajectory-group references."""

    def __init__(
        self,
        *,
        max_ready_groups: int,
        capacity_records: int,
        capacity_bytes: int,
    ) -> None:
        if min(max_ready_groups, capacity_records, capacity_bytes) < 1:
            raise ValueError("trajectory queue capacities must be positive")
        self.max_ready_groups = max_ready_groups
        self.capacity_records = capacity_records
        self.capacity_bytes = capacity_bytes
        self._entries: dict[str, _QueueEntry] = {}
        self._ready: deque[str] = deque()
        self._used_records = 0
        self._used_bytes = 0
        self._finished = False
        self._pending_minimum: tuple[str, int] | None = None
        self._minimum_error: str | None = None
        self.generation = 0
        self._claim_generation = 0
        self._released_leases = 0
        self._lease_lifetime_s = 0.0
        self._max_lease_lifetime_s = 0.0

    def resize(self, *, maxsize: int, generation: int) -> None:
        if maxsize < 1 or generation < 1:
            raise ValueError("trajectory queue resize values must be positive")
        if generation < self.generation:
            return
        if generation == self.generation:
            if maxsize != self.max_ready_groups:
                raise ValueError("trajectory queue resize generation conflicts")
            return
        self.max_ready_groups = maxsize
        self.generation = generation

    def enqueue(self, item: TrajectoryQueueItem) -> TrajectoryEnqueueResult:
        item = _resolve_grouping(item)
        ref = item.ref
        records = len(ref.records)
        byte_count = ref.descriptor.byte_count
        if records > self.capacity_records or byte_count > self.capacity_bytes:
            reason = f"result requires {records} records/{byte_count} bytes"
            if self._pending_minimum is not None:
                return self._fail_pending_minimum(item, reason)
            return TrajectoryEnqueueResult(
                status="oversize",
                reason=reason,
            )
        if self._finished:
            return TrajectoryEnqueueResult(status="closed")
        existing = self._entries.get(ref.result_id)
        if existing is not None:
            if existing.item.ref == ref:
                return TrajectoryEnqueueResult(status="accepted")
            raise TrajectoryLeaseError("trajectory result lease changed while queued")
        if self._minimum_error is not None:
            return TrajectoryEnqueueResult(
                status="minimum_unreachable", reason=self._minimum_error
            )
        blockers = []
        if len(self._entries) >= self.max_ready_groups:
            blockers.append("group capacity")
        if self._used_records + records > self.capacity_records:
            blockers.append("record capacity")
        if self._used_bytes + byte_count > self.capacity_bytes:
            blockers.append("byte capacity")
        if blockers:
            pending = self._pending_minimum
            if (
                pending is not None
                and len(self._ready) < pending[1]
                and self._minimum_cannot_make_progress()
            ):
                return self._fail_pending_minimum(item, ", ".join(blockers))
            return TrajectoryEnqueueResult(status="full")
        self._entries[ref.result_id] = _QueueEntry(item)
        self._ready.append(ref.result_id)
        self._used_records += records
        self._used_bytes += byte_count
        return TrajectoryEnqueueResult(status="accepted")

    def take(self, consumer_id: str, count: int) -> TrajectoryQueueTake:
        """Acquire a positive minimum, take up to a negative limit, or cancel at zero."""
        if not consumer_id:
            raise ValueError("consumer_id must not be empty")
        if count == 0:
            pending = self._pending_minimum
            if pending is not None and pending[0] != consumer_id:
                raise TrajectoryLeaseError(
                    "trajectory minimum acquisition belongs to another consumer"
                )
            self._pending_minimum = None
            return TrajectoryQueueTake(closed=self._finished and not self._ready)
        if self._minimum_error is not None:
            raise TrajectoryCapacityError(self._minimum_error)

        best_effort = count < 0
        limit = abs(count)
        request = (consumer_id, limit)
        if self._pending_minimum not in (None, request):
            raise TrajectoryLeaseError(
                "trajectory queue already has a pending minimum acquisition"
            )
        if best_effort:
            if self._pending_minimum is not None:
                raise TrajectoryLeaseError(
                    "best-effort take cannot replace a pending minimum acquisition"
                )
            take_count = min(limit, len(self._ready))
        elif not self._finished and limit > self.max_ready_groups:
            raise TrajectoryCapacityError(
                f"minimum acquisition requires {limit} trajectory groups; shared "
                f"queue capacity is {self.max_ready_groups} groups"
            )
        elif not self._finished and len(self._ready) < limit:
            self._pending_minimum = request
            return TrajectoryQueueTake()
        else:
            self._pending_minimum = None
            take_count = min(limit, len(self._ready))

        leases: list[TrajectoryQueueLease] = []
        while len(leases) < take_count:
            result_id = self._ready.popleft()
            entry = self._entries[result_id]
            self._claim_generation += 1
            entry.phase = "packing"
            entry.consumer_id = consumer_id
            entry.claim_id = secrets.token_hex(16)
            entry.claim_generation = self._claim_generation
            entry.acquired_at = time.monotonic()
            leases.append(
                TrajectoryQueueLease(
                    claim_id=entry.claim_id,
                    consumer_id=consumer_id,
                    generation=self._claim_generation,
                    item=entry.item,
                )
            )
        return TrajectoryQueueTake(
            leases=tuple(leases), closed=self._finished and not self._ready
        )

    def mark_packed(self, operation: TrajectoryQueuePacking) -> None:
        entries = [self._leased_entry(lease) for lease in operation.leases]
        if any(entry.phase != "packing" for entry in entries):
            raise TrajectoryLeaseError("trajectory claim is not being packed")
        for entry in entries:
            entry.phase = "packed"
            entry.packing_generation_id = operation.generation_id

    def release(self, operation: TrajectoryQueueRelease) -> None:
        entries = [self._leased_entry(lease) for lease in operation.leases]
        for entry in entries:
            if entry.phase == "packing":
                if operation.disposition != "discarded" or operation.generation_id:
                    raise TrajectoryLeaseError(
                        "unpacked trajectory can only be discarded"
                    )
            elif entry.phase == "packed":
                if operation.generation_id != entry.packing_generation_id:
                    raise TrajectoryLeaseError(
                        "trajectory packing generation does not match"
                    )
            else:
                raise TrajectoryLeaseError("trajectory claim was not acquired")
        now = time.monotonic()
        for lease, entry in zip(operation.leases, entries, strict=True):
            assert entry.acquired_at is not None
            lifetime = now - entry.acquired_at
            self._released_leases += 1
            self._lease_lifetime_s += lifetime
            self._max_lease_lifetime_s = max(self._max_lease_lifetime_s, lifetime)
            self._remove(lease.item.ref.result_id)

    def finish(self) -> None:
        self._finished = True

    def close(self) -> tuple[TrajectoryGroupRef, ...]:
        refs = tuple(entry.item.ref for entry in self._entries.values())
        self._entries.clear()
        self._ready.clear()
        self._used_records = 0
        self._used_bytes = 0
        self._finished = True
        self._pending_minimum = None
        self._minimum_error = None
        return refs

    def snapshot(self) -> TrajectoryQueueSnapshot:
        return TrajectoryQueueSnapshot(
            items=tuple(entry.item for entry in self._entries.values()),
            max_ready_groups=self.max_ready_groups,
            generation=self.generation,
            capacity_records=self.capacity_records,
            capacity_bytes=self.capacity_bytes,
            used_records=self._used_records,
            used_bytes=self._used_bytes,
            leased_groups=sum(
                entry.phase != "ready" for entry in self._entries.values()
            ),
            ready_groups=sum(
                entry.phase == "ready" for entry in self._entries.values()
            ),
            packing_groups=sum(
                entry.phase == "packing" for entry in self._entries.values()
            ),
            packed_groups=sum(
                entry.phase == "packed" for entry in self._entries.values()
            ),
            released_leases=self._released_leases,
            lease_lifetime_s=self._lease_lifetime_s,
            max_lease_lifetime_s=self._max_lease_lifetime_s,
        )

    def _leased_entry(self, lease: TrajectoryQueueLease) -> _QueueEntry:
        entry = self._entries.get(lease.item.ref.result_id)
        if (
            entry is None
            or entry.item != lease.item
            or entry.consumer_id != lease.consumer_id
            or entry.claim_id != lease.claim_id
            or entry.claim_generation != lease.generation
        ):
            raise TrajectoryLeaseError("trajectory result has no matching claim")
        return entry

    def _remove(self, result_id: str) -> None:
        entry = self._entries.pop(result_id)
        self._used_records -= len(entry.item.ref.records)
        self._used_bytes -= entry.item.ref.descriptor.byte_count

    def _fail_pending_minimum(
        self, item: TrajectoryQueueItem, blocker: str
    ) -> TrajectoryEnqueueResult:
        assert self._pending_minimum is not None
        count = self._pending_minimum[1]
        ref = item.ref
        packing = sum(entry.phase == "packing" for entry in self._entries.values())
        self._minimum_error = (
            f"minimum acquisition of {count} trajectory groups is unreachable: "
            f"{len(self._ready)} ready/{packing} packing groups use "
            f"{self._used_records}/{self.capacity_records} records and "
            f"{self._used_bytes}/{self.capacity_bytes} bytes; result "
            f"{ref.result_id!r} requires {len(ref.records)} records/"
            f"{ref.descriptor.byte_count} bytes ({blocker})"
        )
        return TrajectoryEnqueueResult(
            status="minimum_unreachable", reason=self._minimum_error
        )

    def _minimum_cannot_make_progress(self) -> bool:
        return all(entry.phase == "ready" for entry in self._entries.values())


def _grouping_key(group: TrajectoryGroup, fallback: str) -> str:
    value = group.metadata.get("grouping_tag", group.metadata.get("scenario_id"))
    return fallback if value is None else str(value)


def _resolve_grouping(item: TrajectoryQueueItem) -> TrajectoryQueueItem:
    ref = item.ref
    scenario_id = item.annotations.metadata.get("scenario_id")
    if ref.descriptor.grouping_key != ref.result_id or scenario_id is None:
        return item
    descriptor = ref.descriptor.model_copy(update={"grouping_key": str(scenario_id)})
    return item.model_copy(
        update={"ref": ref.model_copy(update={"descriptor": descriptor})}
    )


def _policy_token_counts(trajectories: list[Trajectory]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for trajectory in trajectories:
        for version, tokens in _trajectory_policy_token_counts(trajectory).items():
            counts[version] = counts.get(version, 0) + tokens
    return counts


def _trajectory_policy_token_counts(trajectory: Trajectory) -> dict[int, int]:
    counts: dict[int, int] = {}
    items: list[Any] = [
        choice
        for exchange in trajectory.exchanges.chat_completions
        for choice in exchange.response.choices
    ]
    items.extend(trajectory.messages_and_choices)
    for history in trajectory.additional_histories:
        items.extend(history.messages_and_choices)
    for item in items:
        extra = getattr(item, "model_extra", None)
        if not isinstance(extra, Mapping):
            continue
        spans = extra.get("policy_token_spans")
        if spans is None:
            continue
        if not isinstance(spans, list):
            raise RuntimeError("policy_token_spans must be a list")
        cursor = 0
        for span in spans:
            parsed = PolicyTokenSpan.model_validate(span)
            if parsed.start_token != cursor:
                raise RuntimeError(
                    "policy_token_spans must be a contiguous completion partition"
                )
            tokens = parsed.end_token - parsed.start_token
            counts[parsed.policy_version] = (
                counts.get(parsed.policy_version, 0) + tokens
            )
            cursor = parsed.end_token
    return counts
