from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import secrets
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from art.trajectories import MetadataValue, Trajectory, TrajectoryGroup

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
    byte_count: int = Field(ge=0)


class TrajectoryGroupRef(_Contract):
    result_id: str = Field(min_length=1)
    owner_actor_id: str = Field(min_length=1)
    lease_id: str = Field(min_length=1)
    format: Literal["art_trajectory_v1"] = TRAJECTORY_FORMAT
    records: tuple[TrajectoryRecordRef, ...]
    descriptor: TrajectoryGroupDescriptor


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


class TrajectoryEnqueueResult(_Contract):
    status: Literal["accepted", "full", "oversize", "closed"]
    reason: str | None = None


class TrajectoryQueueTake(_Contract):
    item: TrajectoryQueueItem | None = None
    closed: bool = False


class TrajectoryQueueSnapshot(_Contract):
    items: tuple[TrajectoryQueueItem, ...]
    max_ready_groups: int = Field(ge=1)
    capacity_records: int = Field(ge=1)
    capacity_bytes: int = Field(ge=1)
    used_records: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    leased_groups: int = Field(ge=0)


class TrajectoryCapacityError(RuntimeError):
    pass


class TrajectoryLeaseError(RuntimeError):
    pass


class _StoredGroup:
    def __init__(self, header: Any, ref: TrajectoryGroupRef) -> None:
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
        self._records: dict[str, Any] = {}
        self._groups: dict[str, _StoredGroup] = {}
        self._used_bytes = 0

    def put(self, group: TrajectoryGroup) -> TrajectoryGroupRef:
        from msgspec.msgpack import encode

        from .packing import TrajectoryGroupPayload

        payload = TrajectoryGroupPayload.from_group(group)
        record_sizes = tuple(
            len(encode(record.model_dump(mode="python")))
            for record in payload.trajectories
        )
        header = payload.model_copy(update={"trajectories": ()})
        byte_count = sum(record_sizes) + len(encode(header.model_dump(mode="python")))
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
        for record_ref, record in zip(records, payload.trajectories, strict=True):
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
            byte_count=byte_count,
        )
        ref = TrajectoryGroupRef(
            result_id=result_id,
            owner_actor_id=self.owner_actor_id,
            lease_id=secrets.token_hex(16),
            records=records,
            descriptor=descriptor,
        )
        self._groups[result_id] = _StoredGroup(header, ref)
        self._used_bytes += byte_count
        return ref

    def payload(self, ref: TrajectoryGroupRef) -> TrajectoryGroupPayload:
        from .packing import TrajectoryGroupPayload

        stored = self._entry(ref)
        return TrajectoryGroupPayload.model_validate(
            stored.header.model_copy(
                update={
                    "trajectories": tuple(
                        self._records[record.record_id] for record in ref.records
                    )
                }
            )
        )

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
        self.consumer_id: str | None = None


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

    def enqueue(
        self, item: TrajectoryQueueItem, *, max_ready_groups: int
    ) -> TrajectoryEnqueueResult:
        if max_ready_groups < 1:
            raise ValueError("max_ready_groups must be positive")
        self.max_ready_groups = max_ready_groups
        item = _resolve_grouping(item)
        ref = item.ref
        records = len(ref.records)
        byte_count = ref.descriptor.byte_count
        if records > self.capacity_records or byte_count > self.capacity_bytes:
            return TrajectoryEnqueueResult(
                status="oversize",
                reason=f"result requires {records} records/{byte_count} bytes",
            )
        if self._finished:
            return TrajectoryEnqueueResult(status="closed")
        existing = self._entries.get(ref.result_id)
        if existing is not None:
            if existing.item.ref == ref:
                return TrajectoryEnqueueResult(status="accepted")
            raise TrajectoryLeaseError("trajectory result lease changed while queued")
        if (
            len(self._ready) >= self.max_ready_groups
            or self._used_records + records > self.capacity_records
            or self._used_bytes + byte_count > self.capacity_bytes
        ):
            return TrajectoryEnqueueResult(status="full")
        self._entries[ref.result_id] = _QueueEntry(item)
        self._ready.append(ref.result_id)
        self._used_records += records
        self._used_bytes += byte_count
        return TrajectoryEnqueueResult(status="accepted")

    def take(self, consumer_id: str) -> TrajectoryQueueTake:
        if not consumer_id:
            raise ValueError("consumer_id must not be empty")
        if not self._ready:
            return TrajectoryQueueTake(closed=self._finished)
        result_id = self._ready.popleft()
        entry = self._entries[result_id]
        entry.consumer_id = consumer_id
        return TrajectoryQueueTake(item=entry.item)

    def acknowledge(self, result_id: str, consumer_id: str) -> None:
        entry = self._entries.get(result_id)
        if entry is None or entry.consumer_id != consumer_id:
            raise TrajectoryLeaseError(
                "trajectory result has no matching consumer lease"
            )
        self._remove(result_id)

    def finish(self) -> None:
        self._finished = True

    def close(self) -> tuple[TrajectoryGroupRef, ...]:
        refs = tuple(entry.item.ref for entry in self._entries.values())
        self._entries.clear()
        self._ready.clear()
        self._used_records = 0
        self._used_bytes = 0
        self._finished = True
        return refs

    def snapshot(self) -> TrajectoryQueueSnapshot:
        return TrajectoryQueueSnapshot(
            items=tuple(self._entries[result_id].item for result_id in self._ready),
            max_ready_groups=self.max_ready_groups,
            capacity_records=self.capacity_records,
            capacity_bytes=self.capacity_bytes,
            used_records=self._used_records,
            used_bytes=self._used_bytes,
            leased_groups=sum(
                entry.consumer_id is not None for entry in self._entries.values()
            ),
        )

    def _remove(self, result_id: str) -> None:
        entry = self._entries.pop(result_id)
        self._used_records -= len(entry.item.ref.records)
        self._used_bytes -= entry.item.ref.descriptor.byte_count


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
        items = list(trajectory.messages_and_choices)
        for history in trajectory.additional_histories:
            items.extend(history.messages_and_choices)
        for item in items:
            extra = getattr(item, "model_extra", None)
            if not isinstance(extra, Mapping):
                continue
            spans = extra.get("policy_token_spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, Mapping):
                    continue
                try:
                    version = int(span["policy_version"])
                    tokens = int(span["end_token"]) - int(span["start_token"])
                except (KeyError, TypeError, ValueError):
                    continue
                if tokens > 0:
                    counts[version] = counts.get(version, 0) + tokens
    return counts
