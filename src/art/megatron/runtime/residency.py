from __future__ import annotations

from collections.abc import Iterable
from threading import Lock
import time
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

ResidencyTier = Literal["l1_gpu", "l2_cpu", "l3_nvme", "l4_archive"]
LOCAL_RESIDENCY_TIERS: tuple[ResidencyTier, ...] = (
    "l1_gpu",
    "l2_cpu",
    "l3_nvme",
)


class ResidencyKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    topology_fingerprint: str = Field(min_length=1)
    adapter_layout_fingerprint: str = Field(min_length=1)


class TierCapacity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_bytes: int = Field(ge=1)
    high_watermark: float = Field(default=0.90, gt=0.0, le=1.0)
    low_watermark: float = Field(default=0.75, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_watermarks(self) -> "TierCapacity":
        if self.low_watermark >= self.high_watermark:
            raise ValueError("low_watermark must be below high_watermark")
        return self

    @property
    def high_bytes(self) -> int:
        return int(self.max_bytes * self.high_watermark)

    @property
    def low_bytes(self) -> int:
        return int(self.max_bytes * self.low_watermark)


class ResidencyLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    l1_gpu: TierCapacity
    l2_cpu: TierCapacity
    l3_nvme: TierCapacity
    max_concurrent_transitions: int = Field(default=2, ge=1)

    def capacity(self, tier: ResidencyTier) -> TierCapacity | None:
        return None if tier == "l4_archive" else getattr(self, tier)


class ResidencyCopy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: ResidencyTier
    byte_count: int = Field(ge=1)
    immutable_ref: str | None = Field(default=None, min_length=1)
    digest: str | None = Field(default=None, min_length=1)
    ready_at: float = Field(ge=0.0)


class ResidencyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ResidencyKey
    copies: tuple[ResidencyCopy, ...]
    pin_count: int = Field(ge=0)
    last_access: float = Field(ge=0.0)


class ResidencyReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reservation_id: str = Field(min_length=1)
    key: ResidencyKey
    source: ResidencyTier | None
    target: ResidencyTier
    byte_count: int = Field(ge=1)
    created_at: float = Field(ge=0.0)


class ResidencyUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready_bytes: dict[ResidencyTier, int]
    reserved_bytes: dict[ResidencyTier, int]


class ResidencyLedger:
    """Thread-safe byte accounting for copy-safe residency transitions."""

    def __init__(self, limits: ResidencyLimits) -> None:
        self.limits = limits
        self._copies: dict[ResidencyKey, dict[ResidencyTier, ResidencyCopy]] = {}
        self._last_access: dict[ResidencyKey, float] = {}
        self._pins: dict[ResidencyKey, int] = {}
        self._reservations: dict[str, ResidencyReservation] = {}
        self._ready_bytes = {tier: 0 for tier in LOCAL_RESIDENCY_TIERS}
        self._reserved_bytes = {tier: 0 for tier in LOCAL_RESIDENCY_TIERS}
        self._lock = Lock()

    def reserve(
        self,
        key: ResidencyKey,
        *,
        source: ResidencyTier | None,
        target: ResidencyTier,
        byte_count: int,
    ) -> ResidencyReservation:
        if byte_count < 1:
            raise ValueError("residency reservation byte_count must be positive")
        with self._lock:
            copies = self._copies.get(key, {})
            if source is not None and source not in copies:
                raise RuntimeError(f"residency source is not ready: {source}")
            if target in copies or any(
                item.key == key and item.target == target
                for item in self._reservations.values()
            ):
                raise RuntimeError(f"residency target already exists: {target}")
            if len(self._reservations) >= self.limits.max_concurrent_transitions:
                raise RuntimeError("residency transition capacity is exhausted")
            capacity = self.limits.capacity(target)
            if capacity is not None:
                projected = (
                    self._ready_bytes[target]
                    + self._reserved_bytes[target]
                    + byte_count
                )
                if projected > capacity.max_bytes:
                    raise RuntimeError(
                        f"{target} residency capacity exceeded: "
                        f"projected={projected}, max={capacity.max_bytes}"
                    )
                self._reserved_bytes[target] += byte_count
            reservation = ResidencyReservation(
                reservation_id=uuid.uuid4().hex,
                key=key,
                source=source,
                target=target,
                byte_count=byte_count,
                created_at=time.monotonic(),
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def commit(
        self,
        reservation: ResidencyReservation,
        *,
        immutable_ref: str | None = None,
        digest: str | None = None,
    ) -> ResidencyCopy:
        with self._lock:
            current = self._require_reservation(reservation)
            now = time.monotonic()
            copy = ResidencyCopy(
                tier=current.target,
                byte_count=current.byte_count,
                immutable_ref=immutable_ref,
                digest=digest,
                ready_at=now,
            )
            self._copies.setdefault(current.key, {})[current.target] = copy
            capacity = self.limits.capacity(current.target)
            if capacity is not None:
                self._reserved_bytes[current.target] -= current.byte_count
                self._ready_bytes[current.target] += current.byte_count
            self._reservations.pop(current.reservation_id)
            self._last_access[current.key] = now
            return copy

    def abort(self, reservation: ResidencyReservation) -> None:
        with self._lock:
            current = self._require_reservation(reservation)
            if self.limits.capacity(current.target) is not None:
                self._reserved_bytes[current.target] -= current.byte_count
            self._reservations.pop(current.reservation_id)

    def drop(
        self,
        key: ResidencyKey,
        tier: ResidencyTier,
        *,
        deleting_entry: bool = False,
    ) -> None:
        with self._lock:
            if self._pins.get(key, 0):
                raise RuntimeError("cannot drop a pinned residency entry")
            copies = self._copies.get(key)
            if copies is None or tier not in copies:
                raise RuntimeError(f"residency copy is not ready: {tier}")
            if not deleting_entry and len(copies) == 1:
                raise RuntimeError("cannot drop the only ready residency copy")
            copy = copies.pop(tier)
            if self.limits.capacity(tier) is not None:
                self._ready_bytes[tier] -= copy.byte_count
            if not copies:
                self._copies.pop(key)
                self._last_access.pop(key, None)
                self._pins.pop(key, None)

    def pin(self, key: ResidencyKey) -> None:
        with self._lock:
            if key not in self._copies:
                raise RuntimeError("cannot pin an unknown residency entry")
            self._pins[key] = self._pins.get(key, 0) + 1
            self._last_access[key] = time.monotonic()

    def unpin(self, key: ResidencyKey) -> None:
        with self._lock:
            count = self._pins.get(key, 0)
            if count < 1:
                raise RuntimeError("residency entry is not pinned")
            if count == 1:
                self._pins.pop(key)
            else:
                self._pins[key] = count - 1
            self._last_access[key] = time.monotonic()

    def touch(self, key: ResidencyKey) -> None:
        with self._lock:
            if key not in self._copies:
                raise RuntimeError("cannot touch an unknown residency entry")
            self._last_access[key] = time.monotonic()

    def required_reclaim(self, tier: ResidencyTier, incoming_bytes: int = 0) -> int:
        if incoming_bytes < 0:
            raise ValueError("incoming_bytes must be non-negative")
        capacity = self.limits.capacity(tier)
        if capacity is None:
            return 0
        with self._lock:
            projected = (
                self._ready_bytes[tier] + self._reserved_bytes[tier] + incoming_bytes
            )
        return 0 if projected <= capacity.high_bytes else projected - capacity.low_bytes

    def eviction_candidates(
        self,
        tier: ResidencyTier,
        reclaim_bytes: int,
        *,
        protected: Iterable[ResidencyKey] = (),
        require_other_copy: bool = True,
    ) -> tuple[ResidencyKey, ...]:
        if reclaim_bytes <= 0:
            return ()
        protected_set = set(protected)
        with self._lock:
            candidates = sorted(
                (
                    key
                    for key, copies in self._copies.items()
                    if tier in copies
                    and not self._pins.get(key, 0)
                    and key not in protected_set
                    and (not require_other_copy or len(copies) > 1)
                    and not any(item.key == key for item in self._reservations.values())
                ),
                key=lambda key: (self._last_access[key], key.run_id, key.generation_id),
            )
            reclaimed = 0
            selected: list[ResidencyKey] = []
            for key in candidates:
                selected.append(key)
                reclaimed += self._copies[key][tier].byte_count
                if reclaimed >= reclaim_bytes:
                    return tuple(selected)
        raise RuntimeError(
            f"insufficient evictable {tier} residency: "
            f"required={reclaim_bytes}, available={reclaimed}"
        )

    def entry(self, key: ResidencyKey) -> ResidencyEntry:
        with self._lock:
            copies = self._copies.get(key)
            if copies is None:
                raise KeyError(key)
            return ResidencyEntry(
                key=key,
                copies=tuple(copies[tier] for tier in sorted(copies)),
                pin_count=self._pins.get(key, 0),
                last_access=self._last_access[key],
            )

    def entries(self) -> tuple[ResidencyEntry, ...]:
        with self._lock:
            keys = tuple(self._copies)
        return tuple(self.entry(key) for key in keys)

    def usage(self) -> ResidencyUsage:
        with self._lock:
            return ResidencyUsage(
                ready_bytes={**self._ready_bytes, "l4_archive": 0},
                reserved_bytes={**self._reserved_bytes, "l4_archive": 0},
            )

    def _require_reservation(
        self, reservation: ResidencyReservation
    ) -> ResidencyReservation:
        current = self._reservations.get(reservation.reservation_id)
        if current is None or current != reservation:
            raise RuntimeError("residency reservation is stale or unknown")
        return current
