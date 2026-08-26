from __future__ import annotations

from collections.abc import Iterable
from threading import Lock
import time
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

ResidencyTier = Literal["l1_gpu", "l2_cpu", "l3_nvme"]
RESIDENCY_TIERS: tuple[ResidencyTier, ...] = (
    "l1_gpu",
    "l2_cpu",
    "l3_nvme",
)


class ResidencyCapacityUnavailable(RuntimeError):
    """A legal transfer cannot be admitted until current residency changes."""


class ResidencyKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    representation: Literal[
        "weights", "optimizer", "accumulator", "reference", "sampler"
    ] = "weights"
    accumulator_revision: int = Field(default=0, ge=0)
    topology_fingerprint: str = Field(min_length=1)
    adapter_layout_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_representation(self) -> "ResidencyKey":
        if (self.representation == "accumulator") != (self.accumulator_revision > 0):
            raise ValueError("only accumulator residency may have a positive revision")
        return self


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

    def capacity(self, tier: ResidencyTier) -> TierCapacity:
        return getattr(self, tier)


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
    pin_counts: dict[ResidencyTier, int]
    last_access: float = Field(ge=0.0)


class ResidencyReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reservation_id: str = Field(min_length=1)
    key: ResidencyKey
    source: ResidencyTier | None
    target: ResidencyTier
    byte_count: int = Field(ge=1)
    created_at: float = Field(ge=0.0)


class ResidencyDemand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ResidencyKey
    source: ResidencyTier | None
    target: ResidencyTier
    byte_count: int = Field(ge=1)


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
        self._pins: dict[ResidencyKey, dict[ResidencyTier, int]] = {}
        self._reservations: dict[str, ResidencyReservation] = {}
        self._ready_bytes = {tier: 0 for tier in RESIDENCY_TIERS}
        self._reserved_bytes = {tier: 0 for tier in RESIDENCY_TIERS}
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
        return self.reserve_many(
            (
                ResidencyDemand(
                    key=key,
                    source=source,
                    target=target,
                    byte_count=byte_count,
                ),
            )
        )[0]

    def reserve_many(
        self, demands: Iterable[ResidencyDemand]
    ) -> tuple[ResidencyReservation, ...]:
        """Atomically reserve every destination in one demanded transfer set."""
        demands = tuple(demands)
        if not demands:
            return ()
        targets = tuple((demand.key, demand.target) for demand in demands)
        if len(set(targets)) != len(targets):
            raise ValueError("residency demand destinations must be unique")
        with self._lock:
            existing_targets = {
                (item.key, item.target) for item in self._reservations.values()
            }
            projected = {
                tier: self._ready_bytes[tier] + self._reserved_bytes[tier]
                for tier in RESIDENCY_TIERS
            }
            for demand in demands:
                copies = self._copies.get(demand.key, {})
                if demand.source is not None and demand.source not in copies:
                    raise RuntimeError(
                        f"residency source is not ready: {demand.source}"
                    )
                if (
                    demand.target in copies
                    or (
                        demand.key,
                        demand.target,
                    )
                    in existing_targets
                ):
                    raise RuntimeError(
                        f"residency target already exists: {demand.target}"
                    )
                projected[demand.target] += demand.byte_count
            for tier, byte_count in projected.items():
                capacity = self.limits.capacity(tier)
                if byte_count > capacity.max_bytes:
                    raise ResidencyCapacityUnavailable(
                        f"{tier} residency capacity exceeded: "
                        f"projected={byte_count}, max={capacity.max_bytes}"
                    )
            now = time.monotonic()
            reservations = tuple(
                ResidencyReservation(
                    reservation_id=uuid.uuid4().hex,
                    key=demand.key,
                    source=demand.source,
                    target=demand.target,
                    byte_count=demand.byte_count,
                    created_at=now,
                )
                for demand in demands
            )
            for reservation in reservations:
                self._reserved_bytes[reservation.target] += reservation.byte_count
                self._reservations[reservation.reservation_id] = reservation
            return reservations

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
            self._reserved_bytes[current.target] -= current.byte_count
            self._ready_bytes[current.target] += current.byte_count
            self._reservations.pop(current.reservation_id)
            self._last_access[current.key] = now
            return copy

    def abort(self, reservation: ResidencyReservation) -> None:
        with self._lock:
            current = self._require_reservation(reservation)
            self._reserved_bytes[current.target] -= current.byte_count
            self._reservations.pop(current.reservation_id)

    def commit_many(
        self, reservations: Iterable[ResidencyReservation]
    ) -> tuple[ResidencyCopy, ...]:
        reservations = tuple(reservations)
        if not reservations:
            return ()
        with self._lock:
            current = tuple(self._require_reservation(item) for item in reservations)
            if len({item.reservation_id for item in current}) != len(current):
                raise ValueError("residency reservations must be unique")
            now = time.monotonic()
            copies = tuple(
                ResidencyCopy(
                    tier=item.target,
                    byte_count=item.byte_count,
                    ready_at=now,
                )
                for item in current
            )
            for item, copy in zip(current, copies, strict=True):
                self._copies.setdefault(item.key, {})[item.target] = copy
                self._reserved_bytes[item.target] -= item.byte_count
                self._ready_bytes[item.target] += item.byte_count
                self._reservations.pop(item.reservation_id)
                self._last_access[item.key] = now
            return copies

    def abort_many(self, reservations: Iterable[ResidencyReservation]) -> None:
        reservations = tuple(reservations)
        if not reservations:
            return
        with self._lock:
            current = tuple(self._require_reservation(item) for item in reservations)
            if len({item.reservation_id for item in current}) != len(current):
                raise ValueError("residency reservations must be unique")
            for item in current:
                self._reserved_bytes[item.target] -= item.byte_count
                self._reservations.pop(item.reservation_id)

    def drop(
        self,
        key: ResidencyKey,
        tier: ResidencyTier,
        *,
        deleting_entry: bool = False,
    ) -> None:
        with self._lock:
            if self._pin_count(key, tier):
                raise RuntimeError(f"cannot drop a pinned {tier} residency copy")
            if any(
                item.key == key and item.source == tier
                for item in self._reservations.values()
            ):
                raise RuntimeError(f"cannot drop an in-transition {tier} source")
            copies = self._copies.get(key)
            if copies is None or tier not in copies:
                raise RuntimeError(f"residency copy is not ready: {tier}")
            if not deleting_entry and len(copies) == 1:
                raise RuntimeError("cannot drop the only ready residency copy")
            copy = copies.pop(tier)
            self._ready_bytes[tier] -= copy.byte_count
            if not copies:
                self._copies.pop(key)
                self._last_access.pop(key, None)
                self._pins.pop(key, None)

    def advance(
        self,
        source: ResidencyKey,
        target: ResidencyKey,
        tier: ResidencyTier,
        *,
        byte_count: int,
    ) -> ResidencyCopy:
        """Transfer one physical copy to a new immutable state identity."""
        if source == target or byte_count < 1:
            raise ValueError("residency advance requires a new key and positive bytes")
        with self._lock:
            source_copies = self._copies.get(source)
            if source_copies is None or tier not in source_copies:
                raise RuntimeError(f"residency source is not ready: {tier}")
            if target in self._copies or any(
                item.key == target for item in self._reservations.values()
            ):
                raise RuntimeError("residency target identity already exists")
            old = source_copies[tier]
            capacity = self.limits.capacity(tier)
            projected = self._ready_bytes[tier] - old.byte_count + byte_count
            if projected > capacity.max_bytes:
                raise RuntimeError(
                    f"{tier} residency capacity exceeded: "
                    f"projected={projected}, max={capacity.max_bytes}"
                )
            self._ready_bytes[tier] = projected
            source_copies.pop(tier)
            source_pins = self._pins.get(source)
            pins = 0 if source_pins is None else source_pins.pop(tier, 0)
            if source_pins == {}:
                self._pins.pop(source)
            if not source_copies:
                self._copies.pop(source)
                self._last_access.pop(source, None)
            now = time.monotonic()
            copy = ResidencyCopy(tier=tier, byte_count=byte_count, ready_at=now)
            self._copies[target] = {tier: copy}
            self._last_access[target] = now
            if pins:
                self._pins[target] = {tier: pins}
            return copy

    def drop_entry(self, key: ResidencyKey) -> None:
        with self._lock:
            if any(self._pins.get(key, {}).values()):
                raise RuntimeError("cannot drop a pinned residency entry")
            if any(item.key == key for item in self._reservations.values()):
                raise RuntimeError("cannot drop residency with an active transition")
            copies = self._copies.pop(key, None)
            if copies is None:
                raise KeyError(key)
            for tier, copy in copies.items():
                self._ready_bytes[tier] -= copy.byte_count
            self._last_access.pop(key, None)
            self._pins.pop(key, None)

    def has_copy(self, key: ResidencyKey, tier: ResidencyTier) -> bool:
        with self._lock:
            return tier in self._copies.get(key, {})

    def copy(self, key: ResidencyKey, tier: ResidencyTier) -> ResidencyCopy:
        with self._lock:
            try:
                return self._copies[key][tier]
            except KeyError as exc:
                raise KeyError((key, tier)) from exc

    def pin(self, key: ResidencyKey, tier: ResidencyTier) -> None:
        self.pin_many(((key, tier),))

    def pin_many(self, copies: Iterable[tuple[ResidencyKey, ResidencyTier]]) -> None:
        copies = tuple(copies)
        if len(set(copies)) != len(copies):
            raise ValueError("residency copies to pin must be unique")
        with self._lock:
            for key, tier in copies:
                if tier not in self._copies.get(key, {}):
                    raise RuntimeError(f"cannot pin a missing {tier} residency copy")
            now = time.monotonic()
            for key, tier in copies:
                pins = self._pins.setdefault(key, {})
                pins[tier] = pins.get(tier, 0) + 1
                self._last_access[key] = now

    def unpin(self, key: ResidencyKey, tier: ResidencyTier) -> None:
        self.unpin_many(((key, tier),))

    def unpin_many(self, copies: Iterable[tuple[ResidencyKey, ResidencyTier]]) -> None:
        copies = tuple(copies)
        if len(set(copies)) != len(copies):
            raise ValueError("residency copies to unpin must be unique")
        with self._lock:
            for key, tier in copies:
                if self._pin_count(key, tier) < 1:
                    raise RuntimeError(f"{tier} residency copy is not pinned")
            now = time.monotonic()
            for key, tier in copies:
                pins = self._pins[key]
                count = pins[tier]
                if count == 1:
                    pins.pop(tier)
                    if not pins:
                        self._pins.pop(key)
                else:
                    pins[tier] = count - 1
                self._last_access[key] = now

    def touch(self, key: ResidencyKey) -> None:
        with self._lock:
            if key not in self._copies:
                raise RuntimeError("cannot touch an unknown residency entry")
            self._last_access[key] = time.monotonic()

    def required_reclaim(self, tier: ResidencyTier, incoming_bytes: int = 0) -> int:
        if incoming_bytes < 0:
            raise ValueError("incoming_bytes must be non-negative")
        capacity = self.limits.capacity(tier)
        with self._lock:
            non_evictable = self._reserved_bytes[tier] + incoming_bytes
            if non_evictable > capacity.max_bytes:
                raise ResidencyCapacityUnavailable(
                    f"{tier} residency capacity exceeded by in-flight state: "
                    f"required={non_evictable}, max={capacity.max_bytes}"
                )
            projected = self._ready_bytes[tier] + non_evictable
        return (
            0
            if projected <= capacity.high_bytes
            else projected - max(capacity.low_bytes, non_evictable)
        )

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
                    and not self._pin_count(key, tier)
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
        raise ResidencyCapacityUnavailable(
            f"insufficient evictable {tier} residency: "
            f"required={reclaim_bytes}, available={reclaimed}"
        )

    def admission_evictions(
        self,
        tier: ResidencyTier,
        incoming_bytes: int,
        demanded_bytes: int,
        *,
        protected: Iterable[ResidencyKey] = (),
        require_other_copy: bool = True,
    ) -> tuple[ResidencyKey, ...]:
        """Plan mandatory capacity reclaim plus best-effort watermark reclaim."""
        if incoming_bytes < 0 or demanded_bytes < 0:
            raise ValueError("residency admission bytes must be non-negative")
        capacity = self.limits.capacity(tier)
        if demanded_bytes > capacity.max_bytes:
            raise ResidencyCapacityUnavailable(
                f"{tier} demanded working set exceeds capacity: "
                f"demanded={demanded_bytes}, max={capacity.max_bytes}"
            )
        protected_set = set(protected)
        with self._lock:
            projected = (
                self._ready_bytes[tier] + self._reserved_bytes[tier] + incoming_bytes
            )
            mandatory = max(projected - capacity.max_bytes, 0)
            protected_reserved = sum(
                item.byte_count
                for item in self._reservations.values()
                if item.target == tier and item.key in protected_set
            )
            unrelated_reserved = self._reserved_bytes[tier] - protected_reserved
            preferred = mandatory
            if projected > capacity.high_bytes:
                retained_floor = max(
                    capacity.low_bytes, demanded_bytes + unrelated_reserved
                )
                preferred = max(mandatory, projected - retained_floor)
            if preferred <= 0:
                return ()
            candidates = sorted(
                (
                    key
                    for key, copies in self._copies.items()
                    if tier in copies
                    and not self._pin_count(key, tier)
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
                if reclaimed >= preferred:
                    return tuple(selected)
            if reclaimed >= mandatory:
                return tuple(selected)
        raise ResidencyCapacityUnavailable(
            f"insufficient evictable {tier} residency for demanded working set: "
            f"required={mandatory}, available={reclaimed}"
        )

    def entry(self, key: ResidencyKey) -> ResidencyEntry:
        with self._lock:
            copies = self._copies.get(key)
            if copies is None:
                raise KeyError(key)
            return ResidencyEntry(
                key=key,
                copies=tuple(copies[tier] for tier in sorted(copies)),
                pin_counts={
                    tier: self._pin_count(key, tier) for tier in RESIDENCY_TIERS
                },
                last_access=self._last_access[key],
            )

    def entries(self) -> tuple[ResidencyEntry, ...]:
        with self._lock:
            keys = tuple(self._copies)
        return tuple(self.entry(key) for key in keys)

    def usage(self) -> ResidencyUsage:
        with self._lock:
            return ResidencyUsage(
                ready_bytes=dict(self._ready_bytes),
                reserved_bytes=dict(self._reserved_bytes),
            )

    def has_reservation(self, key: ResidencyKey) -> bool:
        with self._lock:
            return any(item.key == key for item in self._reservations.values())

    def accounted_bytes(self, tier: ResidencyTier, keys: Iterable[ResidencyKey]) -> int:
        keys = set(keys)
        with self._lock:
            return sum(
                copies[tier].byte_count
                for key, copies in self._copies.items()
                if key in keys and tier in copies
            ) + sum(
                item.byte_count
                for item in self._reservations.values()
                if item.key in keys and item.target == tier
            )

    def _require_reservation(
        self, reservation: ResidencyReservation
    ) -> ResidencyReservation:
        current = self._reservations.get(reservation.reservation_id)
        if current is None or current != reservation:
            raise RuntimeError("residency reservation is stale or unknown")
        return current

    def _pin_count(self, key: ResidencyKey, tier: ResidencyTier) -> int:
        return self._pins.get(key, {}).get(tier, 0)
