from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import stat
from threading import Lock, RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

_LOCK_NAME = ".art_disk_admission.lock"
_MANIFEST_NAME = ".art_disk_admission.json"
_MAX_CLOSED_RESERVATIONS = 256
_PROCESS_ALIVE = "alive"
_PROCESS_DEAD = "dead"
_PROCESS_UNKNOWN = "unknown"
_LOCKS_GUARD = Lock()
_LOCKS: dict[str, RLock] = {}


class DiskAdmissionError(RuntimeError):
    pass


class DiskCapacityExceeded(DiskAdmissionError):
    pass


class DiskProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_identity: str = Field(min_length=1)
    boot_id: str = Field(min_length=1)
    pid_namespace: str = Field(min_length=1)
    pid: int = Field(gt=0)
    start_time_ticks: int = Field(gt=0)


class DiskCatalogClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    claim_owner: str = Field(min_length=1)
    claim_epoch: int = Field(ge=0)
    claim_revision: int | None = Field(default=None, ge=0)


class DiskReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    reservation_id: str = Field(min_length=1)
    storage_identity: str = Field(min_length=1)
    process: DiskProcessIdentity
    purpose: str = Field(min_length=1)
    tenant_id: str | None = Field(default=None, min_length=1)
    run_id: str | None = Field(default=None, min_length=1)
    slot_id: str | None = Field(default=None, min_length=1)
    slot_epoch: int | None = Field(default=None, ge=0)
    catalog_claim: DiskCatalogClaim | None = None
    planned_bytes: int = Field(gt=0)
    remaining_bytes: int = Field(ge=0)
    owned_paths: tuple[str, ...] = Field(min_length=1)
    state: Literal["active", "completed", "cancelled", "abandoned"] = "active"
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> "DiskReservation":
        if self.remaining_bytes > self.planned_bytes:
            raise ValueError("disk reservation remaining bytes exceed its plan")
        if len(self.owned_paths) != len(set(self.owned_paths)):
            raise ValueError("disk reservation owned paths must be unique")
        if (self.state == "active") != (self.closed_at is None):
            raise ValueError("disk reservation state and close timestamp disagree")
        if self.slot_epoch is not None and (
            self.slot_id is None or self.catalog_claim is None
        ):
            raise ValueError(
                "disk reservation slot epochs require catalog revalidation"
            )
        return self


class DiskAdmissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    shared_storage_mount: Path
    storage_identity: str = Field(min_length=1)
    node_identity: str = Field(min_length=1)
    runtime_free_floor_bytes: int = Field(ge=0)
    progress_update_bytes: int = Field(default=256 << 20, ge=1 << 20)


class DiskAdmissionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    storage_identity: str = Field(min_length=1)
    reservations: dict[str, DiskReservation] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reservations(self) -> "DiskAdmissionManifest":
        for key, value in self.reservations.items():
            if key != value.reservation_id:
                raise ValueError("disk reservation key changed identity")
            if value.storage_identity != self.storage_identity:
                raise ValueError("disk reservation changed storage identity")
        return self


class DiskReservationLease:
    def __init__(
        self, admission: "DiskAdmission", reservation_id: str, planned_bytes: int
    ) -> None:
        self._admission = admission
        self.reservation_id = reservation_id
        self.planned_bytes = planned_bytes
        self._written_bytes = 0
        self._recorded_bytes = 0
        self._closed = False

    def record_written(self, written_bytes: int) -> None:
        if self._closed:
            raise DiskAdmissionError("disk reservation lease is closed")
        if not self._written_bytes <= written_bytes <= self.planned_bytes:
            raise DiskAdmissionError("disk reservation write progress is invalid")
        self._written_bytes = written_bytes
        if (
            written_bytes < self.planned_bytes
            and written_bytes - self._recorded_bytes
            < self._admission.config.progress_update_bytes
        ):
            return
        self._admission._set_remaining(
            self.reservation_id, self.planned_bytes - written_bytes
        )
        self._recorded_bytes = written_bytes

    def complete(self) -> None:
        if self._closed:
            raise DiskAdmissionError("disk reservation lease is closed")
        if self._written_bytes != self.planned_bytes:
            raise DiskAdmissionError("disk reservation completed before its full plan")
        self._admission._close(self.reservation_id, "completed")
        self._closed = True

    def cancel(self) -> None:
        if self._closed:
            return
        self._admission._close(self.reservation_id, "cancelled")
        self._closed = True


class DiskAdmission:
    """Cross-process physical-space admission for one shared filesystem."""

    def __init__(self, config: DiskAdmissionConfig) -> None:
        self.config = config
        self.mount = Path(config.shared_storage_mount).resolve(strict=True)
        if not self.mount.is_dir():
            raise NotADirectoryError(self.mount)
        self.lock_path = self.mount / _LOCK_NAME
        self.manifest_path = self.mount / _MANIFEST_NAME
        self.process = _current_process_identity(config.node_identity)
        self._process_pid = os.getpid()
        self._lock = _path_lock(self.lock_path)
        with self._manifest():
            pass

    def reserve(
        self,
        *,
        incoming_peak_bytes: int,
        purpose: str,
        owned_paths: tuple[str | Path, ...],
        tenant_id: str | None = None,
        run_id: str | None = None,
        slot_id: str | None = None,
        slot_epoch: int | None = None,
        catalog_claim: DiskCatalogClaim | None = None,
    ) -> DiskReservationLease:
        self._require_owner_process()
        if incoming_peak_bytes <= 0:
            raise ValueError("disk reservation peak bytes must be positive")
        if not purpose:
            raise ValueError("disk reservation purpose must be non-empty")
        paths = tuple(str(self._owned_path(value)) for value in owned_paths)
        if not paths:
            raise ValueError("disk reservation must own at least one path")
        now = _now()
        reservation_id = uuid4().hex
        reservation = DiskReservation(
            reservation_id=reservation_id,
            storage_identity=self.config.storage_identity,
            process=self.process,
            purpose=purpose,
            tenant_id=tenant_id,
            run_id=run_id,
            slot_id=slot_id,
            slot_epoch=slot_epoch,
            catalog_claim=catalog_claim,
            planned_bytes=incoming_peak_bytes,
            remaining_bytes=incoming_peak_bytes,
            owned_paths=paths,
            created_at=now,
            updated_at=now,
        )
        with self._manifest() as manifest:
            self._reap_unclaimed_dead(manifest)
            if any(Path(value).exists() for value in paths):
                raise FileExistsError("disk reservation owned path already exists")
            active_paths = tuple(
                Path(path)
                for value in manifest.reservations.values()
                if value.state == "active"
                for path in value.owned_paths
            )
            if any(
                _paths_overlap(Path(incoming), active)
                for incoming in paths
                for active in active_paths
            ):
                raise DiskAdmissionError(
                    "disk reservation owned path overlaps an active reservation"
                )
            active_remaining = sum(
                value.remaining_bytes
                for value in manifest.reservations.values()
                if value.state == "active"
            )
            free = _statvfs_free(self.mount)
            available = free - active_remaining - incoming_peak_bytes
            if available < self.config.runtime_free_floor_bytes:
                raise DiskCapacityExceeded(
                    "disk admission rejected reservation: "
                    f"free={free}, active_remaining={active_remaining}, "
                    f"incoming_peak={incoming_peak_bytes}, "
                    f"runtime_floor={self.config.runtime_free_floor_bytes}"
                )
            manifest.reservations[reservation_id] = reservation
            self._prune_closed(manifest)
        return DiskReservationLease(self, reservation_id, incoming_peak_bytes)

    def active_reservations(self) -> tuple[DiskReservation, ...]:
        with self._manifest() as manifest:
            self._reap_unclaimed_dead(manifest)
            return tuple(
                value
                for value in manifest.reservations.values()
                if value.state == "active"
            )

    def dead_reservations(
        self, catalog_claim_is_active: Callable[[DiskCatalogClaim], bool]
    ) -> tuple[DiskReservation, ...]:
        """Dry-run the claimed reservations eligible for guarded cleanup."""
        with self._manifest() as manifest:
            return tuple(
                value
                for value in manifest.reservations.values()
                if value.state == "active"
                and value.catalog_claim is not None
                and self._process_state(value.process) == _PROCESS_DEAD
                and not catalog_claim_is_active(value.catalog_claim)
                and not _paths_have_open_files(value.owned_paths)
            )

    def reap_dead_reservations(
        self, catalog_claim_is_active: Callable[[DiskCatalogClaim], bool]
    ) -> tuple[str, ...]:
        candidates = self.dead_reservations(catalog_claim_is_active)
        reaped: list[str] = []
        now = _now()
        with self._manifest() as manifest:
            for candidate in candidates:
                value = manifest.reservations.get(candidate.reservation_id)
                if (
                    value is None
                    or value.state != "active"
                    or value.catalog_claim is None
                    or self._process_state(value.process) != _PROCESS_DEAD
                    or catalog_claim_is_active(value.catalog_claim)
                    or _paths_have_open_files(value.owned_paths)
                ):
                    continue
                self._abandon(manifest, value, now)
                reaped.append(value.reservation_id)
            self._prune_closed(manifest)
        return tuple(reaped)

    def _set_remaining(self, reservation_id: str, remaining_bytes: int) -> None:
        self._require_owner_process()
        with self._manifest() as manifest:
            value = self._owned_active(manifest, reservation_id)
            if not 0 <= remaining_bytes <= value.remaining_bytes:
                raise DiskAdmissionError(
                    "disk reservation remaining bytes cannot increase"
                )
            manifest.reservations[reservation_id] = value.model_copy(
                update={"remaining_bytes": remaining_bytes, "updated_at": _now()}
            )

    def _close(
        self,
        reservation_id: str,
        state: Literal["completed", "cancelled"],
    ) -> None:
        self._require_owner_process()
        with self._manifest() as manifest:
            value = self._owned_active(manifest, reservation_id)
            now = _now()
            manifest.reservations[reservation_id] = value.model_copy(
                update={
                    "remaining_bytes": 0,
                    "state": state,
                    "updated_at": now,
                    "closed_at": now,
                }
            )
            self._prune_closed(manifest)

    def _owned_active(
        self, manifest: DiskAdmissionManifest, reservation_id: str
    ) -> DiskReservation:
        value = manifest.reservations.get(reservation_id)
        if value is None:
            raise DiskAdmissionError("disk reservation does not exist")
        if value.state != "active":
            raise DiskAdmissionError("disk reservation is not active")
        if value.process != self.process:
            raise DiskAdmissionError("disk reservation belongs to another process")
        return value

    def _reap_unclaimed_dead(self, manifest: DiskAdmissionManifest) -> None:
        now = _now()
        for value in tuple(manifest.reservations.values()):
            if (
                value.state != "active"
                or value.catalog_claim is not None
                or self._process_state(value.process) != _PROCESS_DEAD
                or _paths_have_open_files(value.owned_paths)
            ):
                continue
            self._abandon(manifest, value, now)

    def _abandon(
        self,
        manifest: DiskAdmissionManifest,
        value: DiskReservation,
        now: datetime,
    ) -> None:
        for owned_path in value.owned_paths:
            path = self._owned_path(owned_path)
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.exists():
                shutil.rmtree(path)
        manifest.reservations[value.reservation_id] = value.model_copy(
            update={
                "remaining_bytes": 0,
                "state": "abandoned",
                "updated_at": now,
                "closed_at": now,
            }
        )

    def _process_state(self, value: DiskProcessIdentity) -> str:
        if (
            value.node_identity != self.process.node_identity
            or value.boot_id != self.process.boot_id
            or value.pid_namespace != self.process.pid_namespace
        ):
            return _PROCESS_UNKNOWN
        try:
            state, start_time_ticks = _proc_identity(value.pid)
        except FileNotFoundError:
            return _PROCESS_DEAD
        except OSError:
            return _PROCESS_UNKNOWN
        if start_time_ticks != value.start_time_ticks or state == "Z":
            return _PROCESS_DEAD
        return _PROCESS_ALIVE

    def _owned_path(self, value: str | Path) -> Path:
        path = Path(value).absolute()
        path = path.parent.resolve(strict=False) / path.name
        if path == self.mount or not path.is_relative_to(self.mount):
            raise ValueError("disk reservation owned path leaves shared storage")
        return path

    def _require_owner_process(self) -> None:
        if os.getpid() != self._process_pid:
            raise DiskAdmissionError("disk admission cannot be reused after fork")

    @contextmanager
    def _manifest(self) -> Iterator[DiskAdmissionManifest]:
        with self._lock, _locked_file(self.lock_path, self.mount):
            manifest, exists = _load_manifest(
                self.manifest_path, self.mount, self.config.storage_identity
            )
            if manifest.storage_identity != self.config.storage_identity:
                raise DiskAdmissionError(
                    "disk admission storage identity changed for this mount"
                )
            before = manifest.model_dump_json() if exists else None
            yield manifest
            after = manifest.model_dump_json()
            if after != before:
                _store_manifest(self.manifest_path, self.mount, after.encode())

    @staticmethod
    def _prune_closed(manifest: DiskAdmissionManifest) -> None:
        closed = sorted(
            (
                value
                for value in manifest.reservations.values()
                if value.state != "active"
            ),
            key=lambda value: value.updated_at,
            reverse=True,
        )
        for value in closed[_MAX_CLOSED_RESERVATIONS:]:
            manifest.reservations.pop(value.reservation_id)


@contextmanager
def _locked_file(path: Path, mount: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        _validate_file(descriptor, mount, "disk admission lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _load_manifest(
    path: Path, mount: Path, storage_identity: str
) -> tuple[DiskAdmissionManifest, bool]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return DiskAdmissionManifest(storage_identity=storage_identity), False
    try:
        _validate_file(descriptor, mount, "disk admission manifest")
        with os.fdopen(os.dup(descriptor), "rb") as source:
            return DiskAdmissionManifest.model_validate_json(source.read()), True
    finally:
        os.close(descriptor)


def _store_manifest(path: Path, mount: Path, payload: bytes) -> None:
    temporary = mount / f".{_MANIFEST_NAME}.{os.getpid()}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(descriptor)
        os.replace(temporary, path)
        _fsync_directory(mount)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _validate_file(descriptor: int, mount: Path, name: str) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise DiskAdmissionError(f"{name} is not a regular file")
    if info.st_dev != mount.stat().st_dev:
        raise DiskAdmissionError(f"{name} is outside shared storage")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(f"{name} must not be group/world accessible")


def _paths_have_open_files(values: tuple[str, ...]) -> bool:
    paths = tuple(Path(value) for value in values)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = tuple((process / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise DiskAdmissionError(
                f"cannot revalidate open files for process {process.name}"
            ) from error
        for descriptor in descriptors:
            try:
                target = Path(os.readlink(descriptor))
            except (FileNotFoundError, OSError):
                continue
            if any(_paths_overlap(target, path) for path in paths):
                return True
    return False


def _path_lock(path: Path) -> RLock:
    key = str(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, RLock())


def _current_process_identity(node_identity: str) -> DiskProcessIdentity:
    state, start_time_ticks = _proc_identity(os.getpid())
    if state == "Z":
        raise RuntimeError("current process is a zombie")
    return DiskProcessIdentity(
        node_identity=node_identity,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        pid=os.getpid(),
        start_time_ticks=start_time_ticks,
    )


def _proc_identity(pid: int) -> tuple[str, int]:
    value = Path(f"/proc/{pid}/stat").read_text()
    fields = value[value.rfind(")") + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError("process identity record is truncated")
    return fields[0], int(fields[19])


def _statvfs_free(path: Path) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _now() -> datetime:
    return datetime.now(timezone.utc)
