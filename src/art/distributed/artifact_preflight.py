from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
import stat
import threading
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArtifactProbeOperation: TypeAlias = Literal[
    "initialize",
    "create",
    "read_created",
    "rename",
    "read_renamed",
    "hold_lock",
    "check_lock_held",
    "release_lock",
    "check_lock_released",
    "delete",
    "finalize",
    "cleanup",
]

_HELD_FLOCKS: dict[tuple[str, str], int] = {}
_FLOCK_GUARD = threading.Lock()


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactProbeSpec(_Contract):
    artifact_root: str = Field(min_length=1)
    runtime_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    host_ids: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)


class ArtifactProbeCommand(_Contract):
    spec: ArtifactProbeSpec
    operation: ArtifactProbeOperation


class ArtifactProbeResult(_Contract):
    host_id: str = Field(min_length=1)
    operation: ArtifactProbeOperation
    path: str = Field(min_length=1)
    error_type: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def _validate_error(self) -> ArtifactProbeResult:
        if (self.error_type is None) != (self.message is None):
            raise ValueError("artifact probe error fields must be set together")
        return self


class ArtifactRootPreflightError(RuntimeError):
    def __init__(self, result: ArtifactProbeResult) -> None:
        self.result = result
        super().__init__(
            f"artifact_root preflight failed on host {result.host_id!r} during "
            f"{result.operation} at {result.path}: {result.error_type}: {result.message}"
        )


def execute_artifact_probe(
    host_id: str, command: ArtifactProbeCommand
) -> ArtifactProbeResult:
    directory = _probe_directory(command.spec)
    try:
        _execute(host_id, command, directory)
        return ArtifactProbeResult(
            host_id=host_id, operation=command.operation, path=str(directory)
        )
    except Exception as error:
        return ArtifactProbeResult(
            host_id=host_id,
            operation=command.operation,
            path=str(directory),
            error_type=type(error).__name__,
            message=str(error) or type(error).__name__,
        )


def _execute(host_id: str, command: ArtifactProbeCommand, directory: Path) -> None:
    spec = command.spec
    try:
        host_index = spec.host_ids.index(host_id)
    except ValueError:
        raise RuntimeError(f"host {host_id!r} is not assigned to this probe") from None
    root = Path(spec.artifact_root)
    created = directory / f"{host_index}.created"
    renamed = directory / f"{host_index}.renamed"
    lock = directory / "advisory.lock"
    lock_key = (spec.runtime_id, host_id)
    operation = command.operation
    if (
        operation in {"initialize", "hold_lock", "release_lock", "finalize"}
        and host_index
    ):
        raise RuntimeError(f"only host {spec.host_ids[0]!r} may {operation} the probe")
    if operation in {"check_lock_held", "check_lock_released"} and not host_index:
        raise RuntimeError(f"host {spec.host_ids[0]!r} owns the probe lock")
    if operation == "initialize":
        if not stat.S_ISDIR(root.stat().st_mode):
            raise NotADirectoryError(f"not a directory: {root}")
        directory.mkdir(mode=0o700)
        _fsync(root)
    elif operation == "create":
        with created.open("xb") as handle:
            handle.write(_payload(spec, host_index))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync(directory)
        _read(created, spec, host_index)
    elif operation == "read_created":
        for index in range(len(spec.host_ids)):
            _read(directory / f"{index}.created", spec, index)
    elif operation == "rename":
        created.rename(renamed)
        _fsync(directory)
        _read(renamed, spec, host_index)
    elif operation == "read_renamed":
        for index in range(len(spec.host_ids)):
            _read(directory / f"{index}.renamed", spec, index)
    elif operation == "hold_lock":
        _hold_flock(lock, lock_key)
    elif operation == "check_lock_held":
        _check_flock(lock, should_block=True)
    elif operation == "release_lock":
        _release_flock(lock_key)
    elif operation == "check_lock_released":
        _check_flock(lock, should_block=False)
    elif operation == "delete":
        renamed.unlink()
        if not host_index and len(spec.host_ids) > 1:
            lock.unlink(missing_ok=True)
        _fsync(directory)
        _absent(created)
        _absent(renamed)
    elif operation == "finalize":
        directory.rmdir()
        _fsync(root)
    elif operation == "cleanup":
        _release_flock(lock_key, required=False)
        try:
            directory.stat()
        except FileNotFoundError:
            return
        paths = (created, renamed, lock) if not host_index else (created, renamed)
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync(directory)


def _probe_directory(spec: ArtifactProbeSpec) -> Path:
    return Path(spec.artifact_root) / f".art-runtime-preflight-{spec.runtime_id}"


def _payload(spec: ArtifactProbeSpec, host_index: int) -> bytes:
    return f"art-runtime-preflight-v1\n{spec.runtime_id}\n{host_index}\n".encode()


def _read(path: Path, spec: ArtifactProbeSpec, host_index: int) -> None:
    if path.read_bytes() != _payload(spec, host_index):
        raise RuntimeError(f"artifact probe payload mismatch at {path}")


def _absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(f"artifact probe path still exists: {path}")


def _hold_flock(path: Path, key: tuple[str, str]) -> None:
    with _FLOCK_GUARD:
        if key in _HELD_FLOCKS:
            raise RuntimeError(f"artifact probe lock is already held: {path}")
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fsync(descriptor)
            _fsync(path.parent)
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        _HELD_FLOCKS[key] = descriptor


def _release_flock(key: tuple[str, str], *, required: bool = True) -> None:
    with _FLOCK_GUARD:
        descriptor = _HELD_FLOCKS.pop(key, None)
        if descriptor is None:
            if required:
                raise RuntimeError("artifact probe lock is not held")
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _check_flock(path: Path, *, should_block: bool) -> None:
    descriptor = os.open(path, os.O_RDWR)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            if not should_block:
                raise RuntimeError(
                    f"artifact probe lock remained held after release: {path}"
                ) from error
        else:
            acquired = True
            if should_block:
                raise RuntimeError(
                    f"artifact probe lock was acquired while owner held it: {path}"
                )
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
