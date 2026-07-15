from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArtifactProbeOperation: TypeAlias = Literal[
    "initialize",
    "create",
    "read_created",
    "rename",
    "read_renamed",
    "delete",
    "finalize",
    "cleanup",
]


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
    operation = command.operation
    if operation in {"initialize", "finalize"} and host_index:
        raise RuntimeError(f"only host {spec.host_ids[0]!r} may {operation} the probe")
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
    elif operation == "delete":
        renamed.unlink()
        _fsync(directory)
        _absent(created)
        _absent(renamed)
    elif operation == "finalize":
        directory.rmdir()
        _fsync(root)
    elif operation == "cleanup":
        try:
            directory.stat()
        except FileNotFoundError:
            return
        for path in (created, renamed):
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


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
