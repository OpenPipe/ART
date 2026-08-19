from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .optimizer_state import (
    OPTIMIZER_MANIFEST,
    OptimizerAdapter,
    OptimizerGenerationManifest,
    adapter_generation_lease,
    commit_optimizer_generation,
    optimizer_generation_lease,
    optimizer_generation_path,
    optimizer_pending_generation_path,
    optimizer_shard_name,
    publish_adapter_checkpoint,
    read_adapter_publication,
    read_committed_optimizer_pointer,
)

BUNDLE_MANIFEST = "bundle.json"


class _BundleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BundleFile(_BundleRecord):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _safe_path(self) -> "BundleFile":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint bundle file path is unsafe")
        return self


class CheckpointBundleManifest(_BundleRecord):
    format: Literal["art_megatron_checkpoint_bundle_v1"] = (
        "art_megatron_checkpoint_bundle_v1"
    )
    adapter: OptimizerAdapter
    optimizer: OptimizerGenerationManifest | None = None
    files: tuple[BundleFile, ...]

    @model_validator(mode="after")
    def _validate_generation(self) -> "CheckpointBundleManifest":
        if self.optimizer is not None and self.optimizer.adapter != self.adapter:
            raise ValueError("optimizer and adapter bundle identities differ")
        expected: set[str] = {f"adapter/{file.name}" for file in self.adapter.files}
        if self.optimizer is not None:
            expected.update(
                "optimizer/"
                + optimizer_shard_name(
                    shard.rank,
                    self.optimizer.topology.world_size,
                    shard.serialization,
                )
                for shard in self.optimizer.shards
            )
        actual = {file.path for file in self.files}
        if actual != expected or len(actual) != len(self.files):
            raise ValueError("checkpoint bundle file coverage is incomplete")
        return self


class RestoredCheckpointBundle(_BundleRecord):
    adapter_path: str
    optimizer_state_path: str | None = None
    generation_id: str
    learner_version: int = Field(ge=0)
    training_session_id: str


def export_checkpoint_bundle(
    adapter_path: str,
    *,
    step: int,
    generation_id: str,
    destination: str | Path,
    optimizer_state_path: str | None,
) -> CheckpointBundleManifest:
    adapter = read_adapter_publication(adapter_path, step=step)
    if adapter is None or adapter.generation_id != generation_id:
        raise RuntimeError("checkpoint adapter generation is unavailable")
    target = Path(destination).absolute()
    if target.exists():
        manifest = read_checkpoint_bundle(target, verify_files=True)
        if (
            manifest.adapter.generation_id != generation_id
            or manifest.adapter.step != step
        ):
            raise RuntimeError("checkpoint archive path contains another generation")
        return manifest
    staging = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=True)
        with ExitStack() as leases:
            leases.enter_context(adapter_generation_lease(adapter))
            optimizer = None
            if optimizer_state_path is not None:
                leases.enter_context(
                    optimizer_generation_lease(optimizer_state_path, generation_id)
                )
                generation = optimizer_generation_path(
                    optimizer_state_path, generation_id
                )
                optimizer = OptimizerGenerationManifest.model_validate_json(
                    (generation / OPTIMIZER_MANIFEST).read_text("utf-8")
                )
                if optimizer.adapter != adapter:
                    raise RuntimeError("optimizer generation names another adapter")
            sources: list[tuple[Path, str]] = [
                (Path(adapter.identity) / file.name, f"adapter/{file.name}")
                for file in adapter.files
            ]
            if optimizer is not None:
                sources.extend(
                    (
                        generation / name,
                        f"optimizer/{name}",
                    )
                    for shard in optimizer.shards
                    for name in (
                        optimizer_shard_name(
                            shard.rank,
                            optimizer.topology.world_size,
                            shard.serialization,
                        ),
                    )
                )
            files = tuple(
                _copy_file(source, staging / relative, relative)
                for source, relative in sources
            )
        manifest = CheckpointBundleManifest(
            adapter=adapter,
            optimizer=optimizer,
            files=files,
        )
        _write_manifest(staging / BUNDLE_MANIFEST, manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        _fsync_directory(target.parent)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def consume_checkpoint_bundle(
    source: str | Path,
    *,
    output_dir: str | Path,
    restore_optimizer: bool,
) -> RestoredCheckpointBundle:
    bundle = Path(source).absolute()
    manifest = read_checkpoint_bundle(bundle)
    bundle_files = {record.path: record for record in manifest.files}
    if restore_optimizer and manifest.optimizer is None:
        raise RuntimeError("checkpoint bundle has no optimizer state")
    output = Path(output_dir).absolute()
    canonical = output / "checkpoints" / f"{manifest.adapter.step:04d}"
    adapter = read_adapter_publication(canonical, step=manifest.adapter.step)
    if adapter is None:
        staging = output / "megatron_runtime" / "staging" / uuid4().hex
        try:
            for record in manifest.adapter.files:
                relative = f"adapter/{record.name}"
                _move_verified_file(
                    bundle / relative,
                    staging / record.name,
                    expected=bundle_files[relative],
                )
            adapter = publish_adapter_checkpoint(
                staging,
                step=manifest.adapter.step,
                training_session_id=manifest.adapter.training_session_id,
                generation_id=manifest.adapter.generation_id,
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    elif (
        adapter.generation_id != manifest.adapter.generation_id
        or adapter.training_session_id != manifest.adapter.training_session_id
        or adapter.files != manifest.adapter.files
    ):
        raise RuntimeError("restored adapter path contains another generation")

    optimizer_path = None
    if restore_optimizer:
        assert manifest.optimizer is not None
        optimizer_path = str(output / "optimizer_states")
        pointer = read_committed_optimizer_pointer(optimizer_path)
        if pointer is None:
            pending = optimizer_pending_generation_path(
                optimizer_path, manifest.optimizer.generation
            )
            try:
                for record in manifest.files:
                    if not record.path.startswith("optimizer/"):
                        continue
                    _move_verified_file(
                        bundle / record.path,
                        pending / PurePosixPath(record.path).name,
                        expected=record,
                    )
                restored_manifest = manifest.optimizer.model_copy(
                    update={"adapter": adapter}
                )
                commit_optimizer_generation(
                    optimizer_path,
                    restored_manifest,
                    expected_pointer=None,
                )
            finally:
                if pending.exists():
                    shutil.rmtree(pending)
        elif (
            pointer.generation != manifest.optimizer.generation
            or pointer.adapter != adapter
        ):
            raise RuntimeError("optimizer restore target contains another generation")

    return RestoredCheckpointBundle(
        adapter_path=adapter.identity,
        optimizer_state_path=optimizer_path,
        generation_id=adapter.generation_id,
        learner_version=adapter.step,
        training_session_id=adapter.training_session_id,
    )


def read_checkpoint_bundle(
    path: str | Path, *, verify_files: bool = False
) -> CheckpointBundleManifest:
    root = Path(path).absolute()
    try:
        manifest = CheckpointBundleManifest.model_validate_json(
            (root / BUNDLE_MANIFEST).read_text("utf-8")
        )
    except Exception as error:
        raise RuntimeError(f"invalid checkpoint bundle: {root}") from error
    if verify_files:
        for record in manifest.files:
            _verify_file(root / record.path, record)
    return manifest


def _copy_file(
    source: Path,
    target: Path,
    relative: str,
    *,
    expected: BundleFile | None = None,
) -> BundleFile:
    if not source.is_file():
        raise RuntimeError(f"checkpoint bundle source is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_file, target.open("xb") as output_file:
        while chunk := input_file.read(8 << 20):
            output_file.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    result = BundleFile(path=relative, size_bytes=size, sha256=digest.hexdigest())
    if expected is not None and result != expected:
        raise RuntimeError(f"checkpoint bundle file changed: {source}")
    return result


def _verify_file(path: Path, expected: BundleFile) -> None:
    if not path.is_file() or path.stat().st_size != expected.size_bytes:
        raise RuntimeError(f"checkpoint bundle file is incomplete: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as value:
        while chunk := value.read(8 << 20):
            digest.update(chunk)
    if digest.hexdigest() != expected.sha256:
        raise RuntimeError(f"checkpoint bundle file hash differs: {path}")


def _move_verified_file(source: Path, target: Path, *, expected: BundleFile) -> None:
    _verify_file(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(
            "checkpoint restore staging must share its target filesystem"
        )
    os.replace(source, target)
    with target.open("rb") as output:
        os.fsync(output.fileno())
    _fsync_directory(target.parent)


def _write_manifest(path: Path, manifest: CheckpointBundleManifest) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
