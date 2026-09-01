from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, BinaryIO, Iterator, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

CheckpointHost = Any

ART_PORTABLE_SNAPSHOT_SOURCE_FACTORY_ENV = "ART_PORTABLE_SNAPSHOT_SOURCE_FACTORY"
ART_PORTABLE_SNAPSHOT_SINK_FACTORY_ENV = "ART_PORTABLE_SNAPSHOT_SINK_FACTORY"
_SHA256 = r"^[0-9a-f]{64}$"
_GENERATION_ID = re.compile(r"^step-(?P<step>\d{8,})-[0-9a-f]{32}$")
_ENTRYPOINT = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):"
    r"(?P<callable>[A-Za-z_]\w*)$"
)
_CHECKPOINT_IDENTITY_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "checkpoint.json",
)
_CHECKPOINT_METADATA_FILES = ("adapter_config.json", "checkpoint.json")
_MAX_PORTABLE_RANKS = 4096
_MAX_PORTABLE_FILES = 65_536


def _rank_zero_phase[T](rank: int, action: Callable[[], T], phase: str) -> T | None:
    result = None
    error: BaseException | None = None
    if rank == 0:
        try:
            result = action()
        except BaseException as exc:
            error = exc

    import torch.distributed as dist

    errors = [None if error is None else repr(error)]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, None if error is None else repr(error))
    if any(errors):
        if error is not None:
            raise error
        raise RuntimeError(
            f"Another rank failed to {phase}: {next(item for item in errors if item)}"
        )
    return result


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PortableSnapshotGeneration(_Contract):
    training_session_id: str = Field(min_length=1)
    policy_step: int = Field(ge=0)
    generation_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_generation(self) -> "PortableSnapshotGeneration":
        match = _GENERATION_ID.fullmatch(self.generation_id)
        if match is None or int(match.group("step")) != self.policy_step:
            raise ValueError("portable snapshot generation ID changed policy step")
        return self


class PortableSnapshotFile(_Contract):
    object_id: str = Field(min_length=1, max_length=1024)
    relative_path: str = Field(min_length=1)
    component: Literal["metadata", "adapter", "optimizer"]
    byte_count: int = Field(gt=0, le=(1 << 63) - 1)
    sha256: str = Field(pattern=_SHA256)
    source_ref: str = Field(min_length=1, max_length=65_536)

    @model_validator(mode="after")
    def _validate_path(self) -> "PortableSnapshotFile":
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or str(path) != self.relative_path
            or self.relative_path in {"", "."}
            or ".." in path.parts
            or re.match(r"^[A-Za-z]:", self.relative_path) is not None
            or "\\" in self.relative_path
            or "\0" in self.relative_path
        ):
            raise ValueError("portable snapshot path must be normalized")
        if self.component != _checkpoint_component(self.relative_path):
            raise ValueError("portable snapshot file component changed")
        return self


class PortableSnapshotRankReceipt(_Contract):
    rank: int = Field(ge=0)
    checkpoint_digest: str = Field(pattern=_SHA256)
    files: tuple[PortableSnapshotFile, ...] = Field(
        min_length=1, max_length=_MAX_PORTABLE_FILES
    )

    @model_validator(mode="after")
    def _validate_files(self) -> "PortableSnapshotRankReceipt":
        paths = tuple(file.relative_path for file in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("portable rank files must be sorted and unique")
        return self


class PortableSnapshotArchive(_Contract):
    """Transport receipt over canonical Megatron checkpoint files."""

    format: Literal["art_trainer_rank_checkpoint_v1"] = "art_trainer_rank_checkpoint_v1"
    generation: PortableSnapshotGeneration
    checkpoint_digest: str = Field(pattern=_SHA256)
    ranks: tuple[PortableSnapshotRankReceipt, ...] = Field(
        min_length=1, max_length=_MAX_PORTABLE_RANKS
    )
    archive_sha256: str = Field(pattern=_SHA256)
    receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _validate_archive(self) -> "PortableSnapshotArchive":
        ranks = tuple(receipt.rank for receipt in self.ranks)
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("portable snapshot ranks must be sorted and unique")
        files = tuple(file for receipt in self.ranks for file in receipt.files)
        if len(files) > _MAX_PORTABLE_FILES:
            raise ValueError("portable snapshot file inventory exceeds its bound")
        paths = tuple(file.relative_path for file in files)
        if len(paths) != len(set(paths)):
            raise ValueError("portable snapshot file ownership overlaps")
        if not set(_CHECKPOINT_IDENTITY_FILES).issubset(paths):
            raise ValueError("portable snapshot lacks canonical checkpoint metadata")
        if self.archive_sha256 != _json_sha256(_archive_payload(self)):
            raise ValueError("portable snapshot archive digest changed")
        if self.receipt_sha256 != _json_sha256(_receipt_payload(self)):
            raise ValueError("portable snapshot receipt digest changed")
        return self


class PortableSnapshotReadFile(_Contract):
    source_rank: int = Field(ge=0)
    relative_path: str = Field(min_length=1)
    byte_count: int = Field(gt=0, le=(1 << 63) - 1)
    sha256: str = Field(pattern=_SHA256)


class PortableSnapshotPreparedFile(_Contract):
    relative_path: str = Field(min_length=1)
    component: Literal["metadata", "adapter", "optimizer"]
    byte_count: int = Field(gt=0, le=(1 << 63) - 1)
    sha256: str = Field(pattern=_SHA256)


class PortableSnapshotCommittedFile(_Contract):
    relative_path: str = Field(min_length=1)
    object_id: str = Field(min_length=1, max_length=1024)
    source_ref: str = Field(min_length=1, max_length=65_536)


class PortableSnapshotTensorOwner(_Contract):
    tensor_name: str = Field(min_length=1, max_length=4096)
    shard_rank: int = Field(ge=0)
    rank: int = Field(ge=0)


class PortableSnapshotExportReceipt(_Contract):
    export_id: str = Field(min_length=1, max_length=255)
    generation: PortableSnapshotGeneration
    archive: PortableSnapshotArchive
    tensor_owners: tuple[PortableSnapshotTensorOwner, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_generation(self) -> "PortableSnapshotExportReceipt":
        if self.archive.generation != self.generation:
            raise ValueError("portable export archive changed generation")
        identities = tuple(
            (owner.tensor_name, owner.shard_rank) for owner in self.tensor_owners
        )
        if identities != tuple(sorted(set(identities))):
            raise ValueError("portable tensor owners must be sorted and unique")
        return self


class PortableSnapshotReadReceipt(_Contract):
    archive_sha256: str = Field(pattern=_SHA256)
    destination_rank: int = Field(ge=0)
    files: tuple[PortableSnapshotReadFile, ...] = Field(
        min_length=1, max_length=_MAX_PORTABLE_FILES
    )

    @model_validator(mode="after")
    def _validate_files(self) -> "PortableSnapshotReadReceipt":
        paths = tuple(file.relative_path for file in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("portable read files must be sorted and unique")
        return self


class PortableSnapshotInstallReceipt(_Contract):
    archive_sha256: str = Field(pattern=_SHA256)
    runtime_fingerprint: str = Field(pattern=_SHA256)
    restore_optimizer: bool = True
    ranks: tuple[PortableSnapshotReadReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ranks(self) -> "PortableSnapshotInstallReceipt":
        ranks = tuple(receipt.destination_rank for receipt in self.ranks)
        if ranks != tuple(range(len(ranks))):
            raise ValueError("portable install must cover ordered destination ranks")
        if any(receipt.archive_sha256 != self.archive_sha256 for receipt in self.ranks):
            raise ValueError("portable destination ranks installed another archive")
        return self

    def validate_archive(self, archive: PortableSnapshotArchive) -> None:
        if self.archive_sha256 != archive.archive_sha256:
            raise RuntimeError("portable install identifies another archive")
        expected = {
            file.relative_path: (
                receipt.rank,
                file.byte_count,
                file.sha256,
            )
            for receipt in archive.ranks
            for file in receipt.files
            if self.restore_optimizer or file.component != "optimizer"
        }
        observed: dict[str, tuple[int, int, str]] = {}
        for receipt in self.ranks:
            for file in receipt.files:
                identity = (file.source_rank, file.byte_count, file.sha256)
                if expected.get(file.relative_path) != identity:
                    raise RuntimeError(
                        "portable install read evidence changed archive inventory: "
                        f"{file.relative_path}"
                    )
                prior = observed.setdefault(file.relative_path, identity)
                if prior != identity:
                    raise RuntimeError(
                        "portable destination ranks disagreed on a source file"
                    )
        if set(observed) != set(expected):
            missing = sorted(set(expected).difference(observed))
            raise RuntimeError(f"portable install omitted archive files: {missing[:8]}")


class PortableSnapshotLoadReceipt(_Contract):
    operation_id: str = Field(min_length=1, max_length=64)
    generation: PortableSnapshotGeneration
    install: PortableSnapshotInstallReceipt


class PortableSnapshotSource(Protocol):
    """Service-injected reader for private immutable checkpoint files."""

    def read_prepared(
        self,
        receipt: PortableSnapshotRankReceipt,
        files: Mapping[str, memoryview],
    ) -> None: ...

    def close(self, *, deadline: float | None = None) -> None: ...


class PortableSnapshotSink(Protocol):
    """Rank-local durable writer injected by the service runtime."""

    def commit_prepared(
        self,
        *,
        export_id: str,
        generation: PortableSnapshotGeneration,
        rank: int,
        checkpoint_digest: str | None,
        directory: Path,
        files: tuple[PortableSnapshotPreparedFile, ...],
    ) -> tuple[PortableSnapshotCommittedFile, ...]: ...

    def close(self, *, deadline: float | None = None) -> None: ...


@dataclass(slots=True)
class PreparedPortableCheckpoint:
    """Authenticated rank-local files retained until trainer installation."""

    archive: PortableSnapshotArchive
    checkpoint: Any
    config: dict[str, object]
    restore_optimizer: bool
    destination_rank: int
    required_files: tuple[str, ...]
    _source: PortableSnapshotSource
    _owned: dict[str, tuple[PortableSnapshotRankReceipt, PortableSnapshotFile]]
    _read: dict[str, PortableSnapshotReadFile]
    _temporary: tempfile.TemporaryDirectory[str] | None

    @property
    def receipt(self) -> PortableSnapshotReadReceipt:
        return PortableSnapshotReadReceipt(
            archive_sha256=self.archive.archive_sha256,
            destination_rank=self.destination_rank,
            files=tuple(self._read[path] for path in sorted(self._read)),
        )

    @contextmanager
    def materialize(self, relative_path: str) -> Iterator[Path]:
        if self._temporary is None:
            raise RuntimeError("prepared portable checkpoint is closed")
        if relative_path not in self.required_files:
            raise RuntimeError(
                f"portable checkpoint file is not required: {relative_path}"
            )
        root = Path(self._temporary.name)
        path = root / relative_path
        retained = relative_path in _CHECKPOINT_METADATA_FILES
        if not path.is_file():
            receipt, file = self._owned[relative_path]
            try:
                _read_rank_files(self._source, receipt, (file,), root)
                self._record_read(receipt, file)
            except BaseException:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise
        try:
            yield path
        finally:
            if not retained:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _record_read(
        self, receipt: PortableSnapshotRankReceipt, file: PortableSnapshotFile
    ) -> None:
        manifest = self.checkpoint.manifest
        if manifest is not None and file.relative_path in manifest["files"]:
            assert self._temporary is not None
            path = Path(self._temporary.name) / file.relative_path
            actual = _checkpoint_blake2b(path)
            expected = manifest["files"][file.relative_path]
            if actual != expected:
                raise RuntimeError(
                    f"portable checkpoint manifest digest changed: {file.relative_path}"
                )
        self._read[file.relative_path] = PortableSnapshotReadFile(
            source_rank=receipt.rank,
            relative_path=file.relative_path,
            byte_count=file.byte_count,
            sha256=file.sha256,
        )

    def close(self) -> None:
        temporary, self._temporary = self._temporary, None
        if temporary is not None:
            temporary.cleanup()

    def __enter__(self) -> "PreparedPortableCheckpoint":
        if self._temporary is None:
            raise RuntimeError("prepared portable checkpoint is closed")
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def build_portable_snapshot_archive(
    *,
    generation: PortableSnapshotGeneration,
    checkpoint_digest: str,
    ranks: tuple[PortableSnapshotRankReceipt, ...],
) -> PortableSnapshotArchive:
    if any(receipt.checkpoint_digest != checkpoint_digest for receipt in ranks):
        raise ValueError("portable ranks disagree on checkpoint digest")
    provisional = PortableSnapshotArchive.model_construct(
        format="art_trainer_rank_checkpoint_v1",
        generation=generation,
        checkpoint_digest=checkpoint_digest,
        ranks=ranks,
        archive_sha256="0" * 64,
        receipt_sha256="0" * 64,
    )
    archive_sha256 = _json_sha256(_archive_payload(provisional))
    with_archive = provisional.model_copy(update={"archive_sha256": archive_sha256})
    return PortableSnapshotArchive(
        generation=generation,
        checkpoint_digest=checkpoint_digest,
        ranks=ranks,
        archive_sha256=archive_sha256,
        receipt_sha256=_json_sha256(_receipt_payload(with_archive)),
    )


def portable_snapshot_source_from_local_runtime(
    *,
    run_id: str,
    rank: int,
    environ: Mapping[str, str] | None = None,
) -> PortableSnapshotSource | None:
    if not run_id or rank < 0:
        raise ValueError("portable snapshot source requires a run and rank")
    entrypoint = (os.environ if environ is None else environ).get(
        ART_PORTABLE_SNAPSHOT_SOURCE_FACTORY_ENV
    )
    if entrypoint is None:
        return None
    match = _ENTRYPOINT.fullmatch(entrypoint)
    if match is None:
        raise RuntimeError(
            f"{ART_PORTABLE_SNAPSHOT_SOURCE_FACTORY_ENV} must be <module>:<callable>"
        )
    try:
        factory = getattr(
            importlib.import_module(match.group("module")), match.group("callable")
        )
    except (AttributeError, ImportError) as error:
        raise RuntimeError("cannot resolve portable snapshot source factory") from error
    if not callable(factory):
        raise RuntimeError("portable snapshot source factory is not callable")
    source = factory(run_id=run_id, rank=rank)
    if any(
        not callable(getattr(source, method, None))
        for method in ("read_prepared", "close")
    ):
        raise RuntimeError("portable snapshot source factory returned an invalid port")
    return cast(PortableSnapshotSource, source)


def portable_snapshot_sink_from_local_runtime(
    *,
    run_id: str,
    rank: int,
    environ: Mapping[str, str] | None = None,
) -> PortableSnapshotSink | None:
    if not run_id or rank < 0:
        raise ValueError("portable snapshot sink requires a run and rank")
    entrypoint = (os.environ if environ is None else environ).get(
        ART_PORTABLE_SNAPSHOT_SINK_FACTORY_ENV
    )
    if entrypoint is None:
        return None
    match = _ENTRYPOINT.fullmatch(entrypoint)
    if match is None:
        raise RuntimeError(
            f"{ART_PORTABLE_SNAPSHOT_SINK_FACTORY_ENV} must be <module>:<callable>"
        )
    try:
        factory = getattr(
            importlib.import_module(match.group("module")), match.group("callable")
        )
    except (AttributeError, ImportError) as error:
        raise RuntimeError("cannot resolve portable snapshot sink factory") from error
    if not callable(factory):
        raise RuntimeError("portable snapshot sink factory is not callable")
    sink = factory(run_id=run_id, rank=rank)
    if any(
        not callable(getattr(sink, method, None))
        for method in ("commit_prepared", "close")
    ):
        raise RuntimeError("portable snapshot sink factory returned an invalid port")
    return cast(PortableSnapshotSink, sink)


def export_portable_checkpoint(
    trainer: CheckpointHost,
    sink: PortableSnapshotSink,
    generation: PortableSnapshotGeneration,
    *,
    export_id: str,
    name: str,
    rank: int,
    components: Callable[
        [Literal["weights", "optimizer"]],
        AbstractContextManager[tuple[Any, ...]],
    ],
) -> PortableSnapshotRankReceipt | None:
    """Commit canonical files from one lower-tier component window at a time."""

    from .portable_checkpoint_stream import export_portable_checkpoint_streamed

    return export_portable_checkpoint_streamed(
        trainer,
        sink,
        generation,
        export_id=export_id,
        name=name,
        rank=rank,
        components=cast(Any, components),
    )


def prepare_portable_checkpoint(
    trainer: CheckpointHost,
    source: PortableSnapshotSource,
    archive: PortableSnapshotArchive,
    *,
    destination_rank: int,
    expected_lora_rank: int,
    expected_lora_target_modules: tuple[str, ...],
    restore_optimizer: bool,
) -> PreparedPortableCheckpoint:
    """Authenticate identity while deferring bounded data-file materialization."""

    owned = {
        file.relative_path: (receipt, file)
        for receipt in archive.ranks
        for file in receipt.files
    }
    temporary = tempfile.TemporaryDirectory(prefix="art-portable-restore-")
    try:
        root = Path(temporary.name)
        from art.megatron import checkpoint as _checkpoint
        checkpoint = None
        required: tuple[str, ...] = ()
        read: dict[str, PortableSnapshotReadFile] = {}
        read_error: BaseException | None = None
        try:
            missing = set(_CHECKPOINT_IDENTITY_FILES).difference(owned)
            if missing:
                raise RuntimeError(
                    f"portable snapshot lacks required files: {sorted(missing)}"
                )
            for relative in _CHECKPOINT_IDENTITY_FILES:
                receipt, file = owned[relative]
                _read_rank_files(source, receipt, (file,), root)
                read[relative] = PortableSnapshotReadFile(
                    source_rank=receipt.rank,
                    relative_path=relative,
                    byte_count=file.byte_count,
                    sha256=file.sha256,
                )
            checkpoint = _checkpoint.prepare_checkpoint(
                str(root), artifact_entries=tuple(sorted(owned))
            )
            if checkpoint.digest != archive.checkpoint_digest:
                raise RuntimeError("portable checkpoint digest changed")
            assert checkpoint.manifest is not None
            expected_files = {"checkpoint.json", *checkpoint.manifest["files"]}
            if set(owned) != expected_files:
                raise RuntimeError("portable archive file inventory changed")
            actual_rank, actual_targets = _adapter_shape(checkpoint.config)
            if actual_rank != expected_lora_rank or set(actual_targets) != set(
                expected_lora_target_modules
            ):
                raise RuntimeError(
                    "portable checkpoint adapter shape differs from run admission"
                )
            required = (
                _checkpoint.required_local_checkpoint_files(trainer, checkpoint)
                if restore_optimizer
                else tuple(
                    sorted(
                        relative
                        for relative, (_receipt, file) in owned.items()
                        if file.component != "optimizer"
                    )
                )
            )
            if missing := set(required).difference(owned):
                raise RuntimeError(
                    f"portable snapshot lacks required files: {sorted(missing)}"
                )
        except BaseException as error:
            read_error = error
        _checkpoint.raise_distributed(
            read_error,
            "prepare portable checkpoint",
            _checkpoint._ensure_group(trainer),
        )
        assert checkpoint is not None
        return PreparedPortableCheckpoint(
            archive=archive,
            checkpoint=checkpoint,
            config=dict(checkpoint.config),
            restore_optimizer=restore_optimizer,
            destination_rank=destination_rank,
            required_files=tuple(sorted(required)),
            _source=source,
            _owned=owned,
            _read=read,
            _temporary=temporary,
        )
    except BaseException:
        temporary.cleanup()
        raise


def install_prepared_portable_checkpoint(
    trainer: CheckpointHost,
    prepared: PreparedPortableCheckpoint,
    *,
    name: str,
) -> None:
    """Install already authenticated files into one trainer checkpoint slot."""

    if prepared._temporary is None:
        raise RuntimeError("prepared portable checkpoint is closed")
    from art.megatron import checkpoint as _checkpoint

    checkpoint = prepared.checkpoint
    if not prepared.restore_optimizer:
        assert checkpoint.manifest is not None
        checkpoint = replace(
            checkpoint,
            manifest=cast(
                Any,
                {
                    **checkpoint.manifest,
                    "optimizer": None,
                    "parameters": {},
                    "steps": {},
                    "files": {
                        relative: digest
                        for relative, digest in checkpoint.manifest["files"].items()
                        if _checkpoint_component(relative) != "optimizer"
                    },
                },
            ),
        )
    _checkpoint.load_checkpoint(
        trainer, checkpoint, name, materialize=prepared.materialize
    )


def commit_prepared_portable_checkpoint(
    trainer: CheckpointHost,
    *,
    staging_name: str,
    name: str,
) -> None:
    """Atomically replace one live slot with an already prepared CPU slot."""

    if not staging_name or not name or staging_name == name:
        raise ValueError("portable checkpoint commit names are invalid")
    from art.megatron import checkpoint as _checkpoint

    group = _checkpoint._ensure_group(trainer)
    _checkpoint._phase(
        lambda: trainer._guard_slot_can_load(trainer._slot_ref(name)),
        "validate prepared checkpoint target",
        group,
    )
    snapshot = _checkpoint._slot_snapshot(trainer)
    try:
        staged = trainer._checkpoint_slots[staging_name]
    except KeyError as error:
        raise RuntimeError("prepared portable checkpoint slot is absent") from error
    previous = trainer._checkpoint_slots.get(name)
    staged_revision = staged.revision

    def commit() -> None:
        _checkpoint._commit_slot(trainer, staging_name, name)
        installed = trainer._checkpoint_slots.pop(staging_name)
        installed.revision = 0 if previous is None else previous.revision + 1
        trainer._checkpoint_slots[name] = installed

    try:
        _checkpoint._phase(commit, "commit prepared portable checkpoint", group)
    except BaseException as error:

        def rollback() -> None:
            _checkpoint._restore_slots(snapshot)
            staged.revision = staged_revision
            trainer._checkpoint_slots[staging_name] = staged
            if previous is None:
                trainer._checkpoint_slots.pop(name, None)
            else:
                trainer._checkpoint_slots[name] = previous

        try:
            _checkpoint._phase(
                rollback, "roll back prepared portable checkpoint", group
            )
        except BaseException as rollback_error:
            raise BaseExceptionGroup(
                "portable checkpoint commit and rollback failed",
                [error, rollback_error],
            ) from None
        raise


def install_portable_checkpoint(
    trainer: CheckpointHost,
    source: PortableSnapshotSource,
    archive: PortableSnapshotArchive,
    *,
    name: str,
    destination_rank: int,
    expected_lora_rank: int,
    expected_lora_target_modules: tuple[str, ...],
    restore_optimizer: bool,
) -> tuple[PortableSnapshotReadReceipt, dict[str, object]]:
    """Read, authenticate, repartition, and atomically install one checkpoint."""

    with prepare_portable_checkpoint(
        trainer,
        source,
        archive,
        destination_rank=destination_rank,
        expected_lora_rank=expected_lora_rank,
        expected_lora_target_modules=expected_lora_target_modules,
        restore_optimizer=restore_optimizer,
    ) as prepared:
        install_prepared_portable_checkpoint(trainer, prepared, name=name)
        return prepared.receipt, prepared.config


def _read_rank_files(
    source: PortableSnapshotSource,
    receipt: PortableSnapshotRankReceipt,
    files: tuple[PortableSnapshotFile, ...],
    root: Path,
) -> None:
    handles: list[BinaryIO] = []
    maps: list[mmap.mmap] = []
    views: dict[str, memoryview] = {}
    try:
        for file in files:
            path = root / file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w+b")
            handle.truncate(file.byte_count)
            mapping = mmap.mmap(
                handle.fileno(), file.byte_count, access=mmap.ACCESS_WRITE
            )
            handles.append(handle)
            maps.append(mapping)
            views[file.relative_path] = memoryview(mapping)
        source.read_prepared(receipt, views)
    finally:
        for view in views.values():
            view.release()
        for mapping in maps:
            mapping.flush()
            mapping.close()
        for handle in handles:
            handle.close()
    for file in files:
        path = root / file.relative_path
        if path.stat().st_size != file.byte_count:
            raise RuntimeError(
                f"portable snapshot file size changed: {file.relative_path}"
            )
        if _file_sha256(path) != file.sha256:
            raise RuntimeError(
                f"portable snapshot file digest changed: {file.relative_path}"
            )


def _archive_payload(archive: PortableSnapshotArchive) -> dict[str, object]:
    files = sorted(
        (
            file.relative_path,
            file.component,
            file.byte_count,
            file.sha256,
        )
        for receipt in archive.ranks
        for file in receipt.files
    )
    return {
        "format": archive.format,
        "generation": archive.generation.model_dump(mode="json"),
        "checkpoint_digest": archive.checkpoint_digest,
        "files": files,
    }


def _adapter_shape(config: Mapping[str, object]) -> tuple[int, tuple[str, ...]]:
    rank = config.get("r")
    targets = config.get("target_modules")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise RuntimeError("portable checkpoint has an invalid LoRA rank")
    if isinstance(targets, str):
        selected = (targets,)
    elif isinstance(targets, list):
        if not all(isinstance(target, str) and target for target in targets):
            raise RuntimeError("portable checkpoint has invalid LoRA targets")
        selected = tuple(cast(list[str], targets))
    else:
        raise RuntimeError("portable checkpoint has invalid LoRA targets")
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("portable checkpoint LoRA targets must be unique")
    return rank, selected


def _checkpoint_component(
    relative_path: str,
) -> Literal["metadata", "adapter", "optimizer"]:
    if relative_path in {"adapter_config.json", "checkpoint.json"}:
        return "metadata"
    if relative_path.startswith("optimizer/"):
        return "optimizer"
    return "adapter"


def _receipt_payload(archive: PortableSnapshotArchive) -> dict[str, object]:
    return {
        **_archive_payload(archive),
        "archive_sha256": archive.archive_sha256,
        "ranks": [receipt.model_dump(mode="json") for receipt in archive.ranks],
    }


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_blake2b(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
