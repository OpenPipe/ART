from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import hashlib
import importlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from art.trainer_rank import TrainerRank

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
    """Transport receipt over canonical TrainerRank files, not another format."""

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
        checkpoint_digest: str,
        directory: Path,
        files: tuple[PortableSnapshotPreparedFile, ...],
    ) -> tuple[PortableSnapshotCommittedFile, ...]: ...

    def close(self, *, deadline: float | None = None) -> None: ...


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
    trainer: TrainerRank,
    sink: PortableSnapshotSink,
    generation: PortableSnapshotGeneration,
    *,
    export_id: str,
    name: str,
    rank: int,
) -> PortableSnapshotRankReceipt | None:
    """Commit canonical TrainerRank files and return private immutable refs."""

    if not export_id or not name or rank < 0:
        raise ValueError("portable checkpoint export identity is invalid")
    from art.trainer_rank import _checkpoint

    digest = hashlib.sha256(f"{name}\0{export_id}".encode()).hexdigest()
    root = Path(tempfile.gettempdir()) / f"art-portable-export-{digest}"
    reservation = root.with_name(f".{root.name}.reserved")

    def clean_shared_paths() -> None:
        for path in (root, reservation):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass

    def clean_rank_snapshots() -> None:
        for path in root.parent.glob(f".{root.name}.snapshot-r{rank}-*"):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass

    clean_rank_snapshots()
    _rank_zero_phase(rank, clean_shared_paths, "prepare portable checkpoint export")
    try:
        trainer.save_checkpoint(str(root), name)

        def commit() -> PortableSnapshotRankReceipt:
            prepared = _checkpoint.prepare_checkpoint(str(root))
            manifest = prepared.manifest
            if manifest is None or manifest["optimizer"] is None:
                raise RuntimeError("portable export requires canonical optimizer state")
            relative_paths = tuple(sorted({"checkpoint.json", *manifest["files"]}))
            files = tuple(
                PortableSnapshotPreparedFile(
                    relative_path=relative,
                    component=_checkpoint_component(relative),
                    byte_count=(root / relative).stat().st_size,
                    sha256=_file_sha256(root / relative),
                )
                for relative in relative_paths
            )
            committed = sink.commit_prepared(
                export_id=export_id,
                generation=generation,
                rank=rank,
                checkpoint_digest=prepared.digest,
                directory=root,
                files=files,
            )
            by_path = {file.relative_path: file for file in committed}
            if len(by_path) != len(committed) or set(by_path) != set(relative_paths):
                raise RuntimeError(
                    "portable sink changed the checkpoint file inventory"
                )
            return PortableSnapshotRankReceipt(
                rank=rank,
                checkpoint_digest=prepared.digest,
                files=tuple(
                    PortableSnapshotFile(
                        object_id=by_path[file.relative_path].object_id,
                        relative_path=file.relative_path,
                        component=file.component,
                        byte_count=file.byte_count,
                        sha256=file.sha256,
                        source_ref=by_path[file.relative_path].source_ref,
                    )
                    for file in files
                ),
            )

        return _rank_zero_phase(rank, commit, "commit portable checkpoint export")
    finally:
        _rank_zero_phase(rank, clean_shared_paths, "clean portable checkpoint export")
        clean_rank_snapshots()


def install_portable_checkpoint(
    trainer: TrainerRank,
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

    owned = {
        file.relative_path: (receipt, file)
        for receipt in archive.ranks
        for file in receipt.files
    }
    read: dict[str, PortableSnapshotReadFile] = {}
    with tempfile.TemporaryDirectory(prefix="art-portable-restore-") as temporary:
        root = Path(temporary)

        def materialize(paths: set[str]) -> None:
            missing = paths.difference(owned)
            if missing:
                raise RuntimeError(
                    f"portable snapshot lacks required files: {sorted(missing)}"
                )
            for receipt in archive.ranks:
                selected = tuple(
                    file
                    for file in receipt.files
                    if file.relative_path in paths and file.relative_path not in read
                )
                if not selected:
                    continue
                _read_rank_files(source, receipt, selected, root)
                read.update(
                    (
                        file.relative_path,
                        PortableSnapshotReadFile(
                            source_rank=receipt.rank,
                            relative_path=file.relative_path,
                            byte_count=file.byte_count,
                            sha256=file.sha256,
                        ),
                    )
                    for file in selected
                )

        from art.trainer_rank import _checkpoint

        prepared = None
        read_error: BaseException | None = None
        try:
            materialize(set(_CHECKPOINT_IDENTITY_FILES))
            entries = tuple(sorted(owned))
            prepared = _checkpoint.prepare_checkpoint(
                str(root), artifact_entries=entries
            )
            if prepared.digest != archive.checkpoint_digest:
                raise RuntimeError("portable checkpoint digest changed")
            assert prepared.manifest is not None
            expected_files = {"checkpoint.json", *prepared.manifest["files"]}
            if set(entries) != expected_files:
                raise RuntimeError("portable archive file inventory changed")
            actual_rank, actual_targets = _adapter_shape(prepared.config)
            if actual_rank != expected_lora_rank or set(actual_targets) != set(
                expected_lora_target_modules
            ):
                raise RuntimeError(
                    "portable checkpoint adapter shape differs from run admission"
                )
            required = (
                _checkpoint.required_local_checkpoint_files(trainer, prepared)
                if restore_optimizer
                else tuple(
                    sorted(
                        relative
                        for relative, (_receipt, file) in owned.items()
                        if file.component != "optimizer"
                    )
                )
            )
            materialize(set(required))
            for relative in set(required) - {"checkpoint.json"}:
                actual = _checkpoint._file_digest(root / relative)
                expected = prepared.manifest["files"][relative]
                if actual != expected:
                    raise RuntimeError(
                        f"portable checkpoint manifest digest changed: {relative}"
                    )
            prepared = _checkpoint.prepare_checkpoint(
                str(root), artifact_entries=entries
            )
            if not restore_optimizer:
                assert prepared.manifest is not None
                # Keep the authenticated canonical checkpoint identity while
                # presenting only its materialized adapter state to the loader.
                adapter_manifest = cast(
                    Any,
                    {
                        **prepared.manifest,
                        "optimizer": None,
                        "parameters": {},
                        "steps": {},
                        "files": {
                            relative: digest
                            for relative, digest in prepared.manifest["files"].items()
                            if _checkpoint_component(relative) != "optimizer"
                        },
                    },
                )
                prepared = replace(
                    prepared,
                    manifest=adapter_manifest,
                    custom=(
                        _checkpoint._load_custom_payload(root, adapter_manifest)
                        if adapter_manifest.get("custom_tensors")
                        else None
                    ),
                )
            elif prepared.manifest is not None and prepared.manifest.get(
                "custom_tensors"
            ):
                prepared = replace(
                    prepared,
                    custom=_checkpoint._load_custom_payload(root, prepared.manifest),
                )
        except BaseException as error:
            read_error = error
        _checkpoint.raise_distributed(
            read_error,
            "prepare portable checkpoint",
            _checkpoint._ensure_group(trainer),
        )
        assert prepared is not None
        _checkpoint.load_checkpoint(trainer, prepared, name)
        config = dict(prepared.config)
    receipt = PortableSnapshotReadReceipt(
        archive_sha256=archive.archive_sha256,
        destination_rank=destination_rank,
        files=tuple(read[path] for path in sorted(read)),
    )
    return receipt, config


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
