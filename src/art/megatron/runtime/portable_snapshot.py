from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from art.trainer_rank import TrainerRank

ART_PORTABLE_SNAPSHOT_SOURCE_FACTORY_ENV = "ART_PORTABLE_SNAPSHOT_SOURCE_FACTORY"
_SHA256 = r"^[0-9a-f]{64}$"
_GENERATION_ID = re.compile(r"^step-(?P<step>\d{8,})-[0-9a-f]{32}$")
_ENTRYPOINT = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):"
    r"(?P<callable>[A-Za-z_]\w*)$"
)
_METADATA_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "checkpoint.json",
)
_MAX_PORTABLE_RANKS = 4096
_MAX_PORTABLE_FILES = 65_536


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
    relative_path: str = Field(min_length=1)
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
            or "\\" in self.relative_path
            or "\0" in self.relative_path
        ):
            raise ValueError("portable snapshot path must be normalized")
        return self


class PortableSnapshotRankReceipt(_Contract):
    rank: int = Field(ge=0)
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
        if not set(_METADATA_FILES).issubset(paths):
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
    ranks: tuple[PortableSnapshotReadReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ranks(self) -> "PortableSnapshotInstallReceipt":
        ranks = tuple(receipt.destination_rank for receipt in self.ranks)
        if ranks != tuple(range(len(ranks))):
            raise ValueError("portable install must cover ordered destination ranks")
        if any(receipt.archive_sha256 != self.archive_sha256 for receipt in self.ranks):
            raise ValueError("portable destination ranks installed another archive")
        return self


class PortableSnapshotSource(Protocol):
    """Service-injected reader for private immutable checkpoint files."""

    def read_prepared(
        self,
        receipt: PortableSnapshotRankReceipt,
        files: Mapping[str, memoryview],
    ) -> None: ...

    def close(self, *, deadline: float | None = None) -> None: ...


def build_portable_snapshot_archive(
    *,
    generation: PortableSnapshotGeneration,
    checkpoint_digest: str,
    ranks: tuple[PortableSnapshotRankReceipt, ...],
) -> PortableSnapshotArchive:
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


def install_portable_checkpoint(
    trainer: TrainerRank,
    source: PortableSnapshotSource,
    archive: PortableSnapshotArchive,
    *,
    name: str,
    destination_rank: int,
    expected_lora_rank: int,
    expected_lora_target_modules: tuple[str, ...],
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
            materialize(set(_METADATA_FILES))
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
            required = _checkpoint.required_local_checkpoint_files(trainer, prepared)
            materialize(set(required))
            for relative in set(required) - {"checkpoint.json"}:
                actual = _checkpoint._file_digest(root / relative)
                expected = prepared.manifest["files"][relative]
                if actual != expected:
                    raise RuntimeError(
                        f"portable checkpoint manifest digest changed: {relative}"
                    )
            if prepared.manifest.get("custom_tensors"):
                prepared = _checkpoint.prepare_checkpoint(str(root))
            else:
                prepared = _checkpoint.prepare_checkpoint(
                    str(root), artifact_entries=entries
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
