from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Protocol

from art.training import (
    CheckpointRef,
    LoadStateRequest,
    LoadStateResult,
    OperationRef,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)

from .operation_handler import MegatronArtifactResourcePlan, MegatronLoadedState
from .runtime.portable_snapshot import (
    PortableSnapshotArchive,
    PortableSnapshotExportReceipt,
    PortableSnapshotLoadReceipt,
)
from .runtime.specs import TrainerGeneration

LOCAL_CHECKPOINT_ARCHIVE_FILENAME = "archive.json"
_MAX_LOCAL_CHECKPOINT_FILES = 65_536
_LOCAL_SAMPLER_MODES = {"none"}


class _CheckpointCoordinator(Protocol):
    async def export_run_checkpoint(
        self, operation: OperationRef
    ) -> PortableSnapshotExportReceipt: ...

    async def install_run_checkpoint(
        self,
        operation: OperationRef,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
        *,
        restore_optimizer: bool,
    ) -> PortableSnapshotLoadReceipt: ...


class MegatronLocalCheckpointOperations:
    """Persist portable Megatron archives in a run-owned local directory."""

    def __init__(
        self,
        coordinator: _CheckpointCoordinator,
        checkpoint_root: str | Path,
        *,
        run_id: str,
        training_session_id: str,
        output_adapter_root: str | Path | None = None,
        optimizer_state_path: str | Path | None = None,
    ) -> None:
        if not run_id or not training_session_id:
            raise ValueError("local checkpoint run identity must not be empty")
        self.coordinator = coordinator
        self.run_id = run_id
        self.training_session_id = training_session_id
        self.checkpoint_root = _prepare_root(Path(checkpoint_root))
        self.output_adapter_root = Path(
            output_adapter_root or self.checkpoint_root
        ).absolute()
        self.optimizer_state_path = Path(
            optimizer_state_path or self.checkpoint_root.parent / "optimizer_states"
        ).absolute()

    async def save_weights_for_sampler(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> SamplerWeightsResult:
        if (
            request.publication.mode not in _LOCAL_SAMPLER_MODES
            or request.run_id != self.run_id
            or operation.run_id != self.run_id
            or request.sequence_id != operation.sequence_id
            or operation.kind != "save_sampler"
            or operation.learner_parent_version != generation.policy_step
            or generation.training_session_id != self.training_session_id
        ):
            raise ValueError("local sampler-save identity changed")
        return SamplerWeightsResult(
            operation_id=operation.operation_id,
            checkpoint=CheckpointRef(
                run_id=operation.run_id,
                learner_version=generation.policy_step,
                checkpoint_id=request.checkpoint_name,
            ),
            lora=generation.adapter_path,
        )

    async def save_state(
        self,
        request: SaveStateRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> SaveStateResult:
        self._validate_save_identity(request, operation, generation)
        receipt = await self.coordinator.export_run_checkpoint(operation)
        archive = PortableSnapshotArchive.model_validate(
            receipt.archive.model_dump(mode="json")
        )
        exported = receipt.generation
        if (
            receipt.export_id != operation.operation_id
            or exported.training_session_id != generation.training_session_id
            or exported.policy_step != generation.policy_step
            or exported.generation_id != generation.generation_id
            or archive.generation != exported
        ):
            raise RuntimeError("portable export changed the save-state generation")

        checkpoint = self._save_checkpoint_directory(request.checkpoint_name)
        payload = archive.model_dump_json().encode("utf-8")
        await asyncio.to_thread(
            _commit_archive,
            checkpoint,
            payload,
            overwrite=request.overwrite,
        )
        return SaveStateResult(
            operation_id=operation.operation_id,
            checkpoint=CheckpointRef(
                run_id=operation.run_id,
                learner_version=generation.policy_step,
                checkpoint_id=str(checkpoint),
            ),
        )

    async def load_state(
        self,
        request: LoadStateRequest,
        operation: OperationRef,
    ) -> MegatronLoadedState:
        self._validate_load_identity(request, operation)
        checkpoint, archive = await asyncio.to_thread(
            self._read_checkpoint, request.checkpoint
        )
        if archive.generation.training_session_id != self.training_session_id:
            raise RuntimeError("local checkpoint belongs to another training session")
        generation = self._load_generation(operation)
        receipt = await self.coordinator.install_run_checkpoint(
            operation,
            generation,
            archive,
            restore_optimizer=request.restore_optimizer,
        )
        observed = receipt.generation
        if (
            receipt.operation_id != operation.operation_id
            or observed.training_session_id != generation.training_session_id
            or observed.policy_step != generation.policy_step
            or observed.generation_id != generation.generation_id
            or receipt.install.archive_sha256 != archive.archive_sha256
            or receipt.install.restore_optimizer is not request.restore_optimizer
        ):
            raise RuntimeError("portable install changed the load-state identity")
        receipt.install.validate_archive(archive)
        return MegatronLoadedState(
            result=LoadStateResult(
                operation_id=operation.operation_id,
                checkpoint=CheckpointRef(
                    run_id=operation.run_id,
                    learner_version=archive.generation.policy_step,
                    checkpoint_id=str(checkpoint),
                ),
                optimizer_restored=request.restore_optimizer,
            ),
            generation=generation,
            optimizer_state_path=str(self.optimizer_state_path),
        )

    async def plan_artifacts(
        self,
        request: SaveWeightsForSamplerRequest | SaveStateRequest | LoadStateRequest,
        generation: TrainerGeneration,
    ) -> MegatronArtifactResourcePlan:
        if request.run_id != self.run_id:
            raise ValueError("local checkpoint plan belongs to another run")
        if generation.training_session_id != self.training_session_id:
            raise ValueError("local checkpoint plan changed training session")
        if isinstance(request, SaveWeightsForSamplerRequest):
            if request.publication.mode not in _LOCAL_SAMPLER_MODES:
                raise RuntimeError("local checkpoint adapter does not own publication")
            return _empty_plan()
        if isinstance(request, LoadStateRequest):
            _checkpoint, archive = await asyncio.to_thread(
                self._read_checkpoint, request.checkpoint
            )
            if archive.generation.training_session_id != self.training_session_id:
                raise RuntimeError(
                    "local checkpoint belongs to another training session"
                )
            return _archive_plan(
                archive,
                restore_optimizer=request.restore_optimizer,
                storage=False,
            )

        checkpoint = self._save_checkpoint_directory(request.checkpoint_name)
        archive_path = checkpoint / LOCAL_CHECKPOINT_ARCHIVE_FILENAME
        if archive_path.exists():
            archive = await asyncio.to_thread(_read_archive, archive_path)
            if _same_generation(archive, generation):
                return _archive_plan(
                    archive,
                    restore_optimizer=True,
                    storage=True,
                )
            if not request.overwrite:
                raise RuntimeError(
                    "local checkpoint already contains another generation"
                )
        objects, byte_count = await asyncio.to_thread(
            _local_file_inventory,
            (Path(generation.adapter_path), self.optimizer_state_path),
        )
        return MegatronArtifactResourcePlan(
            basis="bounded",
            checkpoint_objects=objects,
            lora_bytes=0,
            transfer_bytes=byte_count,
            storage_bytes=byte_count,
        )

    def _validate_save_identity(
        self,
        request: SaveStateRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> None:
        if (
            request.run_id != self.run_id
            or operation.run_id != self.run_id
            or request.sequence_id != operation.sequence_id
            or operation.kind != "save_state"
            or operation.reserved_output_learner_version is not None
            or operation.learner_parent_version != generation.policy_step
            or generation.training_session_id != self.training_session_id
        ):
            raise ValueError("local save-state checkpoint identity changed")

    def _validate_load_identity(
        self, request: LoadStateRequest, operation: OperationRef
    ) -> None:
        if (
            request.run_id != self.run_id
            or operation.run_id != self.run_id
            or request.sequence_id != operation.sequence_id
            or operation.kind != "load_state"
            or operation.reserved_output_learner_version is None
        ):
            raise ValueError("local load-state checkpoint identity changed")

    def _load_generation(self, operation: OperationRef) -> TrainerGeneration:
        version = operation.reserved_output_learner_version
        if version is None:
            raise ValueError("load-state operation has no reserved learner version")
        suffix = hashlib.sha256(operation.operation_id.encode()).hexdigest()[:32]
        return TrainerGeneration(
            training_session_id=self.training_session_id,
            policy_step=version,
            generation_id=f"step-{version:08d}-{suffix}",
            adapter_path=str(self.output_adapter_root / f"{version:04d}"),
        )

    def _save_checkpoint_directory(self, checkpoint_name: str) -> Path:
        relative = Path(checkpoint_name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
        ):
            raise ValueError("local checkpoint name must be one path component")
        return _within_root(self.checkpoint_root, self.checkpoint_root / relative)

    def _read_checkpoint(
        self, checkpoint_reference: str
    ) -> tuple[Path, PortableSnapshotArchive]:
        reference = Path(checkpoint_reference)
        candidate = (
            reference if reference.is_absolute() else self.checkpoint_root / reference
        )
        resolved = _within_root(self.checkpoint_root, candidate)
        if resolved.name == LOCAL_CHECKPOINT_ARCHIVE_FILENAME:
            checkpoint, archive_path = resolved.parent, resolved
        else:
            checkpoint = resolved
            archive_path = checkpoint / LOCAL_CHECKPOINT_ARCHIVE_FILENAME
        if checkpoint == self.checkpoint_root:
            raise ValueError("local checkpoint reference must identify a checkpoint")
        return checkpoint, _read_archive(archive_path)


def _archive_plan(
    archive: PortableSnapshotArchive,
    *,
    restore_optimizer: bool,
    storage: bool,
) -> MegatronArtifactResourcePlan:
    files = tuple(
        file
        for rank in archive.ranks
        for file in rank.files
        if restore_optimizer or file.component != "optimizer"
    )
    byte_count = sum(file.byte_count for file in files)
    return MegatronArtifactResourcePlan(
        basis="exact",
        checkpoint_objects=len(files),
        lora_bytes=0,
        transfer_bytes=byte_count,
        storage_bytes=byte_count if storage else 0,
    )


def _empty_plan() -> MegatronArtifactResourcePlan:
    return MegatronArtifactResourcePlan(
        basis="exact",
        checkpoint_objects=0,
        lora_bytes=0,
        transfer_bytes=0,
        storage_bytes=0,
    )


def _same_generation(
    archive: PortableSnapshotArchive, generation: TrainerGeneration
) -> bool:
    source = archive.generation
    return (
        source.training_session_id,
        source.policy_step,
        source.generation_id,
    ) == (
        generation.training_session_id,
        generation.policy_step,
        generation.generation_id,
    )


def _local_file_inventory(roots: tuple[Path, ...]) -> tuple[int, int]:
    files: dict[Path, int] = {}
    for requested_root in roots:
        root = requested_root.resolve(strict=True)
        candidates = (root,) if root.is_file() else root.rglob("*")
        for candidate in candidates:
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(
                    f"local checkpoint inventory contains a symlink: {candidate}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"local checkpoint inventory contains a non-file: {candidate}"
                )
            files[candidate] = metadata.st_size
            if len(files) > _MAX_LOCAL_CHECKPOINT_FILES:
                raise RuntimeError("local checkpoint file inventory exceeds its bound")
    return len(files), sum(files.values())


def _commit_archive(checkpoint: Path, payload: bytes, *, overwrite: bool) -> None:
    created = False
    try:
        checkpoint.mkdir()
        created = True
    except FileExistsError:
        if not checkpoint.is_dir():
            raise NotADirectoryError(
                f"local checkpoint is not a directory: {checkpoint}"
            )
    if created:
        _fsync_directory(checkpoint.parent)
    archive_path = checkpoint / LOCAL_CHECKPOINT_ARCHIVE_FILENAME
    try:
        existing = _read_regular_file(archive_path)
    except FileNotFoundError:
        existing = None
    if existing == payload:
        return
    if existing is not None and not overwrite:
        raise RuntimeError("local checkpoint already contains another archive")
    if overwrite:
        _write_atomic(archive_path, payload)
        return
    _write_once(archive_path, payload)


def _read_archive(path: Path) -> PortableSnapshotArchive:
    return PortableSnapshotArchive.model_validate_json(_read_regular_file(path))


def _prepare_root(path: Path) -> Path:
    root = path.absolute()
    existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"local state root is not a directory: {root}")
    if not existed:
        _fsync_directory(root.parent)
    return root


def _within_root(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("local checkpoint reference escaped its root") from None
    return resolved


def _write_once(path: Path, payload: bytes) -> None:
    candidate = _write_temporary(path, payload)
    try:
        try:
            os.link(candidate, path)
        except FileExistsError:
            if _read_regular_file(path) != payload:
                raise RuntimeError(
                    "local checkpoint was concurrently committed with another archive"
                ) from None
        _fsync_directory(path.parent)
    finally:
        candidate.unlink(missing_ok=True)


def _write_atomic(path: Path, payload: bytes) -> None:
    candidate = _write_temporary(path, payload)
    try:
        os.replace(candidate, path)
        _fsync_directory(path.parent)
    finally:
        candidate.unlink(missing_ok=True)


def _write_temporary(path: Path, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def _read_regular_file(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"local state path is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
