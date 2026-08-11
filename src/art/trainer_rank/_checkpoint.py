"""Topology-independent dynamic LoRA checkpoint persistence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import threading
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Protocol,
    cast,
)
import uuid

from pydantic import BaseModel, ConfigDict, Field
import torch
import torch.distributed as dist

from art.megatron._collective import (
    collective_errors as _collective_errors,
)
from art.megatron._collective import (
    distributed as _distributed,
)
from art.megatron._collective import (
    dtype_name as _dtype_name,
)
from art.megatron._collective import (
    gather_objects as _gather_objects,
)
from art.megatron._collective import (
    raise_distributed as _raise_distributed,
)
from art.megatron._collective import (
    rank as _rank,
)

if TYPE_CHECKING:
    from art.megatron.lora import LoraShardManifest, LoraShardMeta
    from art.megatron.weights.lora_publish import _LocalLoraExport
    from art.trainer_rank._impl import TrainerRank, _DynamicOptimizer


MANIFEST_FILE = "checkpoint.json"
_LAYER_RE = re.compile(r"\.layers\.(?P<layer>\d+)\.")


class TensorRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str
    tensor: str
    shape: tuple[int, ...]
    dtype: str


class ParameterRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    master: TensorRecord
    exp_avg: TensorRecord
    exp_avg_sq: TensorRecord


class AdamWRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    amsgrad: Literal[False] = False


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Literal[1] = 1
    optimizer_format: Literal["adamw"] = "adamw"
    base_model_name_or_path: str
    optimizer: AdamWRecord | None
    parameters: dict[str, ParameterRecord]
    steps: dict[str, Annotated[float, Field(ge=0)]]
    digest: str


@dataclass(frozen=True)
class PreparedCheckpoint:
    path: Path
    adapter_config: dict[str, object]
    manifest: CheckpointManifest | None
    artifact_keys: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class LocalOptimizerState:
    masters: tuple[torch.Tensor, ...]
    exp_avgs: tuple[torch.Tensor, ...]
    exp_avg_sqs: tuple[torch.Tensor, ...]
    steps: tuple[float, ...]
    config: AdamWRecord


@dataclass(frozen=True)
class _LocalShard:
    metadata: LoraShardMeta
    master: torch.Tensor
    exp_avg: torch.Tensor | None
    exp_avg_sq: torch.Tensor | None
    expert: int | None
    step: float


@dataclass(frozen=True)
class _SnapshotShard:
    metadata: LoraShardMeta
    lora_file: str
    optimizer_files: Mapping[str, str] | None
    step: float | None


@dataclass(frozen=True)
class _PreparedSave:
    sequence: int
    snapshot: Path
    destination: Path
    adapter_config: dict[str, object]
    shards: tuple[_SnapshotShard, ...]
    optimizer: AdamWRecord | None


class _SafeSlice(Protocol):
    def get_shape(self) -> list[int]: ...

    def __getitem__(self, slices: tuple[slice, ...]) -> torch.Tensor: ...


class _SafeTensorFile(Protocol):
    def keys(self) -> list[str]: ...

    def get_slice(self, key: str) -> _SafeSlice: ...

    def get_tensor(self, key: str) -> torch.Tensor: ...


def _checkpoint_metadata(
    path: str | Path,
    *,
    require_optimizer: bool = False,
    verify_payload: bool = True,
    artifact_entries: Iterable[str] | None = None,
    expected_digest: str | None = None,
) -> tuple[Path, dict[str, object], tuple[str, ...], CheckpointManifest | None, str]:
    if (artifact_entries is None) != (expected_digest is None):
        raise ValueError(
            "artifact_entries and expected_digest must be provided together"
        )
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    from art.megatron.model_support.lora_disk import load_adapter_config, safe_open

    adapter_config = cast(dict[str, object], load_adapter_config(root))
    with safe_open(root / "adapter_model.safetensors", framework="pt") as handle:
        artifact_keys = tuple(sorted(handle.keys()))
    manifest_path = root / MANIFEST_FILE
    manifest = (
        CheckpointManifest.model_validate_json(manifest_path.read_text())
        if manifest_path.is_file()
        else None
    )
    if require_optimizer and (manifest is None or manifest.optimizer is None):
        raise RuntimeError("Checkpoint does not contain canonical optimizer state")
    if manifest is not None:
        from art.megatron.model_support.lora_disk import (
            ART_LORA_FORMAT_CONFIG_KEY,
            ART_LORA_FORMAT_MEGATRON,
        )

        if adapter_config.get(ART_LORA_FORMAT_CONFIG_KEY) != ART_LORA_FORMAT_MEGATRON:
            raise RuntimeError("Exact checkpoint is not in ART-native Megatron format")
        _validate_manifest(manifest, artifact_keys)
        if artifact_entries is not None:
            assert expected_digest is not None
            _validate_artifact_view(manifest, artifact_entries, expected_digest)
        elif verify_payload:
            _validate_files(root, manifest)
            actual = _digest(root, manifest.model_copy(update={"digest": ""}))
            if actual != manifest.digest:
                raise RuntimeError(
                    f"Checkpoint digest mismatch for {path}: {actual} != {manifest.digest}"
                )
    digest = (
        manifest.digest
        if manifest is not None
        else _hash_files(root, ("adapter_config.json", "adapter_model.safetensors"))
    )
    return root, adapter_config, artifact_keys, manifest, digest


def validate_checkpoint(
    path: str | Path, *, require_optimizer: bool = False
) -> CheckpointManifest | None:
    """Validate an ART checkpoint without materializing trainer-local state."""
    return _checkpoint_metadata(path, require_optimizer=require_optimizer)[3]


def materialize_lora(
    path: str | Path,
    output_dir: str | Path,
    *,
    require_optimizer: bool = False,
    artifact_entries: Iterable[str] | None = None,
    expected_digest: str | None = None,
) -> None:
    """Copy only the inference adapter view from a validated checkpoint."""
    root, _config, _keys, _manifest, _digest_value = _checkpoint_metadata(
        path,
        require_optimizer=require_optimizer,
        verify_payload=artifact_entries is None,
        artifact_entries=artifact_entries,
        expected_digest=expected_digest,
    )
    inference_files = {"adapter_config.json", "adapter_model.safetensors"}
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"LoRA output directory is not empty: {destination}")
    shutil.copytree(
        root,
        destination,
        dirs_exist_ok=True,
        ignore=lambda _root, names: [
            name for name in names if name not in inference_files
        ],
    )
    from art.megatron.model_support.lora_disk import (
        normalize_lora_checkpoint_to_vllm,
    )

    normalize_lora_checkpoint_to_vllm(destination)


def prepare_checkpoint(path: str) -> PreparedCheckpoint:
    root, adapter_config, artifact_keys, manifest, digest = _checkpoint_metadata(
        path, verify_payload=_is_node_validator()
    )

    return PreparedCheckpoint(
        root,
        adapter_config,
        manifest,
        artifact_keys,
        digest,
    )


def save_checkpoint(
    trainer: TrainerRank,
    output_dir: str,
    checkpoint_name: str,
) -> None:
    prepare_checkpoint_save(trainer, output_dir, checkpoint_name)
    finish_checkpoint_save(trainer, output_dir)


def prepare_checkpoint_save(
    trainer: TrainerRank,
    output_dir: str,
    checkpoint_name: str,
) -> None:
    """Capture immutable rank-local state without retaining live tensors."""
    _ensure_checkpoint_save_state(trainer)
    destination = Path(output_dir)
    if output_dir in trainer._prepared_checkpoint_saves:
        raise RuntimeError(f"Checkpoint save is already pending: {output_dir}")
    with trainer._checkpoint_save_condition:
        trainer._completed_checkpoint_saves.discard(output_dir)
    _ensure_checkpoint_group(trainer)
    _validate_save_state(trainer, checkpoint_name)
    adapter_config = deepcopy(dict(_checkpoint_config(trainer, checkpoint_name)))
    dynamic = trainer._dynamic_optimizers.get(checkpoint_name)
    initialized = dynamic is not None
    if any(value != initialized for value in _gather_objects(initialized)):
        raise trainer._slot_state_error(
            "Checkpoint optimizer initialization differs across ranks"
        )
    optimizer: AdamWRecord | None = None
    with _collective_errors("prepare optimizer snapshot metadata"):
        if dynamic is not None:
            optimizer = _optimizer_config(dynamic)
    if any(value != optimizer for value in _gather_objects(optimizer)):
        raise trainer._slot_state_error(
            "Checkpoint optimizer config differs across ranks"
        )

    sequence = trainer._checkpoint_save_sequence
    if any(value != sequence for value in _gather_objects(sequence)):
        raise RuntimeError("Checkpoint save sequence differs across ranks")
    snapshot = destination.with_name(
        f".{destination.name}.snapshot-r{_rank()}-{uuid.uuid4().hex}"
    )
    prepared: _PreparedSave | None = None
    error: BaseException | None = None
    try:
        snapshot.mkdir(parents=True)
        from art.megatron.weights.lora_publish import collect_local_lora_entries

        local_tensors, metadata = collect_local_lora_entries(
            trainer.runtime.model,
            {},
            owner_rank=_rank(),
            slot_ref=trainer._slot_ref(checkpoint_name),
        )
        local_optimizer = (
            ()
            if dynamic is None
            else _local_optimizer_shards(trainer, checkpoint_name, dynamic)
        )
        optimizer_by_key = {item.metadata.key: item for item in local_optimizer}
        if dynamic is not None and set(optimizer_by_key) != set(local_tensors):
            raise trainer._slot_state_error(
                "Local optimizer tensors differ from the LoRA snapshot: "
                f"optimizer={sorted(optimizer_by_key)} "
                f"lora={sorted(local_tensors)}"
            )

        records: list[_SnapshotShard] = []
        blocks = sorted({item.block for item in metadata})
        for index, block in enumerate(blocks):
            block_metadata = [item for item in metadata if item.block == block]
            lora_file = f"lora-{index:06d}.safetensors"
            _save_file(
                {
                    item.key: local_tensors[item.key].detach().cpu().contiguous()
                    for item in block_metadata
                },
                snapshot / lora_file,
            )
            optimizer_files: dict[str, str] | None = None
            if dynamic is not None:
                optimizer_files = {}
                for component in ("master", "exp_avg", "exp_avg_sq"):
                    relative = f"{component}-{index:06d}.safetensors"
                    _save_file(
                        {
                            item.key: _optimizer_component(
                                optimizer_by_key[item.key], component
                            )
                            .detach()
                            .cpu()
                            .contiguous()
                            for item in block_metadata
                        },
                        snapshot / relative,
                    )
                    optimizer_files[component] = relative
            records.extend(
                _SnapshotShard(
                    item,
                    lora_file,
                    None if optimizer_files is None else dict(optimizer_files),
                    (None if dynamic is None else optimizer_by_key[item.key].step),
                )
                for item in block_metadata
            )

        manifest = {
            "sequence": sequence,
            "adapter_config": adapter_config,
            "optimizer": None if optimizer is None else optimizer.model_dump(),
            "shards": [
                {
                    "metadata": dict(record.metadata._asdict()),
                    "lora_file": record.lora_file,
                    "optimizer_files": record.optimizer_files,
                    "step": record.step,
                }
                for record in records
            ],
        }
        (snapshot / "snapshot.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        prepared = _PreparedSave(
            sequence,
            snapshot,
            destination,
            adapter_config,
            tuple(records),
            optimizer,
        )
    except BaseException as exc:
        error = exc
    try:
        _raise_distributed(error, "prepare immutable checkpoint snapshot")
    except BaseException:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
    assert prepared is not None
    trainer._prepared_checkpoint_saves[output_dir] = prepared
    trainer._checkpoint_save_sequence += 1


def finish_checkpoint_save(trainer: TrainerRank, output_dir: str) -> None:
    """Finalize a prepared checkpoint without reading mutable trainer state."""
    _ensure_checkpoint_save_state(trainer)
    condition = trainer._checkpoint_save_condition
    with condition:
        condition.wait_for(
            lambda: output_dir not in trainer._checkpoint_finishing_saves
        )
        if output_dir in trainer._completed_checkpoint_saves:
            return
        prepared = trainer._prepared_checkpoint_saves.get(output_dir)
        if prepared is None:
            raise RuntimeError(f"Checkpoint save was not prepared: {output_dir}")
        condition.wait_for(
            lambda: prepared.sequence == trainer._checkpoint_finish_sequence
        )
        trainer._checkpoint_finishing_saves.add(output_dir)

    error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        _finish_prepared_save(trainer, prepared)
    except BaseException as exc:
        error = exc
    try:
        shutil.rmtree(prepared.snapshot)
    except FileNotFoundError:
        pass
    except BaseException as exc:
        cleanup_error = exc
    finally:
        trainer._prepared_checkpoint_saves.pop(output_dir, None)
        with condition:
            trainer._checkpoint_finishing_saves.discard(output_dir)
            if error is None and cleanup_error is None:
                trainer._completed_checkpoint_saves.add(output_dir)
            trainer._checkpoint_finish_sequence += 1
            condition.notify_all()
    if error is not None and cleanup_error is not None:
        raise BaseExceptionGroup(
            "checkpoint finalization and snapshot cleanup both failed",
            [error, cleanup_error],
        )
    if error is not None:
        raise error
    if cleanup_error is not None:
        raise cleanup_error


def export_lora(
    trainer: TrainerRank,
    output_dir: str,
    checkpoint_name: str,
    *,
    native_format: bool = False,
) -> int:
    config = _checkpoint_config(trainer, checkpoint_name)
    _export_lora_files(
        trainer, output_dir, checkpoint_name, config, native_format=native_format
    )
    return trainer._checkpoint_revisions.get(checkpoint_name, 0)


def _export_lora_files(
    trainer: TrainerRank,
    output_dir: str,
    checkpoint_name: str,
    config: Mapping[str, object],
    *,
    native_format: bool,
) -> "_LocalLoraExport":
    from art.megatron.weights.lora_publish import save_vllm_lora_from_model

    result: _LocalLoraExport | None = None
    error: BaseException | None = None
    try:
        result = save_vllm_lora_from_model(
            model=trainer.runtime.model,
            adapter_dtypes={},
            handler=trainer.runtime.model_support_handler,
            adapter_config=dict(config),
            output_dir=output_dir,
            rank=trainer.runtime.rank,
            world_size=trainer.runtime.world_size,
            slot_ref=trainer._slot_ref(checkpoint_name),
            native_format=native_format,
        )
    except BaseException as exc:
        error = exc
    _raise_distributed(error, "export LoRA")
    assert result is not None
    return result


def _load_stage_lora(
    trainer: TrainerRank, source: PreparedCheckpoint, adapter_rank: int
) -> dict[str, torch.Tensor]:
    if _native_checkpoint(source):
        return _load_native_lora(trainer, source, adapter_rank)
    local_layers = {
        match.group("layer")
        for key in trainer._local_adapter_keys()
        if (match := _LAYER_RE.search(key)) is not None
    }
    selected = {
        key
        for key in source.artifact_keys
        if (match := _LAYER_RE.search(key)) is None
        or match.group("layer") in local_layers
    }
    from art.megatron.model_support.lora_disk import safe_open

    with safe_open(source.path / "adapter_model.safetensors", framework="pt") as file:
        tensors = {key: file.get_tensor(key) for key in selected}
    return trainer.runtime.model_support_handler.from_vllm_lora_tensors(
        tensors, adapter_config=dict(source.adapter_config)
    )


def _native_checkpoint(source: PreparedCheckpoint) -> bool:
    from art.megatron.model_support.lora_disk import (
        ART_LORA_FORMAT_CONFIG_KEY,
        ART_LORA_FORMAT_MEGATRON,
    )

    return (
        source.adapter_config.get(ART_LORA_FORMAT_CONFIG_KEY)
        == ART_LORA_FORMAT_MEGATRON
    )


def _local_tensor_plan(
    trainer: TrainerRank,
    adapter_rank: int,
) -> dict[str, tuple[LoraShardManifest, tuple[int, ...]]]:
    from art.megatron.lora import LoRA

    plan: dict[str, tuple[LoraShardManifest, tuple[int, ...]]] = {}
    for chunk in trainer.runtime.model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            for suffix, parameter in (("lora_A", module.A_T), ("lora_B", module.B_T)):
                manifest = module._manifest_for_param(parameter)
                keys = module._expected_weight_keys(suffix)
                for expert, key in enumerate(keys):
                    local = parameter[expert] if parameter.ndim == 3 else parameter
                    shape = list(reversed(local.shape))
                    shape[0 if suffix == "lora_A" else -1] = adapter_rank
                    plan[key] = (manifest, tuple(shape))
    return plan


def _read_local_slice(
    handle: _SafeTensorFile,
    key: str,
    manifest: LoraShardManifest,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    view = handle.get_slice(key)
    shape = tuple(view.get_shape())
    slices = [slice(None)] * len(shape)
    if not bool(manifest["sharded"]):
        tensor = view[tuple(slices)]
    else:
        axis = int(manifest["export_shard_dim"])
        world_size = int(manifest["shard_world_size"])
        rank = int(manifest["shard_rank"])
        strategy = str(manifest.get("export_shard_strategy", "uniform"))
        if strategy == "uniform":
            if shape[axis] % world_size:
                raise RuntimeError(
                    f"Checkpoint tensor {key!r} cannot be sharded across {world_size} ranks"
                )
            size = shape[axis] // world_size
            slices[axis] = slice(rank * size, (rank + 1) * size)
            tensor = view[tuple(slices)]
        elif strategy == "componentwise":
            components = tuple(int(size) for size in manifest["component_sizes"])
            if sum(components) != shape[axis] or any(
                size % world_size for size in components
            ):
                raise RuntimeError(
                    f"Checkpoint tensor {key!r} has incompatible component shards"
                )
            parts: list[torch.Tensor] = []
            offset = 0
            for component in components:
                size = component // world_size
                slices[axis] = slice(offset + rank * size, offset + (rank + 1) * size)
                parts.append(view[tuple(slices)])
                offset += component
            tensor = torch.cat(parts, dim=axis)
        else:
            raise RuntimeError(
                f"Checkpoint tensor {key!r} has unsupported shard strategy {strategy!r}"
            )
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Checkpoint tensor {key!r} has local shape {tuple(tensor.shape)}; "
            f"expected {expected_shape}"
        )
    return tensor


def _load_native_lora(
    trainer: TrainerRank, source: PreparedCheckpoint, adapter_rank: int
) -> dict[str, torch.Tensor]:
    from art.megatron.model_support.lora_disk import safe_open

    plan = _local_tensor_plan(trainer, adapter_rank)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source.path / "adapter_model.safetensors", framework="pt") as file:
        keys = set(file.keys())
        for key, (manifest, shape) in plan.items():
            if key in keys:
                tensors[key] = _read_local_slice(file, key, manifest, shape)
    return tensors


def load_checkpoint(
    trainer: TrainerRank, source: PreparedCheckpoint, name: str
) -> None:
    config = trainer._validate_checkpoint_adapter_config(
        name, source.adapter_config, alpha=None
    )
    assert config is not None
    native = _native_checkpoint(source)
    if any(digest != source.digest for digest in _gather_objects(source.digest)):
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} content differs across ranks"
        )
    adapter_model: dict[str, torch.Tensor] = {}
    read_error: BaseException | None = None
    try:
        adapter_model = _load_stage_lora(trainer, source, int(config["r"]))
    except BaseException as exc:
        read_error = exc
    _raise_distributed(read_error, "read checkpoint")
    prepared: dict[str, torch.Tensor] = {}
    validation_error: BaseException | None = None
    try:
        prepared = trainer._preflight_adapter(
            name,
            adapter_model,
            config,
            localized=native,
            canonicalized=native,
        )
        _validate_base_model(trainer, source, config)
        trainer._guard_slot_can_load(trainer._slot_ref(name))
    except BaseException as exc:
        validation_error = exc
    _raise_distributed(validation_error, "validate checkpoint")
    expected_keys = {
        key for rank_keys in _gather_objects(tuple(prepared)) for key in rank_keys
    }
    coverage_error: BaseException | None = None
    try:
        artifact_keys = set(source.artifact_keys)
        if native and expected_keys != artifact_keys:
            raise trainer._slot_state_error(
                "Canonical checkpoint coverage differs from the target runtime: "
                f"missing={sorted(artifact_keys - expected_keys)[:8]} "
                f"unexpected={sorted(expected_keys - artifact_keys)[:8]}"
            )
    except BaseException as exc:
        coverage_error = exc
    _raise_distributed(coverage_error, "validate checkpoint coverage")

    target = trainer._slot_ref(name)
    temporary_name = f"__art_loading_{uuid.uuid4().hex}"
    temporary_ref = trainer._slot_ref(temporary_name)
    dynamic: _DynamicOptimizer | None = None
    loaded = 0
    stage_error: BaseException | None = None
    try:
        loaded = trainer._load_checkpoint_slot(
            temporary_name,
            prepared,
            alpha=float(config["lora_alpha"]),
            _prepared=True,
            _localized=native,
        )
        trainer._validate_loaded_checkpoint_config(temporary_name, config)
        staged_params = tuple(trainer._iter_slot_parameters(temporary_ref))
        trainer._checkpoint_slot_params_by_name[temporary_name] = staged_params
        if source.manifest is not None and source.manifest.optimizer is not None:
            local_optimizer = _load_local_optimizer(
                trainer, source, temporary_name, int(config["r"])
            )
            dynamic = trainer._restore_canonical_optimizer(
                temporary_name, local_optimizer
            )
    except BaseException as exc:
        stage_error = exc
    stage_errors = _gather_objects(None if stage_error is None else repr(stage_error))
    if any(stage_errors):
        _discard_staged_checkpoint(trainer, temporary_name)
        if stage_error is not None:
            raise stage_error
        raise RuntimeError(
            "Another rank failed to stage checkpoint: "
            f"{next(item for item in stage_errors if item)}"
        )

    consistency_error: BaseException | None = None
    params: tuple[torch.nn.Parameter, ...] = ()
    try:
        params = trainer._validate_checkpoint_consistency(
            temporary_name, loaded, expected_keys
        )
    except BaseException as exc:
        consistency_error = exc
    consistency_errors = _gather_objects(
        None if consistency_error is None else repr(consistency_error)
    )
    if any(consistency_errors):
        _discard_staged_checkpoint(trainer, temporary_name)
        if consistency_error is not None:
            raise consistency_error
        raise RuntimeError(
            "Another rank rejected staged checkpoint: "
            f"{next(item for item in consistency_errors if item)}"
        )

    from art.megatron.lora import (
        _restore_lora_slots,
        _snapshot_lora_slots,
        replace_lora_slot_in_model,
        validate_lora_slot_replacement,
    )

    replacement_error: BaseException | None = None
    try:
        validate_lora_slot_replacement(trainer.runtime.model, temporary_ref, target)
    except BaseException as exc:
        replacement_error = exc
    replacement_errors = _gather_objects(
        None if replacement_error is None else repr(replacement_error)
    )
    if any(replacement_errors):
        _discard_staged_checkpoint(trainer, temporary_name)
        if replacement_error is not None:
            raise replacement_error
        raise RuntimeError(
            "Another rank rejected checkpoint commit: "
            f"{next(item for item in replacement_errors if item)}"
        )

    slot_snapshot = _snapshot_lora_slots(trainer.runtime.model)
    had_params = name in trainer._checkpoint_slot_params_by_name
    previous_params = trainer._checkpoint_slot_params_by_name.get(name)
    had_dynamic = name in trainer._dynamic_optimizers
    previous_dynamic = trainer._dynamic_optimizers.get(name)
    had_config = name in trainer._checkpoint_slot_adapter_configs
    previous_config = trainer._checkpoint_slot_adapter_configs.get(name)
    had_revision = name in trainer._checkpoint_revisions
    previous_revision = trainer._checkpoint_revisions.get(name)
    commit_error: BaseException | None = None
    try:
        replace_lora_slot_in_model(trainer.runtime.model, temporary_ref, target)
        trainer._checkpoint_slot_params_by_name.pop(temporary_name, None)
        trainer._checkpoint_slot_params_by_name[name] = params
        if dynamic is None:
            trainer._dynamic_optimizers.pop(name, None)
        else:
            trainer._dynamic_optimizers[name] = dynamic
        trainer._checkpoint_slot_adapter_configs[name] = config
        trainer._checkpoint_revisions[name] = (
            trainer._checkpoint_revisions.get(name, -1) + 1
        )
    except BaseException as exc:
        commit_error = exc
    commit_errors = _gather_objects(
        None if commit_error is None else repr(commit_error)
    )
    if not any(commit_errors):
        return

    rollback_error: BaseException | None = None
    try:
        _restore_lora_slots(slot_snapshot)
        _discard_staged_checkpoint(trainer, temporary_name)
        if had_params:
            assert previous_params is not None
            trainer._checkpoint_slot_params_by_name[name] = previous_params
        else:
            trainer._checkpoint_slot_params_by_name.pop(name, None)
        if had_dynamic:
            assert previous_dynamic is not None
            trainer._dynamic_optimizers[name] = previous_dynamic
        else:
            trainer._dynamic_optimizers.pop(name, None)
        if had_config:
            assert previous_config is not None
            trainer._checkpoint_slot_adapter_configs[name] = previous_config
        else:
            trainer._checkpoint_slot_adapter_configs.pop(name, None)
        if had_revision:
            assert previous_revision is not None
            trainer._checkpoint_revisions[name] = previous_revision
        else:
            trainer._checkpoint_revisions.pop(name, None)
    except BaseException as exc:
        rollback_error = exc
    rollback_errors = _gather_objects(
        None if rollback_error is None else repr(rollback_error)
    )
    if any(rollback_errors):
        if rollback_error is not None:
            raise rollback_error
        raise RuntimeError(
            "Another rank failed to roll back checkpoint commit: "
            f"{next(item for item in rollback_errors if item)}"
        )
    if commit_error is not None:
        raise commit_error
    raise RuntimeError(
        "Another rank failed to commit checkpoint: "
        f"{next(item for item in commit_errors if item)}"
    )


def _discard_staged_checkpoint(trainer: TrainerRank, name: str) -> None:
    from art.megatron.lora import delete_lora_slot_from_model

    delete_lora_slot_from_model(trainer.runtime.model, trainer._slot_ref(name))
    trainer._checkpoint_slot_params_by_name.pop(name, None)


def _ensure_checkpoint_save_state(trainer: TrainerRank) -> None:
    if hasattr(trainer, "_prepared_checkpoint_saves"):
        return
    trainer._checkpoint_process_group = None
    trainer._checkpoint_save_condition = threading.Condition()
    trainer._checkpoint_save_sequence = 0
    trainer._checkpoint_finish_sequence = 0
    trainer._prepared_checkpoint_saves = {}
    trainer._checkpoint_finishing_saves = set()
    trainer._completed_checkpoint_saves = set()


def _ensure_checkpoint_group(trainer: TrainerRank) -> dist.ProcessGroup | None:
    if _distributed() and trainer._checkpoint_process_group is None:
        trainer._checkpoint_process_group = dist.new_group(backend="gloo")
    return trainer._checkpoint_process_group


def _snapshot_identity(metadata: LoraShardMeta) -> tuple[str, int]:
    return metadata.key, int(metadata.manifest.get("shard_rank", 0))


def _load_snapshot_file(
    prepared: _PreparedSave, relative: str
) -> dict[str, torch.Tensor]:
    load_file = importlib.import_module("safetensors.torch").load_file
    return load_file(prepared.snapshot / relative)


def _local_snapshot_digests(
    prepared: _PreparedSave,
) -> tuple[
    list[tuple[LoraShardMeta, str]],
    list[tuple[tuple[str, int], float, str]],
]:
    from art.megatron.weights.lora_publish import _tensor_digest

    lora_records: list[tuple[LoraShardMeta, str]] = []
    optimizer_records: list[tuple[tuple[str, int], float, str]] = []
    by_file: dict[str, list[_SnapshotShard]] = {}
    for record in prepared.shards:
        by_file.setdefault(record.lora_file, []).append(record)
    for relative, records in sorted(by_file.items()):
        tensors = _load_snapshot_file(prepared, relative)
        expected = {record.metadata.key for record in records}
        if set(tensors) != expected:
            raise RuntimeError(
                f"Checkpoint snapshot tensor coverage differs: {relative}"
            )
        lora_records.extend(
            (record.metadata, _tensor_digest(tensors[record.metadata.key]))
            for record in records
        )
        if prepared.optimizer is None:
            continue
        hashers = {
            record.metadata.key: hashlib.blake2b(digest_size=16) for record in records
        }
        for component in ("master", "exp_avg", "exp_avg_sq"):
            files = {
                cast(Mapping[str, str], record.optimizer_files)[component]
                for record in records
            }
            if len(files) != 1:
                raise RuntimeError(
                    f"Checkpoint snapshot {component} files differ within a block"
                )
            component_tensors = _load_snapshot_file(prepared, files.pop())
            if set(component_tensors) != expected:
                raise RuntimeError(f"Checkpoint snapshot {component} coverage differs")
            for key, hasher in hashers.items():
                hasher.update(_tensor_digest(component_tensors[key]).encode())
        for record in records:
            if record.step is None:
                raise RuntimeError(
                    f"Checkpoint snapshot optimizer step is missing for "
                    f"{record.metadata.key!r}"
                )
            optimizer_records.append(
                (
                    _snapshot_identity(record.metadata),
                    record.step,
                    hashers[record.metadata.key].hexdigest(),
                )
            )
    return lora_records, optimizer_records


def _snapshot_plan(
    trainer: TrainerRank,
    prepared: _PreparedSave,
    group: dist.ProcessGroup | None,
) -> tuple[list[LoraShardMeta], tuple[str, ...], dict[str, float]]:
    from art.megatron.weights.lora_publish import (
        _elect_lora_contributors,
        _validate_replica_digests,
    )

    local_lora: list[tuple[LoraShardMeta, str]] = []
    local_optimizer: list[tuple[tuple[str, int], float, str]] = []
    error: BaseException | None = None
    try:
        local_lora, local_optimizer = _local_snapshot_digests(prepared)
    except BaseException as exc:
        error = exc
    _raise_distributed(error, "validate local checkpoint snapshot", group=group)

    gathered_lora = [
        item
        for rank_items in _gather_objects(tuple(local_lora), group=group)
        for item in rank_items
    ]
    metadata: list[LoraShardMeta] = []
    with _collective_errors("elect checkpoint contributors", group=group):
        _validate_replica_digests(gathered_lora)
        metadata = _elect_lora_contributors(
            [item for item, _digest_value in gathered_lora]
        )

    optimizers = _gather_objects(prepared.optimizer, group=group)
    if any(value != prepared.optimizer for value in optimizers):
        raise trainer._slot_state_error(
            "Checkpoint snapshot optimizer config differs across ranks"
        )
    steps: dict[str, float] = {}
    if prepared.optimizer is not None:
        gathered_optimizer = [
            item
            for rank_items in _gather_objects(tuple(local_optimizer), group=group)
            for item in rank_items
        ]
        expected = {_snapshot_identity(item) for item in metadata}
        records: dict[tuple[str, int], list[tuple[float, str]]] = {}
        for identity, step, digest in gathered_optimizer:
            records.setdefault(identity, []).append((step, digest))
        if set(records) != expected:
            raise trainer._slot_state_error(
                "Optimizer shard plan differs from snapshotted LoRA: "
                f"missing={sorted(expected - set(records))[:8]} "
                f"unexpected={sorted(set(records) - expected)[:8]}"
            )
        steps_by_key: dict[str, set[float]] = {}
        for (key, _shard_rank), replicas in records.items():
            if len({digest for _step, digest in replicas}) != 1:
                raise RuntimeError(
                    f"Inconsistent replicated tensor contents for {key!r}"
                )
            shard_steps = {step for step, _digest_value in replicas}
            if len(shard_steps) != 1:
                raise trainer._slot_state_error(
                    f"Replicated optimizer step differs for {key!r}"
                )
            steps_by_key.setdefault(key, set()).update(shard_steps)
        if mismatched := {
            key: values for key, values in steps_by_key.items() if len(values) != 1
        }:
            raise trainer._slot_state_error(
                f"Optimizer shard steps differ: {mismatched}"
            )
        steps = {key: values.pop() for key, values in steps_by_key.items()}

    blocks = tuple(sorted({item.block for item in metadata}))
    return metadata, blocks, steps


def _snapshot_block_tensors(
    prepared: _PreparedSave,
    metadata: Sequence[LoraShardMeta],
    component: Literal["lora", "master", "exp_avg", "exp_avg_sq"],
) -> dict[str, torch.Tensor]:
    selected = [item for item in metadata if item.owner_rank == _rank()]
    if not selected:
        return {}
    records = {
        _snapshot_identity(record.metadata): record for record in prepared.shards
    }
    selected_records = [records[_snapshot_identity(item)] for item in selected]
    if component == "lora":
        files = {record.lora_file for record in selected_records}
    else:
        files = {
            cast(Mapping[str, str], record.optimizer_files)[component]
            for record in selected_records
        }
    if len(files) != 1:
        raise RuntimeError(
            f"Checkpoint snapshot {component} tensors span unexpected files"
        )
    tensors = _load_snapshot_file(prepared, files.pop())
    return {item.key: tensors[item.key] for item in selected}


def _serialize_snapshot_lora(
    prepared: _PreparedSave,
    output: Path,
    metadata: list[LoraShardMeta],
    blocks: Sequence[str],
    group: dist.ProcessGroup | None,
) -> None:
    from art.megatron.model_support.lora_disk import (
        ART_LORA_FORMAT_CONFIG_KEY,
        ART_LORA_FORMAT_MEGATRON,
        _consolidate_safetensors,
        save_adapter_config,
    )
    from art.megatron.weights.lora_publish import (
        _gather_merged_adapter_tensors,
    )

    shards: list[Path] = []
    try:
        for index, block in enumerate(blocks):
            block_metadata = [item for item in metadata if item.block == block]
            local_tensors: dict[str, torch.Tensor] = {}
            error: BaseException | None = None
            try:
                local_tensors = _snapshot_block_tensors(
                    prepared, block_metadata, "lora"
                )
            except BaseException as exc:
                error = exc
            _raise_distributed(
                error,
                f"read checkpoint LoRA block {block}",
                group=group,
            )
            canonical = _gather_merged_adapter_tensors(
                block_metadata,
                local_tensors=local_tensors,
                rank=_rank(),
                device=torch.device("cpu"),
                group=group,
            )
            with _collective_errors(
                f"serialize checkpoint LoRA block {block}",
                group=group,
            ):
                if _rank() == 0:
                    shard = output / f".adapter_model-{index:06d}.safetensors"
                    _save_file(canonical, shard)
                    shards.append(shard)
        with _collective_errors("finalize checkpoint LoRA", group=group):
            if _rank() == 0:
                if not shards:
                    raise RuntimeError("No LoRA tensors were available to checkpoint")
                _consolidate_safetensors(shards, output / "adapter_model.safetensors")
                save_adapter_config(
                    output,
                    {
                        **prepared.adapter_config,
                        ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_MEGATRON,
                    },
                )
    finally:
        if _rank() == 0:
            for shard in shards:
                shard.unlink(missing_ok=True)


def _serialize_snapshot_optimizer(
    trainer: TrainerRank,
    prepared: _PreparedSave,
    output: Path,
    metadata: list[LoraShardMeta],
    blocks: Sequence[str],
    steps: dict[str, float],
    group: dist.ProcessGroup | None,
) -> CheckpointManifest:
    base_model = str(prepared.adapter_config["base_model_name_or_path"])
    if prepared.optimizer is None:
        return CheckpointManifest(
            base_model_name_or_path=base_model,
            optimizer=None,
            parameters={},
            steps={},
            digest="",
        )

    from art.megatron.weights.lora_publish import (
        _gather_merged_adapter_tensors,
    )

    records_by_component: dict[str, dict[str, TensorRecord]] = {}
    for component in ("master", "exp_avg", "exp_avg_sq"):
        component_records: dict[str, TensorRecord] = {}
        for index, block in enumerate(blocks):
            block_metadata = [item for item in metadata if item.block == block]
            local_tensors: dict[str, torch.Tensor] = {}
            error: BaseException | None = None
            try:
                local_tensors = _snapshot_block_tensors(
                    prepared,
                    block_metadata,
                    component,
                )
            except BaseException as exc:
                error = exc
            _raise_distributed(
                error,
                f"read checkpoint optimizer {component} block {block}",
                group=group,
            )
            canonical = _gather_merged_adapter_tensors(
                block_metadata,
                local_tensors=local_tensors,
                rank=_rank(),
                device=torch.device("cpu"),
                group=group,
            )
            with _collective_errors(
                f"serialize checkpoint optimizer {component} block {block}",
                group=group,
            ):
                if _rank() == 0:
                    relative = f"optimizer/{component}-{index:06d}.safetensors"
                    (output / "optimizer").mkdir(parents=True, exist_ok=True)
                    _save_file(canonical, output / relative)
                    component_records.update(
                        (
                            key,
                            TensorRecord(
                                file=relative,
                                tensor=key,
                                shape=tuple(int(dim) for dim in tensor.shape),
                                dtype=_dtype_name(tensor.dtype),
                            ),
                        )
                        for key, tensor in canonical.items()
                    )
        records_by_component[component] = component_records

    parameters: dict[str, ParameterRecord] = {}
    with _collective_errors("validate checkpoint optimizer coverage", group=group):
        if _rank() == 0:
            coverage = {
                component: set(records)
                for component, records in records_by_component.items()
            }
            expected = next(iter(coverage.values()), set())
            if any(keys != expected for keys in coverage.values()):
                raise RuntimeError(
                    f"Canonical optimizer component coverage differs: {coverage}"
                )
            from art.megatron.model_support.lora_disk import safe_open

            with safe_open(
                output / "adapter_model.safetensors", framework="pt"
            ) as handle:
                artifact_keys = set(handle.keys())
            if expected != artifact_keys:
                raise RuntimeError(
                    "Canonical optimizer coverage differs from exported LoRA: "
                    f"optimizer={sorted(expected)} lora={sorted(artifact_keys)}"
                )
            parameters = {
                key: ParameterRecord(
                    master=records_by_component["master"][key],
                    exp_avg=records_by_component["exp_avg"][key],
                    exp_avg_sq=records_by_component["exp_avg_sq"][key],
                )
                for key in sorted(expected)
            }
    return CheckpointManifest(
        base_model_name_or_path=base_model,
        optimizer=prepared.optimizer,
        parameters=parameters,
        steps=steps,
        digest="",
    )


def _finish_prepared_save(
    trainer: TrainerRank,
    prepared: _PreparedSave,
) -> None:
    group = trainer._checkpoint_process_group
    temporary = _temporary_output(str(prepared.destination), group=group)
    error: BaseException | None = None
    try:
        metadata, blocks, steps = _snapshot_plan(trainer, prepared, group)
        _serialize_snapshot_lora(
            prepared,
            temporary,
            metadata,
            blocks,
            group,
        )
        manifest = _serialize_snapshot_optimizer(
            trainer,
            prepared,
            temporary,
            metadata,
            blocks,
            steps,
            group,
        )
        if _rank() == 0:
            digest = _digest(temporary, manifest)
            _write_manifest(
                temporary,
                manifest.model_copy(update={"digest": digest}),
            )
            _commit_output(temporary, prepared.destination, digest)
    except BaseException as exc:
        error = exc
    finally:
        if _rank() == 0 and temporary.exists():
            try:
                shutil.rmtree(temporary)
            except BaseException as exc:
                if error is None:
                    error = exc
    _raise_distributed(error, "finish checkpoint save", group=group)


def _optimizer_component(local: _LocalShard, component: str) -> torch.Tensor:
    value = getattr(local, component)
    if value is None:
        value = torch.zeros_like(local.master)
    if local.expert is not None:
        value = value[local.expert]
    return value.T.contiguous()


def _local_optimizer_shards(
    trainer: TrainerRank,
    name: str,
    dynamic: _DynamicOptimizer,
) -> tuple[_LocalShard, ...]:
    from art.megatron.lora import LoRA, LoraShardMeta, _block_for_key

    params = trainer._checkpoint_slot_params_by_name[name]
    masters = {
        id(param): master
        for param, master in zip(params, dynamic.master_params, strict=True)
    }
    ref = trainer._slot_ref(name)
    items: list[_LocalShard] = []
    for chunk in trainer.runtime.model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            for key, param, expert in module._export_items(ref):
                master = masters.get(id(param))
                if master is None:
                    raise trainer._slot_state_error(
                        f"Cannot map optimizer parameter for {key!r}"
                    )
                state = dynamic.optimizer.state.get(master, {})
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step = state.get("step")
                if exp_avg is exp_avg_sq is step is None:
                    step = 0.0
                elif not all(
                    isinstance(value, torch.Tensor) for value in (exp_avg, exp_avg_sq)
                ):
                    raise trainer._slot_state_error(
                        f"AdamW state for {key!r} is incomplete"
                    )
                local_master = master if expert is None else master[expert]
                exported_shape = tuple(reversed(local_master.shape))
                items.append(
                    _LocalShard(
                        LoraShardMeta(
                            key=key,
                            owner_rank=_rank(),
                            shape=exported_shape,
                            dtype_name=_dtype_name(local_master.dtype),
                            manifest=module._manifest_for_param(param),
                            block=_block_for_key(key),
                        ),
                        master,
                        cast(torch.Tensor | None, exp_avg),
                        cast(torch.Tensor | None, exp_avg_sq),
                        expert,
                        _scalar(step),
                    )
                )
    return tuple(items)


def _load_local_optimizer(
    trainer: TrainerRank,
    source: PreparedCheckpoint,
    name: str,
    adapter_rank: int,
) -> LocalOptimizerState:
    manifest = source.manifest
    assert manifest is not None and manifest.optimizer is not None
    plan = _local_tensor_plan(trainer, adapter_rank)
    localized: dict[str, tuple[torch.Tensor, ...]] = {}
    for component in ("master", "exp_avg", "exp_avg_sq"):
        records = {
            key: cast(TensorRecord, getattr(record, component))
            for key, record in manifest.parameters.items()
            if key in plan
        }
        artifact_tensors = _load_tensors(
            source.path,
            records,
            plans={key: plan[key] for key in records},
        )
        localized[component] = trainer._localize_adapter_tensors(
            artifact_tensors, name, localized=True
        )

    lengths = {len(values) for values in localized.values()}
    if len(lengths) != 1:
        raise trainer._slot_state_error(
            f"Canonical optimizer component lengths differ: {lengths}"
        )
    steps = tuple(
        _parameter_group_step(group, manifest.steps)
        for group in trainer._local_parameter_key_groups(name)
    )
    return LocalOptimizerState(
        masters=localized["master"],
        exp_avgs=localized["exp_avg"],
        exp_avg_sqs=localized["exp_avg_sq"],
        steps=steps,
        config=manifest.optimizer,
    )


def _parameter_group_step(keys: Sequence[str], steps: Mapping[str, float]) -> float:
    missing = [key for key in keys if key not in steps]
    if missing:
        raise RuntimeError(f"Canonical optimizer is missing steps: {missing}")
    values = {steps[key] for key in keys}
    if len(values) != 1:
        raise RuntimeError(
            f"Target optimizer parameter combines different steps for {tuple(keys)!r}"
        )
    return values.pop()


def _checkpoint_config(trainer: TrainerRank, name: str) -> Mapping[str, object]:
    loaded = name in trainer._checkpoint_slot_params_by_name
    if any(value != loaded for value in _gather_objects(loaded)):
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} is not loaded consistently across ranks"
        )
    if not loaded:
        raise ValueError(f"Unknown checkpoint: {name!r}")
    config = trainer._checkpoint_slot_adapter_configs.get(name)
    configs = _gather_objects(config)
    if any(value != config for value in configs):
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} adapter config differs across ranks"
        )
    if config is None:
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} was loaded without adapter_config"
        )
    return config


def _validate_save_state(trainer: TrainerRank, name: str) -> None:
    _checkpoint_config(trainer, name)
    if any(trainer._checkpoint_grad_flags((name,))):
        raise trainer._slot_state_error(
            f"Cannot save checkpoint {name!r} with accumulated gradients"
        )
    local_graph = trainer._has_live_slot_graph(trainer._slot_ref(name))
    if any(bool(value) for value in _gather_objects(local_graph)):
        raise trainer._slot_state_error(
            f"Cannot save checkpoint {name!r} with a live backward graph"
        )


def _validate_base_model(
    trainer: TrainerRank,
    source: PreparedCheckpoint,
    config: Mapping[str, object],
) -> None:
    configured = str(config["base_model_name_or_path"])
    if (
        source.manifest is not None
        and source.manifest.base_model_name_or_path != configured
    ):
        raise trainer._slot_state_error(
            "Checkpoint manifest and adapter config name different base models"
        )
    runtime_model = getattr(trainer.runtime, "model_identifier", None)
    if runtime_model is not None and runtime_model != configured:
        kind = "Exact checkpoint" if source.manifest is not None else "Checkpoint"
        raise trainer._slot_state_error(
            f"{kind} base model {configured!r} differs from runtime model "
            f"{runtime_model!r}"
        )
    supported = tuple(
        getattr(getattr(trainer.runtime, "model_support_spec", None), "model_names", ())
    )
    if supported and configured not in supported:
        raise trainer._slot_state_error(
            f"Checkpoint base model {configured!r} is incompatible with this runtime"
        )


def _optimizer_config(dynamic: _DynamicOptimizer) -> AdamWRecord:
    group = dynamic.optimizer.param_groups[0]
    if bool(group["amsgrad"]):
        raise RuntimeError("Canonical checkpoints do not support AdamW amsgrad state")
    beta1, beta2 = cast(tuple[float, float], group["betas"])
    return AdamWRecord(
        learning_rate=float(group["lr"]),
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
        amsgrad=False,
    )


def _temporary_output(
    output_dir: str, *, group: dist.ProcessGroup | None = None
) -> Path:
    destination = Path(output_dir)
    value = str(
        destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        if _rank() == 0
        else ""
    )
    values = [value]
    if _distributed():
        torch.distributed.broadcast_object_list(values, src=0, group=group)
    temporary = Path(values[0])
    error: BaseException | None = None
    try:
        if _rank() == 0:
            temporary.mkdir(parents=True)
    except BaseException as exc:
        error = exc
    _raise_distributed(error, "create checkpoint staging directory", group=group)
    return temporary


def _commit_output(temporary: Path, destination: Path, digest: str) -> None:
    if destination.exists():
        existing_manifest = destination / MANIFEST_FILE
        if existing_manifest.is_file():
            existing = CheckpointManifest.model_validate_json(
                existing_manifest.read_text()
            )
            if existing.digest == digest:
                _checkpoint_metadata(destination)
                return
        if any(destination.iterdir()):
            raise FileExistsError(f"Checkpoint output is not empty: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)


def _write_manifest(path: Path, manifest: CheckpointManifest) -> None:
    target = path / MANIFEST_FILE
    temporary = target.with_suffix(".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    os.replace(temporary, target)


def _digest(root: Path, manifest: CheckpointManifest) -> str:
    seed = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        *_manifest_files(manifest),
    }
    return _hash_files(root, files, seed=seed)


def _hash_files(root: Path, files: Iterable[str], *, seed: bytes = b"") -> str:
    digest = hashlib.sha256(seed)
    for relative in sorted(set(files)):
        digest.update(relative.encode())
        with (root / _safe_relative_path(relative)).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(manifest: CheckpointManifest) -> set[str]:
    return {
        record.file
        for value in manifest.parameters.values()
        for record in (value.master, value.exp_avg, value.exp_avg_sq)
    }


def _safe_relative_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    windows_path = PureWindowsPath(relative)
    if (
        not relative
        or chr(0) in relative
        or chr(92) in relative
        or ":" in relative
        or path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in relative.split("/"))
        or path.as_posix() != relative
    ):
        raise RuntimeError(f"Unsafe checkpoint tensor path: {relative!r}")
    return path


def _validate_artifact_view(
    manifest: CheckpointManifest,
    artifact_entries: Iterable[str],
    expected_digest: str,
) -> None:
    if not expected_digest or manifest.digest != expected_digest:
        raise RuntimeError(
            "Checkpoint manifest digest differs from the durable artifact digest"
        )
    required = {
        MANIFEST_FILE,
        "adapter_config.json",
        "adapter_model.safetensors",
        *_manifest_files(manifest),
    }
    missing = sorted(required - set(artifact_entries))
    if missing:
        raise RuntimeError(
            f"Checkpoint artifact is missing referenced files: {missing[:8]}"
        )


def _validate_manifest(
    manifest: CheckpointManifest, artifact_keys: Sequence[str]
) -> None:
    for relative in _manifest_files(manifest):
        _safe_relative_path(relative)
    parameter_keys = set(manifest.parameters)
    step_keys = set(manifest.steps)
    if manifest.optimizer is None:
        if parameter_keys or step_keys:
            raise RuntimeError(
                "LoRA-only checkpoint unexpectedly contains optimizer parameters"
            )
        return
    for key, parameter in manifest.parameters.items():
        for component, record in (
            ("master", parameter.master),
            ("exp_avg", parameter.exp_avg),
            ("exp_avg_sq", parameter.exp_avg_sq),
        ):
            if record.dtype != "float32":
                raise RuntimeError(
                    f"Checkpoint optimizer tensor {key!r} {component} must use "
                    f"float32, got {record.dtype!r}"
                )
    if parameter_keys != set(artifact_keys):
        missing = sorted(set(artifact_keys) - parameter_keys)
        extra = sorted(parameter_keys - set(artifact_keys))
        raise RuntimeError(
            "Checkpoint optimizer coverage differs from its LoRA tensors: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    if step_keys != parameter_keys:
        raise RuntimeError(
            "Checkpoint optimizer step coverage differs from its tensors: "
            f"missing={sorted(parameter_keys - step_keys)[:8]} "
            f"extra={sorted(step_keys - parameter_keys)[:8]}"
        )


def _validate_files(root: Path, manifest: CheckpointManifest) -> None:
    safe_open = importlib.import_module("safetensors").safe_open
    files: dict[str, dict[str, TensorRecord]] = {}
    for key, value in manifest.parameters.items():
        for record in (value.master, value.exp_avg, value.exp_avg_sq):
            if record.tensor != key or record.tensor in files.setdefault(
                record.file, {}
            ):
                raise RuntimeError(f"Invalid checkpoint tensor index: {record.file}")
            files[record.file][record.tensor] = record
    for relative, records in files.items():
        candidate = (root / _safe_relative_path(relative)).resolve()
        if root.resolve() not in candidate.parents or not candidate.is_file():
            raise RuntimeError(f"Invalid checkpoint tensor path: {relative}")
        with safe_open(candidate, framework="pt") as handle:
            if set(handle.keys()) != set(records):
                raise RuntimeError(f"Checkpoint tensor index mismatch: {relative}")
            for key, record in records.items():
                view = handle.get_slice(key)
                shape = tuple(view.get_shape())
                empty = view[tuple(slice(0, 0) for _ in shape)]
                if shape != record.shape or _dtype_name(empty.dtype) != record.dtype:
                    raise RuntimeError(
                        f"Checkpoint tensor metadata mismatch: {record.file}"
                    )


def _load_tensors(
    root: Path,
    records: Mapping[str, TensorRecord],
    *,
    plans: Mapping[str, tuple[LoraShardManifest, tuple[int, ...]]] | None = None,
) -> dict[str, torch.Tensor]:
    safe_open = importlib.import_module("safetensors").safe_open
    files: dict[str, list[tuple[str, TensorRecord]]] = {}
    for key, record in records.items():
        files.setdefault(record.file, []).append((key, record))
    tensors: dict[str, torch.Tensor] = {}
    for relative, file_records in files.items():
        with safe_open(root / relative, framework="pt") as handle:
            for key, record in file_records:
                if plans is None:
                    tensor = handle.get_tensor(record.tensor)
                    shape_matches = tuple(tensor.shape) == record.shape
                else:
                    manifest, expected_shape = plans[key]
                    tensor = _read_local_slice(
                        handle, record.tensor, manifest, expected_shape
                    )
                    shape_matches = True
                if not shape_matches or _dtype_name(tensor.dtype) != record.dtype:
                    raise RuntimeError(
                        f"Checkpoint tensor metadata mismatch: {record.file}"
                    )
                tensors[record.tensor] = tensor
    return tensors


def _save_file(tensors: dict[str, torch.Tensor], path: Path) -> None:
    importlib.import_module("safetensors.torch").save_file(tensors, path)


def _is_node_validator() -> bool:
    if not _distributed():
        return True
    local_rank = os.environ.get("LOCAL_RANK")
    return True if local_rank is None else int(local_rank) == 0


def _scalar(value: object) -> float:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeError("AdamW optimizer step is not scalar")
