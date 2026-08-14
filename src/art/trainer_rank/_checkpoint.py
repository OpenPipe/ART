"""Topology-portable persistence for TrainerRank checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
import threading
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast
import uuid

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from art.megatron.lora import LoRA, LoraShardMeta, LoRASlotRef
    from art.trainer_rank._impl import (
        TrainerRank,
        _AdapterConfig,
        _DynamicOptimizer,
    )

FORMAT = 3
DRAFT_FORMAT = 2
LEGACY_FORMAT = 1
MANIFEST_FILE = "checkpoint.json"
_ART_FORMAT_KEY = "art_lora_format"
_ART_FORMAT = "art-trainer-rank-v1"


class OptimizerConfig(TypedDict):
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


class CustomTensorRecord(TypedDict):
    kind: Literal["module", "parameter", "buffer"]
    tensor_keys: list[str]
    trainable_keys: list[str]
    parameter_aliases: list[list[str]]
    buffer_aliases: list[list[str]]
    persistent_buffer_keys: list[str]


class CheckpointManifest(TypedDict):
    format_version: Literal[1, 3]
    base_model_name_or_path: str
    optimizer: OptimizerConfig | None
    parameters: dict[str, list[str]]
    steps: dict[str, float]
    files: dict[str, str]
    digest: str
    custom_tensors: NotRequired[dict[str, CustomTensorRecord]]


@dataclass(frozen=True)
class PreparedCheckpoint:
    path: Path
    config: dict[str, object]
    keys: tuple[str, ...]
    manifest: CheckpointManifest | None
    digest: str
    custom: PreparedCustomPayload | None = None


@dataclass(frozen=True)
class PreparedCustomPayload:
    records: dict[str, CustomTensorRecord]
    tensors: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]


@dataclass(frozen=True)
class LocalOptimizerState:
    masters: tuple[torch.Tensor, ...]
    exp_avgs: tuple[torch.Tensor, ...]
    exp_avg_sqs: tuple[torch.Tensor, ...]
    steps: tuple[float, ...]
    config: OptimizerConfig


@dataclass(frozen=True)
class CustomOptimizerState:
    master: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: float


@dataclass(frozen=True)
class _LocalShard:
    metadata: LoraShardMeta
    file: str


@dataclass(frozen=True)
class _PreparedSave:
    sequence: int
    snapshot: Path
    reservation: Path
    destination: Path
    config: dict[str, object]
    shards: tuple[_LocalShard, ...]
    optimizer: OptimizerConfig | None
    custom_tensors: dict[str, CustomTensorRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class _FinalizedSave:
    sequence: int
    outcome: Literal["finish", "abort"]


type _SlotSnapshot = tuple[
    tuple[
        "LoRA",
        dict["LoRASlotRef", str],
        dict[str, torch.nn.Module],
        dict[str, "LoRASlotRef"],
    ],
    ...,
]


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _gather[T](value: T, group: dist.ProcessGroup | None = None) -> tuple[T, ...]:
    if not _distributed():
        return (value,)
    values: list[T | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return tuple(cast(T, item) for item in values)


def raise_distributed(
    error: BaseException | None,
    phase: str,
    group: dist.ProcessGroup | None = None,
) -> None:
    errors = _gather(None if error is None else repr(error), group)
    if not any(errors):
        return
    if error is not None:
        raise error
    raise RuntimeError(
        f"Another rank failed to {phase}: {next(item for item in errors if item)}"
    )


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or PureWindowsPath(value).drive
        or "\\" in value
    ):
        raise RuntimeError(f"Unsafe checkpoint path: {value!r}")
    return path


def _hash_files(root: Path, files: Iterable[str], *, seed: bytes = b"") -> str:
    digest = hashlib.blake2b(digest_size=32)
    digest.update(seed)
    for relative in sorted(files):
        digest.update(relative.encode())
        with (root / _safe_relative(relative)).open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _manifest_seed(manifest: Mapping[str, object]) -> bytes:
    value = {**manifest, "digest": ""}
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    return hashlib.blake2b(_manifest_seed(manifest), digest_size=32).hexdigest()


def _alias_keys(value: object) -> set[str] | None:
    if not isinstance(value, list):
        return None
    groups = cast(list[object], value)
    if any(
        not isinstance(group, list)
        or not group
        or not all(isinstance(key, str) and key for key in group)
        or len(set(group)) != len(group)
        for group in groups
    ):
        return None
    typed = cast(list[list[str]], groups)
    keys = {key for group in typed for key in group}
    return keys if len(keys) == sum(map(len, typed)) else None


def _validate_manifest(
    manifest: CheckpointManifest,
    *,
    adapter_keys: set[str],
    config: Mapping[str, object],
) -> set[str]:
    format_version = manifest.get("format_version")
    if format_version not in (LEGACY_FORMAT, FORMAT):
        raise RuntimeError("Unsupported ART checkpoint format")
    digest = manifest.get("digest")
    file_digests = manifest.get("files")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(file_digests, dict)
        or any(
            not isinstance(path, str)
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for path, value in file_digests.items()
        )
    ):
        raise RuntimeError("Checkpoint digest is invalid")
    if manifest.get("base_model_name_or_path") != config.get("base_model_name_or_path"):
        raise RuntimeError(
            "Checkpoint manifest and adapter config name different models"
        )
    optimizer = manifest.get("optimizer")
    parameters = manifest.get("parameters")
    steps = manifest.get("steps")
    custom = manifest.get("custom_tensors", {})
    if format_version == LEGACY_FORMAT and custom:
        raise RuntimeError("Legacy checkpoint contains custom tensors")
    if not isinstance(custom, dict):
        raise RuntimeError("Checkpoint custom tensor mapping is invalid")
    if format_version == FORMAT and not custom:
        raise RuntimeError("Custom checkpoint format contains no custom tensors")
    custom_keys: set[str] = set()
    trainable_custom_keys: set[str] = set()
    for name, record in custom.items():
        tensor_keys = record.get("tensor_keys", []) if isinstance(record, dict) else []
        trainable_keys = (
            record.get("trainable_keys", []) if isinstance(record, dict) else []
        )
        parameter_aliases = (
            record.get("parameter_aliases") if isinstance(record, dict) else None
        )
        buffer_aliases = (
            record.get("buffer_aliases") if isinstance(record, dict) else None
        )
        persistent_buffer_keys = (
            record.get("persistent_buffer_keys") if isinstance(record, dict) else None
        )
        kind = record.get("kind") if isinstance(record, dict) else None
        parameter_keys = _alias_keys(parameter_aliases)
        buffer_keys = _alias_keys(buffer_aliases)
        if (
            not isinstance(name, str)
            or not name
            or "." in name
            or "/" in name
            or not isinstance(record, dict)
            or kind not in ("module", "parameter", "buffer")
            or not isinstance(tensor_keys, list)
            or not isinstance(trainable_keys, list)
            or not all(isinstance(key, str) for key in tensor_keys)
            or not all(isinstance(key, str) for key in trainable_keys)
            or len(set(tensor_keys)) != len(tensor_keys)
            or len(set(trainable_keys)) != len(trainable_keys)
            or parameter_keys is None
            or buffer_keys is None
            or parameter_keys & buffer_keys
            or not isinstance(persistent_buffer_keys, list)
            or not all(isinstance(key, str) for key in persistent_buffer_keys)
            or len(set(persistent_buffer_keys)) != len(persistent_buffer_keys)
            or not set(trainable_keys).issubset(parameter_keys)
            or not set(persistent_buffer_keys).issubset(buffer_keys)
            or not parameter_keys.issubset(tensor_keys)
            or not set(persistent_buffer_keys).issubset(tensor_keys)
            or (kind == "buffer" and bool(trainable_keys))
        ):
            raise RuntimeError(
                f"Checkpoint custom tensor mapping is invalid for {name!r}"
            )
        keys = set(tensor_keys)
        if record["kind"] == "module":
            valid_keys = all(
                key.startswith(f"{name}.")
                for key in keys | parameter_keys | buffer_keys
            )
        elif record["kind"] == "parameter":
            valid_keys = (
                keys == {name}
                and parameter_keys == {name}
                and not buffer_keys
                and not persistent_buffer_keys
            )
        else:
            valid_keys = (
                keys == {name}
                and buffer_keys == {name}
                and persistent_buffer_keys == [name]
                and not parameter_keys
            )
        if not valid_keys or custom_keys & keys:
            raise RuntimeError("Checkpoint custom tensor keys overlap")
        custom_keys.update(keys)
        trainable_custom_keys.update(trainable_keys)
    if not isinstance(parameters, dict) or not isinstance(steps, dict):
        raise RuntimeError("Checkpoint optimizer mapping is invalid")
    files: set[str] = set()
    if optimizer is None:
        if parameters or steps:
            raise RuntimeError("LoRA-only checkpoint contains optimizer metadata")
    else:
        required = {"learning_rate", "beta1", "beta2", "eps", "weight_decay"}
        optimizer_values = cast(dict[str, object], optimizer)
        if (
            not isinstance(optimizer, dict)
            or set(optimizer_values) != required
            or any(
                not isinstance(optimizer_values[key], int | float)
                or isinstance(optimizer_values[key], bool)
                for key in required
            )
        ):
            raise RuntimeError("Checkpoint optimizer config is invalid")
        optimizer_keys = adapter_keys
        if set(parameters) != optimizer_keys or set(steps) != optimizer_keys:
            raise RuntimeError(
                "Checkpoint optimizer mapping differs from trainable tensors: "
                f"parameters={sorted(set(parameters) ^ optimizer_keys)[:8]} "
                f"steps={sorted(set(steps) ^ optimizer_keys)[:8]}"
            )
        for key, record in parameters.items():
            if (
                not isinstance(key, str)
                or not isinstance(record, list | tuple)
                or len(record) != 3
                or not all(isinstance(item, str) for item in record)
            ):
                raise RuntimeError(
                    f"Checkpoint optimizer mapping is invalid for {key!r}"
                )
            normalized = [_safe_relative(item).as_posix() for item in record]
            parameters[key] = normalized
            files.update(normalized)
        if any(
            not isinstance(value, int | float) or isinstance(value, bool)
            for value in steps.values()
        ):
            raise RuntimeError("Checkpoint optimizer steps are invalid")
    expected_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        *files,
        *({"custom_tensors.safetensors"} if custom else set()),
        *(
            {"optimizer/custom.safetensors"}
            if optimizer is not None and trainable_custom_keys
            else set()
        ),
    }
    if set(file_digests) != expected_files:
        raise RuntimeError("Checkpoint file digest mapping is invalid")
    if custom:
        files.add("custom_tensors.safetensors")
    if optimizer is not None and trainable_custom_keys:
        files.add("optimizer/custom.safetensors")
    return files


def _load_custom_payload(
    root: Path, manifest: CheckpointManifest
) -> PreparedCustomPayload | None:
    custom = manifest.get("custom_tensors", {})
    if not custom:
        return None
    load = importlib.import_module("safetensors.torch").load_file
    tensors = {
        key: value.detach().cpu().clone()
        for key, value in load(
            root / "custom_tensors.safetensors", device="cpu"
        ).items()
    }
    actual = set(tensors)
    expected = {key for record in custom.values() for key in record["tensor_keys"]}
    if actual != expected:
        raise RuntimeError(
            "Checkpoint custom tensor payload differs from its manifest: "
            f"missing={sorted(expected - actual)[:8]} "
            f"unexpected={sorted(actual - expected)[:8]}"
        )
    trainable = {key for record in custom.values() for key in record["trainable_keys"]}
    optimizer_file = root / "optimizer/custom.safetensors"
    if not trainable:
        if optimizer_file.is_file():
            raise RuntimeError(
                "Checkpoint has custom optimizer state without parameters"
            )
        return PreparedCustomPayload(deepcopy(custom), tensors, {})
    if manifest["optimizer"] is None:
        if optimizer_file.is_file():
            raise RuntimeError(
                "Checkpoint has custom optimizer state without an optimizer"
            )
        return PreparedCustomPayload(deepcopy(custom), tensors, {})
    optimizer = {
        key: value.detach().cpu().clone()
        for key, value in load(optimizer_file, device="cpu").items()
    }
    actual = set(optimizer)
    expected = {
        f"{component}/{key}"
        for key in trainable
        for component in ("master", "exp_avg", "exp_avg_sq", "step")
    }
    if actual != expected:
        raise RuntimeError(
            "Checkpoint custom optimizer payload differs from its manifest: "
            f"missing={sorted(expected - actual)[:8]} "
            f"unexpected={sorted(actual - expected)[:8]}"
        )
    return PreparedCustomPayload(deepcopy(custom), tensors, optimizer)


def prepare_checkpoint(
    path: str, *, artifact_entries: Iterable[str] | None = None
) -> PreparedCheckpoint:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint is not a directory: {path}")
    from art.megatron.model_support.lora_disk import load_adapter_config, safe_open

    config = cast(dict[str, object], load_adapter_config(root))
    adapter = root / "adapter_model.safetensors"
    with safe_open(adapter, framework="pt") as handle:
        keys = tuple(sorted(handle.keys()))
    manifest_path = root / MANIFEST_FILE
    manifest: CheckpointManifest | None = None
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text())
        if isinstance(value, dict) and value.get("format_version") == DRAFT_FORMAT:
            raise RuntimeError(
                "Unsupported draft custom checkpoint format 2; regenerate the "
                "checkpoint with the current TrainerRank implementation"
            )
        if not isinstance(value, dict) or value.get("format_version") not in (
            LEGACY_FORMAT,
            FORMAT,
        ):
            raise RuntimeError("Unsupported ART checkpoint format")
        manifest = cast(CheckpointManifest, value)
        if config.get(_ART_FORMAT_KEY) != _ART_FORMAT:
            raise RuntimeError("Canonical checkpoint adapter format is invalid")
        optimizer_files = _validate_manifest(
            manifest, adapter_keys=set(keys), config=config
        )
        files = {
            "adapter_config.json",
            "adapter_model.safetensors",
            MANIFEST_FILE,
            *optimizer_files,
        }
        expected = manifest["digest"]
        actual = _manifest_digest(manifest)
        if actual != expected:
            raise RuntimeError(f"Checkpoint digest mismatch: {actual} != {expected}")
        if artifact_entries is None:
            downloaded = files - {MANIFEST_FILE}
        else:
            available = {_safe_relative(entry).as_posix() for entry in artifact_entries}
            if missing := sorted(files - available):
                raise RuntimeError(
                    f"Checkpoint artifact is missing entries: {missing[:8]}"
                )
            downloaded = {"adapter_config.json", "adapter_model.safetensors"}
        for relative in downloaded:
            file_actual = _file_digest(root / relative)
            if file_actual != manifest["files"][relative]:
                raise RuntimeError(
                    f"Checkpoint file digest mismatch for {relative}: "
                    f"{file_actual} != {manifest['files'][relative]}"
                )
        custom = (
            _load_custom_payload(root, manifest) if artifact_entries is None else None
        )
    else:
        if artifact_entries is not None:
            raise RuntimeError("Checkpoint artifact lacks a canonical manifest")
        actual = _hash_files(root, ("adapter_config.json", "adapter_model.safetensors"))
        custom = None
    return PreparedCheckpoint(root, config, keys, manifest, actual, custom)


def validate_checkpoint(
    path: str | Path, *, require_optimizer: bool = False
) -> CheckpointManifest | None:
    prepared = prepare_checkpoint(str(path))
    if require_optimizer and (
        prepared.manifest is None or prepared.manifest["optimizer"] is None
    ):
        raise RuntimeError("Checkpoint does not contain optimizer state")
    return prepared.manifest


def materialize_lora(
    path: str | Path,
    output_dir: str | Path,
    *,
    require_optimizer: bool = False,
    artifact_entries: Iterable[str] | None = None,
    expected_digest: str | None = None,
) -> None:
    source = prepare_checkpoint(str(path), artifact_entries=artifact_entries)
    if expected_digest is not None and source.digest != expected_digest:
        raise RuntimeError(
            f"Checkpoint digest mismatch: {source.digest} != {expected_digest}"
        )
    if require_optimizer and (
        source.manifest is None or source.manifest["optimizer"] is None
    ):
        raise RuntimeError("Checkpoint does not contain optimizer state")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"LoRA output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(source.path / name, destination / name)
    from art.megatron.model_support.lora_disk import normalize_lora_checkpoint_to_vllm

    normalize_lora_checkpoint_to_vllm(destination)


def load_custom_tensors(
    source: PreparedCustomPayload | None,
    name: str,
    kind: Literal["module", "parameter", "buffer"],
) -> tuple[CustomTensorRecord, dict[str, torch.Tensor]] | None:
    if source is None:
        return None
    record = source.records.get(name)
    if record is None:
        return None
    if record["kind"] != kind:
        raise RuntimeError(
            f"Checkpoint registers {name!r} as {record['kind']}, not {kind}"
        )
    prefix = f"{name}."
    return record, {
        key.removeprefix(prefix) if kind == "module" else "": source.tensors[key]
        for key in record["tensor_keys"]
    }


def load_custom_optimizer(
    source: PreparedCustomPayload | None,
    keys: Sequence[str],
) -> dict[str, CustomOptimizerState]:
    if source is None or not source.optimizer or not keys:
        return {}
    saved = {
        key for record in source.records.values() for key in record["trainable_keys"]
    }
    selected = saved.intersection(keys)
    if not selected:
        return {}
    payload = source.optimizer
    return {
        key: CustomOptimizerState(
            payload[f"master/{key}"],
            payload[f"exp_avg/{key}"],
            payload[f"exp_avg_sq/{key}"],
            float(payload[f"step/{key}"].item()),
        )
        for key in selected
    }


def _optimizer_config(dynamic: _DynamicOptimizer) -> OptimizerConfig:
    group = dynamic.optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    return {
        "learning_rate": float(group["lr"]),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
    }


def _validate_save_state(trainer: TrainerRank, name: str) -> _AdapterConfig:
    slot = trainer._checkpoint_slots.get(name)
    if slot is None or slot.config is None:
        raise trainer._slot_state_error(f"Unknown checkpoint: {name!r}")
    if trainer._checkpoint_grad_flags((name,))[0]:
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} has accumulated gradients"
        )
    return slot.config


def _custom_snapshot(
    trainer: TrainerRank,
    name: str,
    snapshot: Path,
) -> dict[str, CustomTensorRecord]:
    from art.trainer_rank._impl import (
        _custom_layout,
        _custom_named_parameters,
        _custom_state,
    )

    slot = trainer._checkpoint_slots[name]
    records = (
        {} if slot.custom_payload is None else deepcopy(slot.custom_payload.records)
    )
    tensors: dict[str, torch.Tensor] = {}
    optimizer: dict[str, torch.Tensor] = {}
    if records:
        assert slot.custom_payload is not None
        tensors.update(
            (key, value.clone()) for key, value in slot.custom_payload.tensors.items()
        )
        optimizer.update(
            (key, value.clone()) for key, value in slot.custom_payload.optimizer.items()
        )

    dynamic = slot.optimizer
    masters = (
        {}
        if dynamic is None
        else {
            id(param): master
            for param, master in zip(slot.params, dynamic.master_params, strict=True)
        }
    )
    for custom_name, custom in slot.custom.items():
        previous = records.get(custom_name)
        if previous is not None:
            for key in previous["tensor_keys"]:
                tensors.pop(key, None)
            for key in previous["trainable_keys"]:
                for component in ("master", "exp_avg", "exp_avg_sq", "step"):
                    optimizer.pop(f"{component}/{key}", None)
        state = _custom_state(custom)
        flattened = {
            custom_name if key == "" else f"{custom_name}.{key}": value
            for key, value in state.items()
        }
        trainable = {
            key: param
            for key, param in _custom_named_parameters(custom_name, custom)
            if param.requires_grad
        }
        parameter_aliases, buffer_aliases, persistent_buffer_keys, _ = _custom_layout(
            custom_name, custom
        )
        records[custom_name] = {
            "kind": custom.kind,
            "tensor_keys": sorted(flattened),
            "trainable_keys": sorted(trainable),
            "parameter_aliases": [list(group) for group in parameter_aliases],
            "buffer_aliases": [list(group) for group in buffer_aliases],
            "persistent_buffer_keys": list(persistent_buffer_keys),
        }
        tensors.update(
            (key, value.detach().cpu().contiguous().clone())
            for key, value in flattened.items()
        )
        for key, param in trainable.items():
            master = masters.get(id(param))
            if master is None:
                continue
            assert dynamic is not None
            state = dynamic.optimizer.state.get(master, {})
            optimizer[f"master/{key}"] = master.detach().cpu().contiguous()
            optimizer[f"exp_avg/{key}"] = (
                cast(torch.Tensor, state.get("exp_avg", torch.zeros_like(master)))
                .cpu()
                .contiguous()
            )
            optimizer[f"exp_avg_sq/{key}"] = (
                cast(torch.Tensor, state.get("exp_avg_sq", torch.zeros_like(master)))
                .cpu()
                .contiguous()
            )
            optimizer[f"step/{key}"] = torch.tensor(float(state.get("step", 0.0)))

    if dynamic is not None:
        for record in records.values():
            for key in record["trainable_keys"]:
                if f"master/{key}" in optimizer:
                    continue
                value = tensors[key].float()
                optimizer[f"master/{key}"] = value
                optimizer[f"exp_avg/{key}"] = torch.zeros_like(value)
                optimizer[f"exp_avg_sq/{key}"] = torch.zeros_like(value)
                optimizer[f"step/{key}"] = torch.tensor(0.0)

    if not records:
        return {}
    save = importlib.import_module("safetensors.torch").save_file
    save(tensors, snapshot / "custom_tensors.safetensors")
    if optimizer:
        (snapshot / "optimizer").mkdir(exist_ok=True)
        save(optimizer, snapshot / "optimizer/custom.safetensors")
    return records


def _local_state(
    trainer: TrainerRank,
    name: str,
    snapshot: Path,
) -> tuple[
    tuple[_LocalShard, ...],
    OptimizerConfig | None,
    dict[str, CustomTensorRecord],
]:
    from art.megatron.lora import LoRA
    from art.megatron.weights.lora_publish import collect_local_lora_entries

    ref = trainer._slot_ref(name)
    tensors, metadata = collect_local_lora_entries(
        trainer.runtime.model, {}, owner_rank=_rank(), slot_ref=ref
    )
    dynamic = trainer._checkpoint_slots[name].optimizer
    optimizer = None if dynamic is None else _optimizer_config(dynamic)
    masters = (
        {}
        if dynamic is None
        else {
            id(param): master
            for param, master in zip(
                trainer._checkpoint_slots[name].params,
                dynamic.master_params,
                strict=True,
            )
        }
    )
    by_key = {item.key: item for item in metadata}
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    metadata_by_block: dict[str, list[LoraShardMeta]] = {}
    for item in metadata:
        payloads.setdefault(item.block, {})[f"lora/{item.key}"] = (
            tensors[item.key].cpu().contiguous()
        )
        metadata_by_block.setdefault(item.block, []).append(item)
    if dynamic is not None:
        for chunk in trainer.runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA):
                    continue
                for key, param, expert in module._export_items(ref):
                    item = by_key.get(key)
                    if item is None:
                        continue
                    master = masters[id(param)]
                    state = dynamic.optimizer.state.get(master, {})
                    values = (
                        master,
                        cast(torch.Tensor | None, state.get("exp_avg")),
                        cast(torch.Tensor | None, state.get("exp_avg_sq")),
                    )
                    for component, value in zip(
                        ("master", "exp_avg", "exp_avg_sq"), values, strict=True
                    ):
                        value = torch.zeros_like(master) if value is None else value
                        local = value if expert is None else value[expert]
                        payloads[item.block][f"{component}/{key}"] = (
                            local.T.float().cpu().contiguous()
                        )
                    step = state.get("step", 0.0)
                    payloads[item.block][f"step/{key}"] = torch.tensor(float(step))
    records: list[_LocalShard] = []
    for index, block in enumerate(sorted(payloads)):
        relative = f"block-{index:06d}.safetensors"
        importlib.import_module("safetensors.torch").save_file(
            payloads[block], snapshot / relative
        )
        records.extend(_LocalShard(item, relative) for item in metadata_by_block[block])
    return tuple(records), optimizer, _custom_snapshot(trainer, name, snapshot)


def prepare_checkpoint_save(
    trainer: TrainerRank, output_dir: str, checkpoint_name: str
) -> None:
    with trainer._checkpoint_prepare_lock:
        group = _ensure_group(trainer)
        identity = (output_dir, checkpoint_name)
        if any(value != identity for value in _gather(identity, group)):
            raise RuntimeError("Checkpoint save identity differs across ranks")
        with trainer._checkpoint_save_condition:
            pending = (
                output_dir in trainer._checkpoint_preparing_saves
                or output_dir in trainer._prepared_checkpoint_saves
            )
        if any(value != pending for value in _gather(pending, group)):
            raise RuntimeError(
                f"Checkpoint save state differs across ranks: {output_dir}"
            )
        if pending:
            raise RuntimeError(f"Checkpoint save is already pending: {output_dir}")
        with trainer._checkpoint_save_condition:
            trainer._checkpoint_preparing_saves.add(output_dir)
        try:
            known = (
                checkpoint_name in trainer._checkpoint_slots
                and trainer._checkpoint_slots[checkpoint_name].config is not None
            )
            if not all(_gather(known, group)):
                raise trainer._slot_state_error(
                    f"Unknown checkpoint on at least one rank: {checkpoint_name!r}"
                )
            config = deepcopy(_validate_save_state(trainer, checkpoint_name))
            if any(value != config for value in _gather(config, group)):
                raise trainer._slot_state_error(
                    f"Checkpoint {checkpoint_name!r} configuration differs across ranks"
                )
        except BaseException:
            with trainer._checkpoint_save_condition:
                trainer._checkpoint_preparing_saves.discard(output_dir)
            raise
        destination = Path(output_dir)
        reservation = destination.with_name(f".{destination.name}.reserved")
        snapshot = destination.with_name(
            f".{destination.name}.snapshot-r{_rank()}-{uuid.uuid4().hex}"
        )
        error: BaseException | None = None
        prepared: _PreparedSave | None = None
        shards: tuple[_LocalShard, ...] | None = None
        optimizer: OptimizerConfig | None = None
        custom_tensors: dict[str, CustomTensorRecord] = {}
        reservation_created = False
        with trainer._checkpoint_save_condition:
            sequence = trainer._checkpoint_save_sequence
            trainer._checkpoint_save_sequence += 1
        if any(value != sequence for value in _gather(sequence, group)):
            with trainer._checkpoint_save_condition:
                trainer._checkpoint_preparing_saves.discard(output_dir)
            with trainer._checkpoint_save_condition:
                trainer._checkpoint_save_skipped.add(sequence)
            _advance_save_queue(trainer, sequence)
            raise RuntimeError("Checkpoint save order differs across ranks")
        try:
            if _rank() == 0:
                reservation.mkdir(parents=True)
                reservation_created = True
            snapshot.mkdir(parents=True)
            shards, optimizer, custom_tensors = _local_state(
                trainer, checkpoint_name, snapshot
            )
        except BaseException as exc:
            error = exc
        try:
            raise_distributed(error, "prepare checkpoint", group)
            if any(value != optimizer for value in _gather(optimizer, group)):
                raise trainer._slot_state_error(
                    f"Checkpoint {checkpoint_name!r} optimizer differs across ranks"
                )
            custom_signature = (
                custom_tensors,
                _file_digest(snapshot / "custom_tensors.safetensors")
                if custom_tensors
                else None,
                _file_digest(snapshot / "optimizer/custom.safetensors")
                if (snapshot / "optimizer/custom.safetensors").is_file()
                else None,
            )
            if any(
                value != custom_signature for value in _gather(custom_signature, group)
            ):
                raise trainer._slot_state_error(
                    f"Checkpoint {checkpoint_name!r} custom tensors differ across ranks"
                )
            assert shards is not None
            prepared = _PreparedSave(
                sequence,
                snapshot,
                reservation,
                destination,
                dict(config),
                shards,
                optimizer,
                custom_tensors,
            )
        except BaseException as failure:
            cleanup = _cleanup_paths(
                [snapshot, *([reservation] if reservation_created else [])]
            )
            with trainer._checkpoint_save_condition:
                trainer._checkpoint_preparing_saves.discard(output_dir)
                trainer._checkpoint_save_skipped.add(sequence)
            _advance_save_queue(trainer, sequence)
            cleanup_failure: BaseException | None = None
            try:
                raise_distributed(cleanup, "clean up checkpoint preparation", group)
            except BaseException as exc:
                cleanup_failure = exc
            if cleanup_failure is not None:
                raise BaseExceptionGroup(
                    "checkpoint preparation and cleanup both failed",
                    [failure, cleanup_failure],
                ) from None
            raise failure
        assert prepared is not None
        with trainer._checkpoint_save_condition:
            trainer._prepared_checkpoint_saves[output_dir] = prepared
            trainer._finalized_checkpoint_saves.pop(output_dir, None)
            trainer._checkpoint_preparing_saves.discard(output_dir)
            trainer._checkpoint_save_condition.notify_all()


def _read_snapshot(
    prepared: _PreparedSave, relative: str, prefix: str, keys: Iterable[str]
) -> dict[str, torch.Tensor]:
    load = importlib.import_module("safetensors.torch").load_file
    payload = load(prepared.snapshot / relative)
    return {key: payload[f"{prefix}/{key}"] for key in keys}


def _merge_component(
    prepared: _PreparedSave,
    metadata: Sequence[LoraShardMeta],
    component: str,
    group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    from art.megatron.weights.lora_publish import merge_sharded_adapter_entries

    owned = [item for item in metadata if item.owner_rank == _rank()]
    local: dict[str, torch.Tensor] = {}
    error: BaseException | None = None
    try:
        if owned:
            files = {
                record.file for record in prepared.shards if record.metadata in owned
            }
            for relative in files:
                keys = [
                    item.key
                    for item in owned
                    if next(
                        record.file
                        for record in prepared.shards
                        if record.metadata == item
                    )
                    == relative
                ]
                local.update(_read_snapshot(prepared, relative, component, keys))
    except BaseException as exc:
        error = exc
    raise_distributed(error, f"read checkpoint {component} block", group)
    exchanged: dict[tuple[int, str], torch.Tensor] = {}
    for item in sorted(metadata, key=lambda value: (value.owner_rank, value.key)):
        identity = (item.owner_rank, item.key)
        if _rank() == item.owner_rank:
            tensor = local[item.key].contiguous()
            if _rank() == 0:
                exchanged[identity] = tensor
            else:
                dist.send(tensor, dst=0, group=group)
        elif _rank() == 0:
            dtype = (
                getattr(torch, item.dtype_name)
                if component == "lora"
                else torch.float32
            )
            tensor = torch.empty(item.shape, dtype=dtype)
            dist.recv(tensor, src=item.owner_rank, group=group)
            exchanged[identity] = tensor
    entries: dict[str, list[tuple[dict[str, object], torch.Tensor]]] = {}
    merged: dict[str, torch.Tensor] = {}
    error = None
    if _rank() == 0:
        try:
            for item in metadata:
                entries.setdefault(item.key, []).append(
                    (item.manifest, exchanged[(item.owner_rank, item.key)])
                )
            merged = merge_sharded_adapter_entries(entries)  # type: ignore[arg-type]
        except BaseException as exc:
            error = exc
    raise_distributed(error, f"merge checkpoint {component} block", group)
    return merged


def _consolidate(shards: Sequence[Path], output: Path) -> None:
    sources: dict[str, tuple[Path, int, int, int, dict[str, object]]] = {}
    for shard in shards:
        with shard.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            start, end = value["data_offsets"]
            sources[key] = (
                shard,
                8 + header_size,
                start,
                end,
                {name: item for name, item in value.items() if name != "data_offsets"},
            )
    offset = 0
    header: dict[str, dict[str, object]] = {}
    for key in sorted(sources):
        _path, _base, start, end, metadata = sources[key]
        header[key] = {**metadata, "data_offsets": [offset, offset + end - start]}
        offset += end - start
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with output.open("wb") as target:
        target.write(struct.pack("<Q", len(encoded)))
        target.write(encoded)
        for key in sorted(sources):
            path, base, start, end, _metadata = sources[key]
            with path.open("rb") as source:
                source.seek(base + start)
                remaining = end - start
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(f"Truncated safetensors payload for {key!r}")
                    target.write(chunk)
                    remaining -= len(chunk)


def _rank_zero_phase(
    action: Callable[[], None],
    phase: str,
    group: dist.ProcessGroup | None,
) -> None:
    error: BaseException | None = None
    if _rank() == 0:
        try:
            action()
        except BaseException as exc:
            error = exc
    raise_distributed(error, phase, group)


def _finish(trainer: TrainerRank, prepared: _PreparedSave) -> None:
    from art.megatron.model_support.lora_disk import save_adapter_config

    group = _ensure_finalize_group(trainer)
    metadata = [item for values in _gather(prepared.shards, group) for item in values]
    identities: set[tuple[str, int]] = set()
    selected: list[LoraShardMeta] = []
    for item in sorted(metadata, key=lambda value: value.metadata.owner_rank):
        identity = (item.metadata.key, int(item.metadata.manifest.get("shard_rank", 0)))
        if identity not in identities:
            identities.add(identity)
            selected.append(item.metadata)
    blocks = sorted({item.block for item in selected})
    temporary = prepared.destination.with_name(
        f".{prepared.destination.name}.tmp-{uuid.uuid4().hex}"
    )
    _rank_zero_phase(
        lambda: temporary.mkdir(parents=True), "create checkpoint output", group
    )
    parameters: dict[str, list[str]] = {}
    steps: dict[str, float] = {}
    lora_shards: list[Path] = []
    try:
        for index, block in enumerate(blocks):
            block_metadata = [item for item in selected if item.block == block]
            lora = _merge_component(prepared, block_metadata, "lora", group)
            relative = f".adapter-{index:06d}.safetensors"
            _rank_zero_phase(
                lambda: importlib.import_module("safetensors.torch").save_file(
                    lora, temporary / relative
                ),
                "write checkpoint adapter block",
                group,
            )
            if _rank() == 0:
                lora_shards.append(temporary / relative)
            if prepared.optimizer is None:
                continue
            files: list[str] = []
            for component in ("master", "exp_avg", "exp_avg_sq"):
                tensors = _merge_component(prepared, block_metadata, component, group)
                relative = f"optimizer/{component}-{index:06d}.safetensors"

                def write_optimizer_block() -> None:
                    (temporary / "optimizer").mkdir(exist_ok=True)
                    importlib.import_module("safetensors.torch").save_file(
                        tensors, temporary / relative
                    )

                _rank_zero_phase(
                    write_optimizer_block, "write checkpoint optimizer block", group
                )
                files.append(relative)
            if _rank() == 0:
                for key in (item.key for item in block_metadata):
                    parameters[key] = list(files)
            owned = [item for item in block_metadata if item.owner_rank == _rank()]
            local_steps: dict[str, float] = {}
            error: BaseException | None = None
            try:
                for relative in {
                    record.file
                    for record in prepared.shards
                    if record.metadata in owned
                }:
                    load = importlib.import_module("safetensors.torch").load_file
                    payload = load(prepared.snapshot / relative)
                    local_steps.update(
                        (key.removeprefix("step/"), float(value.item()))
                        for key, value in payload.items()
                        if key.startswith("step/")
                    )
            except BaseException as exc:
                error = exc
            raise_distributed(error, "read checkpoint optimizer steps", group)
            step_values: dict[str, set[float]] = {}
            for values in _gather(local_steps, group):
                for key, value in values.items():
                    step_values.setdefault(key, set()).add(value)
            if mismatched := {
                key: values for key, values in step_values.items() if len(values) != 1
            }:
                raise trainer._slot_state_error(
                    f"Optimizer shard steps differ: {mismatched}"
                )
            steps.update((key, values.pop()) for key, values in step_values.items())

        def commit() -> None:
            safe_open = importlib.import_module("safetensors").safe_open
            _consolidate(lora_shards, temporary / "adapter_model.safetensors")
            for shard in lora_shards:
                shard.unlink()
            save_adapter_config(
                temporary, {**prepared.config, _ART_FORMAT_KEY: _ART_FORMAT}
            )
            if prepared.custom_tensors:
                shutil.copy2(
                    prepared.snapshot / "custom_tensors.safetensors",
                    temporary / "custom_tensors.safetensors",
                )
                custom_optimizer = prepared.snapshot / "optimizer/custom.safetensors"
                if custom_optimizer.is_file():
                    (temporary / "optimizer").mkdir(exist_ok=True)
                    shutil.copy2(
                        custom_optimizer,
                        temporary / "optimizer/custom.safetensors",
                    )
            manifest: CheckpointManifest = {
                "format_version": FORMAT if prepared.custom_tensors else LEGACY_FORMAT,
                "base_model_name_or_path": str(
                    prepared.config["base_model_name_or_path"]
                ),
                "optimizer": prepared.optimizer,
                "parameters": parameters,
                "steps": steps,
                "files": {},
                "digest": "",
            }
            if prepared.custom_tensors:
                manifest["custom_tensors"] = prepared.custom_tensors
            artifact_files = {
                "adapter_config.json",
                "adapter_model.safetensors",
                *(file for record in parameters.values() for file in record),
                *({"custom_tensors.safetensors"} if prepared.custom_tensors else set()),
                *(
                    {"optimizer/custom.safetensors"}
                    if (prepared.snapshot / "optimizer/custom.safetensors").is_file()
                    else set()
                ),
            }
            manifest["files"] = {
                relative: _file_digest(temporary / relative)
                for relative in artifact_files
            }
            manifest["digest"] = _manifest_digest(manifest)
            with safe_open(
                temporary / "adapter_model.safetensors", framework="pt"
            ) as adapter:
                adapter_keys = set(adapter.keys())
            _validate_manifest(
                manifest,
                adapter_keys=adapter_keys,
                config=prepared.config,
            )
            (temporary / MANIFEST_FILE).write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            if prepared.destination.exists():
                if (
                    prepare_checkpoint(str(prepared.destination)).digest
                    != manifest["digest"]
                ):
                    raise FileExistsError(
                        "Checkpoint path already contains different state: "
                        f"{prepared.destination}"
                    )
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, prepared.destination)

        _rank_zero_phase(commit, "commit checkpoint", group)
    except BaseException:
        if _rank() == 0:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _advance_save_queue(trainer: TrainerRank, sequence: int) -> None:
    with trainer._checkpoint_save_condition:
        if sequence == trainer._checkpoint_save_next:
            trainer._checkpoint_save_next += 1
        while trainer._checkpoint_save_next in trainer._checkpoint_save_skipped:
            trainer._checkpoint_save_skipped.remove(trainer._checkpoint_save_next)
            trainer._checkpoint_save_next += 1
        trainer._checkpoint_save_condition.notify_all()


def _cleanup_paths(paths: Iterable[Path]) -> BaseException | None:
    errors: list[BaseException] = []
    for path in paths:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            errors.append(exc)
    return BaseExceptionGroup("checkpoint cleanup failed", errors) if errors else None


def _claim_finalization(
    trainer: TrainerRank,
    output_dir: str,
    action: Literal["finish", "abort"],
) -> _PreparedSave | None:
    with trainer._checkpoint_save_condition:
        while True:
            prepared = trainer._prepared_checkpoint_saves.get(output_dir)
            if prepared is None:
                if output_dir in trainer._finalized_checkpoint_saves:
                    return None
                if action == "abort":
                    return None
                raise RuntimeError(f"Checkpoint save was not prepared: {output_dir}")
            outcome = trainer._checkpoint_save_outcomes.get(output_dir)
            if outcome is not None and outcome != action:
                raise RuntimeError(
                    f"Checkpoint save was already {outcome}ed: {output_dir}"
                )
            if output_dir in trainer._checkpoint_finalizing_saves:
                trainer._checkpoint_save_condition.wait()
                continue
            if outcome is None and prepared.sequence != trainer._checkpoint_save_next:
                raise RuntimeError(
                    "Checkpoint saves must be finalized in preparation order: "
                    f"expected sequence {trainer._checkpoint_save_next}, got "
                    f"{prepared.sequence}"
                )
            trainer._checkpoint_finalizing_saves[output_dir] = action
            return prepared


def _finalize_checkpoint_save(
    trainer: TrainerRank,
    output_dir: str,
    action: Literal["finish", "abort"],
) -> None:
    group = _ensure_finalize_group(trainer)
    with trainer._checkpoint_finalize_lock:
        with trainer._checkpoint_save_condition:
            local = trainer._prepared_checkpoint_saves.get(output_dir)
            finalized = trainer._finalized_checkpoint_saves.get(output_dir)
            sequence = (
                local.sequence
                if local is not None
                else None
                if finalized is None
                else finalized.sequence
            )
            outcome = (
                trainer._checkpoint_save_outcomes.get(output_dir)
                if finalized is None
                else finalized.outcome
            )
        states = _gather((action, output_dir, sequence, outcome), group)
        if any(not isinstance(value, tuple) or len(value) != 4 for value in states):
            raise RuntimeError("Checkpoint finalization protocol is out of sync")
        if any(value[:3] != states[0][:3] for value in states):
            raise RuntimeError("Checkpoint save actions differ across ranks")
        outcomes = {value[3] for value in states}
        if len(outcomes) != 1:
            raise RuntimeError("Checkpoint save outcomes differ across ranks")
        if sequence is None:
            if action == "abort":
                return
            raise RuntimeError(f"Checkpoint save was not prepared: {output_dir}")
        finalized_ranks = _gather(finalized is not None, group)
        if all(finalized_ranks):
            if outcome == "finish" or action == "abort":
                return
            raise RuntimeError(f"Checkpoint save was already {outcome}ed: {output_dir}")
        if outcome is not None and outcome != action:
            raise RuntimeError(f"Checkpoint save was already {outcome}ed: {output_dir}")
        prepared = (
            _claim_finalization(trainer, output_dir, action)
            if finalized is None
            else None
        )
        assert prepared is not None or finalized is not None
        error: BaseException | None = None
        cleanup_failed = True
        try:
            if finalized is None and outcome is None and action == "finish":
                try:
                    assert prepared is not None
                    _finish(trainer, prepared)
                except BaseException as exc:
                    error = exc
            if finalized is None and outcome is None:
                assert prepared is not None
                outcome = action if error is None else "abort"
                with trainer._checkpoint_save_condition:
                    trainer._checkpoint_save_outcomes[output_dir] = outcome
                    if outcome == "abort":
                        trainer._checkpoint_save_skipped.add(prepared.sequence)
                _advance_save_queue(trainer, prepared.sequence)
            cleanup = None
            if prepared is not None:
                paths = [prepared.snapshot]
                if _rank() == 0:
                    paths.append(prepared.reservation)
                cleanup = _cleanup_paths(paths)
            try:
                failures = _gather(
                    (
                        None if error is None else repr(error),
                        None if cleanup is None else repr(cleanup),
                    ),
                    group,
                )
            except BaseException as exc:
                local_failures = [
                    *([error] if error is not None else []),
                    *([cleanup] if cleanup is not None else []),
                    exc,
                ]
                if len(local_failures) == 1:
                    raise local_failures[0]
                raise BaseExceptionGroup(
                    "checkpoint finalization and coordination failed", local_failures
                ) from None
            if any(
                not isinstance(failure, tuple)
                or len(failure) != 2
                or any(
                    value is not None and not isinstance(value, str)
                    for value in failure
                )
                for failure in failures
            ):
                raise RuntimeError("Checkpoint finalization protocol is out of sync")
            cleanup_failed = any(
                cleanup_error is not None for _, cleanup_error in failures
            )
            local_failures = [
                *([error] if error is not None else []),
                *([cleanup] if cleanup is not None else []),
            ]
            if local_failures:
                if len(local_failures) == 1:
                    raise local_failures[0]
                raise BaseExceptionGroup(
                    "checkpoint finalization failed", local_failures
                )
            if remote := next((failure for failure in failures if any(failure)), None):
                raise RuntimeError(
                    f"Another rank failed to {action} checkpoint: {remote}"
                )
        finally:
            with trainer._checkpoint_save_condition:
                trainer._checkpoint_finalizing_saves.pop(output_dir, None)
                if not cleanup_failed:
                    trainer._prepared_checkpoint_saves.pop(output_dir, None)
                    trainer._checkpoint_save_outcomes.pop(output_dir, None)
                    assert outcome is not None
                    trainer._finalized_checkpoint_saves[output_dir] = _FinalizedSave(
                        sequence, outcome
                    )
                trainer._checkpoint_save_condition.notify_all()


def finish_checkpoint_save(trainer: TrainerRank, output_dir: str) -> None:
    _finalize_checkpoint_save(trainer, output_dir, "finish")


def abort_checkpoint_save(trainer: TrainerRank, output_dir: str) -> None:
    _finalize_checkpoint_save(trainer, output_dir, "abort")


def _load_adapter(
    trainer: TrainerRank, source: PreparedCheckpoint, keys: Iterable[str]
) -> dict[str, torch.Tensor]:
    if source.manifest is None:
        from art.megatron.model_support.lora_disk import (
            load_lora_tensors_for_megatron,
        )

        loaded = load_lora_tensors_for_megatron(
            source.path, handler=trainer.runtime.model_support_handler
        )
        return {key: value for key, value in loaded.items() if key in set(keys)}
    safe_open = importlib.import_module("safetensors").safe_open
    with safe_open(source.path / "adapter_model.safetensors", framework="pt") as handle:
        available = set(handle.keys())
        return {key: handle.get_tensor(key) for key in keys if key in available}


def _localized(
    module: LoRA, tensor: torch.Tensor, parameter: torch.nn.Parameter
) -> torch.Tensor:
    return module._localized_weight(tensor, into=parameter).contiguous()


def _slot_snapshot(trainer: TrainerRank) -> _SlotSnapshot:
    from art.megatron.lora import LoRA

    return tuple(
        (
            module,
            dict(module._slot_keys),
            dict(module._slot_modules.items()),
            {key: getattr(slot, "ref") for key, slot in module._slot_modules.items()},
        )
        for chunk in trainer.runtime.model
        for module in chunk.modules()
        if isinstance(module, LoRA)
    )


def _restore_slots(snapshot: _SlotSnapshot) -> None:
    for module, keys, slots, refs in snapshot:
        for key, slot in slots.items():
            setattr(slot, "ref", refs[key])
        module._slot_keys = keys
        module._slot_modules = torch.nn.ModuleDict(slots)


def _commit_slot(trainer: TrainerRank, source: str, destination: str) -> None:
    from art.megatron.lora import LoRA

    source_ref = trainer._slot_ref(source)
    destination_ref = trainer._slot_ref(destination)
    for chunk in trainer.runtime.model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            source_key = module._slot_keys.pop(source_ref, None)
            destination_key = module._slot_keys.pop(destination_ref, None)
            if source_key is None:
                if destination_key is not None:
                    del module._slot_modules[destination_key]
                continue
            slot = module._slot_modules[source_key]
            setattr(slot, "ref", destination_ref)
            target_key = destination_key or source_key
            module._slot_keys[destination_ref] = target_key
            if target_key != source_key:
                module._slot_modules[target_key] = slot
                del module._slot_modules[source_key]


def _optimizer_state(
    trainer: TrainerRank, source: PreparedCheckpoint, name: str
) -> LocalOptimizerState:
    assert source.manifest is not None and source.manifest["optimizer"] is not None
    from art.megatron.lora import LoRA

    ref = trainer._slot_ref(name)
    components: dict[str, list[torch.Tensor]] = {
        "master": [],
        "exp_avg": [],
        "exp_avg_sq": [],
    }
    steps: list[float] = []
    sites: list[tuple[LoRA, str, torch.nn.Parameter, list[str], list[list[str]]]] = []
    file_keys: dict[str, set[str]] = {}
    for chunk in trainer.runtime.model:
        for module in chunk.modules():
            if not isinstance(module, LoRA) or module._slot(ref) is None:
                continue
            for suffix, parameter in module._lora_params(ref):
                suffix = suffix.removesuffix(".weight")
                keys = [
                    key
                    for key in module._expected_weight_keys(suffix)
                    if isinstance(key, str)
                ]
                records = [source.manifest["parameters"][key] for key in keys]
                sites.append((module, suffix, parameter, keys, records))
                for record in records:
                    for filename in record:
                        file_keys.setdefault(filename, set()).update(keys)
    safe_open = importlib.import_module("safetensors").safe_open
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    for filename, keys in file_keys.items():
        with safe_open(source.path / filename, framework="pt") as handle:
            loaded[filename] = {key: handle.get_tensor(key) for key in keys}
    for module, suffix, parameter, keys, records in sites:
        if not keys:
            for component in components.values():
                component.append(torch.zeros_like(parameter))
            steps.append(0.0)
            continue
        for index, component in enumerate(components):
            tensors = {
                key: loaded[record[index]][key]
                for key, record in zip(keys, records, strict=True)
            }
            full = module._adapter_weight(tensors, suffix=suffix)
            components[component].append(_localized(module, full, parameter))
        key_steps = {source.manifest["steps"][key] for key in keys}
        if len(key_steps) != 1:
            raise RuntimeError(f"Optimizer steps differ for {keys}")
        steps.append(key_steps.pop())
    return LocalOptimizerState(
        tuple(components["master"]),
        tuple(components["exp_avg"]),
        tuple(components["exp_avg_sq"]),
        tuple(steps),
        source.manifest["optimizer"],
    )


def _phase[T](
    action: Callable[[], T], phase: str, group: dist.ProcessGroup | None
) -> T:
    result: T | None = None
    error: BaseException | None = None
    try:
        result = action()
    except BaseException as exc:
        error = exc
    raise_distributed(error, phase, group)
    return cast(T, result)


def _validate_base_model(
    trainer: TrainerRank,
    source: PreparedCheckpoint,
    config: Mapping[str, object],
) -> None:
    configured = str(config["base_model_name_or_path"])
    if (
        source.manifest is not None
        and source.manifest["base_model_name_or_path"] != configured
    ):
        raise trainer._slot_state_error(
            "Checkpoint manifest and adapter config name different base models"
        )
    runtime_model = getattr(trainer.runtime, "model_identifier", None)
    if runtime_model is not None and runtime_model != configured:
        raise trainer._slot_state_error(
            f"Checkpoint base model {configured!r} differs from runtime model "
            f"{runtime_model!r}"
        )
    supported = tuple(
        getattr(getattr(trainer.runtime, "model_support_spec", None), "model_names", ())
    )
    if supported and configured not in supported:
        raise trainer._slot_state_error(
            f"Checkpoint base model {configured!r} is incompatible with this runtime"
        )


def _rollback_load(
    trainer: TrainerRank,
    snapshot: _SlotSnapshot,
    temporary: str,
    name: str,
    previous: object,
    group: dist.ProcessGroup | None,
) -> None:
    def rollback() -> None:
        _restore_slots(snapshot)
        trainer._checkpoint_slots.pop(temporary, None)
        if previous is None:
            trainer._checkpoint_slots.pop(name, None)
        else:
            from art.trainer_rank._impl import _CheckpointSlot

            trainer._checkpoint_slots[name] = cast(_CheckpointSlot, previous)

    _phase(rollback, "roll back checkpoint load", group)


def load_checkpoint(
    trainer: TrainerRank, source: PreparedCheckpoint, name: str
) -> None:
    group = _ensure_group(trainer)
    if any(value != source.digest for value in _gather(source.digest, group)):
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} content differs across ranks"
        )
    config = _phase(
        lambda: trainer._validate_checkpoint_adapter_config(
            name, source.config, alpha=None
        ),
        "validate checkpoint config",
        group,
    )
    assert config is not None
    if any(value != config for value in _gather(config, group)):
        raise trainer._slot_state_error(
            f"Checkpoint {name!r} configuration differs across ranks"
        )
    _phase(
        lambda: _validate_base_model(trainer, source, config),
        "validate checkpoint base model",
        group,
    )
    _phase(
        lambda: trainer._guard_slot_can_load(trainer._slot_ref(name)),
        "validate checkpoint target",
        group,
    )
    local_keys = trainer._local_lora_adapter_templates()
    adapter = _phase(
        lambda: _load_adapter(trainer, source, local_keys),
        "read checkpoint adapter",
        group,
    )
    prepared_adapter = _phase(
        lambda: trainer._prepare_adapter_model(
            name, adapter, canonicalized=source.manifest is not None
        ),
        "localize checkpoint adapter",
        group,
    )
    expected = {key for keys in _gather(tuple(prepared_adapter), group) for key in keys}
    if source.manifest is not None and expected != set(source.keys):
        raise trainer._slot_state_error(
            "Checkpoint tensor coverage differs from runtime"
        )
    temporary = f"__art_loading_{uuid.uuid4().hex}"
    snapshot = _slot_snapshot(trainer)
    previous = trainer._checkpoint_slots.get(name)
    try:
        loaded = _phase(
            lambda: trainer._load_checkpoint_slot(
                temporary,
                prepared_adapter,
                alpha=float(config["lora_alpha"]),
                _prepared=True,
            ),
            "stage checkpoint adapter",
            group,
        )
        params = _phase(
            lambda: trainer._validate_checkpoint_consistency(
                temporary, loaded, expected
            ),
            "validate staged checkpoint",
            group,
        )
        from art.trainer_rank._impl import _CheckpointSlot

        trainer._checkpoint_slots[temporary] = _CheckpointSlot(
            params, config, custom_payload=source.custom
        )
        _phase(
            lambda: trainer._validate_loaded_checkpoint_config(temporary, config),
            "validate loaded checkpoint config",
            group,
        )
        if source.manifest is not None and source.manifest["optimizer"] is not None:
            optimizer_state = _phase(
                lambda: _optimizer_state(trainer, source, temporary),
                "read checkpoint optimizer",
                group,
            )
            trainer._checkpoint_slots[temporary].optimizer = _phase(
                lambda: trainer._restore_canonical_optimizer(
                    temporary, optimizer_state
                ),
                "restore checkpoint optimizer",
                group,
            )

        def commit() -> None:
            _commit_slot(trainer, temporary, name)
            staged = trainer._checkpoint_slots.pop(temporary)
            staged.revision = 0 if previous is None else previous.revision + 1
            trainer._checkpoint_slots[name] = staged

        _phase(commit, "commit checkpoint", group)
    except BaseException:
        _rollback_load(trainer, snapshot, temporary, name, previous, group)
        raise


def export_lora(trainer: TrainerRank, output_dir: str, checkpoint_name: str) -> int:
    group = _ensure_group(trainer)
    slot = None
    error: BaseException | None = None
    try:
        slot = trainer._checkpoint_slots.get(checkpoint_name)
        if slot is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint_name!r}")
        if slot.config is None:
            raise trainer._slot_state_error(
                f"Checkpoint {checkpoint_name!r} has no adapter_config"
            )
    except BaseException as exc:
        error = exc
    raise_distributed(error, "validate LoRA export", group)
    assert slot is not None and slot.config is not None
    identity = (dict(slot.config), slot.revision)
    if any(value != identity for value in _gather(identity, group)):
        raise trainer._slot_state_error(
            f"Checkpoint {checkpoint_name!r} differs across ranks"
        )
    from art.megatron.weights.lora_publish import save_vllm_lora_from_model

    error = None
    try:
        save_vllm_lora_from_model(
            model=trainer.runtime.model,
            adapter_dtypes={},
            handler=trainer.runtime.model_support_handler,
            adapter_config=dict(slot.config),
            output_dir=output_dir,
            rank=trainer.runtime.rank,
            world_size=trainer.runtime.world_size,
            slot_ref=trainer._slot_ref(checkpoint_name),
        )
    except BaseException as exc:
        error = exc
    raise_distributed(error, "export LoRA", group)
    return slot.revision


def _ensure_groups(
    trainer: TrainerRank,
) -> tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]:
    if not hasattr(trainer, "_checkpoint_process_group"):
        trainer._checkpoint_process_group = None
    if not hasattr(trainer, "_checkpoint_finalize_process_group"):
        trainer._checkpoint_finalize_process_group = None
    if not hasattr(trainer, "_checkpoint_group_lock"):
        trainer._checkpoint_group_lock = threading.Lock()
    if _distributed():
        with trainer._checkpoint_group_lock:
            if trainer._checkpoint_process_group is None:
                trainer._checkpoint_process_group = dist.new_group(backend="gloo")
            if trainer._checkpoint_finalize_process_group is None:
                trainer._checkpoint_finalize_process_group = dist.new_group(
                    backend="gloo"
                )
    return (
        trainer._checkpoint_process_group,
        trainer._checkpoint_finalize_process_group,
    )


def _ensure_group(trainer: TrainerRank) -> dist.ProcessGroup | None:
    return _ensure_groups(trainer)[0]


def _ensure_finalize_group(trainer: TrainerRank) -> dist.ProcessGroup | None:
    return _ensure_groups(trainer)[1]
