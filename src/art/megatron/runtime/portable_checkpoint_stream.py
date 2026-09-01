from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, cast

import torch

from art.megatron import checkpoint
from art.megatron.lora import LoRA, LoraShardMeta
from art.megatron.weights.lora_publish import collect_local_lora_entries

from .portable_snapshot import (
    PortableSnapshotCommittedFile,
    PortableSnapshotFile,
    PortableSnapshotGeneration,
    PortableSnapshotPreparedFile,
    PortableSnapshotRankReceipt,
    PortableSnapshotSink,
    _checkpoint_component,
    _file_sha256,
)


def export_portable_checkpoint_streamed(
    trainer: Any,
    sink: PortableSnapshotSink,
    generation: PortableSnapshotGeneration,
    *,
    export_id: str,
    name: str,
    rank: int,
    components: Callable[
        [Literal["weights", "optimizer"]],
        AbstractContextManager[tuple[torch.Tensor, ...]],
    ],
) -> PortableSnapshotRankReceipt | None:
    if not export_id or not name or rank < 0:
        raise ValueError("portable checkpoint export identity is invalid")
    if checkpoint._rank() != rank:
        raise RuntimeError("portable checkpoint rank identity changed")
    slot = trainer._checkpoint_slots.get(name)
    if slot is None or slot.config is None or slot.optimizer is None:
        raise RuntimeError("portable export requires weights and optimizer state")
    if slot.custom or slot.custom_payload is not None:
        raise RuntimeError("portable export does not support custom run tensors")

    group = checkpoint._ensure_finalize_group(trainer)
    config = deepcopy(checkpoint._validate_save_state(trainer, name))
    if any(value != config for value in checkpoint._gather(config, group)):
        raise RuntimeError("portable checkpoint configuration differs across ranks")

    committed: list[PortableSnapshotFile] = []
    file_digests: dict[str, str] = {}
    parameters: dict[str, list[str]] = {}
    steps: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix=f"art-portable-r{rank}-") as temporary:
        root = Path(temporary)
        shards, local_metadata = _write_weight_shards(
            trainer, name, root, components
        )
        metadata = _selected_metadata(local_metadata, group)
        blocks = tuple(sorted({item.block for item in metadata}))
        prepared = _prepared(root, config, shards, optimizer=None)
        adapter_shards: list[Path] = []
        for index, block in enumerate(blocks):
            block_metadata = [item for item in metadata if item.block == block]
            tensors = checkpoint._merge_component(
                prepared, block_metadata, "lora", group
            )
            relative = f".adapter-{index:06d}.safetensors"
            checkpoint._rank_zero_phase(
                lambda relative=relative, tensors=tensors: _save_tensors(
                    tensors, root / relative
                ),
                "write portable adapter block",
                group,
            )
            if rank == 0:
                adapter_shards.append(root / relative)

        adapter_path = root / "adapter_model.safetensors"
        checkpoint._rank_zero_phase(
            lambda: checkpoint._consolidate(adapter_shards, adapter_path),
            "consolidate portable adapter",
            group,
        )
        if rank == 0:
            for path in adapter_shards:
                path.unlink()
        _commit_rank_zero_file(
            sink,
            committed,
            file_digests,
            root,
            "adapter_model.safetensors",
            export_id=export_id,
            generation=generation,
            rank=rank,
            checkpoint_digest=None,
            group=group,
        )

        optimizer_config, optimizer_shards, local_steps = _write_optimizer_shards(
            trainer, name, root, local_metadata, components
        )
        if any(
            value != optimizer_config
            for value in checkpoint._gather(optimizer_config, group)
        ):
            raise RuntimeError("portable optimizer configuration differs across ranks")
        prepared = _prepared(root, config, optimizer_shards, optimizer_config)
        for index, block in enumerate(blocks):
            block_metadata = [item for item in metadata if item.block == block]
            paths: list[str] = []
            for component in ("master", "exp_avg", "exp_avg_sq"):
                tensors = checkpoint._merge_component(
                    prepared, block_metadata, component, group
                )
                relative = f"optimizer/{component}-{index:06d}.safetensors"
                checkpoint._rank_zero_phase(
                    lambda relative=relative, tensors=tensors: _save_tensors(
                        tensors, root / relative
                    ),
                    f"write portable {component} block",
                    group,
                )
                _commit_rank_zero_file(
                    sink,
                    committed,
                    file_digests,
                    root,
                    relative,
                    export_id=export_id,
                    generation=generation,
                    rank=rank,
                    checkpoint_digest=None,
                    group=group,
                )
                paths.append(relative)
            if rank == 0:
                for key in (item.key for item in block_metadata):
                    parameters[key] = paths

        gathered_steps: dict[str, set[float]] = {}
        for values in checkpoint._gather(local_steps, group):
            for key, value in values.items():
                gathered_steps.setdefault(key, set()).add(value)
        if mismatched := {
            key: values for key, values in gathered_steps.items() if len(values) != 1
        }:
            raise RuntimeError(f"portable optimizer steps differ: {mismatched}")
        if rank == 0:
            steps.update((key, values.pop()) for key, values in gathered_steps.items())

        config_path = root / "adapter_config.json"
        checkpoint._rank_zero_phase(
            lambda: importlib.import_module(
                "art.megatron.model_support.lora_disk"
            ).save_adapter_config(
                root,
                {
                    **config,
                    checkpoint._ART_FORMAT_KEY: checkpoint._ART_FORMAT,
                },
            ),
            "write portable adapter config",
            group,
        )
        _commit_rank_zero_file(
            sink,
            committed,
            file_digests,
            root,
            config_path.name,
            export_id=export_id,
            generation=generation,
            rank=rank,
            checkpoint_digest=None,
            group=group,
        )

        manifest: checkpoint.CheckpointManifest = {
            "format_version": checkpoint.LEGACY_FORMAT,
            "base_model_name_or_path": str(config["base_model_name_or_path"]),
            "optimizer": optimizer_config,
            "parameters": parameters,
            "steps": steps,
            "files": file_digests,
            "digest": "",
        }
        manifest["digest"] = checkpoint._manifest_digest(manifest)
        checkpoint_path = root / checkpoint.MANIFEST_FILE
        checkpoint._rank_zero_phase(
            lambda: checkpoint_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            ),
            "write portable checkpoint manifest",
            group,
        )
        _commit_rank_zero_file(
            sink,
            committed,
            file_digests,
            root,
            checkpoint.MANIFEST_FILE,
            export_id=export_id,
            generation=generation,
            rank=rank,
            checkpoint_digest=manifest["digest"],
            group=group,
            record_manifest_digest=False,
        )
        if rank != 0:
            return None
        return PortableSnapshotRankReceipt(
            rank=rank,
            checkpoint_digest=manifest["digest"],
            files=tuple(sorted(committed, key=lambda item: item.relative_path)),
        )


def _write_weight_shards(
    trainer: Any,
    name: str,
    root: Path,
    components: Callable[
        [Literal["weights", "optimizer"]],
        AbstractContextManager[tuple[torch.Tensor, ...]],
    ],
) -> tuple[tuple[checkpoint._LocalShard, ...], tuple[LoraShardMeta, ...]]:
    params = trainer.checkpoint_slot_parameters(name)
    with components("weights") as values, _bound_data(params, values):
        tensors, metadata = collect_local_lora_entries(
            trainer.runtime.model,
            {},
            owner_rank=checkpoint._rank(),
            slot_ref=trainer._slot_ref(name),
        )
        by_block: dict[str, dict[str, torch.Tensor]] = {}
        metadata_by_block: dict[str, list[LoraShardMeta]] = {}
        for item in metadata:
            by_block.setdefault(item.block, {})[f"lora/{item.key}"] = (
                tensors[item.key].cpu().contiguous()
            )
            metadata_by_block.setdefault(item.block, []).append(item)
        records: list[checkpoint._LocalShard] = []
        for index, block in enumerate(sorted(by_block)):
            relative = f"block-{index:06d}.safetensors"
            _save_tensors(by_block[block], root / relative)
            records.extend(
                checkpoint._LocalShard(item, relative)
                for item in metadata_by_block[block]
            )
    return tuple(records), tuple(metadata)


def _write_optimizer_shards(
    trainer: Any,
    name: str,
    root: Path,
    metadata: tuple[LoraShardMeta, ...],
    components: Callable[
        [Literal["weights", "optimizer"]],
        AbstractContextManager[tuple[torch.Tensor, ...]],
    ],
) -> tuple[
    checkpoint.OptimizerConfig,
    tuple[checkpoint._LocalShard, ...],
    dict[str, float],
]:
    slot = trainer._checkpoint_slots[name]
    dynamic = slot.optimizer
    assert dynamic is not None
    live = trainer.checkpoint_slot_optimizer_tensors(name)
    with components("optimizer") as values, _bound_data(live, values):
        optimizer_config = checkpoint._optimizer_config(dynamic)
        by_key = {item.key: item for item in metadata}
        masters = {
            id(param): master
            for param, master in zip(slot.params, dynamic.master_params, strict=True)
        }
        payloads: dict[str, dict[str, torch.Tensor]] = {}
        local_steps: dict[str, float] = {}
        for chunk in trainer.runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA):
                    continue
                for key, param, expert in module._export_items(trainer._slot_ref(name)):
                    item = by_key.get(key)
                    if item is None:
                        continue
                    master = masters[id(param)]
                    state = dynamic.optimizer.state.get(master, {})
                    values_by_component = {
                        "master": master,
                        "exp_avg": cast(
                            torch.Tensor,
                            state.get("exp_avg", torch.zeros_like(master)),
                        ),
                        "exp_avg_sq": cast(
                            torch.Tensor,
                            state.get("exp_avg_sq", torch.zeros_like(master)),
                        ),
                    }
                    block = payloads.setdefault(item.block, {})
                    for component, value in values_by_component.items():
                        local = value if expert is None else value[expert]
                        block[f"{component}/{key}"] = (
                            local.T.float().cpu().contiguous()
                        )
                    step = float(state.get("step", 0.0))
                    block[f"step/{key}"] = torch.tensor(step)
                    local_steps[key] = step
        records: list[checkpoint._LocalShard] = []
        for index, block_name in enumerate(sorted(payloads)):
            relative = f"block-{index:06d}.safetensors"
            _save_tensors(payloads[block_name], root / relative)
            records.extend(
                checkpoint._LocalShard(item, relative)
                for item in metadata
                if item.block == block_name and item.owner_rank == checkpoint._rank()
            )
    return optimizer_config, tuple(records), local_steps


def _selected_metadata(
    local: tuple[LoraShardMeta, ...], group: Any
) -> tuple[LoraShardMeta, ...]:
    metadata = [item for values in checkpoint._gather(local, group) for item in values]
    identities: set[tuple[str, int]] = set()
    selected: list[LoraShardMeta] = []
    for item in sorted(metadata, key=lambda value: value.owner_rank):
        identity = (item.key, int(item.manifest.get("shard_rank", 0)))
        if identity not in identities:
            identities.add(identity)
            selected.append(item)
    return tuple(selected)


def _prepared(
    root: Path,
    config: dict[str, object],
    shards: tuple[checkpoint._LocalShard, ...],
    optimizer: checkpoint.OptimizerConfig | None,
) -> checkpoint._PreparedSave:
    return checkpoint._PreparedSave(0, root, root, root, config, shards, optimizer)


@contextmanager
def _bound_data(
    live: tuple[torch.Tensor, ...], values: tuple[torch.Tensor, ...]
) -> Iterator[None]:
    if len(live) != len(values) or any(
        source.shape != target.shape for source, target in zip(values, live, strict=True)
    ):
        raise RuntimeError("portable component image differs from live tensor layout")
    original = tuple(tensor.data for tensor in live)
    try:
        for target, source in zip(live, values, strict=True):
            target.data = source.detach()
        yield
    finally:
        for target, source in zip(live, original, strict=True):
            target.data = source


def _save_tensors(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    importlib.import_module("safetensors.torch").save_file(dict(tensors), path)


def _commit_rank_zero_file(
    sink: PortableSnapshotSink,
    committed: list[PortableSnapshotFile],
    file_digests: dict[str, str],
    root: Path,
    relative: str,
    *,
    export_id: str,
    generation: PortableSnapshotGeneration,
    rank: int,
    checkpoint_digest: str | None,
    group: Any,
    record_manifest_digest: bool = True,
) -> None:
    result: tuple[PortableSnapshotCommittedFile, ...] = ()
    manifest_digest: str | None = None

    def commit() -> None:
        nonlocal manifest_digest, result
        source = root / relative
        prepared = PortableSnapshotPreparedFile(
            relative_path=relative,
            component=_checkpoint_component(relative),
            byte_count=source.stat().st_size,
            sha256=_file_sha256(source),
        )
        with tempfile.TemporaryDirectory(prefix="art-portable-file-") as temporary:
            directory = Path(temporary)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, target)
            if record_manifest_digest:
                manifest_digest = checkpoint._file_digest(target)
            result = sink.commit_prepared(
                export_id=export_id,
                generation=generation,
                rank=rank,
                checkpoint_digest=checkpoint_digest,
                directory=directory,
                files=(prepared,),
            )
        if len(result) != 1 or result[0].relative_path != relative:
            raise RuntimeError("portable sink changed committed file identity")
        item = result[0]
        committed.append(
            PortableSnapshotFile(
                object_id=item.object_id,
                relative_path=relative,
                component=prepared.component,
                byte_count=prepared.byte_count,
                sha256=prepared.sha256,
                source_ref=item.source_ref,
            )
        )
        if record_manifest_digest:
            assert manifest_digest is not None
            file_digests[relative] = manifest_digest

    checkpoint._rank_zero_phase(commit, f"commit portable file {relative}", group)
