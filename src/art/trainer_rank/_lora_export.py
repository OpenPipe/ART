"""Trainer-rank lifecycle for pipelined LoRA publication."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import TYPE_CHECKING, Any, TypeVar
import uuid

import torch

if TYPE_CHECKING:
    from art.megatron.lora import LoraShardMeta, LoRASlotRef
    from art.megatron.training.model_chunks import ModelChunks
    from art.megatron.weights.lora_publish import PackedExpertShardMeta
    from art.trainer_rank._impl import TrainerRank

_K = TypeVar("_K")


@dataclass(frozen=True)
class _PreparedLoraExport:
    inputs: _VllmLoraPublishInputs


@dataclass(frozen=True)
class _VllmLoraPublishPlan:
    rank: int
    device: torch.device
    metadata: list[LoraShardMeta]
    local_tensors: dict[str, torch.Tensor]
    packed_expert_metadata: list[PackedExpertShardMeta]
    local_packed_expert_tensors: dict[str, torch.Tensor]
    handler: Any
    adapter_config: dict[str, Any]


@dataclass(frozen=True)
class _VllmLoraPublishInputs:
    metadata: list[LoraShardMeta]
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor]
    packed_expert_metadata: list[PackedExpertShardMeta]
    packed_expert_tensors_by_owner_key: dict[tuple[int, str], torch.Tensor]
    handler: Any
    adapter_config: dict[str, Any]


class _PinnedCpuStager:
    def __init__(self) -> None:
        self._events: list[torch.cuda.Event] = []
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def stage(self, tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        if self._stream is None or not source.is_cuda:
            return source.cpu()

        source = source.contiguous()
        target = torch.empty_like(source, device="cpu", pin_memory=True)
        source_stream = torch.cuda.current_stream(source.device)
        self._stream.wait_stream(source_stream)
        with torch.cuda.stream(self._stream):
            target.copy_(source, non_blocking=True)
            source.record_stream(self._stream)
            event = torch.cuda.Event()
            event.record(self._stream)
        self._events.append(event)
        return target

    def finish(self) -> None:
        for event in self._events:
            event.synchronize()
        self._events.clear()


def _stage_tensor_mapping(
    tensors: dict[_K, torch.Tensor],
    stager: _PinnedCpuStager,
) -> dict[_K, torch.Tensor]:
    grouped: dict[
        tuple[str, int | None, torch.dtype], list[tuple[_K, torch.Tensor]]
    ] = {}
    for key, tensor in tensors.items():
        group_key = (tensor.device.type, tensor.device.index, tensor.dtype)
        grouped.setdefault(group_key, []).append((key, tensor))

    staged: dict[_K, torch.Tensor] = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: str(item[0]))
        flat = torch.cat(
            [tensor.detach().contiguous().view(-1) for _key, tensor in ordered]
        )
        staged_flat = stager.stage(flat)
        offset = 0
        for key, tensor in ordered:
            numel = tensor.numel()
            if key in staged:
                raise RuntimeError(f"Duplicate staged LoRA tensor: {key}")
            staged[key] = staged_flat.narrow(0, offset, numel).view(tensor.shape)
            offset += numel
    return staged


def _validate_vllm_lora_publish_runtime(
    rank: int, world_size: int
) -> tuple[int, torch.device]:
    from art.megatron.weights import lora_publish

    actual_rank, device = lora_publish._rank_and_device()
    if lora_publish._distributed_ready():
        actual_world_size = torch.distributed.get_world_size()  # type: ignore[possibly-missing-attribute]
        if actual_rank != rank or actual_world_size != world_size:
            raise RuntimeError(
                "LoRA publisher rank/world-size mismatch: "
                f"runtime=({rank}, {world_size}) "
                f"distributed=({actual_rank}, {actual_world_size})"
            )
    else:
        if rank != 0 or world_size != 1:
            raise RuntimeError(
                "Non-distributed LoRA publish requires rank=0 and world_size=1, "
                f"got rank={rank} world_size={world_size}"
            )
        rank = 0
    return rank, device


def _prepare_vllm_lora_publish(
    *,
    model: ModelChunks,
    adapter_dtypes: dict[str, torch.dtype],
    handler: Any,
    adapter_config: dict[str, Any],
    rank: int,
    world_size: int,
    slot_ref: LoRASlotRef | None = None,
    runtime: tuple[int, torch.device] | None = None,
) -> _VllmLoraPublishPlan:
    from art.megatron.lora import LoRAPublishPlanner
    from art.megatron.weights import lora_publish

    rank, device = (
        _validate_vllm_lora_publish_runtime(rank, world_size)
        if runtime is None
        else runtime
    )
    packed_expert_groups = tuple(handler.expert_packed_lora_groups())
    planner = LoRAPublishPlanner(model, slot_ref)
    local_tensors, local_metadata = lora_publish.collect_local_lora_entries(
        model,
        adapter_dtypes,
        owner_rank=rank,
        packed_expert_groups=packed_expert_groups,
        slot_ref=slot_ref,
    )
    (
        local_packed_tensors,
        local_packed_metadata,
    ) = lora_publish.collect_local_packed_expert_entries(
        model,
        adapter_dtypes,
        owner_rank=rank,
        packed_expert_groups=packed_expert_groups,
        slot_ref=slot_ref,
    )
    all_packed_metadata = (
        lora_publish._global_packed_expert_metadata(
            planner, adapter_dtypes, packed_expert_groups
        )
        if rank == 0
        else local_packed_metadata
    )
    all_metadata = (
        lora_publish._global_regular_metadata(
            planner,
            adapter_dtypes,
            packed_expert_groups if all_packed_metadata else (),
        )
        if rank == 0
        else local_metadata
    )
    return _VllmLoraPublishPlan(
        rank=rank,
        device=device,
        metadata=all_metadata,
        local_tensors=local_tensors,
        packed_expert_metadata=all_packed_metadata,
        local_packed_expert_tensors=local_packed_tensors,
        handler=handler,
        adapter_config=dict(adapter_config),
    )


def _exchange_vllm_lora_publish(
    plan: _VllmLoraPublishPlan,
) -> _VllmLoraPublishInputs | None:
    from art.megatron.weights import lora_publish

    exchanged_tensors = lora_publish._exchange_batched_tensors(
        plan.metadata,
        local_tensors=plan.local_tensors,
        rank=plan.rank,
        device=plan.device,
    )
    exchanged_packed_tensors = lora_publish._exchange_batched_tensors(
        plan.packed_expert_metadata,
        local_tensors=plan.local_packed_expert_tensors,
        rank=plan.rank,
        device=plan.device,
    )
    if plan.rank != 0:
        return None
    return _VllmLoraPublishInputs(
        metadata=plan.metadata,
        tensors_by_owner_key=exchanged_tensors,
        packed_expert_metadata=plan.packed_expert_metadata,
        packed_expert_tensors_by_owner_key=exchanged_packed_tensors,
        handler=plan.handler,
        adapter_config=plan.adapter_config,
    )


def _build_vllm_lora_tensors_from_inputs(
    inputs: _VllmLoraPublishInputs,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from art.megatron.weights import lora_publish

    return lora_publish._rank0_vllm_lora_tensors(
        metadata=inputs.metadata,
        tensors_by_owner_key=inputs.tensors_by_owner_key,
        packed_expert_metadata=inputs.packed_expert_metadata,
        packed_expert_tensors_by_owner_key=inputs.packed_expert_tensors_by_owner_key,
        handler=inputs.handler,
        adapter_config=inputs.adapter_config,
    )


def _capture_lora_publish_inputs(
    trainer: TrainerRank,
    checkpoint_name: str,
    adapter_config: dict[str, object],
    group: torch.distributed.ProcessGroup | None,
) -> tuple[_PreparedLoraExport | None, dict[str, float]]:
    from art.trainer_rank import _checkpoint

    timings: dict[str, float] = {}
    started = time.monotonic()
    runtime = _checkpoint._phase(
        lambda: _validate_vllm_lora_publish_runtime(
            trainer.runtime.rank, trainer.runtime.world_size
        ),
        "validate LoRA publish runtime",
        group,
    )
    timings["runtime_validation"] = time.monotonic() - started

    started = time.monotonic()
    plan = _checkpoint._phase(
        lambda: _prepare_vllm_lora_publish(
            model=trainer.runtime.model,
            adapter_dtypes={},
            handler=trainer.runtime.model_support_handler,
            adapter_config=adapter_config,
            rank=trainer.runtime.rank,
            world_size=trainer.runtime.world_size,
            slot_ref=trainer._slot_ref(checkpoint_name),
            runtime=runtime,
        ),
        "plan LoRA publish",
        group,
    )
    timings["plan_collect"] = time.monotonic() - started

    started = time.monotonic()
    inputs = _exchange_vllm_lora_publish(plan)
    timings["exchange"] = time.monotonic() - started

    started = time.monotonic()

    def stage() -> _PreparedLoraExport | None:
        if inputs is not None:
            stager = _PinnedCpuStager()
            staged = replace(
                inputs,
                tensors_by_owner_key=_stage_tensor_mapping(
                    inputs.tensors_by_owner_key, stager
                ),
                packed_expert_tensors_by_owner_key=_stage_tensor_mapping(
                    inputs.packed_expert_tensors_by_owner_key, stager
                ),
            )
            stager.finish()
            return _PreparedLoraExport(staged)
        return None

    prepared = _checkpoint._phase(stage, "stage LoRA publish tensors", group)
    timings["d2h"] = time.monotonic() - started
    return prepared, timings


def _save_lora_publish_inputs(
    output_dir: str, prepared: _PreparedLoraExport
) -> dict[str, float]:
    from art.megatron.model_support.lora_disk import save_vllm_lora_tensors

    started = time.monotonic()
    vllm_tensors, published_config = _build_vllm_lora_tensors_from_inputs(
        prepared.inputs
    )
    timings = {"convert": time.monotonic() - started}
    started = time.monotonic()
    save_vllm_lora_tensors(output_dir, vllm_tensors, published_config)
    timings["serialize"] = time.monotonic() - started
    return timings


def prepare_lora_export(
    trainer: TrainerRank,
    export_id: str,
    checkpoint_name: str,
    *,
    owner_id: str,
) -> tuple[int, dict[str, float]]:
    from art.trainer_rank import _checkpoint

    started = time.monotonic()
    group = _checkpoint._ensure_group(trainer)
    snapshots: dict[str, tuple[str, _PreparedLoraExport]] = getattr(
        trainer, "_prepared_lora_exports", {}
    )
    duplicate = (
        RuntimeError(f"LoRA export {export_id!r} is already prepared")
        if export_id in snapshots
        else None
    )
    _checkpoint.raise_distributed(duplicate, "validate LoRA export ID", group)
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
    _checkpoint.raise_distributed(error, "validate LoRA export", group)
    assert slot is not None and slot.config is not None
    identity = (dict(slot.config), slot.revision)
    if any(value != identity for value in _checkpoint._gather(identity, group)):
        raise trainer._slot_state_error(
            f"Checkpoint {checkpoint_name!r} differs across ranks"
        )
    slot_validation = time.monotonic() - started

    prepared = None
    capture_timings: dict[str, float] = {}
    error = None
    try:
        prepared, capture_timings = _capture_lora_publish_inputs(
            trainer, checkpoint_name, dict(slot.config), group
        )
    except BaseException as exc:
        error = exc
    _checkpoint.raise_distributed(error, "prepare LoRA export", group)
    if prepared is not None:
        snapshots[export_id] = (owner_id, prepared)
        trainer._prepared_lora_exports = snapshots
    return slot.revision, {"slot_validation": slot_validation, **capture_timings}


def finish_lora_export(
    trainer: TrainerRank, export_id: str, output_dir: str, *, owner_id: str
) -> dict[str, float]:
    snapshots: dict[str, tuple[str, _PreparedLoraExport]] = getattr(
        trainer, "_prepared_lora_exports", {}
    )
    try:
        owner, prepared = snapshots[export_id]
    except KeyError:
        raise ValueError(f"Unknown prepared LoRA export: {export_id!r}") from None
    if owner != owner_id:
        raise ValueError(f"LoRA export {export_id!r} belongs to another owner")
    snapshots.pop(export_id)
    return _save_lora_publish_inputs(output_dir, prepared)


def abort_lora_export(trainer: TrainerRank, export_id: str, *, owner_id: str) -> None:
    snapshots: dict[str, tuple[str, _PreparedLoraExport]] = getattr(
        trainer, "_prepared_lora_exports", {}
    )
    if (prepared := snapshots.get(export_id)) is not None and prepared[0] == owner_id:
        snapshots.pop(export_id)


def export_lora(trainer: TrainerRank, output_dir: str, checkpoint_name: str) -> int:
    from art.trainer_rank import _checkpoint

    group = _checkpoint._ensure_group(trainer)
    export_id = uuid.uuid4().hex
    owner_id = uuid.uuid4().hex
    revision, _timings = prepare_lora_export(
        trainer, export_id, checkpoint_name, owner_id=owner_id
    )
    error: BaseException | None = None
    try:
        if trainer.runtime.rank == 0:
            finish_lora_export(trainer, export_id, output_dir, owner_id=owner_id)
    except BaseException as exc:
        error = exc
    finally:
        abort_lora_export(trainer, export_id, owner_id=owner_id)
    _checkpoint.raise_distributed(error, "export LoRA", group)
    return revision
