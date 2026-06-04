from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from megatron.core import parallel_state as ps
from pydantic import BaseModel, ConfigDict
import torch

from art.loss import LossInputs, shift_tensor
from art.megatron.context_parallel.runtime import prepare_cp_micro
from art.megatron.context_parallel.types import (
    ContextParallelConfig,
    DispatchedPackedTensors,
    ParallelTopology,
    PreparedMegatronBatch,
)
from art.megatron.shared_prefix_state import create_shared_prefix_state
from art.megatron.training.trace import (
    packed_sequence_token_uids,
    sft_sequence_token_uids,
)
from art.preprocessing.pack import PackedTensors


class CpBatchLookaheadState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pending_prepared_micro: PreparedMegatronBatch | None = None


class PreparedRLMicroInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_tokens: torch.Tensor
    model_input_pos: torch.Tensor
    model_labels: torch.Tensor
    attention_state: Any
    packed_seq_params: Any | None = None
    loss_inputs: LossInputs | DispatchedPackedTensors
    ref_logprobs: torch.Tensor | None = None
    local_token_uids: torch.Tensor | None = None


class PreparedSFTMicroInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_state: Any
    packed_seq_params: Any | None = None
    local_token_uids: torch.Tensor | None = None


@torch.no_grad()
def select_indexed_inputs(packed_tensors: PackedTensors, index: int) -> PackedTensors:
    def selected_tensor(value: torch.Tensor) -> torch.Tensor:
        selected = value[index : index + 1]
        # File-backed slices keep the mmap alive and can make job cleanup fail.
        if getattr(selected.untyped_storage(), "filename", None):
            return selected.clone()
        return selected

    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: selected_tensor(value)
            for key, value in packed_tensors.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _clone_packed_tensors(inputs: PackedTensors) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value.clone()
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _zero_contribution_inputs(template: PackedTensors) -> PackedTensors:
    dummy = _clone_packed_tensors(template)
    dummy["assistant_mask"].zero_()
    return dummy


@torch.no_grad()
def _clone_sft_tensors(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


@torch.no_grad()
def _zero_contribution_sft_inputs(
    template: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    dummy = _clone_sft_tensors(template)
    dummy["labels"].fill_(-100)
    return dummy


def resolve_global_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int | None,
) -> int:
    dp_world_size = ps.get_data_parallel_world_size()
    if global_grad_accumulation_sequences is None:
        return dp_world_size
    return global_grad_accumulation_sequences


def resolve_local_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int | None,
) -> int:
    resolved_global_grad_accumulation_sequences = (
        resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences=global_grad_accumulation_sequences
        )
    )
    dp_world_size = ps.get_data_parallel_world_size()
    if (
        resolved_global_grad_accumulation_sequences <= 0
        or resolved_global_grad_accumulation_sequences % dp_world_size != 0
    ):
        raise RuntimeError(
            "Invalid global grad accumulation / DP world size combination: "
            f"global_grad_accumulation_sequences={resolved_global_grad_accumulation_sequences}, "
            f"dp_world_size={dp_world_size}"
        )
    return resolved_global_grad_accumulation_sequences // dp_world_size


def build_micro_sample_indices(
    step_index: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int | None,
) -> list[int | None]:
    dp_rank = ps.get_data_parallel_rank()
    resolved_global_grad_accumulation_sequences = (
        resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences=global_grad_accumulation_sequences
        )
    )
    dp_world_size = ps.get_data_parallel_world_size()
    local_grad_accumulation_sequences = resolve_local_grad_accumulation_sequences(
        global_grad_accumulation_sequences=resolved_global_grad_accumulation_sequences,
    )
    base_global_sample_index = step_index * resolved_global_grad_accumulation_sequences
    global_step_indices: list[int | None] = []
    for offset in range(resolved_global_grad_accumulation_sequences):
        global_sample_index = base_global_sample_index + offset
        global_step_indices.append(
            global_sample_index if global_sample_index < num_sequences else None
        )
    return [
        global_step_indices[offset * dp_world_size + dp_rank]
        for offset in range(local_grad_accumulation_sequences)
    ]


def select_micro_inputs(
    packed_tensors: PackedTensors,
    sample_indices: list[int | None],
    zero_template: PackedTensors,
) -> list[PackedTensors]:
    return [
        _clone_packed_tensors(zero_template)
        if sample_index is None
        else select_indexed_inputs(packed_tensors, sample_index)
        for sample_index in sample_indices
    ]


def select_sft_micro_inputs(
    trajectory_tensors: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    zero_template: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    return [
        _clone_sft_tensors(zero_template)
        if sample_index is None
        else _clone_sft_tensors(trajectory_tensors[sample_index])
        for sample_index in sample_indices
    ]


def _select_next_step_first_micro(
    *,
    packed_tensors: PackedTensors,
    zero_template: PackedTensors,
    step_index: int,
    num_steps: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int,
) -> PackedTensors | None:
    next_step_index = step_index + 1
    if next_step_index >= num_steps:
        return None
    next_micro_indices = build_micro_sample_indices(
        step_index=next_step_index,
        num_sequences=num_sequences,
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    return select_micro_inputs(
        packed_tensors,
        [next_micro_indices[0]],
        zero_template,
    )[0]


def _move_inputs_to_device(inputs: PackedTensors, device: torch.device) -> None:
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)  # type: ignore[index]


def _count_trainable_tokens(inputs: LossInputs | DispatchedPackedTensors) -> float:
    assistant_mask = inputs.align_inputs().assistant_mask
    return float(assistant_mask.sum().item())


def _local_trainable_token_count_tensor(
    micro_inputs: list[LossInputs | DispatchedPackedTensors],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(_count_trainable_tokens(micro) for micro in micro_inputs)
    return torch.tensor([local_token_total], device=device, dtype=torch.float32)


def _causal_attention_state(
    seq_len: int,
    device: torch.device,
    *,
    build_gdn_execution_spec: bool,
    attention_head_dim: int | None = None,
    attention_value_head_dim: int | None = None,
) -> Any:
    group_ids = torch.zeros((1, seq_len), dtype=torch.int64, device=device)
    parent_ids = torch.zeros_like(group_ids)
    return create_shared_prefix_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
        build_gdn_execution_spec=build_gdn_execution_spec,
        attention_head_dim=attention_head_dim,
        attention_value_head_dim=attention_value_head_dim,
    )


def _next_micro_lookahead(
    micro_inputs: list[Any],
    micro_order: int,
    trailing_micro: Any | None = None,
) -> Any | None:
    next_micro_order = micro_order + 1
    if next_micro_order < len(micro_inputs):
        return micro_inputs[next_micro_order]
    return trailing_micro


def _prepare_dense_rl_micro(
    micro: PackedTensors,
    *,
    device: torch.device,
    provider: Any,
    model_support_handler: Any,
    ref_logprobs: torch.Tensor | None,
) -> PreparedRLMicroInputs:
    _move_inputs_to_device(micro, device)
    shifted_labels = shift_tensor(micro["tokens"], -100)
    shifted_assistant_mask = shift_tensor(micro["assistant_mask"], False)
    shifted_labels = torch.where(
        shifted_assistant_mask,
        shifted_labels,
        torch.full_like(shifted_labels, -100),
    )
    return PreparedRLMicroInputs(
        model_tokens=micro["tokens"],
        model_input_pos=micro["input_pos"],
        model_labels=shifted_labels,
        attention_state=create_shared_prefix_state(
            group_ids=micro["group_ids"],
            parent_ids=micro["parent_ids"],
            build_gdn_execution_spec=bool(
                getattr(model_support_handler, "build_gdn_execution_spec", False)
            ),
            attention_head_dim=getattr(provider, "kv_channels", None),
            attention_value_head_dim=getattr(provider, "kv_channels", None),
        ),
        loss_inputs=LossInputs(inputs=micro),
        ref_logprobs=ref_logprobs,
        local_token_uids=packed_sequence_token_uids(micro, device=device),
    )


def _prepare_rl_cp_micro_full(
    micro: PackedTensors,
    *,
    device: torch.device,
    topology: ParallelTopology,
    model_support_handler: Any,
    trace_token_uids: bool,
    ref_logprobs: torch.Tensor | None,
) -> PreparedMegatronBatch:
    """Prepare RL CP inputs without moving planning metadata to CUDA first.

    CP lookahead relies on the CPU running this after backward has enqueued GPU
    work. Moving the full packed micro to CUDA before planning forces later D2H
    metadata reads and collapses that overlap.
    """
    return prepare_cp_micro(
        micro=micro,
        topology=topology,
        config=ContextParallelConfig(),
        cp_group=ps.get_context_parallel_group(check_initialized=False),
        cp_rank=ps.get_context_parallel_rank(),
        build_gdn_execution_spec=bool(
            getattr(model_support_handler, "build_gdn_execution_spec", False)
        ),
        trace_token_uids=trace_token_uids,
        target_device=device,
        ref_logprobs=ref_logprobs,
    )


def _prepared_rl_micro_from_cp_batch(
    prepared: PreparedMegatronBatch,
    *,
    ref_logprobs: torch.Tensor | None,
) -> PreparedRLMicroInputs:
    return PreparedRLMicroInputs(
        model_tokens=prepared.tensors.tokens,
        model_input_pos=prepared.tensors.input_pos,
        model_labels=prepared.tensors.labels,
        attention_state=prepared.attention_state,
        packed_seq_params=prepared.packed_seq_params,
        loss_inputs=prepared.tensors,
        ref_logprobs=prepared.tensors.ref_logprobs
        if ref_logprobs is not None
        else None,
        local_token_uids=prepared.tensors.token_uids,
    )


def _empty_new_logprobs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if int(labels.numel()) != 0:
        raise ValueError("empty-logprob path requires empty local labels")
    if logits.ndim < 3 or int(logits.shape[-1]) == 0:
        raise ValueError(
            f"expected empty local logits [B, S, V], got {tuple(logits.shape)}"
        )
    candidate = logits[..., 0]
    if tuple(candidate.shape) == tuple(labels.shape):
        return candidate
    candidate = candidate.transpose(0, 1).contiguous()
    if tuple(candidate.shape) != tuple(labels.shape):
        raise ValueError(
            "empty local logits shape must match labels after removing vocab dim, "
            f"got logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    return candidate


def _prepare_current_rl_micro(
    micro: PackedTensors,
    *,
    device: torch.device,
    topology: ParallelTopology,
    provider: Any,
    model_support_handler: Any,
    ref_logprobs: torch.Tensor | None,
    trace_token_uids: bool,
    pending_prepared_micro: PreparedMegatronBatch | None,
) -> tuple[PreparedRLMicroInputs, PreparedMegatronBatch | None]:
    if int(topology.cp) <= 1:
        return (
            _prepare_dense_rl_micro(
                micro,
                device=device,
                provider=provider,
                model_support_handler=model_support_handler,
                ref_logprobs=ref_logprobs,
            ),
            pending_prepared_micro,
        )
    prepared = pending_prepared_micro
    if prepared is None:
        prepared = _prepare_rl_cp_micro_full(
            micro,
            device=device,
            topology=topology,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
            ref_logprobs=ref_logprobs,
        )
    return _prepared_rl_micro_from_cp_batch(prepared, ref_logprobs=ref_logprobs), None


def _prepare_next_rl_cp_micro(
    next_micro: PackedTensors | None,
    *,
    device: torch.device,
    topology: ParallelTopology,
    model_support_handler: Any,
    trace_token_uids: bool,
    ref_logprobs: torch.Tensor | None = None,
) -> PreparedMegatronBatch | None:
    if next_micro is None or int(topology.cp) <= 1:
        return None
    return _prepare_rl_cp_micro_full(
        next_micro,
        device=device,
        topology=topology,
        model_support_handler=model_support_handler,
        trace_token_uids=trace_token_uids,
        ref_logprobs=ref_logprobs,
    )


def _count_sft_trainable_tokens(
    inputs: dict[str, torch.Tensor] | PreparedSFTMicroInputs,
) -> float:
    if isinstance(inputs, PreparedSFTMicroInputs):
        return float(inputs.loss_mask.sum().item())
    attention_mask = inputs["attention_mask"].reshape(-1)
    actual_len = int(attention_mask.sum().item())
    labels = inputs["labels"].reshape(-1)[:actual_len].unsqueeze(0)
    shifted_labels = shift_tensor(labels, -100)
    return float((shifted_labels != -100).sum().item())


def _local_trainable_sft_token_count_tensor(
    micro_inputs: Sequence[dict[str, torch.Tensor] | PreparedSFTMicroInputs],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(
        _count_sft_trainable_tokens(micro) for micro in micro_inputs
    )
    return torch.tensor([local_token_total], device=device, dtype=torch.float32)


def _prepare_dense_sft_micro(
    micro: dict[str, torch.Tensor],
    *,
    device: torch.device,
    provider: Any,
    model_support_handler: Any,
) -> PreparedSFTMicroInputs:
    attention_mask = micro["attention_mask"].reshape(-1)
    seq_len = max(int(attention_mask.sum().item()), 1)
    input_ids = micro["input_ids"].reshape(-1)[:seq_len].unsqueeze(0).to(device)
    labels = micro["labels"].reshape(-1)[:seq_len].unsqueeze(0).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    shifted_labels = shift_tensor(labels, -100)
    loss_mask = shifted_labels != -100
    return PreparedSFTMicroInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=shifted_labels,
        loss_mask=loss_mask,
        attention_state=_causal_attention_state(
            seq_len,
            device,
            build_gdn_execution_spec=bool(
                getattr(model_support_handler, "build_gdn_execution_spec", False)
            ),
            attention_head_dim=getattr(provider, "kv_channels", None),
            attention_value_head_dim=getattr(provider, "kv_channels", None),
        ),
        local_token_uids=sft_sequence_token_uids(micro, device=device)[
            :, : int(input_ids.shape[1])
        ],
    )


def _sft_inputs_to_sparse_packed_tensors(
    inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> PackedTensors:
    input_ids = inputs["input_ids"].reshape(-1)
    attention_mask = inputs["attention_mask"].reshape(-1)
    labels = inputs["labels"].reshape(-1)
    actual_len = max(int(attention_mask.sum().item()), 1)
    total_tokens = int(input_ids.numel())

    group_ids = torch.full((1, total_tokens), -1, device=device, dtype=torch.long)
    parent_ids = torch.full((1, total_tokens), -1, device=device, dtype=torch.long)
    group_ids[:, :actual_len] = 0
    parent_ids[:, :actual_len] = 0

    assistant_mask = (labels != -100).unsqueeze(0).to(device=device, dtype=torch.bool)
    return PackedTensors(
        tokens=input_ids.unsqueeze(0).to(device=device, dtype=torch.long),
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=torch.arange(total_tokens, device=device, dtype=torch.long).unsqueeze(
            0
        ),
        assistant_mask=assistant_mask,
        logprobs=torch.full(
            (1, total_tokens),
            float("nan"),
            device=device,
            dtype=torch.float32,
        ),
        advantages=torch.zeros((1, total_tokens), device=device, dtype=torch.float32),
        weights=assistant_mask.to(dtype=torch.float32),
        pixel_values=[None],
        image_grid_thw=[None],
        moe_routing_replay=None,
    )


def _prepare_sft_cp_micro_full(
    micro: dict[str, torch.Tensor],
    *,
    device: torch.device,
    topology: ParallelTopology,
    model_support_handler: Any,
    trace_token_uids: bool,
) -> PreparedMegatronBatch:
    """Prepare SFT CP inputs through the same CPU-planning boundary as RL CP.

    The synthetic sparse-packed metadata is constructed on CPU and only the
    rank-local dispatched tensors are moved to `device`. Constructing it on CUDA
    would make shared-prefix planning read metadata back from the GPU.
    """
    sparse_micro = _sft_inputs_to_sparse_packed_tensors(
        micro,
        device=torch.device("cpu"),
    )
    return prepare_cp_micro(
        micro=sparse_micro,
        topology=topology,
        config=ContextParallelConfig(),
        cp_group=ps.get_context_parallel_group(check_initialized=False),
        cp_rank=ps.get_context_parallel_rank(),
        build_gdn_execution_spec=bool(
            getattr(model_support_handler, "build_gdn_execution_spec", False)
        ),
        trace_token_uids=trace_token_uids,
        target_device=device,
    )


def _prepared_sft_micro_from_cp_batch(
    prepared: PreparedMegatronBatch,
) -> PreparedSFTMicroInputs:
    loss_mask = prepared.tensors.assistant_mask
    return PreparedSFTMicroInputs(
        input_ids=prepared.tensors.tokens,
        position_ids=prepared.tensors.input_pos,
        labels=prepared.tensors.labels.masked_fill(~loss_mask, -100),
        loss_mask=loss_mask,
        attention_state=prepared.attention_state,
        packed_seq_params=prepared.packed_seq_params,
        local_token_uids=prepared.tensors.token_uids,
    )


def _prepare_current_sft_micro(
    micro: dict[str, torch.Tensor],
    *,
    device: torch.device,
    topology: ParallelTopology,
    provider: Any,
    model_support_handler: Any,
    trace_token_uids: bool,
    pending_prepared_micro: PreparedMegatronBatch | None,
) -> tuple[PreparedSFTMicroInputs, PreparedMegatronBatch | None]:
    if int(topology.cp) <= 1:
        return (
            _prepare_dense_sft_micro(
                micro,
                device=device,
                provider=provider,
                model_support_handler=model_support_handler,
            ),
            pending_prepared_micro,
        )
    prepared = pending_prepared_micro
    if prepared is None:
        prepared = _prepare_sft_cp_micro_full(
            micro,
            device=device,
            topology=topology,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
        )
    return _prepared_sft_micro_from_cp_batch(prepared), None


def _prepare_next_sft_cp_micro(
    next_micro: dict[str, torch.Tensor] | None,
    *,
    device: torch.device,
    topology: ParallelTopology,
    model_support_handler: Any,
    trace_token_uids: bool,
) -> PreparedMegatronBatch | None:
    if next_micro is None or int(topology.cp) <= 1:
        return None
    return _prepare_sft_cp_micro_full(
        next_micro,
        device=device,
        topology=topology,
        model_support_handler=model_support_handler,
        trace_token_uids=trace_token_uids,
    )
