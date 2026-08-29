from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from megatron.core import parallel_state as ps
from pydantic import BaseModel, ConfigDict
import torch

from art.loss import LossInputs, shift_tensor
from art.megatron.context_parallel.runtime import (
    context_parallel_rank_model_token_counts,
    prepare_cp_micro,
    preplan_megatron_context_parallel_state,
)
from art.megatron.context_parallel.types import (
    ContextParallelConfig,
    CpBlockMaskVariant,
    DispatchedPackedTensors,
    ParallelTopology,
    PreparedMegatronBatch,
    TrainingMicrobatchWorkload,
)
from art.megatron.flex_attn.compiled import flash_sparse_block_size_for_head_dim
from art.megatron.prefix_tree import parse_prefix_tree
from art.megatron.prefix_tree_state import create_prefix_tree_state
from art.megatron.selective_lm_head import LmHeadTokenSelection
from art.megatron.training.trace import (
    packed_sequence_token_uids,
    sft_sequence_token_uids,
)
from art.preprocessing.pack import PackedTensors


class CpBatchLookaheadState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pending_prepared_micro: PreparedMegatronBatch | None = None


class CpBatchPreplanner(BaseModel):
    """One trainer rank's immutable CPU planning context."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    topology: ParallelTopology
    config: ContextParallelConfig
    cp_rank: int
    build_gdn_execution_spec: bool
    gdn_planner_config: Any | None = None

    @classmethod
    def from_runtime(
        cls, runtime: Any, *, device: torch.device
    ) -> "CpBatchPreplanner | None":
        from art.megatron.train import _infer_parallel_topology

        topology = _infer_parallel_topology(runtime.model)
        if int(topology.cp) <= 1:
            return None
        handler = runtime.model_support_handler
        return cls(
            topology=topology,
            config=_context_parallel_config_for_provider(
                runtime.provider, device, handler
            ),
            cp_rank=int(ps.get_context_parallel_rank()),
            build_gdn_execution_spec=bool(
                getattr(handler, "build_gdn_execution_spec", False)
            ),
            gdn_planner_config=_gdn_planner_config_for_provider(
                runtime.provider, handler
            ),
        )

    def preplan(
        self,
        packed_tensors: PackedTensors,
        *,
        global_grad_accumulation_sequences: int | None,
    ) -> int:
        num_sequences, sequence_length = map(int, packed_tensors["tokens"].shape)
        accumulation = resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences
        )
        sample_indices = {
            0 if index is None else index
            for step_index in range((num_sequences + accumulation - 1) // accumulation)
            for index in build_micro_sample_indices(
                step_index,
                num_sequences,
                accumulation,
            )
        }
        for index in sorted(sample_indices):
            preplan_megatron_context_parallel_state(
                group_ids=packed_tensors["group_ids"][index : index + 1],
                parent_ids=packed_tensors["parent_ids"][index : index + 1],
                original_seq_len=sequence_length,
                topology=self.topology,
                config=self.config,
                cp_rank=self.cp_rank,
                build_gdn_execution_spec=self.build_gdn_execution_spec,
                gdn_planner_config=self.gdn_planner_config,
            )
        return len(sample_indices)


class PreparedRLMicroInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_tokens: torch.Tensor
    model_input_pos: torch.Tensor
    model_labels: torch.Tensor
    attention_state: Any
    packed_seq_params: Any | None = None
    loss_inputs: LossInputs | DispatchedPackedTensors
    lm_head_selection: LmHeadTokenSelection
    ref_logprobs: torch.Tensor | None = None
    target_tokens: torch.Tensor | None = None
    loss_weights: torch.Tensor | None = None
    behavior_logprobs: torch.Tensor | None = None
    token_advantages: torch.Tensor | None = None
    local_token_uids: torch.Tensor | None = None
    workload: TrainingMicrobatchWorkload


class PreparedSFTMicroInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    lm_head_selection: LmHeadTokenSelection
    attention_state: Any
    packed_seq_params: Any | None = None
    local_token_uids: torch.Tensor | None = None
    workload: TrainingMicrobatchWorkload


def _map_packed_tensors(
    inputs: PackedTensors,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> PackedTensors:
    return cast(
        PackedTensors,
        {
            key: transform(value) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        },
    )


@torch.no_grad()
def select_indexed_inputs(packed_tensors: PackedTensors, index: int) -> PackedTensors:
    def selected_tensor(value: torch.Tensor) -> torch.Tensor:
        selected = value[index : index + 1]
        # File-backed slices keep the mmap alive and can make job cleanup fail.
        if getattr(selected.untyped_storage(), "filename", None):
            return selected.clone()
        return selected

    selected = _map_packed_tensors(packed_tensors, selected_tensor)
    selected["moe_routing_replay"] = None
    selected["pixel_values"] = [None]
    selected["image_grid_thw"] = [None]
    return selected


@torch.no_grad()
def _clone_packed_tensors(inputs: PackedTensors) -> PackedTensors:
    cloned = _map_packed_tensors(inputs, torch.Tensor.clone)
    cloned["moe_routing_replay"] = None
    cloned["pixel_values"] = [None]
    cloned["image_grid_thw"] = [None]
    return cloned


@torch.no_grad()
def _zero_contribution_inputs(template: PackedTensors) -> PackedTensors:
    dummy = _clone_packed_tensors(template)
    dummy["assistant_mask"].zero_()
    if "target_tokens" in dummy:
        dummy["target_tokens"].fill_(-100)
        dummy["loss_weights"].zero_()
        dummy["behavior_logprobs"].zero_()
        dummy["token_advantages"].zero_()
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
    return [
        indices[dp_rank]
        for indices in build_micro_sample_indices_by_dp_rank(
            step_index=step_index,
            num_sequences=num_sequences,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
    ]


def build_micro_sample_indices_by_dp_rank(
    step_index: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int | None,
) -> list[list[int | None]]:
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
        global_step_indices[offset * dp_world_size : (offset + 1) * dp_world_size]
        for offset in range(local_grad_accumulation_sequences)
    ]


def build_rl_hybridep_token_counts(
    *,
    packed_tensors: PackedTensors,
    step_index: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int | None,
    topology: ParallelTopology,
    provider: Any,
    model_support_handler: Any,
) -> list[int]:
    sample_rows = build_micro_sample_indices_by_dp_rank(
        step_index=step_index,
        num_sequences=num_sequences,
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    sequence_length = int(packed_tensors["tokens"].shape[1])
    if int(topology.cp) <= 1:
        return [sequence_length for _ in sample_rows]

    config = _context_parallel_config_for_provider(
        provider,
        torch.device("cuda", torch.cuda.current_device()),
        model_support_handler,
    )
    build_gdn = bool(getattr(model_support_handler, "build_gdn_execution_spec", False))
    gdn_planner_config = _gdn_planner_config_for_provider(
        provider, model_support_handler
    )

    def rank_counts(sample_index: int | None) -> tuple[int, ...]:
        index = 0 if sample_index is None else sample_index
        return context_parallel_rank_model_token_counts(
            group_ids=packed_tensors["group_ids"][index : index + 1],
            parent_ids=packed_tensors["parent_ids"][index : index + 1],
            topology=topology,
            config=config,
            original_seq_len=sequence_length,
            build_gdn_execution_spec=build_gdn,
            gdn_planner_config=gdn_planner_config,
        )

    return [
        max(count for sample_index in indices for count in rank_counts(sample_index))
        for indices in sample_rows
    ]


def build_sft_hybridep_token_counts(
    *,
    trajectory_tensors: list[dict[str, torch.Tensor]],
    step_index: int,
    global_grad_accumulation_sequences: int | None,
    topology: ParallelTopology,
    provider: Any,
    model_support_handler: Any,
) -> list[int]:
    sample_rows = build_micro_sample_indices_by_dp_rank(
        step_index=step_index,
        num_sequences=len(trajectory_tensors),
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )

    def sample(sample_index: int | None) -> dict[str, torch.Tensor]:
        return trajectory_tensors[0 if sample_index is None else sample_index]

    if int(topology.cp) <= 1:
        return [
            max(int(sample(index)["input_ids"].numel()) for index in indices)
            for indices in sample_rows
        ]

    config = _context_parallel_config_for_provider(
        provider,
        torch.device("cuda", torch.cuda.current_device()),
        model_support_handler,
    )
    build_gdn = bool(getattr(model_support_handler, "build_gdn_execution_spec", False))
    gdn_planner_config = _gdn_planner_config_for_provider(
        provider, model_support_handler
    )

    def rank_counts(sample_index: int | None) -> tuple[int, ...]:
        sparse = _sft_inputs_to_sparse_packed_tensors(
            sample(sample_index), device=torch.device("cpu")
        )
        return context_parallel_rank_model_token_counts(
            group_ids=sparse["group_ids"],
            parent_ids=sparse["parent_ids"],
            topology=topology,
            config=config,
            original_seq_len=int(sparse["tokens"].shape[1]),
            build_gdn_execution_spec=build_gdn,
            gdn_planner_config=gdn_planner_config,
        )

    return [
        max(count for sample_index in indices for count in rank_counts(sample_index))
        for indices in sample_rows
    ]


def select_micro_inputs(
    packed_tensors: PackedTensors,
    sample_indices: list[int | None],
    zero_template: PackedTensors,
) -> list[PackedTensors]:
    return [
        (
            _clone_packed_tensors(zero_template)
            if sample_index is None
            else select_indexed_inputs(packed_tensors, sample_index)
        )
        for sample_index in sample_indices
    ]


def select_sft_micro_inputs(
    trajectory_tensors: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    zero_template: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    return [
        (
            _clone_sft_tensors(zero_template)
            if sample_index is None
            else _clone_sft_tensors(trajectory_tensors[sample_index])
        )
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
    inputs.update(_map_packed_tensors(inputs, lambda tensor: tensor.to(device)))


def _count_trainable_tokens(inputs: LossInputs | DispatchedPackedTensors) -> float:
    assistant_mask = inputs.align_inputs().assistant_mask
    return float(assistant_mask.sum().item())


def _local_trainable_token_count_tensor(
    micro_inputs: list[LossInputs | DispatchedPackedTensors],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(_count_trainable_tokens(micro) for micro in micro_inputs)
    return torch.tensor(local_token_total, device=device, dtype=torch.int)


def _art_flex_sliding_windows(provider: Any) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            int(window) for window in getattr(provider, "art_flex_sliding_windows", ())
        )
    )


def _art_flex_cp_block_mask_variants(
    provider: Any,
    device: torch.device,
) -> tuple[CpBlockMaskVariant, ...]:
    head_dims = getattr(provider, "art_flex_head_dims_by_window", {})
    value_head_dims = getattr(provider, "art_flex_value_head_dims_by_window", {})
    default_head_dim = getattr(provider, "kv_channels", None)
    variants: list[CpBlockMaskVariant] = []
    seen: set[tuple[int | None, tuple[int, int]]] = set()
    for window in (None, *_art_flex_sliding_windows(provider)):
        head_dim = (
            head_dims.get(window, default_head_dim)
            if isinstance(head_dims, dict)
            else default_head_dim
        )
        value_head_dim = (
            value_head_dims.get(window, head_dim)
            if isinstance(value_head_dims, dict)
            else head_dim
        )
        block_size = (
            (128, 128)
            if head_dim is None
            else flash_sparse_block_size_for_head_dim(
                head_dim=int(head_dim),
                head_dim_v=int(head_dim if value_head_dim is None else value_head_dim),
                device=device,
            )
        )
        key = (None if window is None else int(window), block_size)
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            CpBlockMaskVariant(
                sliding_window=None if window is None else int(window),
                block_size=block_size,
            )
        )
    return tuple(variants)


def _context_parallel_config_for_provider(
    provider: Any,
    device: torch.device,
    model_support_handler: Any,
) -> ContextParallelConfig:
    head_dim = getattr(provider, "kv_channels", None)
    if head_dim is None:
        return ContextParallelConfig(
            workload_profile=model_support_handler.context_parallel_workload_profile(
                provider
            )
        )
    return ContextParallelConfig(
        attention_sparse_block_size=flash_sparse_block_size_for_head_dim(
            head_dim=int(head_dim),
            head_dim_v=int(head_dim),
            device=device,
        ),
        workload_profile=model_support_handler.context_parallel_workload_profile(
            provider
        ),
    )


def _causal_attention_state(
    seq_len: int,
    device: torch.device,
    *,
    provider: Any,
    model_support_handler: Any,
    sliding_windows: tuple[int, ...] = (),
    build_gdn_execution_spec: bool,
    attention_head_dim: int | None = None,
    attention_value_head_dim: int | None = None,
) -> Any:
    group_ids = torch.zeros((1, seq_len), dtype=torch.int64, device="cpu")
    parent_ids = torch.zeros_like(group_ids)
    return create_prefix_tree_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
        target_device=device,
        input_pos=torch.arange(seq_len, dtype=torch.int64).unsqueeze(0),
        sliding_windows=sliding_windows,
        build_gdn_execution_spec=build_gdn_execution_spec,
        model_support_handler=model_support_handler,
        attention_head_dim=attention_head_dim,
        attention_value_head_dim=attention_value_head_dim,
        gdn_planner_config=_gdn_planner_config_for_provider(
            provider,
            model_support_handler,
        ),
    )


def _gdn_planner_config_for_provider(
    provider: Any, model_support_handler: Any
) -> Any | None:
    if not bool(getattr(model_support_handler, "build_gdn_execution_spec", False)):
        return None
    from art.megatron.gdn.gdn_prefix_tree import GdnPlannerConfig

    return GdnPlannerConfig.from_provider(provider)


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
    attention_state = create_prefix_tree_state(
        group_ids=micro["group_ids"],
        parent_ids=micro["parent_ids"],
        target_device=device,
        input_pos=micro["input_pos"],
        sliding_windows=_art_flex_sliding_windows(provider),
        build_gdn_execution_spec=bool(
            getattr(model_support_handler, "build_gdn_execution_spec", False)
        ),
        model_support_handler=model_support_handler,
        attention_head_dim=getattr(provider, "kv_channels", None),
        attention_value_head_dim=getattr(provider, "kv_channels", None),
        gdn_planner_config=_gdn_planner_config_for_provider(
            provider,
            model_support_handler,
        ),
    )
    tokenized = "target_tokens" in micro
    shifted_assistant_mask = (
        micro["assistant_mask"]
        if tokenized
        else shift_tensor(micro["assistant_mask"], False)
    )
    shifted_labels = (
        micro["target_tokens"][..., 0]
        if tokenized
        else torch.where(
            shifted_assistant_mask,
            shift_tensor(micro["tokens"], -100),
            torch.full_like(micro["tokens"], -100),
        )
    )
    lm_head_selection = LmHeadTokenSelection.from_labels(
        shifted_labels,
        target_device=device,
    )
    workload = TrainingMicrobatchWorkload(
        logical_nonpadding_tokens=sum(
            row.valid_tokens
            for row in parse_prefix_tree(
                group_ids=micro["group_ids"], parent_ids=micro["parent_ids"]
            )
        ),
        loss_bearing_tokens=int(shifted_assistant_mask.sum().item()),
        executed_token_equivalents=int(micro["tokens"].numel()),
        nominal_schedule_capacity_tokens=int(micro["tokens"].numel()),
    )
    shifted_labels = shifted_labels.to(device)
    _move_inputs_to_device(micro, device)
    return PreparedRLMicroInputs(
        model_tokens=micro["tokens"],
        model_input_pos=micro["input_pos"],
        model_labels=shifted_labels,
        attention_state=attention_state,
        loss_inputs=LossInputs(inputs=micro),
        lm_head_selection=lm_head_selection,
        ref_logprobs=ref_logprobs,
        target_tokens=micro.get("target_tokens"),
        loss_weights=micro.get("loss_weights"),
        behavior_logprobs=micro.get("behavior_logprobs"),
        token_advantages=micro.get("token_advantages"),
        local_token_uids=packed_sequence_token_uids(micro, device=device),
        workload=workload,
    )


def _prepare_rl_cp_micro_full(
    micro: PackedTensors,
    *,
    device: torch.device,
    topology: ParallelTopology,
    provider: Any,
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
        config=_context_parallel_config_for_provider(
            provider, device, model_support_handler
        ),
        cp_group=ps.get_context_parallel_group(check_initialized=False),
        cp_rank=ps.get_context_parallel_rank(),
        build_gdn_execution_spec=bool(
            getattr(model_support_handler, "build_gdn_execution_spec", False)
        ),
        gdn_planner_config=_gdn_planner_config_for_provider(
            provider,
            model_support_handler,
        ),
        trace_token_uids=trace_token_uids,
        block_mask_variants=_art_flex_cp_block_mask_variants(provider, device),
        target_device=device,
        ref_logprobs=ref_logprobs,
        model_support_handler=model_support_handler,
        attention_head_dim=getattr(provider, "kv_channels", None),
        attention_value_head_dim=getattr(provider, "kv_channels", None),
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
        lm_head_selection=prepared.tensors.lm_head_selection,
        ref_logprobs=(
            prepared.tensors.ref_logprobs if ref_logprobs is not None else None
        ),
        target_tokens=prepared.tensors.target_tokens,
        loss_weights=prepared.tensors.loss_weights,
        behavior_logprobs=prepared.tensors.behavior_logprobs,
        token_advantages=prepared.tensors.token_advantages,
        local_token_uids=prepared.tensors.token_uids,
        workload=prepared.workload,
    )


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
            provider=provider,
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
    provider: Any,
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
        provider=provider,
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
    return torch.tensor(local_token_total, device=device, dtype=torch.int)


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
    labels = micro["labels"].reshape(-1)[:seq_len].unsqueeze(0)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    shifted_labels = shift_tensor(labels, -100)
    loss_mask = shifted_labels != -100
    workload = TrainingMicrobatchWorkload(
        logical_nonpadding_tokens=int(attention_mask.sum().item()),
        loss_bearing_tokens=int(loss_mask.sum().item()),
        executed_token_equivalents=seq_len,
        nominal_schedule_capacity_tokens=int(micro["input_ids"].numel()),
    )
    lm_head_selection = LmHeadTokenSelection.from_labels(
        shifted_labels,
        target_device=device,
    )
    shifted_labels = shifted_labels.to(device)
    loss_mask = loss_mask.to(device)
    return PreparedSFTMicroInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=shifted_labels,
        loss_mask=loss_mask,
        lm_head_selection=lm_head_selection,
        attention_state=_causal_attention_state(
            seq_len,
            device,
            sliding_windows=_art_flex_sliding_windows(provider),
            build_gdn_execution_spec=bool(
                getattr(model_support_handler, "build_gdn_execution_spec", False)
            ),
            provider=provider,
            model_support_handler=model_support_handler,
            attention_head_dim=getattr(provider, "kv_channels", None),
            attention_value_head_dim=getattr(provider, "kv_channels", None),
        ),
        local_token_uids=sft_sequence_token_uids(micro, device=device)[
            :, : int(input_ids.shape[1])
        ],
        workload=workload,
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
    provider: Any,
    model_support_handler: Any,
    trace_token_uids: bool,
) -> PreparedMegatronBatch:
    """Prepare SFT CP inputs through the same CPU-planning boundary as RL CP.

    The synthetic sparse-packed metadata is constructed on CPU and only the
    rank-local dispatched tensors are moved to `device`. Constructing it on CUDA
    would make prefix-tree planning read metadata back from the GPU.
    """
    sparse_micro = _sft_inputs_to_sparse_packed_tensors(
        micro,
        device=torch.device("cpu"),
    )
    return prepare_cp_micro(
        micro=sparse_micro,
        topology=topology,
        config=_context_parallel_config_for_provider(
            provider, device, model_support_handler
        ),
        cp_group=ps.get_context_parallel_group(check_initialized=False),
        cp_rank=ps.get_context_parallel_rank(),
        build_gdn_execution_spec=bool(
            getattr(model_support_handler, "build_gdn_execution_spec", False)
        ),
        gdn_planner_config=_gdn_planner_config_for_provider(
            provider,
            model_support_handler,
        ),
        trace_token_uids=trace_token_uids,
        block_mask_variants=_art_flex_cp_block_mask_variants(provider, device),
        target_device=device,
        model_support_handler=model_support_handler,
        attention_head_dim=getattr(provider, "kv_channels", None),
        attention_value_head_dim=getattr(provider, "kv_channels", None),
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
        lm_head_selection=prepared.tensors.lm_head_selection,
        attention_state=prepared.attention_state,
        packed_seq_params=prepared.packed_seq_params,
        local_token_uids=prepared.tensors.token_uids,
        workload=prepared.workload,
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
            provider=provider,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
        )
    return _prepared_sft_micro_from_cp_batch(prepared), None


def _prepare_next_sft_cp_micro(
    next_micro: dict[str, torch.Tensor] | None,
    *,
    device: torch.device,
    topology: ParallelTopology,
    provider: Any,
    model_support_handler: Any,
    trace_token_uids: bool,
) -> PreparedMegatronBatch | None:
    if next_micro is None or int(topology.cp) <= 1:
        return None
    return _prepare_sft_cp_micro_full(
        next_micro,
        device=device,
        topology=topology,
        provider=provider,
        model_support_handler=model_support_handler,
        trace_token_uids=trace_token_uids,
    )
