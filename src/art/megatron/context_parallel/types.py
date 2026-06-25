from __future__ import annotations

from enum import Enum
from typing import Any

from megatron.core.packed_seq_params import PackedSeqParams
from pydantic import BaseModel, ConfigDict, Field
import torch

from .layout_index import TokenLayoutIndex
from .loss_inputs import ContextParallelLossInputs


class AttnMaskKind(str, Enum):
    FULL = "full"
    CAUSAL = "causal"


class TokenRange(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int

    def size(self) -> int:
        return self.end - self.start

    def is_empty(self) -> bool:
        return self.end <= self.start


class AttnSlice(BaseModel):
    model_config = ConfigDict(frozen=True)

    q_range: TokenRange
    k_range: TokenRange
    mask_kind: AttnMaskKind
    row_index: int
    family_index: int | None = None


class PackedRowAttentionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_index: int
    valid_tokens: int
    slices: tuple[AttnSlice, ...]


class PackedBatchAttentionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: tuple[PackedRowAttentionSpec, ...]


class SharedPrefixBuilderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    ignore_padding_group_id: int = -1
    require_contiguous_group_runs: bool = True


class PlannerCpOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    cp_size: int
    block_size: int | None = None
    planner_chunk_size: int | None = None
    planner_chunk_budget_base: int | None = None
    planner_chunk_budget_per_cp_rank: int | None = None
    planner_assignment_strategy: str | None = None
    planner_stripe_group_size: int | None = None
    planner_max_search_steps: int | None = None
    planner_candidate_chunk_limit: int | None = None
    planner_max_remote_waves: int | None = None
    planner_stage_overhead_ms: float | None = None
    planner_comm_stage_overhead_ms: float | None = None
    planner_interval_overhead_ms: float | None = None
    planner_merge_q_token_ms: float | None = None
    planner_fetch_token_ms: float | None = None
    planner_reduce_token_ms: float | None = None
    planner_local_pair_ms: float | None = None
    planner_remote_pair_ms: float | None = None
    planner_local_backward_pair_ms: float | None = None
    planner_remote_backward_pair_ms: float | None = None
    planner_remote_stage_token_floor: int | None = None
    planner_remote_stage_pair_floor: int | None = None
    planner_remote_stage_underfill_ms: float | None = None
    planner_tuned_backend: str | None = None
    planner_tuned_hardware: str | None = None
    planner_tuned_cp_sizes: tuple[int, ...] | None = None


class ContextParallelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_size: int = 128
    attention_sparse_block_size: tuple[int, int] | None = None
    planner_chunk_size: int = 512
    planner_chunk_budget_base: int = 128
    planner_chunk_budget_per_cp_rank: int = 16
    planner_assignment_strategy: str = "search"
    planner_stripe_group_size: int = 16
    planner_max_search_steps: int = 8
    planner_candidate_chunk_limit: int = 8
    planner_max_remote_waves: int = 4
    planner_stage_overhead_ms: float = 0.287151
    planner_comm_stage_overhead_ms: float = 0.143576
    planner_interval_overhead_ms: float = 0.11486
    planner_merge_q_token_ms: float = 0.00011486
    planner_fetch_token_ms: float = 0.000287151
    planner_reduce_token_ms: float = 0.000287151
    planner_local_pair_ms: float = 0.000000045944
    planner_remote_pair_ms: float = 0.000000048816
    planner_local_backward_pair_ms: float = 0.000000137832
    planner_remote_backward_pair_ms: float = 0.000000149318
    planner_remote_stage_token_floor: int = 4096
    planner_remote_stage_pair_floor: int = 4_000_000
    planner_remote_stage_underfill_ms: float = 0.287151
    planner_tuned_backend: str | None = "art_context_parallel"
    planner_tuned_hardware: str | None = "NVIDIA H200"
    planner_tuned_cp_sizes: tuple[int, ...] = (2,)
    planner_cp_overrides: tuple[PlannerCpOverride, ...] = ()


class ParallelTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int = 1
    cp: int = 1
    dp: int = 1
    pp: int = 1
    sp: bool = False


class ContextParallelRuntimeKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    topology: ParallelTopology
    config: ContextParallelConfig
    row_signatures: tuple[str, ...]


class KvFetchPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    send_ranges_by_peer: tuple[tuple[TokenRange, ...], ...]


class DkvReducePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    recv_ranges_by_peer: tuple[tuple[TokenRange, ...], ...]


class StagePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_index: int
    source_rank: int
    source_ranks: tuple[int, ...] = ()
    is_local_stage: bool
    wave_index: int | None = None
    slices: tuple[AttnSlice, ...]
    global_q_ranges: tuple[TokenRange, ...] = ()
    global_k_ranges: tuple[TokenRange, ...] = ()
    owner_local_q_ranges: tuple[TokenRange, ...]
    owner_local_k_ranges: tuple[TokenRange, ...]
    mask_metadata: "ExactMaskMetadata | None" = None
    remote_buffer_range: TokenRange | None = None
    q_len: int
    k_len: int
    kv_fetch_plan: KvFetchPlan | None = None
    dkv_reduce_plan: DkvReducePlan | None = None


class RankRuntimePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    original_seq_len: int
    token_layout_index: TokenLayoutIndex
    local_valid_lengths: tuple[int, ...]
    local_row_ranges: tuple[TokenRange | None, ...]
    local_token_count: int
    stage_plans: tuple[StagePlan, ...]
    backward_stage_indices: tuple[int, ...] = ()
    remote_kv_fetch_plan: KvFetchPlan
    remote_dkv_reduce_plan: DkvReducePlan


class ContextParallelRuntimePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    topology: ParallelTopology
    config: ContextParallelConfig
    token_layout_index: TokenLayoutIndex
    rank_plans: tuple[RankRuntimePlan, ...]


class DispatchedPackedTensors(ContextParallelLossInputs):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: torch.Tensor
    labels: torch.Tensor
    input_pos: torch.Tensor
    assistant_mask: torch.Tensor
    group_ids: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    valid_lengths: tuple[int, ...]
    original_logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    loss_all_reduce_group: Any | None = None
    token_uids: torch.Tensor | None = None


class ContextParallelExecutionCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    block_masks: dict[Any, Any] = Field(default_factory=dict)
    range_indices: dict[Any, torch.Tensor] = Field(default_factory=dict)
    range_meta: dict[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = Field(
        default_factory=dict
    )
    stage_execution_specs: dict[Any, "StageExecutionSpec"] = Field(default_factory=dict)


class CpBlockMaskVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    sliding_window: int | None = None
    block_size: tuple[int, int]


class StageExecutionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    q_len: int
    k_len: int
    compile_key: str
    mask_metadata: "ExactMaskMetadata | None" = None


class PlannerProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_backend: str
    runtime_hardware: str | None = None
    runtime_cp_size: int
    tuned_backend: str | None = None
    tuned_hardware: str | None = None
    tuned_cp_sizes: tuple[int, ...] = ()
    backend_match: bool
    hardware_match: bool
    cp_size_match: bool
    using_best_effort: bool
    warning_message: str | None = None
    warning_emitted: bool = False


class ArtContextParallelState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime_key: ContextParallelRuntimeKey
    rank_plan: RankRuntimePlan
    cp_group: Any
    config: ContextParallelConfig
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    input_pos: torch.Tensor
    block_mask_variants: tuple[CpBlockMaskVariant, ...] = ()
    gdn_execution_spec: Any | None = None
    gdn_execution_plan: Any | None = None
    gdn_hidden_layout: str = "attention"
    gdn_input_layout: str | None = None
    gdn_output_layout: str | None = None
    gdn_attention_original_shape: tuple[int, int, int] | None = None
    gdn_attention_original_shapes: dict[int, tuple[int, int, int]] = Field(
        default_factory=dict
    )
    gdn_attention_token_uids: torch.Tensor | None = None
    gdn_active_module: Any | None = None
    planner_provenance: PlannerProvenance
    trace_token_uids: torch.Tensor | None = None
    execution_cache: ContextParallelExecutionCache = Field(
        default_factory=ContextParallelExecutionCache
    )


class PreparedMegatronBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensors: DispatchedPackedTensors
    packed_seq_params: PackedSeqParams | None = None
    attention_state: Any
    rank_plan: RankRuntimePlan | None = None
    pad_multiple: int = 1


class FlexMaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    q_len: int
    k_len: int
    block_size: int | tuple[int, int]
    slices: tuple[AttnSlice, ...]
    exact_mask: "ExactMaskMetadata"


class ExactMaskMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    q_token_indices: torch.Tensor
    k_token_indices: torch.Tensor
    cache_key: str
