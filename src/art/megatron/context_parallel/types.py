from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from megatron.core.packed_seq_params import PackedSeqParams
from pydantic import BaseModel, ConfigDict
import torch

from .layout_index import TokenLayoutIndex
from .loss_inputs import ContextParallelLossInputs


class AttnMaskKind(str, Enum):
    FULL = "full"
    CAUSAL = "causal"


@dataclass(frozen=True)
class TokenRange:
    start: int
    end: int

    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class AttnSlice:
    q_range: TokenRange
    k_range: TokenRange
    mask_kind: AttnMaskKind
    row_index: int
    family_index: int | None = None


@dataclass(frozen=True)
class PackedRowAttentionSpec:
    row_index: int
    valid_tokens: int
    slices: tuple[AttnSlice, ...]


@dataclass(frozen=True)
class PackedBatchAttentionSpec:
    rows: tuple[PackedRowAttentionSpec, ...]


@dataclass(frozen=True)
class ContextParallelConfig:
    block_size: int = 128
    attention_sparse_block_size: tuple[int, int] | None = None
    planner_chunk_size: int = 512
    planner_chunk_budget_base: int = 128
    planner_chunk_budget_per_cp_rank: int = 16
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


@dataclass(frozen=True)
class ParallelTopology:
    tp: int = 1
    cp: int = 1
    dp: int = 1
    pp: int = 1
    sp: bool = False


@dataclass(frozen=True)
class KvFetchPlan:
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    send_ranges_by_peer: tuple[tuple[TokenRange, ...], ...]


@dataclass(frozen=True)
class DkvReducePlan:
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    recv_ranges_by_peer: tuple[tuple[TokenRange, ...], ...]


@dataclass(frozen=True)
class StagePlan:
    stage_index: int
    source_rank: int
    is_local_stage: bool
    slices: tuple[AttnSlice, ...]
    owner_local_q_ranges: tuple[TokenRange, ...]
    owner_local_k_ranges: tuple[TokenRange, ...]
    q_len: int
    k_len: int
    source_ranks: tuple[int, ...] = ()
    wave_index: int | None = None
    global_q_ranges: tuple[TokenRange, ...] = ()
    global_k_ranges: tuple[TokenRange, ...] = ()
    mask_metadata: "ExactMaskMetadata | None" = None
    remote_buffer_range: TokenRange | None = None
    kv_fetch_plan: KvFetchPlan | None = None
    dkv_reduce_plan: DkvReducePlan | None = None


@dataclass(frozen=True)
class RankRuntimePlan:
    rank: int
    original_seq_len: int
    token_layout_index: TokenLayoutIndex
    local_valid_lengths: tuple[int, ...]
    local_row_ranges: tuple[TokenRange | None, ...]
    stage_plans: tuple[StagePlan, ...]
    remote_dkv_reduce_plan: DkvReducePlan
    backward_stage_indices: tuple[int, ...] = ()


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


@dataclass
class ContextParallelExecutionCache:
    block_mask_context: Any | None = None
    block_masks: dict[Any, Any] = field(default_factory=dict)
    range_indices: dict[Any, torch.Tensor] = field(default_factory=dict)
    range_meta: dict[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = field(
        default_factory=dict
    )
    stage_execution_specs: dict[Any, "StageExecutionSpec"] = field(default_factory=dict)


@dataclass(frozen=True)
class StageExecutionSpec:
    q_len: int
    k_len: int
    compile_key: str
    mask_metadata: "ExactMaskMetadata | None" = None


@dataclass
class ArtContextParallelState:
    rank_plan: RankRuntimePlan
    cp_group: Any
    config: ContextParallelConfig
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    gdn_execution_spec: Any | None = None
    gdn_execution_plan: Any | None = None
    gdn_hidden_layout: str = "attention"
    gdn_input_layout: str | None = None
    gdn_output_layout: str | None = None
    gdn_attention_original_shape: tuple[int, int, int] | None = None
    gdn_attention_original_shapes: dict[int, tuple[int, int, int]] = field(
        default_factory=dict
    )
    gdn_attention_token_uids: torch.Tensor | None = None
    gdn_active_module: Any | None = None
    trace_token_uids: torch.Tensor | None = None
    execution_cache: ContextParallelExecutionCache = field(
        default_factory=ContextParallelExecutionCache
    )


@dataclass
class PreparedMegatronBatch:
    tensors: DispatchedPackedTensors
    attention_state: Any
    packed_seq_params: PackedSeqParams | None = None
    rank_plan: RankRuntimePlan | None = None
    pad_multiple: int = 1


@dataclass(frozen=True)
class FlexMaskSpec:
    q_len: int
    k_len: int
    block_size: int | tuple[int, int]
    slices: tuple[AttnSlice, ...]
    exact_mask: "ExactMaskMetadata"


@dataclass(frozen=True)
class ExactMaskMetadata:
    q_token_indices: torch.Tensor
    k_token_indices: torch.Tensor
    cache_key: str
