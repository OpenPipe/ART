"""Private TrainerRank implementation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from dataclasses import field as dataclass_field
from functools import lru_cache, partial
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import struct
import threading
import time
import traceback
from types import MethodType, TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    NotRequired,
    Self,
    SupportsIndex,
    TypedDict,
    TypeVar,
    cast,
    overload,
)
import weakref

import torch
from torch.autograd.function import FunctionCtx
import torch.distributed as dist
from typing_extensions import TypeIs

from art.megatron.prefix_tree_packing import (
    PrefixTreePack,
    _local_position_pairs,
    estimate_prefix_tree_packed_tokens,
)
from art.trainer_rank import _gdn_memory, _planner_evidence, _planner_misses
from art.trainer_rank._backward_work import BackwardWork
from art.trainer_rank._backward_work import region as _backward_region
from art.trainer_rank._memory_policy import (
    ForwardMemoryCost,
    MemoryPlacement,
    host_memory_budget,
    local_rank_count,
    placement_cost,
)
from art.trainer_rank._options import (
    ForwardOptions,
    ResolvedForwardOptions,
    Unset,
    _Unset,
    resolve_forward_options,
)
from art.trainer_rank._planner_cost import (
    ModelGeometry,
    ParallelShape,
    select_scoring,
)
from art.trainer_rank._prefix_tree_materializer import materialize_prefix_tree_layout
from art.trainer_rank._prefix_tree_planner import (
    CanonicalPrefixTree,
    PrefixTreeLayout,
    canonical_token_rows_fingerprint,
)
from art.trainer_rank._rng import TrainerRNG, caller_group
from art.trainer_rank._telemetry import phase as _telemetry_phase
from art.trainer_rank._versions import (
    CheckpointVersion,
    CheckpointVersions,
    VersionedGradient,
)

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.packed_seq_params import PackedSeqParams

    from art.megatron.context_parallel.types import (
        ArtContextParallelState,
        ParallelTopology,
    )
    from art.megatron.lora import LoRASlotRef, LoRAVersion
    from art.megatron.prefix_tree_state import PrefixTreeAttentionState
    from art.megatron.train import TrainingRuntime
    from art.trainer_rank._checkpoint import (
        CustomOptimizerState,
        PreparedCheckpoint,
        PreparedCustomPayload,
        _FinalizedSave,
        _PreparedSave,
        _SnapshotSpill,
    )
    from art.trainer_rank._lora_export import _VllmLoraPublishInputs

    from ._heads import ModuleHandle


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    grad_clip_norm: float = 0.1


@dataclass(frozen=True)
class TopK:
    logprobs: torch.Tensor
    tokens: torch.Tensor


LogprobsT = TypeVar("LogprobsT", bound=torch.Tensor | None, covariant=True)
TopKT = TypeVar("TopKT", bound=TopK | None, covariant=True)
LogitsT = TypeVar("LogitsT", bound=torch.Tensor | None, covariant=True)
HiddenStatesT = TypeVar("HiddenStatesT", bound=torch.Tensor | None, covariant=True)
T = TypeVar("T")
ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)

_MEMORY_PROFILE_TRUST_GROWTH = 8

# Internal calibrated planner policy. These are deliberately not user knobs:
# prefix sharing, head chunking, and memory margins are planner decisions.
_MEMORY_SAFETY_FACTOR = 1.10
_MEMORY_RESERVE_FRACTION = 0.03
_HEAD_CHUNK_TOKENS = 512
# An unprofiled full-recompute gradient wave's first execution keeps two fixed
# 32 MiB transients live at its peak (Qwen3.6-35B-A3B CP2: the RoPE frequencies
# and a frozen linear's output, at 2k to 20k tokens); warm waves do not.
_COLD_RECOMPUTE_TRANSIENT_BYTES = 64 * 2**20
# The TP x SP traces (dense Qwen3.8-27B, TP2 and TP4) held a third: TE's 32 MiB
# cuBLAS workspace, allocated by the process's first wave.
_SEQUENCE_PARALLEL_COLD_TRANSIENT_BYTES = 3 * 32 * 2**20
_PLANNER_REFINEMENT_BUDGET = 2_000
_LAYOUT_SELECTION_CACHE_LIMIT = 64

# Test-only layout anchor forcing for paired acceptance measurement. Both
# variables must be set; the hook is inert in production.
logger = logging.getLogger(__name__)

_TEST_HOOKS_ENV = "ART_TRAINER_RANK_TEST_HOOKS"
_TEST_ANCHOR_ENV = "ART_TRAINER_RANK_TEST_ANCHOR"
# Test-only usable-memory cap (bytes) for split/decline acceptance cells.
_TEST_MEMORY_LIMIT_ENV = "ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES"

_U64_STRUCT = struct.Struct("<Q")

# Layout selected when the cost-optimal layout cannot be admitted: full sharing
# minimizes packed tokens, and its packed count is monotone in wave width, so
# it is the feasibility predicate the width search relies on.
_MEMORY_MINIMAL_ANCHOR = "full_sharing"
_CHECKPOINT_PREFETCH_EXECUTOR: tuple[int, ThreadPoolExecutor] | None = None
_CHECKPOINT_PREFETCH_EXECUTOR_LOCK = threading.Lock()


def _checkpoint_prefetch_executor() -> ThreadPoolExecutor:
    global _CHECKPOINT_PREFETCH_EXECUTOR
    pid = os.getpid()
    with _CHECKPOINT_PREFETCH_EXECUTOR_LOCK:
        if (
            _CHECKPOINT_PREFETCH_EXECUTOR is None
            or _CHECKPOINT_PREFETCH_EXECUTOR[0] != pid
        ):
            _CHECKPOINT_PREFETCH_EXECUTOR = (
                pid,
                ThreadPoolExecutor(thread_name_prefix="art-checkpoint-prefetch"),
            )
        return _CHECKPOINT_PREFETCH_EXECUTOR[1]


class _AdapterConfig(TypedDict):
    base_model_name_or_path: str
    revision: NotRequired[str | None]
    r: int
    lora_alpha: float
    target_modules: str | list[str]
    num_attention_heads: NotRequired[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    hidden_size: NotRequired[int]


type AdapterSelection = str | None | _Unset


@dataclass(frozen=True)
class _LocalLoRASlotRef:
    name: str | None


@dataclass(frozen=True)
class ForwardOutput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    """Per-request tensors in flattened input order, replicated across TP/CP."""

    target_logprobs: LogprobsT
    top_k: TopKT
    logits: LogitsT
    hidden_states: HiddenStatesT
    checkpoint: str | None = None
    no_grad: bool = False


@dataclass(slots=True)
class ForwardInput(Generic[LogprobsT, TopKT, LogitsT, HiddenStatesT]):
    """Flattened inputs with optional [tokens, model layers, top-k] expert IDs.

    Routes align with input tokens without a shift and must contain no missing
    (-1) IDs. Shared token prefixes use the first sequence's routes. Omit routes
    for normal model routing. Forward logprobs always describe raw model scores.
    """

    input_tokens: torch.Tensor
    routed_experts: torch.Tensor | None = None
    target_tokens: torch.Tensor | None = None
    top_k: int | None = None
    logits: bool = False
    hidden_states: bool = False
    no_grad: bool | None = None
    checkpoint: AdapterSelection = Unset
    options: ForwardOptions | None = None

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, None, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, None, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, TopK, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, None, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, None, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, TopK, None, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, None, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, None, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, TopK, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, TopK, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, None, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[False] = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, TopK, torch.Tensor, None]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[False] = False,
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, TopK, None, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: None = None,
        logits: Literal[True],
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, None, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: None = None,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[None, TopK, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor,
        top_k: int,
        logits: Literal[True],
        hidden_states: Literal[True],
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor, TopK, torch.Tensor, torch.Tensor]": ...

    @overload
    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> "ForwardInput[torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None]": ...

    def __new__(
        cls,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> Self:
        return object.__new__(cls)

    def __init__(
        self,
        *,
        input_tokens: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        logits: bool = False,
        hidden_states: bool = False,
        no_grad: bool | None = None,
        checkpoint: AdapterSelection = Unset,
        options: ForwardOptions | None = None,
    ) -> None:
        self.routed_experts = routed_experts
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.logits = logits
        self.hidden_states = hidden_states
        self.no_grad = no_grad
        self.checkpoint = checkpoint
        self.options = options
        self.__post_init__()

    def __getnewargs_ex__(self) -> tuple[tuple[()], dict[str, torch.Tensor]]:
        return (), {"input_tokens": self.input_tokens}

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")


type AnyForwardInput = ForwardInput[
    torch.Tensor | None,
    TopK | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
type AnyForwardOutput = ForwardOutput[
    torch.Tensor | None,
    TopK | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
type ForwardInputs = AnyForwardInput | Iterable["ForwardInputs"]
type ForwardOutputs = AnyForwardOutput | Sequence["ForwardOutputs"]
ForwardInputsT = TypeVar("ForwardInputsT", bound=ForwardInputs)
ForwardOutputsT = TypeVar("ForwardOutputsT", bound=ForwardOutputs)
MicroBatchInputsT = TypeVar("MicroBatchInputsT", bound=ForwardInputs, covariant=True)
MicroBatchOutputsT = TypeVar("MicroBatchOutputsT", bound=ForwardOutputs, covariant=True)


@dataclass(frozen=True)
class MicroBatch(Generic[MicroBatchInputsT, MicroBatchOutputsT]):
    inputs: Sequence[MicroBatchInputsT]
    outputs: Sequence[MicroBatchOutputsT]
    indices: Sequence[int]
    stats: "MicroBatchStats"

    def select(self, xs: Sequence[T]) -> Sequence[T]:
        return [xs[i] for i in self.indices]


@dataclass(frozen=True)
class MicroBatchStats:
    global_start: int
    global_stop: int
    global_count: int
    local_count: int
    packed_tokens: int
    logical_tokens: int
    estimated_required_bytes: int
    available_bytes: int
    rejected_candidates: int
    cold_start: bool
    subforward_count: int = 1


class TrainerRankMemoryError(RuntimeError):
    """Bounded conservative planning could not safely admit this call.

    This is not an infeasibility proof: it means the planner's bounded search
    and conservative admission margin found no plan predicted to fit. The
    error reports only what the caller can act on.
    """

    predicted_peak_bytes: int
    usable_limit_bytes: int
    suggestion: str

    def __init__(
        self,
        message: str,
        *,
        predicted_peak_bytes: int = 0,
        usable_limit_bytes: int = 0,
        suggestion: str = "",
    ) -> None:
        super().__init__(message)
        self.predicted_peak_bytes = predicted_peak_bytes
        self.usable_limit_bytes = usable_limit_bytes
        self.suggestion = suggestion

    def __reduce__(self) -> tuple[object, ...]:
        # Keyword-only fields do not survive the default Exception reduce;
        # carry them as state so pickling across process boundaries keeps the
        # actionable numbers.
        return (
            type(self),
            (self.args[0] if self.args else "",),
            {
                "predicted_peak_bytes": self.predicted_peak_bytes,
                "usable_limit_bytes": self.usable_limit_bytes,
                "suggestion": self.suggestion,
            },
        )


class TrainerRankRuntimeSupportError(RuntimeError):
    """The current topology or runtime capability is not yet supported."""


class TrainerRankPartialExecutionError(TrainerRankMemoryError):
    """A subforward of an admitted split failed while executing.

    Distinct from an up-front refusal: model execution already began, so the
    call cannot transparently choose another split. Any earlier subforwards'
    graphs were released; runtime state (e.g. RNG) may have advanced.
    """


class TrainerRankSlotStateError(RuntimeError):
    pass


@dataclass
class _CacheRecoveryState:
    # Local, lifetime measurements; reductions never replace these ledgers.
    cost: float = 0.0
    work: float = 0.0
    high: float = 0.0
    first_consumed: bool = False
    invalid: bool = False
    owner: object | None = None
    backward: BackwardWork | None = None
    lock: Any = dataclass_field(default_factory=threading.RLock)


class _PlannerObservation(dict[str, Any]):
    """Weakly track retained execution/OOM context until execution cleanup."""


@dataclass(frozen=True)
class _MemoryCheck:
    estimated_required_bytes: int
    available_bytes: int
    fits: bool
    cpu_required_bytes: int = 0
    cpu_available_bytes: int = 0
    cpu_fits: bool = True
    fallback_costs: dict[str, Any] | None = None
    sample: _planner_evidence.MemorySample | None = dataclass_field(
        default=None, compare=False, repr=False
    )
    decision: dict[str, Any] | None = dataclass_field(
        default=None, compare=False, repr=False
    )
    # Extrema above are diagnostics; fits is the conjunction of local fits.
    # Keep each local producer through refreshes of different DP items.
    local_required_bytes: int | None = dataclass_field(
        default=None, compare=False, repr=False
    )
    local_available_bytes: int | None = dataclass_field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True)
class _MemoryProfile:
    bytes_per_token: float
    packed_tokens: int
    # Active execution rows only; total-input telemetry can include inactive rows.
    logical_per_packed: float = 1.0
    # Historical forward-retained/forward-peak ratio for calibration telemetry,
    # max-merged only from forward-return observations. Admission uses the
    # independent retained compute rate below, not this ratio of past peaks.
    retained_fraction: float | None = None
    # Separate the retained compute rate from the peak, which also learns
    # caller-owned backward workspace. Requested outputs are charged explicitly.
    retained_compute_bytes_per_token: float | None = None
    # A signature's first executed plan also pays one-time costs (compilation,
    # first-use workspaces), which a small first wave spreads over few tokens.
    # Admission uses the lower of the fit over every observation and the same
    # fit over later flat waves' whole caller-phase peaks (forward plus the
    # caller's loss and backward in the yield), which prices a wave smaller
    # than the smallest of them as if it were that large.
    warm_bytes_per_token: float | None = None
    warm_packed_tokens: int | None = None
    warm_logical_per_packed: float | None = None
    caller_plans: int = 0


@dataclass(frozen=True)
class _CandidateMicroBatch(Generic[ForwardInputsT]):
    inputs: Sequence[ForwardInputsT]
    indices: tuple[int, ...]
    plan: "_AnyForwardPlan"
    check: _MemoryCheck
    stats_global_count: int
    rejected_candidates: int
    cold_start: bool
    fallback: _CandidateMicroBatch[ForwardInputsT] | None = None


class _GatherContextParallelRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        tensor: torch.Tensor,
        positions: torch.Tensor,
        length: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.save_for_backward(positions)
        # Each source row has one owner. Scatter + SUM handles unequal/empty
        # shards without padded all-gather buffers, including for full logits.
        output = tensor.new_zeros((length, *tensor.shape[1:]))
        output.index_copy_(0, positions, tensor)
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (positions,) = cast(tuple[torch.Tensor, ...], getattr(ctx, "saved_tensors"))
        # The caller's loss is replicated over CP, just as over TP after the
        # sequence-parallel gather with tensor_parallel_output_grad=False.
        # Route one copy to its owner, without summing CP copies of the loss.
        return grad_outputs[0].index_select(0, positions), None, None, None


class _CustomSlotGraphSentinel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        tensor: torch.Tensor,
        marker: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(marker)
        return tensor

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        saved_tensors = cast(tuple[torch.Tensor, ...], getattr(ctx, "saved_tensors"))
        (marker,) = saved_tensors

        def finish() -> None:
            try:
                getattr(ctx, "saved_tensors")
            except RuntimeError:
                marker.fill_(True)

        torch.autograd.Variable._execution_engine.queue_callback(finish)
        return grad_outputs[0], None


def _track_slot_graph_tensor(
    tensor: torch.Tensor, marker: torch.Tensor
) -> torch.Tensor:
    # This Function saves only a CPU, non-gradient control marker. Preserve its
    # wrapper identity even under caller hooks, without intercepting activations.
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: value, lambda value: value
    ):
        return cast(torch.Tensor, _CustomSlotGraphSentinel.apply(tensor, marker))


@dataclass(eq=False)
class _CustomTensorTracker:
    trainer: weakref.ReferenceType[TrainerRank]
    ref: LoRASlotRef
    name: str
    generation: object
    active: bool = False
    buffer_revision: int = 0

    def validate(self) -> TrainerRank:
        trainer = self.trainer()
        slot = (
            None
            if trainer is None or self.ref.name is None
            else trainer._checkpoint_slots.get(self.ref.name)
        )
        current = None if slot is None else slot.custom.get(self.name)
        if self.active and (
            current is None or current.generation is not self.generation
        ):
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {self.name!r} is stale because checkpoint "
                f"{self.ref.name!r} was replaced. Register it again from the current "
                "checkpoint before use."
            )
        if trainer is None:
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {self.name!r} no longer has a TrainerRank"
            )
        return trainer

    def record(self, marker: torch.Tensor) -> None:
        trainer = self.validate()
        trainer._slot_graphs().setdefault(self.ref, []).append(weakref.ref(marker))


class _TrackedTensor(torch.Tensor):
    _art_tracker: _CustomTensorTracker

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        tracker: _CustomTensorTracker,
    ) -> Self:
        value = torch.Tensor._make_subclass(cls, data, require_grad=False)
        value._art_tracker = tracker
        return value

    def __deepcopy__(self, memo: dict[int, object]) -> torch.Tensor:
        existing = memo.get(id(self))
        if isinstance(existing, torch.Tensor):
            return existing
        with torch._C.DisableTorchFunctionSubclass():
            result = self.as_subclass(torch.Tensor).detach().clone()
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> str | tuple[Any, ...]:
        with torch._C.DisableTorchFunctionSubclass():
            plain = self.as_subclass(torch.Tensor).detach().clone()
        return cast(str | tuple[Any, ...], plain.__reduce_ex__(proto))

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        return _tracked_tensor_function(func, types, args, kwargs or {})


class _TrackedParameter(torch.nn.Parameter):
    _art_tracker: _CustomTensorTracker

    def __new__(
        cls,
        data: torch.Tensor,
        tracker: _CustomTensorTracker,
        requires_grad: bool = True,
    ) -> Self:
        value = super().__new__(cls, data, requires_grad=requires_grad)
        value._art_tracker = tracker
        return value

    def __init__(
        self,
        data: torch.Tensor,
        tracker: _CustomTensorTracker,
        requires_grad: bool = True,
    ) -> None:
        del data, tracker, requires_grad

    def register_hook(self, hook: Any) -> Any:
        """Run once on summed captured uses per backward; removal affects old graphs."""
        from ._parameter_hooks import register_parameter_hook

        self._art_tracker.validate()
        return register_parameter_hook(self, hook)

    def register_post_accumulate_grad_hook(self, hook: Any) -> Any:
        from ._parameter_hooks import reject_post_accumulate_hook

        return reject_post_accumulate_hook()

    def __setattr__(self, name: str, value: object) -> None:
        if name == "grad":
            with torch._C.DisableTorchFunctionSubclass():
                super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __deepcopy__(self, memo: dict[int, object]) -> torch.nn.Parameter:
        existing = memo.get(id(self))
        if isinstance(existing, torch.nn.Parameter):
            return existing
        with torch._C.DisableTorchFunctionSubclass():
            data = self.as_subclass(torch.Tensor).detach().clone()
            requires_grad = self.requires_grad
        result = torch.nn.Parameter(data, requires_grad=requires_grad)
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> str | tuple[Any, ...]:
        with torch._C.DisableTorchFunctionSubclass():
            data = self.as_subclass(torch.Tensor).detach().clone()
            requires_grad = self.requires_grad
        return cast(
            str | tuple[Any, ...],
            torch.nn.Parameter(data, requires_grad=requires_grad).__reduce_ex__(proto),
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        return _tracked_tensor_function(func, types, args, kwargs or {})


@dataclass(frozen=True)
class _DynamicOptimizer:
    optimizer: torch.optim.Optimizer
    master_params: tuple[torch.nn.Parameter, ...]


@dataclass(frozen=True)
class _CustomObject:
    kind: Literal["module", "parameter", "buffer"]
    value: torch.nn.Module | torch.nn.Parameter | torch.Tensor
    generation: object
    handle: torch.nn.Module | None = None


@dataclass
class _CheckpointSlot:
    params: tuple[torch.nn.Parameter, ...] = ()
    config: _AdapterConfig | None = None
    optimizer: _DynamicOptimizer | None = None
    revision: int = 0
    custom: dict[str, _CustomObject] = dataclass_field(default_factory=dict)
    custom_payload: "PreparedCustomPayload | None" = None
    snapshot: bool = False
    generation: int = 0


@dataclass(frozen=True)
class MaterializedCheckpoint:
    """A logical checkpoint and its rank-local materialized directory."""

    path: str
    directory: str


@dataclass
class PushedCheckpoint:
    _trainer: "TrainerRank"
    _path: str | None
    _directory: str | None
    _entered: bool = False
    _closed: bool = False

    def __enter__(self) -> "PushedCheckpoint":
        if self._entered or self._closed:
            raise RuntimeError("Pushed checkpoint context cannot be entered twice")
        self._trainer._push_checkpoint_sync(self._path, self._directory)
        self._entered = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._exit(exception)
        return False

    def _exit(self, body_error: BaseException | None) -> None:
        try:
            self._pop()
        except BaseException as pop_error:
            if body_error is not None:
                raise BaseExceptionGroup(
                    "checkpoint context body and cleanup both failed",
                    [body_error, pop_error],
                ) from None
            raise

    def _pop(self) -> None:
        if not self._entered:
            return
        ref = self._trainer._slot_ref(self._path)
        if not self._trainer._slot_stack or self._trainer._slot_stack[-1] != ref:
            raise RuntimeError("Pushed checkpoint stack changed before context exit")
        self._trainer.pop_checkpoint()
        self._entered = False
        self._closed = True


@dataclass(frozen=True)
class _ForwardItem:
    request: AnyForwardInput
    input_ids: torch.Tensor
    labels: torch.Tensor | None
    routed_experts: torch.Tensor | None = None


@dataclass(frozen=True)
class _PreparedPackedForward:
    tokens: torch.Tensor
    position_ids: torch.Tensor
    attention_state: "PrefixTreeAttentionState | ArtContextParallelState"
    packed_seq_params: "PackedSeqParams | None"
    positions_by_item: tuple[torch.Tensor, ...]
    source_positions_by_item: tuple[torch.Tensor, ...]
    context_parallel_group: dist.ProcessGroup | None = None
    token_uids: torch.Tensor | None = None


type _RowMatch = tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]


@dataclass(frozen=True)
class _MemorySignature:
    topology: tuple[int, int, int, int]
    # (coefficient version, calibrated table id): the score that planned it.
    planner_coefficients: tuple[int, str | None]
    slot_group_count: int
    request_mix: tuple[str, ...]
    grad_enabled: bool
    grad_modes: tuple[bool, ...]
    memory_placement: tuple[tuple[str, str], ...] = ()
    slot_shapes: tuple[tuple[bool, tuple[tuple[int, ...], ...]], ...] = ()
    # Short single-target requests keep the logical extrapolation, but share
    # the profile learned from longer requests of the same signature.
    short_requests: bool = dataclass_field(default=False, compare=False)


@dataclass(frozen=True)
class _ForwardGroupPlan:
    slot_ref: "LoRASlotRef | None"
    grad_enabled: bool
    request_indices: tuple[int, ...]
    items: tuple[_ForwardItem, ...]
    packed: PrefixTreePack
    memory_placement: MemoryPlacement | None = None
    layout: PrefixTreeLayout | None = None
    input_row_fingerprints: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class _FlatForwardPlan:
    request_count: int
    output_metadata: tuple[tuple[str | None, bool], ...]
    groups: tuple[_ForwardGroupPlan, ...]
    # Physical token count: every group's packed length rounded up to the TP
    # multiple that execution pads it to (see ``_physical_tokens``).
    packed_tokens: int
    logical_tokens: int
    output_bytes: int
    signature: _MemorySignature
    selected_max_depth: int = 0
    inactive_logical_tokens: int = 0

    @property
    def active_logical_tokens(self) -> int:
        # Keep total-input telemetry while pricing only executed requests.
        return self.logical_tokens - self.inactive_logical_tokens

    @property
    def grad_segment_count(self) -> int:
        return sum(
            len(group.packed.segments) for group in self.groups if group.grad_enabled
        )

    @property
    def subforward_count(self) -> int:
        return 1


@dataclass(frozen=True)
class _SplitForwardPlan:
    """One public forward executed as sequential subforwards.

    All returned graphs stay live together, so each subforward was admitted
    against the retained memory of the ones before it. ``request_indices``
    maps every subforward's outputs back to the caller's flat request order.
    """

    subforwards: tuple[_FlatForwardPlan, ...]
    request_indices: tuple[tuple[int, ...], ...]
    request_count: int

    @property
    def subforward_count(self) -> int:
        return len(self.subforwards)

    @property
    def groups(self) -> tuple[_ForwardGroupPlan, ...]:
        return tuple(group for plan in self.subforwards for group in plan.groups)

    @property
    def packed_tokens(self) -> int:
        return sum(plan.packed_tokens for plan in self.subforwards)

    @property
    def logical_tokens(self) -> int:
        return sum(plan.logical_tokens for plan in self.subforwards)

    @property
    def output_bytes(self) -> int:
        return sum(plan.output_bytes for plan in self.subforwards)

    @property
    def selected_max_depth(self) -> int:
        return max(plan.selected_max_depth for plan in self.subforwards)

    @property
    def signature(self) -> _MemorySignature:
        return self.subforwards[0].signature


_AnyForwardPlan = _FlatForwardPlan | _SplitForwardPlan


@dataclass(frozen=True)
class _GroupLayout:
    """One packed group's CP layouts on every rank, for layout-aware pricing."""

    attention_rows: tuple[int, ...]
    gdn_rows: tuple[int, ...] | None
    # What each rank's recomputed CP attention keeps for backward beyond its
    # own-row activations (``retained_stage_record_bytes``).
    attention_retained: tuple[int, ...]
    # Each rank's GDN recurrent states held at once (none if unknown).
    gdn_states: tuple[int, ...] = ()


@dataclass(frozen=True)
class _SubforwardCost:
    """Memory terms of one candidate subforward while all graphs stay live.

    ``required`` is its transient peak while running (the planner estimate);
    ``retained`` is what stays allocated after it returns because its graph
    is kept for backward; ``ephemeral`` is the difference.
    """

    required: int
    retained: int
    # Before the safety factor, separate from observed forward retention.
    checkpoint_retained: int = 0
    checkpoint_workspace: int = 0
    checkpoint_input_gradient: int = 0
    # Already in required; backward allowance must not reorder forward execution.
    checkpoint_peak_increment: int = 0
    # HybridEP buffer growth before the safety factor. It is in required, not
    # retained, and persists across a split, which charges the largest once.
    hybridep_growth: int = 0
    # Adapter gradients the recompute backward holds beyond the boundaries it
    # has released (``_checkpoint_adapter_gradient_bytes``), before the safety
    # factor. Split children training the same slots share them, so a split
    # charges the largest once; ``..._slots`` names those slots (sorted JSON
    # of kind/name pairs, "" when none), keeping the cost JSON-serializable.
    checkpoint_adapter_gradient: int = 0
    checkpoint_adapter_gradient_slots: str = ""

    @property
    def ephemeral(self) -> int:
        return self.required - self.retained


_MEMORY_ERROR_SUGGESTION = (
    "Use smaller top-level items, reduce output requests, or call "
    "forward with already-DP-local smaller inputs."
)


def _memory_error(
    *,
    context: str,
    message: str,
    packed_tokens: int,
    logical_tokens: int,
    check: _MemoryCheck,
) -> TrainerRankMemoryError:
    # Peak and limit are group extrema (largest demand, smallest budget) that may
    # come from different ranks; the rank_* pair is this rank's own comparison.
    local = (
        ""
        if check.local_required_bytes is None or check.local_available_bytes is None
        else f"rank_required_gb={check.local_required_bytes / 1024**3:.3f} "
        f"rank_available_gb={check.local_available_bytes / 1024**3:.3f} "
    )
    return TrainerRankMemoryError(
        f"{context}: {message}. "
        f"packed_tokens={packed_tokens} "
        f"logical_tokens={logical_tokens} "
        f"predicted_peak_gb={check.estimated_required_bytes / 1024**3:.3f} "
        f"usable_limit_gb={check.available_bytes / 1024**3:.3f}. "
        + (
            f"CPU retained bytes={check.cpu_required_bytes}, "
            f"per-rank CPU headroom={check.cpu_available_bytes}. "
            if not check.cpu_fits
            else ""
        )
        + f"{local}{_MEMORY_ERROR_SUGGESTION}",
        predicted_peak_bytes=check.estimated_required_bytes,
        usable_limit_bytes=check.available_bytes,
        suggestion=_MEMORY_ERROR_SUGGESTION,
    )


@dataclass(frozen=True)
class _ForwardRefusal:
    """Why admission failed, with the last relevant local plan and check."""

    plan: _AnyForwardPlan
    check: _MemoryCheck
    message: str
    overridable: bool = True
    candidate: Any = None
    check_matches_plan: bool = True

    def error(self, context: str) -> TrainerRankMemoryError:
        return _memory_error(
            context=context,
            message=self.message,
            packed_tokens=self.plan.packed_tokens,
            logical_tokens=self.plan.logical_tokens,
            check=self.check,
        )


def _wave_geometry(item_count: int, start: int, dp_size: int) -> tuple[int, int, int]:
    """Return (remaining, min_width, granularity) for a wave starting at start."""

    remaining = item_count - start
    min_width = min(dp_size, remaining)
    base_granularity = 1 if remaining < 64 else 8 if remaining < 256 else 32
    granularity = max(1, ((base_granularity + dp_size - 1) // dp_size) * dp_size)
    return remaining, min_width, granularity


def _normalize_wave_width(
    width: int, min_width: int, remaining: int, granularity: int
) -> int:
    width = max(min_width, min(width, remaining))
    if width in (min_width, remaining) or granularity <= 1:
        return width
    if width < granularity:
        return width
    return max(min_width, (width // granularity) * granularity)


def _local_wave_indices(
    start: int, width: int, dp_rank: int, dp_size: int
) -> tuple[int, ...]:
    """This DP rank's strided share of the global wave [start, start + width)."""

    return tuple(range(start + dp_rank, start + width, dp_size))


class _PlannerFacts(NamedTuple):
    """Topology and model facts the layout scorer prices against."""

    cp_size: int
    tp_size: int
    layers: int
    gdn_layers: int
    uses_gdn: bool
    # Expert parallelism (MoE models): EP and expert-TP sizes. Not priced by
    # the current terms, but part of the execution class and the cache key.
    ep_size: int
    etp_size: int
    # Score for this runtime: a fitted table inside its calibrated execution
    # class (version 2, identified by table id), the fallback outside it
    # (version 1); see select_scoring. The tables were calibrated on training
    # forwards (forward + backward); groups that run without gradients keep
    # the fallback until forward-only execution is validated separately, so
    # the grad mode is part of the identity.
    coefficient_version: int
    coefficient_table: str | None
    grad_enabled: bool = True


# Content hash, planner facts (including the score identity), anchor.
_LayoutKey = tuple[str, _PlannerFacts, str | None]


def _gdn_layer_count(model: torch.nn.Module) -> int:
    """Number of gated-delta-net layers in the model (0 when unavailable)."""

    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return 0
    return sum(isinstance(module, GatedDeltaNet) for module in model.modules())


def _mixer_top_gaps(model: torch.nn.Module) -> tuple[int | None, int | None] | None:
    """Layers between the top decoder layer and the highest attention / GDN layer.

    ``None`` for a mixer kind the decoder lacks; ``None`` overall when the
    decoder layers are unreadable.
    """
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return None
    try:
        layers = list(_language_model(model).decoder.layers)
    except (AttributeError, RuntimeError, TypeError):
        return None
    gdn = [
        any(isinstance(module, GatedDeltaNet) for module in layer.modules())
        for layer in layers
    ]
    top = len(layers) - 1
    return (
        next((top - index for index in range(top, -1, -1) if not gdn[index]), None),
        next((top - index for index in range(top, -1, -1) if gdn[index]), None),
    )


def _lora_modules_per_layer(model: torch.nn.Module) -> int:
    """The most LoRA modules any one decoder layer holds (0 when unreadable)."""
    try:
        from art.megatron.lora import LoRA

        layers = list(_language_model(model).decoder.layers)
    except (AttributeError, ImportError, RuntimeError, TypeError):
        return 0
    return max(
        (sum(type(module) is LoRA for module in layer.modules()) for layer in layers),
        default=0,
    )


def _moe_layer_count(model: torch.nn.Module) -> int:
    """Number of mixture-of-experts layers in the model (0 when unavailable)."""

    try:
        from megatron.core.transformer.moe.moe_layer import BaseMoELayer
    except ImportError:
        return 0
    return sum(isinstance(module, BaseMoELayer) for module in model.modules())


def _dense_fc1_adapted(model: torch.nn.Module) -> bool:
    """Whether every decoder layer's MLP FC1 is ART's gated LoRA wrapper, which
    keeps a 2F adapter output and the sum beside the 2F base output, even for
    inactive slots."""

    try:
        from art.megatron.lora import SharedExpertsLinearFC1LoRA

        layers = _language_model(model).decoder.layers
    except (AttributeError, ImportError, RuntimeError):
        return False
    fc1s = [
        getattr(getattr(layer, "mlp", None), "linear_fc1", None) for layer in layers
    ]
    return len(fc1s) > 0 and all(
        type(fc1) is SharedExpertsLinearFC1LoRA
        and getattr(fc1, "non_gated", True) is False
        for fc1 in fc1s
    )


def _expert_parallel_shape(provider: object) -> tuple[int, int]:
    """(EP, ETP) of the initialized runtime, else the provider's configuration."""

    try:
        from megatron.core import parallel_state as ps

        if ps.model_parallel_is_initialized():
            return (
                int(ps.get_expert_model_parallel_world_size()),
                int(ps.get_expert_tensor_parallel_world_size()),
            )
    except (AssertionError, AttributeError, ImportError, RuntimeError):
        pass
    return (
        int(getattr(provider, "expert_model_parallel_size", 1) or 1),
        int(getattr(provider, "expert_tensor_parallel_size", 1) or 1),
    )


def _split_chunks(
    order: Sequence[int],
    requests: Sequence[AnyForwardInput],
    count: int,
) -> tuple[tuple[int, ...], ...]:
    """Cut an ordered request list into ``count`` contiguous, token-balanced chunks."""

    count = max(1, min(count, len(order)))
    masses = [int(requests[index].input_tokens.numel()) for index in order]
    total = sum(masses)
    chunks: list[list[int]] = []
    current: list[int] = []
    cumulative = 0
    for position, index in enumerate(order):
        current.append(index)
        cumulative += masses[position]
        remaining_chunks = count - len(chunks) - 1
        remaining_items = len(order) - position - 1
        boundary = cumulative * count >= total * (len(chunks) + 1)
        if remaining_chunks > 0 and (boundary or remaining_items <= remaining_chunks):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return tuple(tuple(chunk) for chunk in chunks)


def _moe_dispatch_preprocess(
    dispatcher: Any,
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
):
    result = type(dispatcher).dispatch_preprocess(
        dispatcher, hidden_states, routing_map, probs
    )
    # MCore uses this cache only for dtype. Keep the real routing outputs
    # differentiable without retaining their checkpoint graph.
    dispatcher.probs = probs.new_empty(0)
    return result


def _moe_combine_postprocess(dispatcher: Any, hidden_states: torch.Tensor):
    result = type(dispatcher).combine_postprocess(dispatcher, hidden_states)
    # Backward saves its own maps; the next dispatch recreates these caches.
    dispatcher.routing_map = None
    dispatcher.reversed_local_input_permutation_mapping = None
    return result


def _release_hybridep_state(manager: Any) -> None:
    # The dispatched probabilities hold this layer's checkpoint graph, with its
    # recomputed input and that input's gradient, until the next dispatch.
    manager.routing_map = manager.token_probs = manager.dispatched_probs = None


def _hybridep_combine_postprocess(dispatcher: Any, hidden_states: torch.Tensor):
    result = type(dispatcher).combine_postprocess(dispatcher, hidden_states)
    # Backward saves its own; the next setup and dispatch recreate these.
    _release_hybridep_state(dispatcher._comm_manager)
    return result


def _configure_moe_dispatcher_caches(model: Sequence[torch.nn.Module]) -> None:
    for chunk in model:
        for module in chunk.modules():
            dispatcher = getattr(module, "token_dispatcher", None)
            if dispatcher is None:
                continue
            from megatron.core.transformer.moe.token_dispatcher import (
                MoEAlltoAllTokenDispatcher,
                MoEFlexTokenDispatcher,
                _HybridEPManager,
            )

            if (
                type(dispatcher) is MoEFlexTokenDispatcher
                and type(getattr(dispatcher, "_comm_manager", None)) is _HybridEPManager
                and "combine_postprocess" not in vars(dispatcher)
                and getattr(dispatcher.config, "cuda_graph_impl", "none") == "none"
            ):
                # HybridEP keeps its routing inputs and dispatched probabilities
                # after combine; CUDA graph capture reads them back instead.
                setattr(
                    dispatcher,
                    "combine_postprocess",
                    partial(_hybridep_combine_postprocess, dispatcher),
                )
                _release_hybridep_state(dispatcher._comm_manager)
                continue
            if type(dispatcher) is not MoEAlltoAllTokenDispatcher or (
                "dispatch_preprocess" in vars(dispatcher)
            ):
                continue

            # Persist through caller-owned backward/checkpoint recomputation;
            # other dispatcher instances and custom implementations stay intact.
            setattr(
                dispatcher,
                "dispatch_preprocess",
                partial(_moe_dispatch_preprocess, dispatcher),
            )
            if (probs := getattr(dispatcher, "probs", None)) is not None:
                dispatcher.probs = probs.new_empty(0)
            if (
                "combine_postprocess" not in vars(dispatcher)
                and getattr(dispatcher.config, "cuda_graph_impl", "none") == "none"
            ):
                # CUDA graph capture exposes these maps as explicit outputs.
                setattr(
                    dispatcher,
                    "combine_postprocess",
                    partial(_moe_combine_postprocess, dispatcher),
                )


def _shared_expert_output_bytes_per_token(layer: torch.nn.Module) -> int:
    """One supported shared return held across routed compute, not all saves.

    Gated backward can also save a distinct pre-gate result. This mode-neutral
    component intentionally omits that separate term; compiled storage may alias.
    """
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    from art.megatron.lora import (
        LoRA,
        SelfAttentionLinearProjLoRA,
        SharedExpertsLinearFC1LoRA,
        SharedExpertsLinearFC2LoRA,
    )

    shared = getattr(layer, "shared_experts", None)
    if shared is None or type(shared) is not SharedExpertMLP:
        return 0
    config = getattr(layer, "config", None)
    shared_config = getattr(shared, "config", None)
    expected = {
        "params_dtype": torch.bfloat16,
        "moe_shared_expert_overlap": False,
        "sequence_parallel": False,
        "fp32_residual_connection": False,
        "add_bias_linear": False,
        "use_te_activation_func": False,
        "bias_activation_fusion": False,
        "gated_linear_unit": True,
        "cuda_graph_impl": "none",
    }
    if (
        getattr(layer, "use_shared_expert", None) is not True
        or getattr(layer, "shared_expert_overlap", None) is not False
        or getattr(layer, "shared_experts_recompute", None) is not False
        or getattr(layer, "moe_layer_recompute", None) is not False
        or getattr(layer, "fwd_execution_map", None)
        != ["route", "expert_compute", "postprocess"]
        or any(
            name in vars(layer)
            for name in (
                "shared_experts_compute",
                "route",
                "preprocess",
                "dispatch",
                "routed_experts_compute",
                "combine",
                "postprocess",
            )
        )
        or any(
            type(getattr(c, name, None)) is not type(value) or getattr(c, name) != value
            for c in (config, shared_config)
            for name, value in expected.items()
        )
        or any(
            getattr(c, name, None)
            for c in (config, shared_config)
            for name in ("fp8", "fp4", "moe_latent_size")
        )
        or any(
            (type(getattr(config, name, None)) is not int or getattr(config, name) != 1)
            for name in (
                "tensor_model_parallel_size",
                "pipeline_model_parallel_size",
                "expert_tensor_parallel_size",
            )
        )
        # CP shards rows and shared experts are not expert-parallel, so each
        # local token returns the same shared output; Qwen3.6-35B-A3B traces
        # show the same gated pair per token at CP1, CP2 and EP2.
        or any(
            (type(getattr(config, name, None)) is not int or getattr(config, name) < 1)
            for name in ("context_parallel_size", "expert_model_parallel_size")
        )
    ):
        return 0
    hidden = getattr(config, "hidden_size", None)
    width = getattr(config, "moe_shared_expert_intermediate_size", None)
    if (
        type(hidden) is not int
        or hidden <= 0
        or type(width) is not int
        or width <= 0
        or getattr(shared_config, "hidden_size", None) != hidden
        or getattr(shared_config, "ffn_hidden_size", None) != width
        or getattr(shared_config, "moe_shared_expert_intermediate_size", None) != width
        or getattr(shared_config, "activation_func", None)
        is not torch.nn.functional.silu
        or getattr(shared, "activation_func", None) is not torch.nn.functional.silu
        or type(getattr(shared, "use_shared_expert_gate", None)) is not bool
    ):
        return 0
    fc1, fc2 = getattr(shared, "linear_fc1", None), getattr(shared, "linear_fc2", None)
    row = getattr(fc2, "row_parallel_lora", None)
    lora = getattr(row, "lora", None)
    base1, base2 = getattr(fc1, "linear_fc1", None), getattr(row, "linear_proj", None)
    sites = (
        (shared, SharedExpertMLP),
        (fc1, SharedExpertsLinearFC1LoRA),
        (fc2, SharedExpertsLinearFC2LoRA),
        (row, SelfAttentionLinearProjLoRA),
        (lora, LoRA),
        (base2, TERowParallelLinear),
        (getattr(fc1, "gate_lora", None), LoRA),
        (getattr(fc1, "up_lora", None), LoRA),
    )
    if (
        type(base1) not in (TEColumnParallelLinear, TELayerNormColumnParallelLinear)
        or any(type(site) is not cls for site, cls in sites)
        or any(
            "forward" in vars(site)
            or cast(Any, site)._forward_hooks
            or cast(Any, site)._forward_pre_hooks
            for site in (base1, *(site for site, _ in sites))
        )
        or getattr(fc1, "non_gated", None) is not False
        or getattr(fc1, "out_features", None) != 2 * width
        or getattr(getattr(row, "provider", None), "tensor_model_parallel_size", None)
        != 1
        or getattr(getattr(row, "provider", None), "sequence_parallel", None)
        is not False
    ):
        return 0
    weights = (
        (getattr(base1, "weight", None), (2 * width, hidden)),
        (getattr(base2, "weight", None), (hidden, width)),
    )
    for adapter, inputs, outputs in (
        (cast(Any, fc1).gate_lora, hidden, width),
        (cast(Any, fc1).up_lora, hidden, width),
        (lora, width, hidden),
    ):
        a, b = getattr(adapter, "A_T", None), getattr(adapter, "B_T", None)
        if (
            not isinstance(a, torch.Tensor)
            or not isinstance(b, torch.Tensor)
            or a.ndim != 2
            or b.ndim != 2
            or a.shape[1] <= 0
            or a.shape[1] != b.shape[0]
        ):
            return 0
        weights += ((a, (inputs, a.shape[1])), (b, (a.shape[1], outputs)))
    if shared.use_shared_expert_gate:
        weights += ((getattr(shared, "gate_weight", None), (1, hidden)),)
    if any(
        not isinstance(weight, torch.Tensor)
        or weight.dtype is not torch.bfloat16
        or tuple(weight.shape) != shape
        for weight, shape in weights
    ):
        return 0
    return hidden * 2


def _slot_lora_tensors(
    lora: Any, slot_ref: "LoRASlotRef | None" = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read the selected owner directly, without changing the execution context."""
    if slot_ref is None:
        return lora.A_T, lora.B_T
    if slot_ref.name is None:
        return None
    from art.megatron.lora import LoRA

    slot = LoRA._slot(lora, slot_ref)
    return None if slot is None else (slot.A_T, slot.B_T)


def _expert_lora_weight_storage(
    lora: Any, slot_ref: "LoRASlotRef | None" = None
) -> tuple[int, int, int] | None:
    """New padded weights, transposes and effective rank for the Quack path.

    Original contiguous parameters are already in the allocator baseline. This
    excludes padding-concatenation temporaries and all backward GEMM workspace.
    """
    from art.megatron.lora import LoRA

    if type(lora) is not LoRA or "_slot" in vars(lora):
        return None
    tensors = _slot_lora_tensors(lora, slot_ref)
    if tensors is None:
        return None
    a, b = tensors
    if (
        "forward" in vars(lora)
        or "active_lora_tensors" in vars(lora)
        or lora._forward_hooks
        or lora._forward_pre_hooks
        or not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.ndim != 3
        or b.ndim != 3
        or a.dtype not in (torch.float16, torch.bfloat16)
        or b.dtype != a.dtype
        or not a.is_contiguous()
        or not b.is_contiguous()
        or a.shape[0] != b.shape[0]
        or a.shape[2] != b.shape[1]
        or min(*a.shape, *b.shape) <= 0
        or min(a.shape[1], b.shape[2]) <= 1
        or (a.shape[2] >= 8 and a.shape[2] % 8)
    ):
        return None
    effective = max(8, a.shape[2])
    transposes = a.shape[0] * effective * (a.shape[1] + b.shape[2]) * a.element_size()
    return (transposes if a.shape[2] < 8 else 0, transposes, effective)


# Routed rows per rank at EP>1, relative to balanced routing (see below).
_EP_ROUTED_ROW_ALLOWANCE = 1.5


def _moe_dispatcher_supported(
    dispatcher: Any, ep: int, *, alltoall: type, flex: type, hybridep: type
) -> bool:
    """EP1 all-to-all, or ART's HybridEP flex dispatcher across the EP group."""
    if ep == 1:
        return type(dispatcher) is alltoall
    return (
        type(dispatcher) is flex
        and type(getattr(dispatcher, "_comm_manager", None)) is hybridep
        and getattr(dispatcher, "ep_size", None) == ep
        and getattr(dispatcher, "tp_size", None) == 1
    )


def _hybridep_rows_per_rank(capacity: int, ranks: int) -> int:
    """HybridEP's allocated rows per rank: TMA-aligned, at least 512, padded to
    the 64-row combine chunk."""
    multiple = 4 // math.gcd(4, ranks)
    rows = max(-(-capacity // multiple) * multiple, 512)
    return -(-rows // 64) * 64


def _hybridep_buffer_bytes(capacity: int, ranks: int, hidden: int, experts: int) -> int:
    """Intranode HybridEP buffers for a per-rank token capacity.

    ``ranks`` is the ETPxEP communication group and ``experts`` its expert
    columns. Dispatch outputs alias the combine inputs (the shared-buffer
    default), sized for every rank's tokens routed to one rank: BF16 tokens,
    FP32 probabilities over the columns and FP32 FP8 scaling factors, which are
    allocated even without FP8. The routing-map allgather keeps one byte per
    column.
    """
    if capacity <= 0:
        return 0
    tokens = _hybridep_rows_per_rank(capacity, ranks) * ranks
    return tokens * (2 * hidden + 5 * experts + 4 * (hidden // 128))


# Transformer Engine's Hopper cuBLAS workspaces: one per grouped-GEMM stream
# (four) plus the plain GEMM's, each 32 MiB + 1 KiB.
_TE_CUBLAS_WORKSPACE_BYTES = 5 * (32 * 2**20 + 1024)


# Largest LoRA rank the dense stage prices (rank-wide intermediates included).
_DENSE_LORA_RANK_LIMIT = 256
# Packages whose module forwards the dense stage was traced through.
_TRACED_PACKAGES = (
    "torch.nn.",
    "transformer_engine.",
    "megatron.core.",
    "megatron.bridge.",
    "art.megatron.",
)


def _dense_no_grad_row_elements(
    ffn: int, hidden: int, activation_factor: int = 0
) -> int:
    """A dense no-grad layer's peak per row, in elements: its FC1 stage (the
    base GEMM output, the adapter output and their sum: 6F, or the SwiGLU
    live set if wider) beside the residual pair, embedding and norm output
    (4H). Qwen3.8-27B TP1/CP1 traces at 7k-174k rows."""
    return max(6, activation_factor) * ffn + 4 * hidden


def _dense_mlp_recompute_bytes_per_token(
    model: Sequence[torch.nn.Module],
    slot_ref: "LoRASlotRef | None" = None,
    *,
    hidden_size: int | None = None,
) -> tuple[int, int]:
    """Per-row dense MLP bytes: (gradient recompute stage, no-grad transient).

    Qwen3.8-27B CP2 allocator traces (dense, gated SwiGLU, LoRA on FC1 and FC2):

    - A recomputed layer's peak sits in its FC1 stage: the base output, the
      LoRA gate/up output and their sum (2F each) plus one F-wide tensor, 7F
      per row (6.83F measured, cold and warm, on main with #925). Early real
      q062 runs once showed one more FC1 triplet live (6F, as a recompile can
      leave a graph's outputs live); those traces show none, so it is not
      priced. Beside it the attention's q/k norm outputs and statistics
      (0.21H measured) are priced as H/4: with the residual, norm output and
      input gradient priced elsewhere, a layer's norms are 3.25H.
    - A no-grad layer holds the CP1 no-grad stage
      (``_dense_no_grad_row_elements``, 6F + 4H): a CP2 rank measured
      6F + 4.04H, priced with H/4 more.

    Both add the rank intermediates of every adapter in the layer. Every
    decoder layer, all it runs and ``slot_ref``'s adapters must match the
    traced execution (ART's own GDN layer, mixer and norm wrappers included);
    otherwise (0, 0) keeps today's allowances.
    """
    if len(model) != 1:
        return 0, 0
    try:
        decoder = _language_model(model[0]).decoder
    except (AttributeError, RuntimeError):
        return 0, 0
    layers = getattr(decoder, "layers", None)
    if not layers or not all(hasattr(layer, "mlp") for layer in layers):
        return 0, 0
    try:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.mlp import MLP
        from megatron.core.transformer.transformer_block import TransformerBlock
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.gdn.operator import (
            _empty_safe_norm_forward,
            _gdn_island_layer_forward,
            _prefix_tree_forward,
        )
        from art.megatron.lora import (
            LoRA,
            SelfAttentionLinearProjLoRA,
            SharedExpertsLinearFC1LoRA,
            SharedExpertsLinearFC2LoRA,
        )
    except ImportError:
        # Without the traced owner types nothing can match; keep the allowance.
        return 0, 0

    def plain(module: Any, wrapper: Any = None, delegate: str = "") -> bool:
        """No hooks, a class forward from the traced packages, no instance
        callable shadowing a class callable (an executed helper such as
        ``_forward_mlp``), and no forward but the class's or ART's traced
        wrapper, which must still call the class's own forward."""
        forward = vars(module).get("forward")
        if (
            module._forward_hooks
            or module._forward_pre_hooks
            or not type(module).forward.__module__.startswith(_TRACED_PACKAGES)
            or any(
                name not in ("forward", delegate)
                and callable(value)
                and callable(getattr(type(module), name, None))
                for name, value in vars(module).items()
            )
        ):
            return False
        if forward is None:
            return True
        inner = vars(module).get(delegate)
        # Training compile replaces the delegate with Dynamo's wrapper (the
        # traced run was compiled); judge the callable it wraps.
        while hasattr(inner, "_torchdynamo_orig_callable"):
            inner = inner._torchdynamo_orig_callable
        return (
            wrapper is not None
            and type(forward) is MethodType
            and forward.__self__ is module
            and forward.__func__ is wrapper
            and type(inner) is MethodType
            and inner.__self__ is module
            and inner.__func__ is type(module).forward
        )

    # The traced hybrid's exact mixer types (Qwen3.5-family attention is
    # Megatron Bridge's Qwen3VLSelfAttention); subclasses are unmeasured.
    mixers: set[type] = {SelfAttention, GatedDeltaNet}
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
            Qwen3VLSelfAttention,
        )
    except ImportError:
        pass
    else:
        mixers.add(Qwen3VLSelfAttention)

    def traced(module: Any) -> bool:
        """``plain``, or ART's empty-safe norm wrapper around the class forward."""
        return plain(module) or plain(
            module, _empty_safe_norm_forward, "_art_empty_safe_norm_physical_forward"
        )

    # The decoder and its final norm (run after the layers) are traced too.
    final = getattr(decoder, "final_layernorm", None)
    if (
        type(decoder) is not TransformerBlock
        or not plain(decoder)
        or (final is not None and not all(map(traced, final.modules())))
    ):
        return 0, 0
    expected = {
        "gated_linear_unit": True,
        "params_dtype": torch.bfloat16,
        "add_bias_linear": False,
        "sequence_parallel": False,
        # Fused SwiGLU (bias_swiglu_impl), as Megatron Bridge's Qwen3.5
        # providers configure it and the traced run executed.
        "bias_activation_fusion": True,
        "use_te_activation_func": False,
        "cpu_offloading": False,
        "cuda_graph_impl": "none",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    }
    width = hidden = rank = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        config = getattr(mlp, "config", None)
        fc1, fc2 = getattr(mlp, "linear_fc1", None), getattr(mlp, "linear_fc2", None)
        row = getattr(fc2, "row_parallel_lora", None)
        adapters = (
            getattr(fc1, "gate_lora", None),
            getattr(fc1, "up_lora", None),
            getattr(row, "lora", None),
        )
        sites = (
            (mlp, MLP),
            (fc1, SharedExpertsLinearFC1LoRA),
            (getattr(fc1, "linear_fc1", None), TELayerNormColumnParallelLinear),
            (fc2, SharedExpertsLinearFC2LoRA),
            (row, SelfAttentionLinearProjLoRA),
            (getattr(row, "linear_proj", None), TERowParallelLinear),
            *((adapter, LoRA) for adapter in adapters),
        )
        ffn = getattr(config, "ffn_hidden_size", None)
        size = getattr(config, "hidden_size", None)
        mixer = getattr(layer, "self_attention", None)
        if (
            type(layer) is not TransformerLayer
            or not plain(
                layer, _gdn_island_layer_forward, "_art_gdn_island_physical_forward"
            )
            or type(mixer) not in mixers
            or not plain(mixer, _prefix_tree_forward, "_art_physical_forward")
            or any(type(site) is not cls for site, cls in sites)
            or not all(plain(site) for site, _ in sites)
            or type(ffn) is not int
            or ffn <= 0
            or type(size) is not int
            or size <= 0
            or (hidden_size is not None and size != hidden_size)
            or getattr(fc1, "non_gated", None) is not False
            or getattr(fc1, "out_features", None) != 2 * ffn
            or any(
                type(getattr(config, name, None)) is not type(value)
                or getattr(config, name) != value
                for name, value in expected.items()
            )
            or getattr(config, "fp8", None)
            or getattr(config, "fp4", None)
            or getattr(config, "activation_func", None) is not torch.nn.functional.silu
            or getattr(mlp, "activation_func", None) is not torch.nn.functional.silu
            or getattr(config, "activation_func_clamp_value", None) is not None
            or getattr(config, "glu_linear_offset", 0.0) != 0.0
        ):
            return 0, 0
        # Everything else the layer runs, the mixer's children included, must
        # be the traced execution too: no hooks, and no forward but ART's
        # empty-safe norm wrapper. Every adapter must be an exact LoRA whose
        # selector is the one execution uses, within the priced rank.
        layer_rank = 0
        for child in layer.modules():
            if child is layer or child is mixer:
                continue
            if not traced(child):
                return 0, 0
            if not isinstance(child, LoRA):
                continue
            if type(child) is not LoRA or any(
                name in vars(child) for name in ("_slot", "active_lora_tensors")
            ):
                return 0, 0
            tensors = _slot_lora_tensors(child, slot_ref)
            if tensors is None:
                if slot_ref is None or slot_ref.name is None:
                    return 0, 0
                continue  # This slot has no adapter here: base output only.
            a, b = tensors
            if (
                not isinstance(a, torch.Tensor)
                or not isinstance(b, torch.Tensor)
                or a.ndim != 2
                or b.ndim != 2
                or a.shape[1] != b.shape[0]
                or not 0 < a.shape[1] <= _DENSE_LORA_RANK_LIMIT
            ):
                return 0, 0
            layer_rank += int(a.shape[1])
        rank = max(rank, layer_rank)
        width, hidden = max(width, ffn), max(hidden, size)
    # Each of a layer's adapters, the mixer's too, keeps its rank-wide input
    # product and gradient.
    adapters = 2 * rank
    return (7 * width + hidden // 4 + adapters) * 2, (
        _dense_no_grad_row_elements(width, hidden) + hidden // 4 + adapters
    ) * 2


def _moe_output_bytes_per_token(
    model: Sequence[torch.nn.Module],
    shape: ParallelShape,
    *,
    checkpoint_grad: bool = False,
    converted_stages: list[tuple[int, int]] | None = None,
    slot_ref: "LoRASlotRef | None" = None,
) -> int:
    """Known routed-expert working set, not a complete model/compiled bound."""
    # CP shards rows, not the per-token working set. At EP>1 only ART's
    # HybridEP flex dispatcher is modeled; TP and ETP are not.
    if (shape.tp, shape.etp) != (1, 1):
        return 0
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoELayer
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
        MoEFlexTokenDispatcher,
        _HybridEPManager,
    )

    from art.megatron.lora import LoRA, MLPExpertsLinearFC1LoRA, MLPExpertsLinearFC2LoRA

    # HybridEP hands each rank the pairs routed to its local experts, already
    # permuted. Balanced routing gives local tokens x top-k, as at EP1; a
    # pretrained CP2/EP2 run put about 1.35x that on one rank.
    routed_allowance = _EP_ROUTED_ROW_ALLOWANCE if shape.ep > 1 else 1
    coefficient = 0
    for chunk in model:
        for layer in chunk.modules():
            if not isinstance(layer, BaseMoELayer):
                continue
            experts = getattr(layer, "experts", None)
            fc2: Any = getattr(experts, "linear_fc2", None)
            lora: Any = getattr(fc2, "lora", None)
            dispatcher: Any = getattr(layer, "token_dispatcher", None)
            sites = (
                (layer, MoELayer),
                (experts, TEGroupedMLP),
                (fc2, MLPExpertsLinearFC2LoRA),
                (lora, LoRA),
                (getattr(fc2, "linear_fc2", None), TERowParallelGroupedLinear),
                (getattr(layer, "router", None), TopKRouter),
            )
            if any(
                type(site) is not expected
                or "forward" in vars(site)
                or getattr(site, "_forward_hooks", None)
                or getattr(site, "_forward_pre_hooks", None)
                for site, expected in sites
            ) or not _moe_dispatcher_supported(
                dispatcher,
                shape.ep,
                alltoall=MoEAlltoAllTokenDispatcher,
                flex=MoEFlexTokenDispatcher,
                hybridep=_HybridEPManager,
            ):
                return 0
            config = layer.config
            if (
                config.moe_expert_capacity_factor is not None
                or config.moe_pad_expert_input_to_capacity
                or config.moe_router_padding_for_quantization
                or config.fp8
                or config.fp4
                or config.moe_latent_size is not None
                or config.cuda_graph_impl != "none"
                or any(
                    name in vars(dispatcher)
                    for name in ("preprocess", "dispatch_postprocess")
                )
                or (
                    "dispatch_preprocess" in vars(dispatcher)
                    and not (
                        slot_ref is not None
                        and slot_ref.name is not None
                        and type(dispatcher.dispatch_preprocess) is partial
                        and dispatcher.dispatch_preprocess.func
                        is _moe_dispatch_preprocess
                        and dispatcher.dispatch_preprocess.args == (dispatcher,)
                        and not dispatcher.dispatch_preprocess.keywords
                    )
                )
                or (
                    "routing" in vars(layer.router)
                    and not hasattr(layer.router, "_art_rank_original_routing")
                )
            ):
                return 0
            tensors = _slot_lora_tensors(lora, slot_ref)
            # Enclosing row storage is still charged for an inactive adapter;
            # only selected tensors create converted weights.
            inputs, weights = tensors if tensors is not None else (lora.A_T, lora.B_T)
            if (
                weights.dtype not in (torch.float16, torch.bfloat16)
                or weights.shape[-1] != fc2.out_features
                or fc2.out_features != config.hidden_size
                or getattr(layer.router, "topk", None) != config.moe_router_topk
            ):
                return 0
            features = 2 * fc2.out_features
            enclosing_fc1 = None
            if (
                isinstance(inputs, torch.Tensor)
                and inputs.ndim == weights.ndim == 3
                and inputs.dtype == weights.dtype
                and inputs.shape[0] == weights.shape[0]
                and inputs.shape[-1] == weights.shape[-2]
                and inputs.shape[-2] > 0
            ):
                # Eager FC2 keeps x, base_out and adapter_out while producing
                # their sum. Compilers may reuse storage; other workspace is
                # not covered. Unknown input metadata keeps the prior pair.
                features = inputs.shape[-2] + 3 * fc2.out_features
                fc1 = getattr(experts, "linear_fc1", None)
                if (
                    type(fc1) is MLPExpertsLinearFC1LoRA
                    and "forward" not in vars(fc1)
                    and not getattr(fc1, "_forward_hooks", None)
                    and not getattr(fc1, "_forward_pre_hooks", None)
                    and fc1.fused_gate_up
                    and not fc1.non_gated
                    and fc1.out_features == 2 * inputs.shape[-2]
                    # HybridEP keeps one dispatched H-wide input where the
                    # EP1 all-to-all keeps two; charging two is conservative.
                    and getattr(dispatcher, "ep_size", None) == shape.ep
                    and getattr(dispatcher, "tp_size", None) == 1
                    and getattr(dispatcher, "num_local_experts", 0) > 1
                    and getattr(config, "moe_permute_fusion", False)
                    and getattr(experts, "offload_expert_fc1", None) is False
                    and getattr(experts, "offload_moe_act", None) is False
                    and getattr(experts, "activation_recompute", None) is False
                ):
                    # The two dispatched H-wide inputs and FC1 gate/up sum
                    # remain live at the FC2 sum, including in the observed
                    # compiled path. This is one stage, not a backward bound.
                    features += 2 * fc2.out_features + fc1.out_features
                    enclosing_fc1 = fc1
            shared = _shared_expert_output_bytes_per_token(layer)
            if (
                checkpoint_grad
                and shared
                and getattr(layer.shared_experts, "use_shared_expert_gate", False)
                is True
            ):
                # Gate-score backward saves a distinct pre-gate X. Charge it
                # beside this layer's returned X, not another layer's maximum.
                shared += shared
            routed_rows = math.ceil(config.moe_router_topk * routed_allowance)
            row_bytes = routed_rows * features * weights.element_size() + shared
            coefficient = max(coefficient, row_bytes)
            storage = _expert_lora_weight_storage(lora, slot_ref)
            if converted_stages is not None and storage is not None:
                padded, transposes, effective = storage
                saved_fc1, rank_fc1 = 0, 0
                routed_size = routed_rows * weights.element_size()
                if enclosing_fc1 is not None:
                    adapter = getattr(enclosing_fc1, "lora", None)
                    base = getattr(enclosing_fc1, "linear_fc1", None)
                    first = _expert_lora_weight_storage(adapter, slot_ref)
                    first_tensors = (
                        _slot_lora_tensors(adapter, slot_ref)
                        if first is not None
                        else None
                    )
                    if (
                        first is not None
                        and adapter is not None
                        and base is not None
                        and type(base) is TEColumnParallelGroupedLinear
                        and "forward" not in vars(base)
                        and not base._forward_hooks
                        and not base._forward_pre_hooks
                        and first_tensors is not None
                        and first_tensors[0].dtype == weights.dtype
                        and first_tensors[0].shape[:2]
                        == (weights.shape[0], fc2.out_features)
                        and first_tensors[1].shape[2] == enclosing_fc1.out_features
                    ):
                        first_padding, first_transposes, first_rank = first
                        # FC1 retains both routed H inputs and its base O1
                        # while producing adapter O1. Its sum is not live yet.
                        converted_stages.append(
                            (
                                routed_size
                                * (
                                    2 * fc2.out_features
                                    + 2 * enclosing_fc1.out_features
                                    + first_rank
                                )
                                + shared,
                                first_padding + first_transposes,
                            )
                        )
                        # At the subsequent sum, only grad-enabled execution
                        # retains padding/tmp; the two transposes have died.
                        converted_stages.append(
                            (
                                routed_size
                                * (
                                    2 * fc2.out_features
                                    + 3 * enclosing_fc1.out_features
                                    + (first_rank if checkpoint_grad else 0)
                                )
                                + shared,
                                first_padding if checkpoint_grad else 0,
                            )
                        )
                        if checkpoint_grad:
                            saved_fc1, _, rank_fc1 = first
                # At the second GEMM, the FC2 sum does not exist yet: replace
                # that H with tmp. Both weight transposes are still local.
                converted_stages.append(
                    (
                        row_bytes
                        + routed_size * (effective + rank_fc1 - fc2.out_features),
                        padded + transposes + saved_fc1,
                    )
                )
                if checkpoint_grad:
                    # The transposes die at return, but padding/tmp are saved
                    # through backward. Exact fused FC1 saves also remain live.
                    converted_stages.append(
                        (
                            row_bytes + routed_size * (effective + rank_fc1),
                            padded + saved_fc1,
                        )
                    )
                    if rank_fc1 and inputs is not None and inputs.shape[2] < effective:
                        # At FC2 backward return, nominal gradient copies
                        # coexist with effective gradients and FC1 saves.
                        # Unpadded returns alias; original parameters are not
                        # new storage. This is a checkpoint eager-stage floor.
                        experts_count, input_width, rank = inputs.shape
                        nominal = experts_count * rank * weights.element_size()
                        copies = nominal * (
                            input_width + (fc2.out_features if experts_count > 1 else 0)
                        )
                        converted_stages.append(
                            (
                                routed_size
                                * (
                                    2 * input_width
                                    + 2 * fc2.out_features
                                    + 2 * effective
                                    + rank_fc1
                                ),
                                padded
                                + transposes
                                + copies
                                + saved_fc1
                                + 2 * (experts_count + 1) * 4,
                            )
                        )
    return coefficient


# ``_memory``, ``_micro_batch_planner``, ``_optimizer`` and ``_slots`` hold
# TrainerRank method bodies and read ``_impl`` names such as ``Unset`` at
# definition time, so they are imported once those exist and before the class
# body binds their functions.
from art.trainer_rank import _memory, _micro_batch_planner, _optimizer, _slots


class TrainerRank:
    def __init__(
        self, runtime: TrainingRuntime, *, options: ForwardOptions | None = None
    ) -> None:
        planner_options = _planner_misses.parse_options(os.environ)
        self._allow_oversized_batches = planner_options.allow_oversized_batches
        self._planner_reporter = _planner_misses.Reporter(planner_options.threshold_pct)
        self._planner_observation_context: ContextVar[dict[str, Any] | None] = (
            ContextVar("trainer_rank_planner_observation", default=None)
        )
        self._planner_observation_generation = 0
        self._planner_active_observations: weakref.WeakValueDictionary[
            int, _PlannerObservation
        ] = weakref.WeakValueDictionary()
        self._planner_overlap_generation = 0
        self._planner_observation: dict[str, Any] | None = None
        pp_size = int(getattr(runtime.provider, "pipeline_model_parallel_size", 1) or 1)
        if pp_size > 1 or len(runtime.model) > 1:
            raise TrainerRankRuntimeSupportError(
                "TrainerRank does not use the MCore forward/backward schedule and "
                "therefore requires PP=1 with exactly one local model chunk; "
                f"got pp={pp_size}, chunks={len(runtime.model)}"
            )
        # Tensor parallelism is admitted: the vocab-parallel head, sequence-
        # parallel gather, TP padding of packed batches and sharded LoRA
        # gradient reduction pre-date the planner, memory checks all-reduce
        # within the TP x CP group, the memory profile is keyed by topology so
        # TP calibrates itself online, and the fitted layout cost model prices
        # TP explicitly. The cold retained-activation floor also distinguishes
        # tensor/sequence-parallel storage from gathered LoRA inputs.
        self._forward_options = options
        resolve_forward_options(options)
        self.runtime: TrainingRuntime = runtime
        self.device: torch.device = next(runtime.model[0].parameters()).device
        self._rng = TrainerRNG(self.device)
        self._param_dtype_size = _dtype_size(next(runtime.model[0].parameters()).dtype)
        try:
            metadata_model = _language_model(runtime.model[0])
        except RuntimeError:
            metadata_model = None
        self._hidden_size = _hidden_size(metadata_model, runtime.provider)
        self._padded_vocab_size = (
            None if metadata_model is None else _padded_vocab_size(metadata_model)
        )
        self._num_layers = int(
            getattr(getattr(metadata_model, "config", None), "num_layers", 0)
            or getattr(runtime.provider, "num_layers", 1)
            or 1
        )
        memory_config = getattr(metadata_model, "config", None) or runtime.provider

        def memory_field(name: str, default: Any = None) -> Any:
            return getattr(
                memory_config, name, getattr(runtime.provider, name, default)
            )

        self._recompute_granularity = memory_field("recompute_granularity", None)
        self._recompute_method = memory_field("recompute_method", None)
        self._recompute_num_layers = memory_field("recompute_num_layers", None)
        self._recompute_modules: frozenset[str] = frozenset(
            memory_field("recompute_modules", ()) or ()
        )
        self._sequence_parallel = bool(memory_field("sequence_parallel", False))
        self._attention_output_gate = bool(memory_field("attention_output_gate", False))
        # Native fused SwiGLU retains gate/up and the output (3F). Eager
        # unfused SwiGLU also retains SiLU and offset tensors (5F). Compilation
        # may fall back, so only the native fusion setting earns this discount.
        self._mlp_activation_factor = (
            3
            if memory_field("bias_activation_fusion", False)
            and not memory_field("use_te_activation_func", False)
            else 5 + 2 * (memory_field("activation_func_clamp_value", None) is not None)
        )
        # Layers that run the gated-delta-net path (Qwen3.5-4B: 24 of 32); the
        # cost model prices GDN state hand-offs per GDN layer, not per layer.
        self._gdn_layers = _gdn_layer_count(runtime.model[0])
        self._mixer_top_gaps = _mixer_top_gaps(runtime.model[0])
        if self._gdn_layers == 0 and bool(
            getattr(runtime.model_support_handler, "build_gdn_execution_spec", False)
        ):
            self._gdn_layers = self._num_layers
            self._mixer_top_gaps = (None, 0)
        self._lora_modules_per_layer = _lora_modules_per_layer(runtime.model[0])
        # A fitted layout cost table applies only to the execution classes it
        # was calibrated on (device class, dtype, model geometry, parallel
        # shape); other runtimes keep the previous score.
        parameter = next(runtime.model[0].parameters())
        capability: tuple[int, int] | None = None
        device_memory: int | None = None
        if parameter.device.type == "cuda":
            capability = torch.cuda.get_device_capability(parameter.device)
            device_memory = int(
                torch.cuda.get_device_properties(parameter.device).total_memory
            )
        self._planner_device_identity = {
            "capability": capability,
            "total_memory_bytes": device_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        spec = getattr(runtime, "model_support_spec", None)
        self._moe_layers = _moe_layer_count(runtime.model[0])
        self._dense_fc1_adapted = _dense_fc1_adapted(runtime.model[0])
        # Dense models whose every layer is the traced gated MLP price that
        # stage at CP2 instead, and with it one input gradient.
        (
            self._dense_recompute_bytes_per_token,
            self._dense_no_grad_bytes_per_token,
        ) = (
            (0, 0)
            if self._moe_layers
            else _dense_mlp_recompute_bytes_per_token(
                runtime.model, hidden_size=self._hidden_size
            )
        )
        self._checkpointed_moe_layers = sum(
            getattr(module, "moe_layer_recompute", False) is True
            for module in runtime.model[0].modules()
        )
        is_moe = bool(
            self._moe_layers
            or getattr(spec, "is_moe", False)
            or getattr(runtime.model_support_handler, "is_moe", False)
        )
        self._geometry = ModelGeometry.from_config(
            getattr(metadata_model, "config", None) or runtime.provider,
            has_gdn=self._gdn_layers > 0,
            has_moe=is_moe,
        )
        _dp, tp_size, cp_size, _pp = self._topology_key()
        ep_size, etp_size = _expert_parallel_shape(runtime.provider)
        self._parallel_shape = ParallelShape(
            tp=tp_size, cp=cp_size, ep=ep_size, etp=etp_size
        )
        forward_stages: list[tuple[int, int]] = []
        gradient_stages: list[tuple[int, int]] = []
        self._moe_output_bytes_per_token = (
            _moe_output_bytes_per_token(
                runtime.model, self._parallel_shape, converted_stages=forward_stages
            )
            if self._moe_layers
            else 0
        )
        # A declined component is distinct from a qualified cache later damaged.
        self._moe_memory_supported = self._moe_output_bytes_per_token > 0
        # Both modes inspect original owners before dispatcher caches are installed.
        self._moe_checkpoint_grad_bytes_per_token = (
            _moe_output_bytes_per_token(
                runtime.model,
                self._parallel_shape,
                checkpoint_grad=True,
                converted_stages=gradient_stages,
            )
            if self._moe_layers
            else 0
        )
        # Discard partial walks if a later layer has an unsupported owner.
        self._moe_forward_stages = (
            tuple(forward_stages) if self._moe_output_bytes_per_token else ()
        )
        self._moe_gradient_stages = (
            tuple(gradient_stages) if self._moe_checkpoint_grad_bytes_per_token else ()
        )
        selection = select_scoring(
            device_capability=capability,
            device_memory_bytes=device_memory,
            param_dtype=str(parameter.dtype),
            geometry=self._geometry,
            shape=self._parallel_shape,
        )
        self._coefficient_version = selection.version
        self._coefficient_table = selection.table_id
        self._planner_coefficients = selection.coefficients
        # Two-stage selection on re-ranked shapes (part of the table identity).
        self._planner_reranker = selection.reranker
        if selection.table_id is None:
            logger.warning(
                "TrainerRank layout cost model: runtime (capability=%s, device "
                "memory=%s, dtype=%s, geometry=%s, shape=%s) matches no calibrated "
                "execution class; using the version-%d score",
                capability,
                device_memory,
                parameter.dtype,
                self._geometry,
                self._parallel_shape,
                self._coefficient_version,
            )
        self._default_slot_ref: LoRASlotRef | None = None
        self._slot_stack: list[LoRASlotRef] = []
        self._checkpoint_slots: dict[str, _CheckpointSlot] = {}
        self._snapshot_checkpoint_names: set[str] = set()
        self._checkpoint_sources: dict[
            str,
            tuple[str, Callable[[], AbstractContextManager[PreparedCheckpoint]], bool],
        ] = {}
        self._checkpoint_snapshot_lru: OrderedDict[str, None] = OrderedDict()
        self._checkpoint_snapshot_cache_size = 2
        self._checkpoint_slot_writes: dict[str, int] = {}
        self._prepared_lora_exports: dict[str, tuple[str, _VllmLoraPublishInputs]] = {}
        self._checkpoint_prefetches: dict[str, Future[PreparedCheckpoint]] = {}
        self._checkpoint_prefetch_sources: dict[str, str] = {}
        self._checkpoint_prefetch_lock = threading.Lock()
        self._checkpoint_mutation_lock = threading.RLock()
        self._checkpoint_process_group: dist.ProcessGroup | None = None
        self._checkpoint_finalize_process_group: dist.ProcessGroup | None = None
        self._checkpoint_group_lock = threading.Lock()
        self._checkpoint_snapshot_spill: _SnapshotSpill | None = None
        self._checkpoint_prepare_lock = threading.Lock()
        self._checkpoint_finalize_lock = threading.Lock()
        self._checkpoint_save_condition = threading.Condition()
        self._checkpoint_save_sequence = 0
        self._checkpoint_save_next = 0
        self._checkpoint_save_skipped: set[int] = set()
        self._checkpoint_preparing_saves: set[str] = set()
        self._checkpoint_save_outcomes: dict[str, Literal["finish", "abort"]] = {}
        self._prepared_checkpoint_saves: dict[str, _PreparedSave] = {}
        self._finalized_checkpoint_saves: dict[str, _FinalizedSave] = {}
        self._pending_slot_graphs: dict[
            LoRASlotRef, list[weakref.ReferenceType[torch.Tensor]]
        ] = {}
        self._pending_hybridep_graphs: list[weakref.ReferenceType[torch.Tensor]] = []
        self._hybridep_graph_tracking = False
        self._hybridep_buffer_id: int | None = None
        self._hybridep_rows_high_water = 0
        self._cache_recovery_state = _CacheRecoveryState()
        self._memory_profiles: dict[_MemorySignature, _MemoryProfile] = {}
        self._graph_forward_times: OrderedDict[tuple[Any, ...], tuple[float, ...]] = (
            OrderedDict()
        )
        # Tracked peak-counter resets, and the latest (resets, peak) reading.
        self._peak_resets = 0
        self._peak_reading: tuple[int, int] | None = None
        self._split_memory_floors: dict[bytes, int] = {}
        self._split_memory_floor_status = "not_observed"
        self._last_global_micro_batch_size: int | None = None
        self._skipped_forward_waves: dict[object, tuple[int, int, int]] = {}
        # Bounded LRU: steady-state hits are temporally local (identical
        # content on consecutive calls); fresh-token training steps must not
        # accumulate entries for the lifetime of the rank.
        self._layout_selection_cache: OrderedDict[
            _LayoutKey,
            tuple[CanonicalPrefixTree, PrefixTreeLayout],
        ] = OrderedDict()
        self._tree_cache: OrderedDict[str, CanonicalPrefixTree] = OrderedDict()
        self._layout_cache_lock = threading.Lock()
        self._speculative_planner: ThreadPoolExecutor | None = None
        self._speculative_planning_future: Future[None] | None = None
        self._planning_seconds_accum = 0.0
        self._speculative_planning_seconds = 0.0
        self._last_forward_telemetry_snapshot: dict[str, Any] | None = None
        _configure_moe_dispatcher_caches(runtime.model)
        self.zero_grad()

    def zero_grad(self) -> None:
        for chunk in self.runtime.model:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                zero_grad_buffer()
        optimizer = self.runtime.optimizer
        if optimizer is not None:
            optimizer.zero_grad()
        for slot in self._checkpoint_slots.values():
            for param in slot.params:
                param.grad = None
        self._version_state().clear()
        self._prune_slot_graphs()

    def _version_state(self) -> CheckpointVersions:
        state = getattr(self, "_checkpoint_versions", None)
        if state is None:
            state = self._checkpoint_versions = CheckpointVersions(self)
        return state

    def _capture_checkpoint_version(self, name: str) -> CheckpointVersion:
        return self._version_state().capture(name)

    def _validate_checkpoint_version(
        self, version: CheckpointVersion, max_gradient_staleness: int = 2
    ) -> None:
        self._version_state().validate(version, max_gradient_staleness)

    def _snapshot_parameter(
        self,
        parameter: torch.nn.Parameter,
        version: CheckpointVersion,
        max_gradient_staleness: int = 2,
    ) -> torch.nn.Parameter:
        state = self._version_state()
        state.validate(version, max_gradient_staleness)
        with torch._C.DisableTorchFunctionSubclass():
            result = torch.nn.Parameter(
                parameter.detach().clone(), requires_grad=parameter.requires_grad
            )
        state.track(result, parameter, version, max_gradient_staleness)
        return result

    def _commit_versioned_gradients(
        self, gradients: Sequence[VersionedGradient]
    ) -> None:
        self._version_state().accumulate(gradients)

    def _gradient_transaction(
        self, *, before_commit: Callable[[Callable[[], None]], None] | None = None
    ) -> Any:
        return self._version_state().transaction(before_commit=before_commit)

    def _capture_lora_version(
        self,
        ref: LoRASlotRef | None,
        max_gradient_staleness: int = 2,
        *,
        origin: CheckpointVersion | None = None,
    ) -> LoRAVersion | None:
        if ref is None or ref.name is None or not torch.is_grad_enabled():
            return None
        from art.megatron.lora import LoRA, LoRASlot, LoRAVersion

        state = self._version_state()
        weight_version = state.capture(ref.name)
        version = weight_version if origin is None else origin
        if version.checkpoint != ref.name:
            raise ValueError("LoRA replay origin belongs to a different checkpoint")
        state.validate(version, max_gradient_staleness)
        key = (weight_version, version, max_gradient_staleness)
        if cached := state.lora.get(key):
            return cached
        slots: dict[int, LoRASlot] = {}
        for chunk in self.runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA) or id(module) in slots:
                    continue
                current = module._slot(ref)
                if current is None:
                    continue
                captured = slots[id(module)] = LoRASlot(
                    ref=ref,
                    a_t=current.A_T,
                    b_t=current.B_T,
                    alpha=current.alpha,
                    a_template=current.A_T,
                    b_template=current.B_T,
                    requires_grad=current.A_T.requires_grad,
                )
                for snapshot, parameter in zip(
                    (captured.A_T, captured.B_T),
                    (current.A_T, current.B_T),
                    strict=True,
                ):
                    snapshot.requires_grad_(parameter.requires_grad)
                    state.track(snapshot, parameter, version, max_gradient_staleness)
        captured_version = LoRAVersion(
            ref,
            version,
            slots,
            lambda: state.validate(version, max_gradient_staleness),
            weight_version,
        )
        state.lora[key] = captured_version
        return captured_version

    def _lora_version_capture_bytes(
        self,
        ref: LoRASlotRef | None,
        max_gradient_staleness: int = 2,
        *,
        origin: CheckpointVersion | None = None,
    ) -> int:
        if ref is None or ref.name is None or not torch.is_grad_enabled():
            return 0
        state = self._version_state()
        version = state.capture(ref.name)
        key = (version, version if origin is None else origin, max_gradient_staleness)
        if key in state.lora:
            return 0
        return sum(
            param.numel() * param.element_size()
            for param in self._checkpoint_slots[ref.name].params
            if not getattr(param, "_art_custom_checkpoint_param", False)
        )

    def _lora_gradient_staging_bytes(self, ref: LoRASlotRef | None) -> int:
        """Reserve current, staged and replacement gradients across later backwards.

        Several forwards can be admitted before their first backward creates
        ``.grad``. Existing gradients are already in the sampled memory baseline.
        Registered custom heads share this transaction, including streamed remote
        cotangents. Later registrations and arbitrary head activations are not
        predicted by an earlier model forward.
        """
        if ref is None or ref.name is None:
            return 0
        return _gradient_staging_bytes(self._checkpoint_slots[ref.name].params)

    def _pending_backward_memory(
        self, *, checkpoints: Iterable[str] = (), exclude_staging: Iterable[str] = ()
    ) -> tuple[int, int]:
        """Return additional restore and gradient bytes beside sampled live storage."""
        cache = getattr(self, "_graph_cache", None)
        states = () if cache is None else tuple(cache.state(h) for h in cache.handles())
        names = set(checkpoints) | {
            version.checkpoint
            for state in states
            for version in getattr(state, "checkpoint_versions", ())
        }
        excluded = set(exclude_staging)
        staging = sum(
            self._lora_gradient_staging_bytes(self._slot_ref(name))
            for name in names - excluded
            if name in self._checkpoint_slots
        )
        # Head losses can remain live without a model cache record. Preserve their
        # registration reserve without charging unrelated unused LoRA targets.
        staging += _gradient_staging_bytes(
            parameter
            for name, slot in self._checkpoint_slots.items()
            if name not in names | excluded
            for parameter in slot.params
            if getattr(parameter, "_art_custom_checkpoint_param", False)
        )
        return (
            max(
                (getattr(state, "restore_workspace_bytes", 0) for state in states),
                default=0,
            ),
            staging,
        )

    def module(
        self,
        name: str,
        factory: Callable[[], ModuleT],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> ModuleHandle:
        """Return a checkpoint-owned module, registering it on first access.

        Registration is collective across TrainerRank processes. The returned module is
        bound to the resolved checkpoint and is not selected by later push/pop calls.
        """
        value = self._custom_object(name, "module", factory, checkpoint=checkpoint)
        return cast("ModuleHandle", value)

    def parameter(
        self,
        name: str,
        factory: Callable[[], torch.Tensor | torch.nn.Parameter],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> torch.nn.Parameter:
        """Register or retrieve a checkpoint-owned trainable tensor.

        The tensor is replicated across TrainerRank processes.
        """
        value = self._custom_object(name, "parameter", factory, checkpoint=checkpoint)
        return cast(torch.nn.Parameter, value)

    def buffer(
        self,
        name: str,
        factory: Callable[[], torch.Tensor],
        *,
        checkpoint: AdapterSelection = Unset,
    ) -> torch.Tensor:
        """Register or retrieve a checkpoint-owned persistent buffer.

        The tensor is replicated across TrainerRank processes.
        """
        value = self._custom_object(name, "buffer", factory, checkpoint=checkpoint)
        return cast(torch.Tensor, value)

    def _custom_object(
        self,
        name: str,
        kind: Literal["module", "parameter", "buffer"],
        factory: Callable[[], object],
        *,
        checkpoint: AdapterSelection,
    ) -> object:
        self._guard_forward_collective(kind)
        from . import _checkpoint

        group = self._checkpoint_group()
        error: BaseException | None = None
        checkpoint_name: str | None = None
        try:
            if not isinstance(name, str) or not name or "." in name or "/" in name:
                raise ValueError(
                    "Custom checkpoint object name must be non-empty and contain "
                    "neither '.' nor '/'"
                )
            if not callable(factory):
                raise TypeError("Custom checkpoint object factory must be callable")
            checkpoint_name = self._resolve_custom_checkpoint(checkpoint)
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(
            error, "validate custom object registration", group
        )
        assert checkpoint_name is not None
        identity = (checkpoint_name, name, kind)
        if any(value != identity for value in _checkpoint._gather(identity, group)):
            raise TrainerRankSlotStateError(
                "Custom checkpoint object registration differs across ranks"
            )
        self._rng.synchronize(caller_group())
        slot = self._checkpoint_slots[checkpoint_name]
        existing = slot.custom.get(name)
        registered = None if existing is None else existing.kind
        if any(value != registered for value in _checkpoint._gather(registered, group)):
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {name!r} is not registered consistently "
                "across ranks"
            )
        if existing is not None:
            if existing.kind != kind:
                raise TrainerRankSlotStateError(
                    f"Checkpoint {checkpoint_name!r} already registers {name!r} "
                    f"as a {existing.kind}, not a {kind}"
                )
            return existing.handle if existing.handle is not None else existing.value
        custom: _CustomObject | None = None
        try:
            value = factory()
            generation = object()
            if kind == "module":
                if not isinstance(value, torch.nn.Module):
                    raise TypeError("module() factory must return torch.nn.Module")
                from ._heads import move_module

                value = move_module(deepcopy(value), self.device)
                if slot.snapshot:
                    value.requires_grad_(False)
            elif kind == "parameter":
                if not isinstance(value, torch.Tensor):
                    raise TypeError("parameter() factory must return torch.Tensor")
                value = torch.nn.Parameter(
                    value.detach().to(device=self.device).clone(),
                    requires_grad=not slot.snapshot,
                )
            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError("buffer() factory must return torch.Tensor")
                value = value.detach().to(device=self.device).clone()
            custom = _CustomObject(kind, value, generation)
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(error, f"construct custom object {name!r}", group)
        assert custom is not None
        self._initialize_custom_object(checkpoint_name, name, custom)
        tracker: _CustomTensorTracker | None = None
        named_params: tuple[tuple[str, torch.nn.Parameter], ...] = ()
        new_params: tuple[torch.nn.Parameter, ...] = ()
        extended_optimizer: _DynamicOptimizer | None = None
        error = None
        try:
            tracker = _CustomTensorTracker(
                weakref.ref(self),
                self._slot_ref(checkpoint_name),
                name,
                custom.generation,
            )
            custom = _track_custom_object(custom, tracker)
            named_params = tuple(
                (key, parameter)
                for key, parameter in _custom_named_parameters(name, custom)
                if parameter.requires_grad
            )
            new_params = tuple(parameter for _key, parameter in named_params)
            self._tag_custom_parameters(new_params)
            if slot.optimizer is not None and new_params:
                extended_optimizer = self._extend_dynamic_optimizer(
                    checkpoint_name, named_params
                )
            self._admit_custom_gradient_storage(checkpoint_name, new_params)
        except BaseException as exc:
            error = exc
        try:
            _checkpoint.raise_distributed(
                error, f"stage custom checkpoint object {name!r}", group
            )
        except BaseException:
            # A retained registration traceback must not own rejected tensors.
            value = custom = tracker = extended_optimizer = None
            named_params = new_params = ()
            raise
        assert tracker is not None and custom is not None
        if extended_optimizer is not None:
            slot.optimizer = extended_optimizer
        slot.custom[name] = custom
        slot.params += new_params
        tracker.active = True
        return custom.handle if custom.handle is not None else custom.value

    def _admit_custom_gradient_storage(
        self, checkpoint: str, parameters: Sequence[torch.nn.Parameter]
    ) -> None:
        """Price known new targets beside existing graph restore reservations."""
        try:
            if not self._graph_memory_policy_enabled():
                return
            workspace, staging = self._pending_backward_memory(
                checkpoints=(checkpoint,)
            )
            required = _gradient_staging_bytes(parameters) + staging + workspace
            available = self._available_memory_bytes()
            if required > available:
                raise TrainerRankMemoryError(
                    f"Registering custom parameters needs {required} GPU bytes for "
                    f"gradient staging and existing graph restoration; available={available}"
                )
        finally:
            parameters = ()

    def _initialize_custom_object(
        self,
        checkpoint: str,
        name: str,
        custom: _CustomObject,
    ) -> None:
        from . import _checkpoint

        slot = self._checkpoint_slots[checkpoint]
        group = self._checkpoint_group()
        error: BaseException | None = None
        values: dict[str, torch.Tensor] = {}
        try:
            loaded = _checkpoint.load_custom_tensors(
                slot.custom_payload, name, custom.kind
            )
            if loaded is not None:
                record, tensors = loaded
                _validate_custom_schema(name, custom, record)
                _load_custom_state(custom, tensors)
            values = _custom_state(custom)
        except BaseException as exc:
            error = exc
        _checkpoint.raise_distributed(
            error, f"initialize custom object {name!r}", group
        )
        signature = _custom_signature(name, custom, values)
        signatures = _checkpoint._gather(signature, group)
        if any(value != signatures[0] for value in signatures):
            raise TrainerRankSlotStateError(
                f"Custom checkpoint object {name!r} differs across ranks"
            )
        payloads = _checkpoint._gather(
            {key: value.detach().cpu() for key, value in values.items()}, group
        )
        _load_custom_state(custom, payloads[0])

    @staticmethod
    def _tag_custom_parameters(params: Sequence[torch.nn.Parameter]) -> None:
        for param in params:
            setattr(param, "_art_custom_checkpoint_param", True)
            setattr(param, "allreduce", True)
            setattr(param, "lora_tp_sharded", False)
            setattr(param, "lora_shard_domain", "tp")
            setattr(param, "grad_sync_domain", "tp_default")
            setattr(param, "grad_sync_op", "avg")

    @staticmethod
    async def _await_checkpoint_prefetch(
        future: Future[T],
    ) -> T:
        return await asyncio.shield(asyncio.wrap_future(future))

    def snapshot_checkpoint(self, source: str, destination: str) -> bool:
        """Clone a loaded checkpoint into a forward-only resident snapshot."""
        self._guard_forward_collective("snapshot_checkpoint")
        from . import _checkpoint

        self._ensure_checkpoint_slots((source,))
        return _checkpoint.snapshot_checkpoint(self, source, destination)

    def push_checkpoint(
        self, checkpoint: str | MaterializedCheckpoint | None
    ) -> PushedCheckpoint:
        logical, directory = self._checkpoint_source(checkpoint)
        return PushedCheckpoint(self, logical, directory)

    def save_checkpoint(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        self.prepare_checkpoint_save(output_dir, checkpoint_path)
        self.finish_checkpoint_save(output_dir)

    def prepare_checkpoint_save(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> None:
        self._guard_forward_collective("prepare_checkpoint_save")
        from . import _checkpoint

        _checkpoint.prepare_checkpoint_save(
            self, output_dir, self._resolve_checkpoint_name(checkpoint_path)
        )

    def finish_checkpoint_save(self, output_dir: str) -> None:
        self._guard_forward_collective("finish_checkpoint_save")
        from . import _checkpoint

        _checkpoint.finish_checkpoint_save(self, output_dir)

    def abort_checkpoint_save(self, output_dir: str) -> None:
        self._guard_forward_collective("abort_checkpoint_save")
        from . import _checkpoint

        _checkpoint.abort_checkpoint_save(self, output_dir)

    def export_lora(
        self,
        output_dir: str,
        checkpoint_path: str | Literal["active"] = "active",
    ) -> int:
        self._guard_forward_collective("export_lora")
        from . import _lora_export

        return _lora_export.export_lora(
            self, output_dir, self._resolve_checkpoint_name(checkpoint_path)
        )

    def _prepare_lora_export(
        self,
        export_id: str,
        checkpoint_path: str | Literal["active"] = "active",
        *,
        owner_id: str,
    ) -> tuple[int, dict[str, float]]:
        from . import _lora_export

        return _lora_export.prepare_lora_export(
            self,
            export_id,
            self._resolve_checkpoint_name(checkpoint_path),
            owner_id=owner_id,
        )

    def _finish_lora_export(
        self, export_id: str, output_dir: str, *, owner_id: str
    ) -> dict[str, float]:
        from . import _lora_export

        return _lora_export.finish_lora_export(
            self, export_id, output_dir, owner_id=owner_id
        )

    def _abort_lora_export(self, export_id: str, *, owner_id: str) -> None:
        from . import _lora_export

        _lora_export.abort_lora_export(self, export_id, owner_id=owner_id)

    @staticmethod
    def _checkpoint_source_key(path: str) -> str:
        return str(Path(path).resolve())

    @staticmethod
    def _checkpoint_source(
        checkpoint: str | MaterializedCheckpoint | None,
    ) -> tuple[str | None, str | None]:
        if isinstance(checkpoint, MaterializedCheckpoint):
            return checkpoint.path, checkpoint.directory
        return checkpoint, checkpoint

    @staticmethod
    def _slot_state_error(message: str) -> TrainerRankSlotStateError:
        return TrainerRankSlotStateError(message)

    @overload
    def forward_batches(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
            ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
        ]
    ]: ...

    @overload
    def forward_batches(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
            Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        ]
    ]: ...

    @overload
    def forward_batches(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            Sequence[Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]],
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]],
        ]
    ]: ...

    @overload
    def forward_batches(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[
        MicroBatch[
            Sequence[
                Sequence[
                    Sequence[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ],
            Sequence[
                Sequence[
                    Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ],
        ]
    ]: ...

    def forward_batches(
        self,
        inputs: Iterable[ForwardInputs],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
        yield_empty: bool = False,
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        """Forward replicated inputs in adaptive data-parallel microbatches.

        Per-input checkpoints and `no_grad` values override the method defaults.
        `no_grad=None` inherits the ambient PyTorch grad mode; `True` disables
        grads and `False` enables them.
        Input and target tensors may be on a different device from the trainer;
        ART moves its packed model inputs and labels internally without mutating
        the caller-owned `ForwardInput` objects.

        Per-position outputs contain the full flattened input sequence in source
        order, including with context parallelism. Logical callbacks execute
        once per DP rank; use `backward(loss)` to route cotangents to internal
        TP/CP participants. Direct physical callers must invoke matching
        forwards and backwards on their TP/CP peers. `reduce` combines only
        distinct data-parallel batches.

        Model PyTorch randomness advances separately from caller randomness,
        seeded when the physical TrainerRank is constructed. Direct physical
        callers continue their TP/CP leader's default CPU and trainer-device CUDA
        streams before each yield/forward return and custom-object factory. This
        keeps matching random masks and custom-head dropout consistent without
        synchronizing DP workers. Python/NumPy RNGs, explicit generators, other
        devices, concurrent RNG use and rank-dependent control flow are outside
        this contract. Checkpoint saves do not persist RNG state; activation
        checkpointing must preserve RNG for correct recomputation.

        Empty local microbatches are skipped unless `yield_empty=True`. Every
        rank must use the same setting. When a wave skips ranks, TrainerRank
        collective methods raise if called from its loop body; fully populated
        waves permit them. Use `yield_empty=True` for per-wave collectives,
        including reductions on ranks with no outputs. Exhaust or close a retained
        iterator before making collective calls after an early exit. Guards apply
        on the iterator's thread; raw torch.distributed calls are not guarded.
        Collective calls must still match across ranks.

        Admission learns each call's whole peak, including the caller's loss
        and backward. For grad-enabled single-target requests of at least 64
        tokens under one-layer full recompute, it prices model memory by packed
        rows and charges 6 KiB per logical token for the head plus the caller's
        loss saves and backward transients. A caller whose per-token head and
        loss memory peaks above that is unsupported: shared plans can exceed
        their estimate. Backward is learned only when it runs inside the yield;
        a backward deferred past it is not. A wave with another TrainerRank
        forward inside its yield cannot lower later estimates below the
        signature's first wave.
        """
        if not isinstance(yield_empty, bool):
            raise TypeError("yield_empty must be a bool")
        enabled = torch.is_grad_enabled() if no_grad is None else not no_grad
        inputs = cast(
            Iterable[ForwardInputs], self._capture_forward_options(inputs, options)
        )
        batches = self._forward_batches(
            inputs, checkpoint=checkpoint, yield_empty=yield_empty
        )
        return self._yield_forward_batches(
            batches, enabled=enabled, yield_empty=yield_empty
        )

    def _yield_forward_batches(
        self,
        batches: Generator[MicroBatch[ForwardInputs, ForwardOutputs], None, None],
        *,
        enabled: bool,
        yield_empty: bool,
    ) -> Iterator[MicroBatch[ForwardInputs, ForwardOutputs]]:
        token = object()
        try:
            while True:
                self._guard_forward_collective("forward_batches")
                with torch.set_grad_enabled(enabled), self._rng.model():
                    try:
                        batch = next(batches)
                    except StopIteration:
                        return
                if not yield_empty and not batch.outputs:
                    continue
                self._rng.synchronize(caller_group())
                if (
                    not yield_empty
                    and batch.stats.global_count < self._dp_rank_and_size()[1]
                ):
                    self._skipped_forward_waves[token] = (
                        threading.get_ident(),
                        batch.stats.global_start,
                        batch.stats.global_stop,
                    )
                try:
                    yield batch
                finally:
                    self._skipped_forward_waves.pop(token, None)
                    del batch
        finally:
            batches.close()

    def _capture_forward_options(
        self, inputs: ForwardInputs, options: ForwardOptions | None
    ) -> ForwardInputs:
        from ._graphs import _snapshot

        # Input enumeration already happens at submission; only execution is
        # lazy. Own the submitted tensor storage before returning an iterator.
        materialized = _snapshot(_materialize(inputs))
        constructor = getattr(self, "_forward_options", None)

        def capture(value: ForwardInputs) -> ForwardInputs:
            if isinstance(value, ForwardInput):
                if constructor is None and options is None:
                    return replace(value)
                resolved = resolve_forward_options(constructor, options, value.options)
                return replace(value, options=ForwardOptions(**vars(resolved)))
            return _rebuild_forward_tree(value, [capture(child) for child in value])

        return capture(materialized)

    def _guard_forward_collective(self, operation: str) -> None:
        for thread, start, stop in tuple(self._skipped_forward_waves.values()):
            if thread == threading.get_ident():
                raise RuntimeError(
                    f"{operation} cannot run during forward_batches wave "
                    f"[{start}, {stop}): yield_empty=False skips some data-parallel "
                    "ranks. Move collective calls after the iterator or use "
                    "yield_empty=True on every rank."
                )

    @_backward_region
    def _release_cached_memory_for_backward(
        self, plan: _AnyForwardPlan, *, error: BaseException | None = None
    ) -> None:
        # Every WORLD wave reaches this before the public iterator skips empty
        # outputs. Forward has already executed: never replan or retry here.
        with self._cache_recovery_episode(error=error) as (owner, started):
            exchange_error: BaseException | None = None
            try:
                failed, gradients = self._recovery_reduce(
                    [
                        float(error is not None),
                        float(any(group.grad_enabled for group in plan.groups)),
                    ],
                    op="MAX",
                    sync_across_dp=True,
                )
            except BaseException as exc:
                if error is None:
                    raise
                exchange_error = exc
            if error is not None:
                raise self._memory_error_with_reduction_note(error, exchange_error)
            if failed:
                raise RuntimeError("Forward failed on another rank before handoff")
            if not gradients:
                return
            self._try_cache_recovery(
                None,
                sync_across_dp=True,
                owner=owner,
                started=started,
                handoff_grad=any(group.grad_enabled for group in plan.groups),
            )

    @overload
    def forward(
        self,
        inputs: ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
    ]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
    ]: ...

    @overload
    def forward(
        self,
        inputs: Iterable[
            Iterable[
                Iterable[
                    Iterable[ForwardInput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]
                ]
            ]
        ],
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> Sequence[
        Sequence[
            Sequence[Sequence[ForwardOutput[LogprobsT, TopKT, LogitsT, HiddenStatesT]]]
        ]
    ]: ...

    def forward(
        self,
        inputs: ForwardInputs,
        *,
        options: ForwardOptions | None = None,
        checkpoint: AdapterSelection = Unset,
        no_grad: bool | None = None,
    ) -> ForwardOutputs:
        """Forward inputs already local to this data-parallel rank.

        Outputs contain full sequences in source order on every TP/CP rank,
        with the same loss and reduction contract as `forward_batches`.

        Per-input checkpoints and `no_grad` values override the method defaults.
        `no_grad=None` inherits the ambient PyTorch grad mode; `True` disables
        grads and `False` enables them.
        Input and target tensors may be on a different device from the trainer;
        ART moves its packed model inputs and labels internally without mutating
        the caller-owned `ForwardInput` objects.
        """
        self._guard_forward_collective("forward")
        backward = self._backward_work()
        if backward is not None:
            backward.harvest()
        enabled = torch.is_grad_enabled() if no_grad is None else not no_grad
        with torch.set_grad_enabled(enabled):
            # Caller iterators may draw their own inputs; only ART's internal
            # execution belongs to the private model stream.
            materialized = self._capture_forward_options(inputs, options)
            requests = list(_flatten(materialized))
        error: BaseException | None = None
        try:
            with torch.set_grad_enabled(enabled), self._rng.model():
                self._reset_planning_telemetry()
                plan, check = self._plan_admissible_forward(
                    requests, checkpoint=checkpoint, context="forward"
                )
                tracked_outputs = self._execute_admitted_plan(
                    plan, check=check, context="forward"
                )
                outputs = _unflatten(materialized, iter(tracked_outputs))
        except BaseException as exc:
            error = exc
        # Failed peers must leave this frontier before the command layer's
        # error exchange, just as successful peers do. Caller RNG is restored
        # by model() before this collective, including on execution failure.
        try:
            self._rng.synchronize(caller_group())
        except BaseException as sync_error:
            if error is None:
                raise
            self._memory_error_with_reduction_note(
                error, sync_error, operation="RNG synchronization"
            )
        if error is not None:
            raise error
        return outputs

    def _execute_admitted_plan(
        self, plan: _AnyForwardPlan, *, check: _MemoryCheck, context: str
    ) -> list[AnyForwardOutput]:
        if isinstance(plan, _FlatForwardPlan):
            outputs, _baseline = self._run_flat_plan_with_memory_tracking(
                plan, check=check, context=context
            )
        else:
            outputs, _baseline, _peak = self._execute_split_plan_with_memory_tracking(
                plan, check=check, context=context
            )
        # This API calibrates only forward, unlike the yielded microbatch API.
        self._complete_planner_observation(phase="forward")
        return outputs

    @contextmanager
    def _forward_handoff(self, *, advance: bool) -> Iterator[None]:
        # Only intermediate TP/CP frontiers can race the next model collective.
        group = caller_group() if advance else None
        if group is None or dist.get_world_size(group) == 1:
            yield
            return
        error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            error = exc
        try:
            (failed,) = self._recovery_reduce(
                [float(error is not None)], op="MAX", sync_across_dp=False
            )
        except BaseException as exchange_error:
            if error is None:
                raise
            self._memory_error_with_reduction_note(
                error, exchange_error, operation="forward handoff"
            )
        else:
            if error is None and failed:
                raise RuntimeError("Forward handoff failed on another rank")
        if error is not None:
            raise error

    def _discard_forward_graphs(
        self, previous: tuple[str, ...], error: BaseException
    ) -> None:
        cache = getattr(self, "_graph_cache", None)
        if cache is not None:
            previous_handles = set(previous)
            for handle in cache.handles():
                if handle not in previous_handles:
                    try:
                        cache.release(handle)
                    except BaseException as cleanup_error:
                        self._memory_error_with_reduction_note(
                            error, cleanup_error, operation="forward graph release"
                        )

    @_backward_region
    def _execute_split_plan_with_memory_tracking(
        self, plan: _SplitForwardPlan, *, check: _MemoryCheck, context: str
    ) -> tuple[list[AnyForwardOutput], int | None, int]:
        state = self._recovery_state()
        work_before = state.work
        self._begin_planner_observation(plan, check)
        previous = self._graph_cache.handles() if hasattr(self, "_graph_cache") else ()
        outputs: list[AnyForwardOutput] = []
        output: AnyForwardOutput | None = None
        merged: list[AnyForwardOutput | None] = []
        self._planner_observing_split = True
        try:
            baseline, peak = None, 0
            merged = [None] * plan.request_count
            for ordinal, (subforward, indices) in enumerate(
                zip(plan.subforwards, plan.request_indices, strict=True)
            ):
                with self._forward_handoff(advance=ordinal + 1 < plan.subforward_count):
                    try:
                        outputs, child_baseline = (
                            self._run_flat_plan_with_memory_tracking(
                                subforward, check=check, context=context
                            )
                        )
                        if child_baseline is not None:
                            if baseline is None:
                                baseline = child_baseline
                            peak = max(
                                peak, int(torch.cuda.max_memory_allocated(self.device))
                            )
                    except TrainerRankMemoryError as error:
                        # Model execution already began, so no replanning is possible
                        # and the caller must not mistake this for an up-front refusal.
                        raise TrainerRankPartialExecutionError(
                            f"{context}: subforward {ordinal + 1} of "
                            f"{plan.subforward_count} failed during execution "
                            f"({ordinal} of {plan.subforward_count} completed). {error}",
                            predicted_peak_bytes=error.predicted_peak_bytes,
                            usable_limit_bytes=error.usable_limit_bytes,
                            suggestion=error.suggestion,
                        ) from error
                    for index, output in zip(indices, outputs, strict=True):
                        merged[index] = output
            if any(output is None for output in merged):
                raise AssertionError("split execution did not cover every request")
            return cast(list[AnyForwardOutput], merged), baseline, peak
        except BaseException as error:
            outputs.clear()
            merged.clear()
            output = None
            self._discard_forward_graphs(previous, error)
            state.work = work_before
            raise
        finally:
            self._planner_observing_split = False

    def _expert_parallel_active(self) -> bool:
        try:
            from megatron.core import parallel_state as ps

            return int(ps.get_expert_model_parallel_world_size()) > 1
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return False

    def _head_projection_rows(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        positions: Sequence[torch.Tensor] | None = None,
        lower_bound: bool = False,
        uncapped: bool = False,
    ) -> int:
        """Per-group logical bounds or exact packed union; no device-label read.

        Capped at one head chunk unless ``uncapped`` (every projected row, for
        pricing each chunk's statistics path).

        A single sequence's valid rows cannot alias each other. Across requests
        they may share: max is a lower bound, sum an upper bound. Ignore labels
        only when every label on that input row is -100, as projection does.
        Device-label validity is unknown: use all rows for capacity, zero only
        for the rejection lower bound; never copy labels from the device here.
        """
        if not self._head_workspace_bytes(1):
            return 0
        if positions is not None and any(row.device.type != "cpu" for row in positions):
            positions = None  # Capacity bound without reading device positions.
        counts: list[int] = []
        projected: set[int] = set()
        for index, request in enumerate(requests):
            offsets = None
            if request.logits or request.top_k is not None:
                count = int(request.input_tokens.numel())
            elif request.target_tokens is not None:
                count = int(request.input_tokens.numel())
                if request.target_tokens.device.type != "cpu":
                    if lower_bound:
                        count = 0
                else:
                    labels = request.target_tokens.to(dtype=torch.long)
                    valid = (labels != -100).reshape(count, -1).any(dim=1)
                    offsets = torch.nonzero(valid, as_tuple=False).reshape(-1)
                    count = int(offsets.numel())
            else:
                continue
            if positions is None:
                counts.append(count)
            else:
                row = positions[index]
                if offsets is not None:
                    row = row.index_select(0, offsets)
                for position in row.tolist():
                    projected.add(int(position))
                    if len(projected) >= _HEAD_CHUNK_TOKENS and not uncapped:
                        return _HEAD_CHUNK_TOKENS
        rows = (
            (max(counts, default=0) if lower_bound else sum(counts))
            if positions is None
            else len(projected)
        )
        return rows if uncapped else min(_HEAD_CHUNK_TOKENS, rows)

    def _head_target_chunk_rows(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        positions: Sequence[torch.Tensor] | None = None,
        lower_bound: bool = False,
    ) -> int:
        """Largest projected chunk reached by labelled rows, or row bounds.

        Mixed outputs do not remove target backward. Its dense indexing result
        spans the whole projected chunk, including rows requested only as logits
        or top-k. Ignored labels still execute backward if another output
        projects their rows. Without a layout, valid rows give a rejection lower
        bound; possible overlap with any labelled request gives capacity.
        """
        targets = tuple(
            replace(request, logits=False, top_k=None) for request in requests
        )
        if positions is None or any(row.device.type != "cpu" for row in positions):
            if lower_bound:
                return self._head_projection_rows(targets, lower_bound=True)
            return (
                self._head_projection_rows(requests)
                if any(
                    request.target_tokens is not None and request.input_tokens.numel()
                    for request in requests
                )
                else 0
            )
        projected: set[int] = set()
        labelled: set[int] = set()
        for request, row in zip(requests, positions, strict=True):
            target_row = row[:0]
            if request.target_tokens is not None and int(row.numel()):
                labelled.update(row.tolist())
                labels = request.target_tokens
                if labels.device.type == "cpu":
                    valid = (
                        (labels.to(dtype=torch.long) != -100)
                        .reshape(len(row), -1)
                        .any(dim=1)
                    )
                    target_row = row.index_select(
                        0, torch.nonzero(valid, as_tuple=False).reshape(-1)
                    )
                elif not lower_bound:
                    target_row = row
            projected.update(
                (
                    row if request.logits or request.top_k is not None else target_row
                ).tolist()
            )
        targeted = labelled & projected
        if not targeted:
            return 0
        first_target = min(targeted)
        first_index = sum(position < first_target for position in projected)
        chunk_start = first_index // _HEAD_CHUNK_TOKENS * _HEAD_CHUNK_TOKENS
        return min(_HEAD_CHUNK_TOKENS, len(projected) - chunk_start)

    def _sequence_parallel_floor_covered(self, tp: int, cp: int) -> bool:
        """Whether the explicit TP x SP checkpoint floor applies to this model.

        Traced on dense Qwen3.8-27B (64 layers, GDN and gated attention) at
        TP4 and at TP2 with sequence parallelism and CP1. The floor covers
        each rank's boundary shards and the recomputed layer's measured
        workspace with explicit terms (``_sequence_parallel_workspace_bytes``),
        so it needs readable mixer geometry but, unlike the earlier repeated-
        boundary cover, no depth/width bound: shallower and wider dense models
        are now eligible and priced by the same terms (untraced). Other TP
        sizes, CP, MoE and replicated QKV (KV groups below TP) keep today's
        pricing.
        """
        geometry = self._geometry
        if tp not in (2, 4) or cp != 1 or self._moe_layers or geometry.moe_experts:
            return False
        if self._num_layers > self._gdn_layers and (
            geometry.num_attention_heads <= 0
            or geometry.kv_channels <= 0
            # Replicated QKV keeps a global QKV output on every rank.
            or not tp <= geometry.num_query_groups
        ):
            return False
        gdn_widths = (
            geometry.gdn_key_heads,
            geometry.gdn_key_head_dim,
            geometry.gdn_value_heads,
            geometry.gdn_value_head_dim,
            geometry.gdn_conv_kernel,  # Prices each segment's conv history.
        )
        return not (self._gdn_layers and min(gdn_widths) <= 0)

    def _subforward_cost(
        self,
        *,
        packed_tokens: int,
        output_bytes: int,
        signature: _MemorySignature,
        logical_tokens: int,
        gdn_segments: int = 0,
        group_rows: tuple[tuple[int, bool], ...] = (),
        group_layouts: tuple[_GroupLayout, ...] | None = None,
        slot_refs: tuple["LoRASlotRef | None", ...] | None = None,
        head_workspace_bytes: int = 0,
        checkpoint_floor: tuple[int, int] = (0, 0),
        retained_tokens: int | None = None,
        hybridep_growth_bytes: int = 0,
    ) -> _SubforwardCost:
        dense = self._dense_layout_floors(group_rows, slot_refs, group_layouts)
        if dense is None:
            floor = self._checkpoint_memory_floor(group_rows, slot_refs, gdn_segments)
        else:
            # The largest rank's boundaries and the rest of the largest rank total.
            boundaries = max(retained for retained, _ in dense)
            floor = boundaries, max(map(sum, dense)) - boundaries
        checkpoint_memory = self._sequence_parallel_lora_floor(
            floor, group_rows, signature
        )
        required = self._estimate_required_memory_bytes_from_values(
            packed_tokens=packed_tokens,
            output_bytes=output_bytes,
            signature=signature,
            logical_tokens=logical_tokens,
            gdn_segments=gdn_segments,
            group_rows=group_rows,
            slot_refs=slot_refs,
            head_workspace_bytes=head_workspace_bytes,
            checkpoint_floor=checkpoint_floor,
            retained_tokens=retained_tokens,
            include_checkpoint_input_gradient=False,
            checkpoint_memory=checkpoint_memory,
        )
        checkpoint_retained, checkpoint_workspace = checkpoint_memory
        retained = self._retained_memory_bytes(
            signature,
            packed_tokens=packed_tokens,
            logical_tokens=logical_tokens,
            output_bytes=output_bytes,
            required=required,
            checkpoint_retained_bytes=max(
                checkpoint_retained,
                checkpoint_floor[0],
            ),
        )
        # One logical BF16 input gradient per eligible full/uniform/1 boundary
        # (TP x SP: one input-gradient shard; the workspace is explicit).
        # This partial peak allowance is not evidence of simultaneous distinct
        # backing stores, nor a bound for compiler saves or other backward work.
        # Keep it out of forward retention, including the cold fallback above.
        gradient = self._checkpoint_input_gradient_bytes(
            group_rows, checkpoint_retained
        )
        if dense is not None:
            # A covered dense model's recomputed layer holds one H-wide input
            # gradient at its peak (Qwen3.8-27B CP2 traces); each rank pairs
            # its adapter gradients with its own boundaries.
            gradient = (
                sum(rows for rows, grad in group_rows if grad) * self._hidden_size * 2
            )
        gradient_slots = self._gradient_slots(group_rows, slot_refs)
        adapter_gradient = (
            0
            if not gradient
            else self._checkpoint_adapter_gradient_bytes(
                self._checkpoint_gradient_groups(group_rows, slot_refs)
            )
            if dense is None
            else self._dense_adapter_gradient_extra(
                checkpoint_memory, dense, group_rows, slot_refs, group_layouts
            )
        )
        checkpoint_retained = output_bytes + max(
            checkpoint_retained, checkpoint_floor[0]
        )
        checkpoint_workspace = max(
            checkpoint_workspace, head_workspace_bytes, checkpoint_floor[1]
        )
        if gradient and self._memory_profiles.get(signature) is None:
            checkpoint_workspace += self._cold_recompute_transient_bytes()
        if gradient and dense is not None:
            # TE's cuBLAS workspace persists beside whichever stage peaks.
            checkpoint_workspace += self._te_workspace_growth_bytes()
        forward_required = required
        if gradient:
            required = max(
                required,
                int(
                    (
                        checkpoint_retained
                        + checkpoint_workspace
                        + gradient
                        + adapter_gradient
                    )
                    * _MEMORY_SAFETY_FACTOR
                ),
            )
        # HybridEP buffer growth stays allocated through the forward and
        # backward peaks, but is not forward retention.
        return _SubforwardCost(
            required=required + int(hybridep_growth_bytes * _MEMORY_SAFETY_FACTOR),
            retained=retained,
            checkpoint_retained=checkpoint_retained,
            checkpoint_workspace=checkpoint_workspace,
            checkpoint_input_gradient=gradient,
            checkpoint_peak_increment=required - forward_required,
            hybridep_growth=hybridep_growth_bytes,
            checkpoint_adapter_gradient=adapter_gradient,
            checkpoint_adapter_gradient_slots=json.dumps(
                [
                    [ref.kind, ref.name]
                    for ref in sorted(
                        gradient_slots,
                        key=lambda ref: (
                            ref.kind,
                            ref.name is not None,
                            ref.name or "",
                        ),
                    )
                ]
            )
            if adapter_gradient
            else "",
        )

    def last_forward_telemetry(self) -> dict[str, Any]:
        """Concise planner telemetry for the most recent planned forward.

        ``planning_ms`` is critical-path planning accumulated across the whole
        public call (all waves of ``forward_batches``, including the
        synchronous cost of submitting speculative work);
        ``speculative_planning_ms`` is worker CPU time hidden under the
        caller's GPU work; ``selected_max_depth`` describes the most recently
        materialized plan; ``predicted_peak_bytes`` / ``usable_limit_bytes``
        are the admitted plan's memory check (for a split: every retained
        graph plus the largest subforward's ephemeral share). A call refused
        with ``TrainerRankMemoryError`` is still reflected, with the binding
        check that refused it.

        ``packing_plan_sha256`` commits to ordered local prefix-pack geometry
        and TP padding, before CP dispatch. ``subforward_packing_plan_sha256``
        lists the child commitments in execution order, with
        ``subforward_request_indices`` supplying the outer mapping. These are
        not unique event IDs: identical child geometries share a digest.
        ``input_tokens_sha256`` recomposes existing row hashes in original flat
        request order, independent of packing; it is unavailable if any request
        was inactive. These are planning-time commitments, not model/target
        label/caller loss-mask fingerprints or evidence of completed execution.
        """

        if self._last_forward_telemetry_snapshot is None:
            raise RuntimeError("no forward has completed planning yet")
        return dict(self._last_forward_telemetry_snapshot)

    def reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> None:
        """Reduce in place over data-parallel batches, excluding TP/CP replicas."""
        self._guard_forward_collective("reduce")
        from megatron.core import parallel_state as ps

        # Public outputs are CP-replicated; internal shard reductions still include CP.
        dist.all_reduce(
            tensor,
            op=op,
            group=ps.get_data_parallel_group(with_context_parallel=False),
        )

    def backward(
        self,
        loss: torch.Tensor | Sequence[torch.Tensor],
        gradient: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
        *,
        retain_graph: bool = False,
    ) -> None:
        """Collect a complete local backward before committing model cotangents."""

        from ._commands import _coordinate_call

        preflight = partial(_coordinate_call, group=self._forward_memory_group())
        with self._gradient_transaction(before_commit=preflight):
            packets = []
            cache = self._forward_graph_cache()

            def collect() -> None:
                packets.extend(
                    (packet.handle, packet.gradients)
                    for packet in self._forward_cotangent_collector().backward(
                        loss, gradient, retain_graph=retain_graph
                    )
                )
                cache.validate_many(packets)

            preflight(collect)
            cache.backward_many(
                packets, retain_graph=retain_graph, coordinate=preflight
            )

    def _compact_lora_slot_keys(self) -> None:
        from art.megatron.lora import LoRA

        for chunk in self.runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA):
                    continue
                slots = [
                    (ref, module._slot_modules[key])
                    for ref, key in module._slot_keys.items()
                ]
                module._slot_keys = {
                    ref: f"slot_{index}" for index, (ref, _slot) in enumerate(slots)
                }
                module._slot_modules = torch.nn.ModuleDict(
                    {f"slot_{index}": slot for index, (_ref, slot) in enumerate(slots)}
                )

    def _prepare_adapter_model(
        self,
        name: str,
        adapter_model: Mapping[str, torch.Tensor],
        *,
        canonicalized: bool = False,
    ) -> dict[str, torch.Tensor]:
        templates = self._local_lora_adapter_templates()
        keys = set(adapter_model)
        expected = set(templates)
        if dist.is_available() and dist.is_initialized():
            gathered: list[set[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, expected, group=self._checkpoint_group())
            expected = set().union(*(value for value in gathered if value is not None))
        if unknown := sorted(keys - expected):
            preview = ", ".join(repr(key) for key in unknown[:8])
            more = "" if len(unknown) <= 8 else f", ... +{len(unknown) - 8} more"
            raise ValueError(
                f"Checkpoint {name!r} contains keys that do not match installed "
                f"LoRA wrapper sites: {preview}{more}. Configure the Megatron "
                "runtime with matching LoRA target modules before loading."
            )
        local_state = {
            key: tensor for key, tensor in adapter_model.items() if key in templates
        }
        adapter_model = (
            local_state
            if canonicalized
            else self.runtime.model_support_handler.canonicalize_loaded_lora_state(
                local_state, self.runtime.model
            )
        )
        if set(adapter_model) != set(local_state):
            raise TrainerRankSlotStateError(
                "Model-specific LoRA canonicalization changed the adapter key set "
                f"for checkpoint {name!r}."
            )
        return {
            key: tensor.to(
                device=templates[key].device,
                dtype=templates[key].dtype,
                non_blocking=True,
            )
            for key, tensor in adapter_model.items()
        }

    def _local_lora_adapter_templates(self) -> dict[str, torch.Tensor]:
        templates: dict[str, torch.Tensor] = {}
        for chunk in self.runtime.model:
            for module in chunk.modules():
                expected_weight_keys = getattr(module, "_expected_weight_keys", None)
                if not callable(expected_weight_keys):
                    continue
                for suffix, parameter_name in (
                    ("lora_A", "A_T"),
                    ("lora_B", "B_T"),
                ):
                    parameter = getattr(module, parameter_name, None)
                    if not isinstance(parameter, torch.Tensor):
                        continue
                    templates.update(
                        (str(key), parameter) for key in expected_weight_keys(suffix)
                    )
        return templates

    def _local_parameter_key_groups(self, name: str) -> tuple[tuple[str, ...], ...]:
        ref = self._slot_ref(name)
        return tuple(
            tuple(str(key) for key in expected(str(suffix).removesuffix(".weight")))
            for chunk in self.runtime.model
            for module in chunk.modules()
            if (lora_params := getattr(module, "_lora_params", None)) is not None
            if (expected := getattr(module, "_expected_weight_keys", None)) is not None
            for suffix, _param in lora_params(ref)
        )

    @staticmethod
    def _slot_ref(name: str | None) -> "LoRASlotRef":
        try:
            from art.megatron.lora import LoRASlotRef
        except ModuleNotFoundError as exc:
            if exc.name is None or not exc.name.startswith("megatron"):
                raise

            return cast("LoRASlotRef", _LocalLoRASlotRef(name=name))

        return LoRASlotRef(kind="checkpoint", name=name)

    def _reduce_dynamic_grads(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        scale_grads: float,
    ) -> tuple[torch.Tensor, ...]:
        from megatron.core import parallel_state as ps

        from art.megatron.training.finalize_grads import (
            coalesced_all_reduce,
            tensor_parallel_grad_sync,
        )

        buckets: dict[
            tuple[int, str, torch.dtype, torch.device],
            tuple[dist.ProcessGroup, dist.ReduceOp.RedOpType, list[torch.Tensor]],
        ] = {}

        def add(
            group: dist.ProcessGroup,
            op: dist.ReduceOp.RedOpType,
            grad: torch.Tensor,
        ) -> None:
            key = (id(group), str(op), grad.dtype, grad.device)
            buckets.setdefault(key, (group, op, []))[2].append(grad)

        grads = tuple(
            (
                torch.zeros_like(param, dtype=torch.float32)
                if param.grad is None
                else param.grad.detach().float().mul(scale_grads)
            )
            for param in params
        )
        for param, grad in zip(params, grads, strict=True):
            if bool(getattr(param, "allreduce", True)):
                group = ps.get_data_parallel_group(with_context_parallel=True)
                if getattr(param, "_art_custom_checkpoint_param", False):
                    # Custom heads consume replicated full-sequence outputs;
                    # average their CP copies while still summing DP batches.
                    grad.div_(ps.get_context_parallel_world_size())
            else:
                group = ps.get_expert_data_parallel_group()
            if group is not None and group.size() > 1:
                add(group, dist.ReduceOp.SUM, grad)

            sync = tensor_parallel_grad_sync(param, name="dynamic LoRA")
            if sync is not None:
                group, reduce_op = sync
                add(group, reduce_op, grad)

        for group, op, bucket_grads in buckets.values():
            coalesced_all_reduce(bucket_grads, group=group, op=op)
        return grads

    def _validate_replicated_top_level_count(
        self, count: int, *, yield_empty: bool
    ) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        configurations: list[tuple[int, bool] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(configurations, (int(count), yield_empty))
        if len(set(configurations)) == 1:
            return
        raise ValueError(
            "forward_batches requires the same top-level input count and "
            "yield_empty setting on every "
            "distributed rank. Pass already-DP-local inputs to forward instead. "
            f"Observed (count, yield_empty) by rank: {configurations}."
        )

    def _dp_rank_and_size(self) -> tuple[int, int]:
        try:
            from megatron.core import parallel_state as ps

            return int(ps.get_data_parallel_rank()), int(
                ps.get_data_parallel_world_size()
            )
        except (AssertionError, ImportError, RuntimeError, ValueError):
            return 0, 1

    def _forced_test_anchor(self) -> str | None:
        if os.environ.get(_TEST_HOOKS_ENV) != "1":
            return None
        return os.environ.get(_TEST_ANCHOR_ENV) or None

    def _group_active_request_indices(
        self,
        requests: Sequence[AnyForwardInput],
        *,
        checkpoint: AdapterSelection = Unset,
        ensure_slots: bool = True,
    ) -> tuple[tuple[tuple["LoRASlotRef | None", bool], tuple[int, ...]], ...]:
        if ensure_slots:
            self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
        groups: dict[
            tuple[LoRASlotRef | None, bool, ResolvedForwardOptions, bool], list[int]
        ] = {}
        for index, request in enumerate(requests):
            if (
                request.target_tokens is not None
                or request.logits
                or request.top_k is not None
                or request.hidden_states
            ):
                groups.setdefault(
                    (
                        self._resolve_slot_ref(request, checkpoint=checkpoint),
                        (
                            torch.is_grad_enabled()
                            if request.no_grad is None
                            else not request.no_grad
                        ),
                        _resolved_request_policy(request.options),
                        request.routed_experts is not None,
                    ),
                    [],
                ).append(index)
        return tuple(
            ((slot_ref, grad), tuple(indices))
            for (slot_ref, grad, _options, _routed), indices in groups.items()
        )

    @_backward_region
    def _run_flat_plan_with_memory_tracking(
        self,
        plan: _FlatForwardPlan,
        *,
        check: _MemoryCheck,
        context: str,
    ) -> tuple[list[AnyForwardOutput], int | None]:
        if not getattr(self, "_planner_observing_split", False):
            self._begin_planner_observation(plan, check)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            baseline = int(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
            self._peak_resets = self.__dict__.get("_peak_resets", 0) + 1
        else:
            baseline = None
        observation = getattr(self, "_planner_observation", None)
        if observation is not None:
            if observation["baseline"] is None:
                observation["baseline"] = baseline
            observation["phase"] = "forward"
        started = self._recovery_clock() if baseline is not None else None
        try:
            with _telemetry_phase(
                "forward",
                self._telemetry_signature(plan),
                dedup_signature=self._telemetry_plan_signature(plan),
                synchronized=torch.cuda.is_available() and self.device.type == "cuda",
            ):
                outputs = self._execute_flat_plan(plan)
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError as exc:
            self.report_planner_oom(exc)
            raise _memory_error(
                context=context,
                message="CUDA OOM occurred despite the planner estimate",
                packed_tokens=plan.packed_tokens,
                logical_tokens=plan.logical_tokens,
                # Preserve the selected admission, including the whole rung's
                # budget for a split. Rechecking here observes failed state and
                # can enter collectives that successful peers never reach.
                check=check,
            ) from exc
        finished = self._recovery_clock() if baseline is not None else None
        seconds = None if started is None or finished is None else finished - started
        if baseline is not None:
            self._update_peak_memory_profile(
                plan, baseline, int(torch.cuda.memory_allocated(self.device))
            )
        if seconds is not None and plan.packed_tokens > 0:
            try:
                self._record_graph_forward_time(plan, seconds)
                self._record_recovery_work(context, seconds)
            except Exception:
                self._recovery_state().invalid = True
        if observation is not None:
            observation["phase"] = "forward_and_caller"
        return outputs, baseline

    def _begin_planner_observation(
        self, plan: _AnyForwardPlan, check: _MemoryCheck
    ) -> None:
        reporter = getattr(self, "_planner_reporter", None)
        if reporter is None or reporter.threshold_pct is None:
            return
        self.finish_planner_observation()
        self._planner_observation_generation += 1
        if self._planner_active_observations:
            # Both the suspended caller and the newly entered forward share
            # allocator counters; neither interval may claim a completed peak.
            self._planner_overlap_generation = self._planner_observation_generation
        # A snapshot failure must still leave a durable minimal OOM report.
        observation = _PlannerObservation(
            {
                "predicted": check.estimated_required_bytes,
                "admission": check.estimated_required_bytes,
                "replay": lambda: {
                    "incomplete_reasons": ["planner_snapshot_unavailable"]
                },
                "baseline": None,
                "peak": 0,
                "phase": "forward",
                "comparable": False,
                "generation": self._planner_observation_generation,
                "window_open": True,
                "decision": check.decision,
            }
        )
        self._planner_observation = observation
        self._planner_active_observations[observation["generation"]] = observation
        self._fill_planner_snapshot(plan, check, observation)

    @property
    def _planner_observation(self) -> dict[str, Any] | None:
        variable = getattr(self, "_planner_observation_context", None)
        context = None if variable is None else variable.get()
        return (
            getattr(self, "_planner_unscoped_observation", None)
            if context is None
            else context.get("observation")
        )

    @_planner_observation.setter
    def _planner_observation(self, value: dict[str, Any] | None) -> None:
        variable = getattr(self, "_planner_observation_context", None)
        context = None if variable is None else variable.get()
        if context is None:
            self._planner_unscoped_observation = value
        else:
            context["observation"] = value

    @contextmanager
    def planner_observation_scope(self, context: dict[str, Any]) -> Iterator[None]:
        """Use one holder per execution, retaining it across async-generator steps.

        A scope changes only diagnostic ownership. It does not serialize model
        work or make the process-wide CUDA peak counter execution-local.
        """
        token = self._planner_observation_context.set(context)
        try:
            yield
        finally:
            self._planner_observation_context.reset(token)

    @staticmethod
    def _telemetry_plan_signature(plan: _AnyForwardPlan) -> dict[str, object]:
        return {
            "topology": plan.signature.topology,
            "planner_coefficients": plan.signature.planner_coefficients,
            "slot_group_count": plan.signature.slot_group_count,
            "request_mix": plan.signature.request_mix,
            "grad_enabled": plan.signature.grad_enabled,
            "grad_modes": plan.signature.grad_modes,
        }

    @classmethod
    def _telemetry_signature(cls, plan: _AnyForwardPlan) -> dict[str, object]:
        return {
            **cls._telemetry_plan_signature(plan),
            **cls._packing_fingerprints(plan),
            "request_count": plan.request_count,
            "packed_tokens": plan.packed_tokens,
            "logical_tokens": plan.logical_tokens,
            "group_packed_tokens": tuple(
                int(group.packed.tokens.numel()) for group in plan.groups
            ),
            "group_segment_counts": tuple(
                len(group.packed.segments) for group in plan.groups
            ),
        }

    @staticmethod
    def _packing_fingerprints(plan: _AnyForwardPlan) -> dict[str, object]:
        """Host metadata only: never copy/read tensor values or query CUDA.

        Geometry excludes content, checkpoint names and later CP kernel plans.
        Input hashes reuse canonical little-endian int64 row commitments; no
        additional token hashing is done, including for rejected candidates.
        Keep this high-cardinality evidence outside compile-plan deduplication.
        """

        def digest(value: object) -> str:
            return hashlib.sha256(
                json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()

        split = isinstance(plan, _SplitForwardPlan)
        children = plan.subforwards if split else (plan,)
        mappings = (
            plan.request_indices if split else (tuple(range(plan.request_count)),)
        )
        rows: list[tuple[int, str] | None] = [None] * plan.request_count
        fingerprints = []
        for child, mapping in zip(children, mappings, strict=True):
            groups = []
            tp = child.signature.topology[1]
            for group in child.groups:
                length = int(group.packed.tokens.numel())
                groups.append(
                    (
                        group.request_indices,
                        group.grad_enabled,
                        length,
                        ((length + tp - 1) // tp) * tp,
                        tuple(
                            (
                                segment.sequence_indices,
                                segment.start,
                                segment.end,
                                segment.packed_start,
                                segment.group_id,
                                segment.parent_id,
                            )
                            for segment in group.packed.segments
                        ),
                    )
                )
                for index, row in zip(
                    group.request_indices, group.input_row_fingerprints, strict=False
                ):
                    rows[mapping[index]] = row
            fingerprints.append(
                digest(
                    (
                        "art.prefix-pack/v1",
                        child.signature.topology,
                        child.request_count,
                        groups,
                    )
                )
            )
        complete_rows = tuple(row for row in rows if row is not None)
        return {
            "packing_fingerprint_schema": "art.prefix-pack/v1",
            "packing_plan_sha256": digest(
                (
                    "art.prefix-pack-split/v1",
                    plan.request_count,
                    tuple(zip(mappings, fingerprints, strict=True)),
                )
            )
            if split
            else fingerprints[0],
            "subforward_packing_plan_sha256": tuple(fingerprints),
            "input_tokens_sha256": canonical_token_rows_fingerprint(complete_rows)
            if len(complete_rows) == plan.request_count
            else None,
        }

    def _execute_flat_plan(self, plan: _FlatForwardPlan) -> list[AnyForwardOutput]:
        outputs = [
            ForwardOutput(None, None, None, None, checkpoint, no_grad)
            for checkpoint, no_grad in plan.output_metadata
        ]
        self._validate_hybridep_topology()
        hybridep = (
            self._configure_hybridep(
                tuple(group.packed for group in plan.groups), topology=self._topology()
            )
            if plan.groups
            else None
        )
        previous = self._graph_cache.handles() if hasattr(self, "_graph_cache") else ()
        item_outputs: list[AnyForwardOutput] = []
        output: AnyForwardOutput | None = None
        try:
            for group_index, group in enumerate(plan.groups):
                with self._forward_handoff(advance=group_index + 1 < len(plan.groups)):
                    if hybridep is not None:
                        self._set_hybridep_rows(hybridep[0][group_index])
                    with torch.set_grad_enabled(group.grad_enabled):
                        item_outputs = self._execute_graph_group(group)
                    for index, output in zip(
                        group.request_indices, item_outputs, strict=True
                    ):
                        outputs[index] = output
        except BaseException as error:
            outputs.clear()
            item_outputs.clear()
            output = None
            self._discard_forward_graphs(previous, error)
            raise
        finally:
            if hybridep is not None:
                self._set_hybridep_rows(hybridep[1])
        return outputs

    def _forward_graph_cache(self):
        from ._graphs import GraphCache

        if not hasattr(self, "_graph_cache"):
            self._graph_cache = GraphCache()
        return self._graph_cache

    def _forward_cotangent_collector(self):
        from ._tensors import CotangentCollector

        if not hasattr(self, "_cotangent_collector"):
            self._cotangent_collector = CotangentCollector()
        return self._cotangent_collector

    def _execute_graph_group(self, group: _ForwardGroupPlan) -> list[AnyForwardOutput]:
        from art.megatron.lora import use_lora_slot

        from ._corrections import capture_forward_corrections
        from ._options import resolve_forward_options
        from ._tensors import (
            TensorPacket,
            flatten_tensors,
            managed_tree,
            unflatten_tensors,
        )

        options = resolve_forward_options(
            getattr(self, "_forward_options", None),
            input=group.items[0].request.options,
        )
        placement = getattr(group, "memory_placement", None)
        retention = (
            options.backward_state if placement is None else placement.backward_state
        )
        retention = "gpu" if retention == "auto" else retention
        output_device = (
            options.output_device if placement is None else placement.output_device
        )
        output_device = "cpu" if output_device == "cpu" else None
        ref = group.slot_ref
        topology = self._topology()
        spec = None

        def execute(captured: _ForwardGroupPlan) -> tuple[torch.Tensor, ...]:
            nonlocal spec
            if self._topology() != topology:
                raise TrainerRankRuntimeSupportError(
                    "Forward replay requires its original parallel topology"
                )
            hybrid = self._configure_hybridep((captured.packed,), topology=topology)
            try:
                if hybrid is not None:
                    self._set_hybridep_rows(hybrid[0][0])
                prepared = self._prepare_packed_forward(captured.packed)
                from art.megatron.routed_experts import prepare_routes, use_routes

                routes = prepare_routes(
                    captured.items,
                    captured.packed,
                    prepared,
                    getattr(self, "_routing_bindings", ()),
                    self.device,
                )
                with use_routes(routes):
                    outputs = self._forward_packed(captured.items, prepared)
                outputs = [
                    replace(
                        output,
                        checkpoint=None if ref is None else ref.name,
                        no_grad=not captured.grad_enabled,
                    )
                    for output in outputs
                ]
                tensors, captured_spec = flatten_tensors(outputs)
                if spec is not None and captured_spec != spec:
                    raise RuntimeError("Forward replay changed its output tree")
                spec = captured_spec
                # Observe physical backward, including replay, before the cache
                # replaces these outputs with detached caller cotangent proxies.
                backward = self._backward_work()
                if backward is not None:
                    backward.attach(outputs)
                return tensors
            finally:
                if hybrid is not None:
                    self._set_hybridep_rows(hybrid[1])

        if not group.grad_enabled:
            with torch.no_grad(), use_lora_slot(ref):
                tensors = execute(group)
            assert spec is not None
            outputs = unflatten_tensors(spec, tensors)
            return (
                outputs
                if output_device is None
                else managed_tree(outputs, device=output_device)
            )

        version = self._capture_lora_version(ref, options.max_gradient_staleness)
        # Saved views of externally owned parameters must not duplicate whole
        # frozen model weights or immutable LoRA captures into every CPU graph.
        parameters = [
            tensor
            for chunk in self.runtime.model
            for tensor in (
                *chunk.parameters(),
                *(buffer for _, buffer in chunk.named_buffers()),
            )
        ]
        if version is not None:
            parameters.extend(
                parameter
                for slot in version.slots.values()
                for parameter in slot.parameters()
            )
        storages = {
            (parameter.device, parameter.untyped_storage().data_ptr())
            for parameter in parameters
        }
        tracker = None
        devices = ()
        if self.device.type == "cuda":
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

            tracker = get_cuda_rng_tracker()
            devices = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device(),
            )
        cache = self._forward_graph_cache()
        handle = None
        try:
            handle, tensors = cache.run(
                execute,
                group,
                context_factory=lambda: use_lora_slot(ref, version=version),
                validate_backward=None if version is None else version.validate,
                retention=retention,
                checkpoint_versions=() if version is None else (version.version,),
                options=options,
                cuda_devices=devices,
                rng_tracker=tracker,
                keep_on_device=lambda tensor: (
                    (tensor.device, tensor.untyped_storage().data_ptr()) in storages
                ),
                output_device=output_device,
                execution_peak_bytes=getattr(placement, "execution_peak_bytes", 0),
            )
            if topology.cp > 1 and retention != "replay":
                residual = cache.state(handle).non_offloadable_bytes
                if residual is not None:
                    profiles = getattr(self, "_graph_residency", None)
                    if profiles is None:
                        self._graph_residency = profiles = OrderedDict()
                    key = self._graph_residency_key(group)
                    profiles[key] = max(residual, profiles.pop(key, 0))
                    if len(profiles) > 256:
                        profiles.popitem(last=False)
            assert spec is not None
            if version is not None:

                @contextmanager
                def current_context():
                    current = self._capture_lora_version(
                        ref, options.max_gradient_staleness, origin=version.version
                    )
                    assert current is not None
                    previous_storages = storages.copy()
                    storages.update(
                        (parameter.device, parameter.untyped_storage().data_ptr())
                        for slot in current.slots.values()
                        for parameter in slot.parameters()
                    )
                    try:
                        with use_lora_slot(ref, version=current):
                            yield
                    finally:
                        storages.clear()
                        storages.update(previous_storages)

                cache.set_corrections(
                    handle,
                    capture_forward_corrections(
                        unflatten_tensors(spec, tensors), tensors, options
                    ),
                    is_stale=lambda: (
                        self._capture_checkpoint_version(
                            version.version.checkpoint
                        ).revision
                        != version.weight_version.revision
                    ),
                    current_context_factory=current_context,
                )
            packet = TensorPacket(
                handle, spec, tensors, tuple(tensor.requires_grad for tensor in tensors)
            )
            outputs = self._forward_cotangent_collector().attach(
                packet,
                managed=output_device is not None,
                on_release=partial(cache.release, handle),
            )
            # Track the caller graph outside saved-state hooks, which detach markers.
            # Consumption also ends the lifetime of unused sibling outputs.
            return self._track_slot_graph_outputs(ref, outputs)
        except BaseException:
            # No caller owns a failed handoff; a partial bridge may also release.
            try:
                if handle is not None:
                    cache.release(handle)
            finally:
                # A retained traceback must not own this failed call's captures.
                # Clear its closure cell, never the shared version or live graphs.
                del version
                packet = outputs = None
                tensors = ()
                parameters.clear()
            raise

    def _forward_output_metadata(
        self,
        request: AnyForwardInput,
        *,
        checkpoint: AdapterSelection,
    ) -> tuple[str | None, bool]:
        selection = (
            request.checkpoint if request.checkpoint is not Unset else checkpoint
        )
        if selection is Unset:
            ref = self._slot_stack[-1] if self._slot_stack else self._default_slot_ref
            name = None if ref is None else ref.name
        else:
            name = selection
        enabled = (
            torch.is_grad_enabled() if request.no_grad is None else not request.no_grad
        )
        return name, not enabled

    def _hybridep_graphs(self) -> list[weakref.ReferenceType[torch.Tensor]]:
        graphs = getattr(self, "_pending_hybridep_graphs", None)
        if graphs is None:
            graphs = []
            self._pending_hybridep_graphs = graphs
        return graphs

    def _has_live_hybridep_graphs(self) -> bool:
        graphs = self._hybridep_graphs()
        graphs[:] = [marker for marker in graphs if _graph_marker_is_live(marker)]
        return bool(graphs)

    def _topology_key(self) -> tuple[int, int, int, int]:
        try:
            topology = self._topology()
            return cast(
                tuple[int, int, int, int],
                tuple(
                    int(getattr(topology, name)) for name in ("dp", "tp", "cp", "pp")
                ),
            )
        except (AssertionError, AttributeError, ImportError, RuntimeError, ValueError):
            return (1, 1, 1, 1)

    def _physical_tokens(self, packed_tokens: int) -> int:
        """Physical length of one packed group: padded to a multiple of TP.

        Execution pads every group independently (``_pad_packed_batch``), so
        admission, the cheap bounds and the memory profile all count tokens
        the same way; the omission would otherwise grow with the number of
        groups, not stay below TP.
        """

        multiple = max(1, self._topology_key()[1])
        return packed_tokens + (-packed_tokens % multiple)

    def _graph_memory_policy_enabled(self) -> bool:
        return self.device.type == "cuda" and hasattr(self, "_forward_graph_cache")

    def _available_cpu_memory_bytes(self) -> int:
        world = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        return host_memory_budget(
            local_world_size=local_rank_count(world_size=world)
        ).available_bytes

    @staticmethod
    def _graph_forward_time_key(plan: _FlatForwardPlan) -> tuple[Any, ...]:
        return (
            replace(plan.signature, memory_placement=()),
            plan.packed_tokens,
            plan.logical_tokens,
            tuple(group.packed.segments for group in plan.groups),
        )

    def _record_graph_forward_time(
        self, plan: _FlatForwardPlan, seconds: float
    ) -> None:
        if (
            not math.isfinite(seconds)
            or seconds <= 0
            or any(
                group.memory_placement is not None
                and group.memory_placement.backward_state != "gpu"
                for group in plan.groups
            )
        ):
            return
        profiles = getattr(self, "_graph_forward_times", None)
        if profiles is None:
            self._graph_forward_times = profiles = OrderedDict()
        key = self._graph_forward_time_key(plan)
        profiles[key] = (*profiles.pop(key, ())[-2:], seconds)
        if len(profiles) > 256:
            profiles.popitem(last=False)

    def _graph_residency_key(self, group: _ForwardGroupPlan) -> tuple[Any, ...]:
        return (
            self._topology_key(),
            group.packed.segments,
            group.grad_enabled,
            tuple(
                (
                    item.request.input_tokens.numel(),
                    None
                    if item.request.target_tokens is None
                    else item.request.target_tokens.numel(),
                    item.request.top_k,
                    item.request.logits,
                    item.request.hidden_states,
                )
                for item in group.items
            ),
        )

    def _graph_memory_units(
        self, plan: _AnyForwardPlan
    ) -> Iterator[
        tuple[int, tuple[int, ...], ForwardMemoryCost, ResolvedForwardOptions]
    ]:
        """Use existing aggregate profiles when all physical groups share policy."""
        flats = plan.subforwards if isinstance(plan, _SplitForwardPlan) else (plan,)
        staged_slots, captured = set(), set()
        for flat_index, flat in enumerate(flats):
            policies = [
                _resolved_request_policy(g.items[0].request.options)
                for g in flat.groups
            ]
            partitions = (
                [tuple(range(len(flat.groups)))]
                if len(set(policies)) <= 1
                else [(i,) for i in range(len(flat.groups))]
            )
            for indices in partitions:
                if not indices:
                    continue
                groups = tuple(flat.groups[i] for i in indices)
                requests = [item.request for group in groups for item in group.items]
                policy = policies[indices[0]]
                priced = (
                    flat
                    if len(indices) == len(flat.groups)
                    else replace(
                        flat,
                        groups=groups,
                        packed_tokens=sum(
                            self._physical_tokens(int(g.packed.tokens.numel()))
                            for g in groups
                        ),
                        logical_tokens=sum(
                            int(r.input_tokens.numel()) for r in requests
                        ),
                        inactive_logical_tokens=0,
                        output_bytes=self._estimate_group_request_output_bytes(
                            requests
                        ),
                        signature=self._memory_signature_from_requests(
                            requests,
                            slot_group_count=len(groups),
                            grad_modes=tuple(g.grad_enabled for g in groups),
                            slot_groups=tuple(
                                (g.slot_ref, g.grad_enabled) for g in groups
                            ),
                        ),
                    )
                )
                # CPU/replay observations cannot lower the GPU retention model.
                priced = replace(
                    priced, signature=replace(priced.signature, memory_placement=())
                )
                cost = self._plan_cost(priced)
                timings = getattr(self, "_graph_forward_times", {}).get(
                    self._graph_forward_time_key(priced), ()
                )
                output = priced.output_bytes
                retained = max(output, cost.retained)
                transient = max(0, cost.required - retained)
                residual = 0
                if priced.signature.topology[2] > 1:
                    profiles = getattr(self, "_graph_residency", {})
                    observed = [
                        profiles.get(self._graph_residency_key(g)) for g in groups
                    ]
                    # CP owns raw stage graphs outside saved-variable hooks.
                    # Until this exact layout is observed, grant no release credit.
                    residual = (
                        int(sum(observed) * _MEMORY_SAFETY_FACTOR)
                        if all(value is not None for value in observed)
                        else retained
                    )
                    retained = max(retained, residual)
                version_bytes = getattr(self, "_lora_version_capture_bytes", None)
                persistent = staging = 0
                for group in groups:
                    # Later children and groups reuse the first one's live capture.
                    key = (group.slot_ref, policy.max_gradient_staleness)
                    if group.grad_enabled and version_bytes and key not in captured:
                        captured.add(key)
                        persistent += version_bytes(*key)
                    if group.grad_enabled and group.slot_ref not in staged_slots:
                        staged_slots.add(group.slot_ref)
                        staging += self._lora_gradient_staging_bytes(group.slot_ref)
                credit = cost.checkpoint_adapter_gradient
                if credit:
                    # The walk repeats a slot's inventory per gradient group
                    # (e.g. routed and unrouted); one batch stages each target once.
                    credit = min(
                        credit,
                        sum(
                            self._pending_adapter_gradient_bytes(
                                g.slot_ref
                                for g in groups
                                if g.grad_enabled and g.slot_ref is not None
                            )
                        ),
                    )
                yield (
                    flat_index,
                    indices,
                    ForwardMemoryCost(
                        peak_bytes=max(cost.required, retained + transient),
                        retained_bytes=retained,
                        cpu_resident_bytes=residual,
                        output_bytes=output,
                        replay_bytes=sum(
                            _snapshot_tensor_bytes(group)
                            + 64 * 1024
                            + _correction_state_bytes(group, policy)
                            for group in groups
                            if group.grad_enabled
                        ),
                        backward_required=any(group.grad_enabled for group in groups),
                        persistent_bytes=persistent,
                        gradient_staging_bytes=staging,
                        staged_gradient_bytes=credit,
                        replay_seconds=max(timings) if len(timings) >= 2 else None,
                        correction_workspace_bytes=cost.required
                        if any(
                            correction.policy == "always"
                            for correction in policy.stale_gradient_corrections
                        )
                        else 0,
                    ),
                    policy,
                )

    def _graph_memory_candidates(
        self,
        units: Sequence[
            tuple[int, tuple[int, ...], ForwardMemoryCost, ResolvedForwardOptions]
        ],
        *,
        sync_across_dp: bool,
    ) -> Iterator[
        tuple[
            Literal["gpu", "cpu", "replay"],
            Literal["model", "cpu"],
            dict[str, Any] | None,
        ]
    ]:
        # Avoid timing work and its collective entirely on the GPU headroom path.
        yield "gpu", "model", None
        yield "gpu", "cpu", None
        eligible = [
            cost
            for _, _, cost, options in units
            if cost.backward_required
            and options.backward_state == "auto"
            and options.allow_cpu_offload
            and options.allow_replay
        ]
        stats = getattr(getattr(self, "_graph_cache", None), "transfer_stats", None)
        transfer_bytes = sum(
            cost.retained_bytes - cost.output_bytes for cost in eligible
        )
        trusted = bool(stats) and all(
            cost.replay_seconds is not None for cost in eligible
        )
        if stats is not None and trusted:
            trusted = all(
                math.isfinite(value) and value > 0
                for value in (
                    stats.offload_bytes,
                    stats.offload_seconds,
                    stats.restore_bytes,
                    stats.restore_seconds,
                )
            ) and (
                min(stats.offload_seconds, stats.restore_seconds) >= 0.001
                and transfer_bytes <= 2 * min(stats.offload_bytes, stats.restore_bytes)
            )
        cpu_seconds = 0.0
        if stats is not None and trusted:
            cpu_seconds = transfer_bytes * (
                stats.offload_seconds / stats.offload_bytes
                + stats.restore_seconds / stats.restore_bytes
            )
        replay_seconds = sum(cost.replay_seconds or 0.0 for cost in eligible)
        cpu_seconds, replay_seconds, missing = self._recovery_reduce(
            [cpu_seconds, replay_seconds, float(bool(eligible) and not trusted)],
            op="MAX",
            sync_across_dp=sync_across_dp,
        )
        prefer_replay = (
            not missing and replay_seconds > 0 and replay_seconds * 1.1 < cpu_seconds
        )
        evidence = {
            "source": "insufficient_samples"
            if missing
            else "measured_forward_and_transfers",
            "cpu_extra_seconds": cpu_seconds,
            "replay_extra_seconds": replay_seconds,
            "preferred": "replay" if prefer_replay else "cpu",
        }
        for state in ("replay", "cpu") if prefer_replay else ("cpu", "replay"):
            yield state, "model", evidence
            yield state, "cpu", evidence

    @overload
    def _admit_graph_memory(
        self, plan: _FlatForwardPlan, *, sync_across_dp: bool = False
    ) -> tuple[_FlatForwardPlan, _MemoryCheck]: ...

    @overload
    def _admit_graph_memory(
        self, plan: _SplitForwardPlan, *, sync_across_dp: bool = False
    ) -> tuple[_SplitForwardPlan, _MemoryCheck]: ...

    def _admit_graph_memory(
        self, plan: _AnyForwardPlan, *, sync_across_dp: bool = False
    ) -> tuple[_AnyForwardPlan, _MemoryCheck]:
        """Try a bounded placement ladder without changing root/group structure."""
        units = list(self._graph_memory_units(plan))
        cpu_available = self._available_cpu_memory_bytes()
        planned_checkpoints = {
            group.slot_ref.name
            for group in plan.groups
            if group.grad_enabled
            and group.slot_ref is not None
            and group.slot_ref.name is not None
        }
        prior_workspace, prior_staging = self._pending_backward_memory(
            exclude_staging=planned_checkpoints
        )
        flats = plan.subforwards if isinstance(plan, _SplitForwardPlan) else (plan,)
        for state, device, evidence in self._graph_memory_candidates(
            units, sync_across_dp=sync_across_dp
        ):
            placements = []
            for _, _, cost, options in units:
                selected_state = options.backward_state
                if selected_state == "auto":
                    selected_state = state
                    if selected_state == "replay" and not options.allow_replay:
                        selected_state = "cpu"
                    if selected_state == "cpu" and not options.allow_cpu_offload:
                        selected_state = "gpu"
                placements.append(
                    placement_cost(
                        (cost,),
                        backward_state=selected_state,
                        output_device=device
                        if options.output_device == "auto"
                        else options.output_device,
                    )
                )
            selected_groups = [list(flat.groups) for flat in flats]
            for (flat_index, indices, _, _), placement in zip(
                units, placements, strict=True
            ):
                for index in indices:
                    selected_groups[flat_index][index] = replace(
                        selected_groups[flat_index][index], memory_placement=placement
                    )
            selected_flats = []
            for flat, groups in zip(flats, selected_groups, strict=True):
                modes = tuple(
                    (
                        cast(MemoryPlacement, g.memory_placement).backward_state,
                        cast(MemoryPlacement, g.memory_placement).output_device,
                    )
                    for g in groups
                )
                selected_flats.append(
                    replace(
                        flat,
                        groups=tuple(groups),
                        signature=replace(
                            flat.signature,
                            memory_placement=modes
                            if any(mode != ("gpu", "model") for mode in modes)
                            else (),
                        ),
                    )
                )
            selected = (
                replace(plan, subforwards=tuple(selected_flats))
                if isinstance(plan, _SplitForwardPlan)
                else selected_flats[0]
            )
            required = (
                prior_staging
                + sum(p.gpu_retained_bytes + p.gpu_backward_bytes for p in placements)
                + max(
                    prior_workspace,
                    max(
                        (
                            p.gpu_required_bytes
                            - p.gpu_retained_bytes
                            - p.gpu_backward_bytes
                            for p in placements
                        ),
                        default=0,
                    ),
                )
            )
            if isinstance(selected, _SplitForwardPlan):
                key = self._split_memory_key(selected)
                required = max(
                    required,
                    0
                    if key is None
                    else int(
                        self._split_memory_floors.get(key, 0) * _MEMORY_SAFETY_FACTOR
                    ),
                )
            check = self._memory_check_required(required, sync_across_dp=sync_across_dp)
            cpu_required = sum(p.cpu_required_bytes for p in placements)
            # One fixed reduction per candidate keeps policy choice identical on
            # every physical participant, including locally empty DP ownership.
            cpu_margin = self._recovery_reduce(
                [float(cpu_available - cpu_required)],
                op="MIN",
                sync_across_dp=sync_across_dp,
            )[0]
            check = replace(
                check,
                fits=check.fits and cpu_margin >= 0,
                cpu_required_bytes=cpu_required,
                cpu_available_bytes=cpu_available,
                cpu_fits=cpu_margin >= 0,
                fallback_costs=evidence,
            )
            if check.fits:
                return selected, check
        return selected, check

    def _reclaim_graph_memory(
        self, check: _MemoryCheck, *, sync_across_dp: bool
    ) -> bool:
        if not self._graph_memory_policy_enabled():
            return False
        cache = getattr(self, "_graph_cache", None)
        actions = []
        # DP partitions can own different numbers of graphs; only TP/CP peers
        # coordinate individual records. WORLD sees one final success exchange.
        with self._planning_status(sync_across_dp):
            handles = () if cache is None else cache.handles()
            counts = self._recovery_reduce(
                [float(len(handles)), -float(len(handles))],
                op="MAX",
                sync_across_dp=False,
            )
            if counts[0] != -counts[1]:
                raise RuntimeError(
                    "Physical participants have different graph cache lengths"
                )
            cpu_available = max(
                0, self._available_cpu_memory_bytes() - check.cpu_required_bytes
            )
            for handle in handles:
                assert cache is not None
                state = cache.state(handle)
                can_offload, can_replay, cpu_margin = self._recovery_reduce(
                    [
                        float(state.offloadable and state.retention == "gpu"),
                        float(state.replayable and state.retention != "replay"),
                        float(cpu_available - state.offload_bytes),
                    ],
                    op="MIN",
                    sync_across_dp=False,
                )
                if can_offload and cpu_margin >= 0 and check.cpu_fits:
                    actions.append((cache.offload, handle))
                    cpu_available -= state.offload_bytes
                elif can_replay:
                    actions.append((cache.evict, handle))
            # Finish all collective decisions before a transfer/allocation can
            # fail, so peers still reach the final error exchange on failure.
            error: BaseException | None = None
            try:
                for operation, handle in actions:
                    operation(handle)
                if actions:
                    # Native allocator accounting only credits physical free bytes.
                    # Release reclaimed graph storage once, on this refusal path.
                    torch.cuda.empty_cache()
            except BaseException as exc:
                error = exc
            try:
                succeeded = self._recovery_reduce(
                    [float(error is None)], op="MIN", sync_across_dp=False
                )[0]
            except BaseException as exchange_error:
                if error is None:
                    raise
                raise self._memory_error_with_reduction_note(error, exchange_error)
            if error is not None:
                raise error
            if not succeeded:
                raise RuntimeError("Graph reclamation failed on another physical rank")
        return bool(
            self._recovery_reduce(
                [float(bool(actions))], op="MAX", sync_across_dp=sync_across_dp
            )[0]
        )

    @contextmanager
    def _cache_recovery_episode(
        self, *, error: BaseException | None = None
    ) -> Iterator[tuple[object, float | None]]:
        state = None
        started = None
        owner = None
        primary = error
        try:
            try:
                state = self._recovery_state()
                started = self._recovery_clock()
                owner = object()
                with state.lock:
                    if state.owner is None:
                        state.owner = owner
            except BaseException:
                # A saved forward failure must still reach the WORLD status
                # vote. Unknown accounting cost cannot enable later recovery.
                if state is None:
                    state = getattr(self, "_cache_recovery_state", None)
                    if state is None:
                        state = self._cache_recovery_state = _CacheRecoveryState()
                state.invalid = True
                if error is None:
                    raise
            yield owner, started
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if state is not None:
                # Includes control, release, resample and repeated search; no idle.
                try:
                    finished = self._recovery_clock()
                    elapsed = (
                        None
                        if started is None or finished is None
                        else finished - started
                    )
                    with state.lock:
                        if (
                            elapsed is None
                            or not math.isfinite(elapsed)
                            or elapsed <= 0
                            or not math.isfinite(state.cost + elapsed)
                        ):
                            state.invalid = True
                        else:
                            state.cost += elapsed
                            state.high = max(state.high, elapsed)
                        _planner_evidence.record(
                            self,
                            "recovery",
                            "accounted",
                            elapsed_seconds=elapsed,
                            local_cumulative_seconds=state.cost,
                            local_forward_work_seconds=state.work,
                            local_high_seconds=state.high,
                            invalid=state.invalid,
                        )
                except Exception:
                    state.invalid = True
                except BaseException as exc:
                    state.invalid = True
                    if primary is None:
                        primary = exc
                        raise
                finally:
                    try:
                        with state.lock:
                            if state.owner is owner:
                                state.owner = None
                    except Exception:
                        state.invalid = True
                    except BaseException:
                        state.invalid = True
                        if primary is None:
                            raise

    def _recovery_clock(self) -> float | None:
        try:
            value = time.perf_counter()
            if type(value) is float and math.isfinite(value):
                return value
        except Exception:
            pass
        self._recovery_state().invalid = True
        return None

    def _record_recovery_work(self, context: str, seconds: float) -> None:
        if context not in ("forward_batches", "forward"):
            return
        state = self._recovery_state()
        try:
            with state.lock:
                if (
                    not math.isfinite(seconds)
                    or seconds < 0
                    or not math.isfinite(state.work + seconds)
                ):
                    state.invalid = True
                else:
                    state.work += seconds
        except Exception:
            state.invalid = True

    def _recovery_state(self) -> _CacheRecoveryState:
        state = getattr(self, "_cache_recovery_state", None)
        if state is None:
            state = self._cache_recovery_state = _CacheRecoveryState()
        return state

    def _backward_work(self) -> BackwardWork | None:
        state = None
        started = None
        primary = False
        try:
            state = self._recovery_state()
            started = time.perf_counter_ns()
            with state.lock:
                if state.backward is None and not state.invalid:
                    state.backward = BackwardWork(state.lock, self.device)
                return state.backward
        except BaseException as exc:
            # Unknown accounting cost must never become free recovery budget,
            # including a failure before state lookup returned.
            if state is None:
                state = getattr(self, "_cache_recovery_state", None)
                if state is None:
                    state = self._cache_recovery_state = _CacheRecoveryState()
            state.invalid = True
            if isinstance(exc, Exception):
                return None
            primary = True
            raise
        finally:
            if state is not None and state.backward is not None:
                # Includes lazy setup and lock wait; constructor overlap is an
                # intentional conservative charge, not an exact subtraction.
                state.backward._charge(started, primary=primary)

    def _recovery_reduce(
        self,
        values: list[float],
        *,
        op: Literal["MAX", "MIN", "SUM"],
        sync_across_dp: bool,
    ) -> list[float]:
        if not (dist.is_available() and dist.is_initialized()):
            return values
        tensor = torch.tensor(
            values,
            device=self.device if self.device.type == "cuda" else "cpu",
            dtype=torch.float64,
        )
        dist.all_reduce(
            tensor,
            op=getattr(dist.ReduceOp, op),
            group=None if sync_across_dp else self._forward_memory_group(),
        )
        return [float(value) for value in tensor.tolist()]

    @staticmethod
    def _memory_error_with_reduction_note(
        error: BaseException,
        exchange_error: BaseException | None,
        *,
        operation: str = "memory reduction",
    ) -> BaseException:
        # Raise outside the exchange handler to preserve the local error's chain.
        # A secondary poisoned-communicator failure is diagnostic, not the primary.
        if exchange_error is not None and exchange_error is not error:
            try:
                BaseException.add_note(
                    error,
                    f"Secondary {operation} failure:\n"
                    + "".join(traceback.format_exception(exchange_error)),
                )
            except BaseException:
                # Exception rendering must never replace the original failure.
                pass
        return error

    def _try_cache_recovery(
        self,
        check: _MemoryCheck | None,
        *,
        sync_across_dp: bool,
        owner: object,
        started: float | None,
        handoff_grad: bool = False,
    ) -> bool:
        state = self._recovery_state()
        backward = self._backward_work()
        if backward is not None:
            backward.harvest()
        now = self._recovery_clock()
        elapsed = None if now is None or started is None else now - started
        with state.lock:
            invalid = (
                state.invalid
                or (backward is not None and backward.invalid)
                or state.owner is not owner
                or elapsed is None
                or not math.isfinite(elapsed)
                or elapsed < 0
            )
            projected = state.cost if elapsed is None else state.cost + elapsed
            # B survives forward rollback. O deliberately includes observer spans
            # also covered by recovery; reductions retain the original topology.
            work = state.work + (0.0 if backward is None else backward.work_ns / 1e9)
            accounted_cost = projected + (
                0.0 if backward is None else backward.cost_ns / 1e9
            )
            invalid |= any(
                not math.isfinite(value) or value < 0
                for value in (
                    projected,
                    state.work,
                    work,
                    state.high,
                    projected + state.high,
                    accounted_cost + state.high,
                )
            )
            local_cost = [0.0, 0.0] if invalid else [accounted_cost, state.high]
        # SUM costs deliberately overcharges parallel ranks; unlike MAX of
        # lifetime costs, it cannot miss episodes with different slow ranks.
        costs = self._recovery_reduce(
            local_cost, op="SUM", sync_across_dp=sync_across_dp
        )
        invalid |= any(not math.isfinite(value) for value in (*costs, sum(costs)))
        values = self._recovery_reduce(
            [
                float(check.estimated_required_bytes) if check is not None else 0.0,
                0.0 if invalid else work,
                float(state.first_consumed),
                float(invalid),
            ],
            op="MAX",
            sync_across_dp=sync_across_dp,
        )
        required = int(values[0])
        state.first_consumed = bool(values[2])
        state.invalid |= bool(values[3])
        _planner_evidence.record(
            self,
            "recovery",
            "budget_observed",
            local_projected_seconds=accounted_cost,
            local_high_seconds=state.high,
            local_work_seconds=work,
            local_recovery_seconds=projected,
            local_forward_work_seconds=state.work,
            reduced_cost_seconds=costs[0],
            reduced_high_seconds=costs[1],
            reduced_work_seconds=values[1],
            required_bytes=required,
            first_consumed=state.first_consumed,
            budget_fraction=0.05,
            invalid=state.invalid,
        )
        if state.invalid:
            _planner_evidence.record(
                self, "recovery", "skipped", reason="invalid_budget_or_owner"
            )
            return False
        error: BaseException | None = None
        needed = False
        cap_blocks = False
        try:
            available = self._available_memory_bytes() if check is not None else 0
            if (
                (available < required if check is not None else handoff_grad)
                and self.device.type == "cuda"
                and torch.cuda.is_available()
                and torch.cuda.get_allocator_backend() == "native"
            ):
                free, total = torch.cuda.mem_get_info(self.device)
                needed = int(free) < required + int(total * _MEMORY_RESERVE_FRACTION)
                if check is None:
                    needed &= int(torch.cuda.memory_reserved(self.device)) > int(
                        torch.cuda.memory_allocated(self.device)
                    )
                elif os.environ.get(_TEST_HOOKS_ENV) == "1":
                    limit = os.environ.get(_TEST_MEMORY_LIMIT_ENV)
                    if limit:
                        cap_blocks = required > max(
                            0,
                            int(limit) - int(torch.cuda.memory_allocated(self.device)),
                        )
        except BaseException as exc:
            error, available = exc, -1
        exchange_error: BaseException | None = None
        try:
            sampled = self._recovery_reduce(
                [
                    float(available),
                    -float(needed),
                    float(not cap_blocks),
                    float(not state.invalid),
                ],
                op="MIN",
                sync_across_dp=sync_across_dp,
            )
        except BaseException as exc:
            if error is None:
                raise
            exchange_error = exc
        if error is not None:
            _planner_evidence.record(
                self, "recovery", "failed", reason="sampling_or_reduction"
            )
            raise self._memory_error_with_reduction_note(error, exchange_error)
        if sampled[0] < 0:
            raise RuntimeError("Memory recovery sampling failed on another rank")
        state.invalid |= not bool(sampled[3])
        _planner_evidence.record(
            self,
            "recovery",
            "sampled",
            local_available_bytes=available if check is not None else None,
            reduced_available_bytes=sampled[0] if check is not None else None,
            local_needed=needed,
            any_needed=bool(-sampled[1]),
            local_cap_blocks=cap_blocks,
            cap_permits=bool(sampled[2]),
            invalid=state.invalid,
        )
        if check is not None and required <= sampled[0]:
            _planner_evidence.record(
                self, "recovery", "skipped", reason="fresh_sample_fits"
            )
            return True
        if sampled[1] == 0 or sampled[2] == 0 or state.invalid:
            _planner_evidence.record(
                self, "recovery", "skipped", reason="native_recovery_not_permitted"
            )
            return False
        if state.first_consumed and not (
            costs[1] > 0 and sum(costs) <= 0.05 * values[1]
        ):
            _planner_evidence.record(
                self, "recovery", "refused", reason="recovery_budget"
            )
            return False
        # Every peer enters this block. Only locally deficient native ranks call
        # the process-wide allocator operation. Test-only caps cannot trigger it.
        state.first_consumed = True
        attempted = False
        before_available = available
        try:
            if needed:
                # The control exchange can change allocator state. Check the
                # physical condition again immediately before the sole call.
                free, total = torch.cuda.mem_get_info(self.device)
                if int(free) < required + int(total * _MEMORY_RESERVE_FRACTION):
                    if check is not None:
                        attempted = True
                        _planner_evidence.record(
                            self,
                            "cache_release",
                            "attempted",
                            physical_free_bytes=int(free),
                            physical_total_bytes=int(total),
                            required_bytes=required,
                            reserve_bytes=int(total * _MEMORY_RESERVE_FRACTION),
                        )
                        torch.cuda.empty_cache()
                        _planner_evidence.record(self, "cache_release", "completed")
                    else:
                        allocated = int(torch.cuda.memory_allocated(self.device))
                        reserved = int(torch.cuda.memory_reserved(self.device))
                        if reserved > allocated:
                            # A soft trigger, not calibrated library demand. The
                            # native release affects unused caches process-wide.
                            evidence = dict(
                                device=str(self.device),
                                reserve_trigger_bytes=int(
                                    total * _MEMORY_RESERVE_FRACTION
                                ),
                                physical_free_before_bytes=int(free),
                                allocated_bytes=allocated,
                                reserved_before_bytes=reserved,
                            )
                            with _telemetry_phase(
                                "gradient_handoff_cache_release", evidence
                            ):
                                attempted = True
                                with torch.cuda.device(self.device):
                                    torch.cuda.empty_cache()
                                evidence["physical_free_after_bytes"] = int(
                                    torch.cuda.mem_get_info(self.device)[0]
                                )
                                evidence["reserved_after_bytes"] = int(
                                    torch.cuda.memory_reserved(self.device)
                                )
            available = self._available_memory_bytes() if check is not None else 0
        except BaseException as exc:
            error, available = exc, -1
        exchange_error: BaseException | None = None
        try:
            sampled = self._recovery_reduce(
                [float(available), -float(attempted)],
                op="MIN",
                sync_across_dp=sync_across_dp,
            )
        except BaseException as exc:
            if error is None:
                raise
            exchange_error = exc
        else:
            state.first_consumed |= bool(sampled[1])
        if error is not None:
            _planner_evidence.record(
                self,
                "recovery",
                "failed",
                reason="release_or_resample",
                attempted=attempted,
            )
            raise self._memory_error_with_reduction_note(error, exchange_error)
        if sampled[0] < 0:
            raise RuntimeError("Memory recovery failed on another rank")
        # Admission rebuilds pure search caches even when the sample decreased.
        # Handoff ignores this return: denied/insufficient recovery still yields
        # the completed outputs, with no claim that backward will fit.
        _planner_evidence.record(
            self,
            "recovery",
            "completed",
            attempted=attempted,
            local_before_available_bytes=before_available
            if check is not None
            else None,
            local_after_available_bytes=available if check is not None else None,
            observed_available_delta_bytes=(
                available - before_available if check is not None else None
            ),
            reduced_available_bytes=sampled[0] if check is not None else None,
        )
        return True

    def _one_layer_recompute(self) -> bool:
        """ART's default full/uniform/1 recompute, which Megatron runs in training.

        Forward keeps only layer inputs and backward recomputes one layer at a
        time. Megatron checks the decoder's own mode and config, and eval mode
        skips recompute even with gradients enabled, so read both live.
        """
        recorded = self.__dict__.get("_recorded_one_layer_recompute")
        if recorded is not None:
            return recorded  # Planner-report replay has no live model.
        target = ("full", "uniform", 1)
        names = ("recompute_granularity", "recompute_method", "recompute_num_layers")

        def active(chunk: torch.nn.Module) -> bool:
            try:
                decoder = _language_model(chunk).decoder
                config = decoder.config
            except (AttributeError, RuntimeError):
                # No decoder config (stub or non-GPT chunk): stored settings,
                # and every module must be training.
                stored = tuple(getattr(self, "_" + name) for name in names)
                return stored == target and all(m.training for m in chunk.modules())
            settings = tuple(getattr(config, name, None) for name in names)
            return settings == target and decoder.training is True

        return all(active(chunk) for chunk in self.runtime.model)

    def _all_ranks_true(self, local: bool) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return local
        value = torch.tensor(
            int(local),
            device=self.device if self.device.type == "cuda" else "cpu",
            dtype=torch.int32,
        )
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def _forward_item(self, request: AnyForwardInput) -> _ForwardItem:
        if request.top_k is not None:
            _validate_top_k(request.top_k, _language_model(self.runtime.model[0]))
        input_ids = request.input_tokens.reshape(-1).to(dtype=torch.long)
        if int(input_ids.numel()) == 0:
            raise ValueError("input_tokens must not be empty")
        labels = None
        if request.target_tokens is not None:
            labels = request.target_tokens.to(dtype=torch.long)
            if int(labels.numel()) == 0:
                raise ValueError("target_tokens must not be empty")
            input_shape = tuple(request.input_tokens.shape)
            if tuple(labels.shape) == input_shape:
                labels = labels.reshape(-1)
            elif (
                labels.ndim > request.input_tokens.ndim
                and tuple(labels.shape[: request.input_tokens.ndim]) == input_shape
            ):
                labels = labels.reshape(
                    int(input_ids.numel()), *labels.shape[request.input_tokens.ndim :]
                )
            elif labels.ndim < 1 or int(labels.shape[0]) != int(input_ids.numel()):
                raise ValueError(
                    "target_tokens must match input_tokens or add trailing target "
                    f"dimensions: input_tokens={input_shape} "
                    f"target_tokens={tuple(labels.shape)}"
                )
        routes = request.routed_experts
        if routes is not None:
            from art.megatron.routed_experts import router_bindings, validate_routes

            if not hasattr(self, "_routing_bindings"):
                self._routing_bindings = router_bindings(self.runtime.model)
            routes = validate_routes(
                routes,
                int(input_ids.numel()),
                self.runtime.provider.num_layers,
                self._routing_bindings,
            )
        return _ForwardItem(
            request=request, input_ids=input_ids, labels=labels, routed_experts=routes
        )

    def _forward_packed(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
    ) -> list[AnyForwardOutput]:
        hidden_by_row = self._gather_sequence_parallel_hidden(
            self._decoder_hidden(prepared)
        )
        outputs = self._project_head(items, prepared, hidden_by_row)
        group = prepared.context_parallel_group
        if group is None:
            return outputs
        tensors = [
            tensor
            for output in outputs
            for tensor in (
                output.target_logprobs,
                output.logits,
                output.hidden_states,
                None if output.top_k is None else output.top_k.logprobs,
                None if output.top_k is None else output.top_k.tokens,
            )
            if tensor is not None
        ]
        grad_flags = [False for _ in tensors]
        if torch.is_grad_enabled():
            flags = torch.tensor(
                [
                    tensor.is_floating_point()
                    and (tensor.requires_grad or hidden_by_row.requires_grad)
                    for tensor in tensors
                ],
                device=hidden_by_row.device,
                dtype=torch.int32,
            )
            dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
            grad_flags = flags.tolist()
        needs_grad = iter(grad_flags)

        def gather(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if next(needs_grad) and not tensor.requires_grad:
                # Empty shards must still enter decoder backward collectives.
                # A frozen decoder with a trainable head only needs a leaf.
                tensor = tensor + hidden_by_row.reshape(-1)[:1].sum() * 0.0
                tensor.requires_grad_(True)
            return _GatherContextParallelRows.apply(tensor, positions, length, group)

        for index, (item, output, source_positions) in enumerate(
            zip(items, outputs, prepared.source_positions_by_item, strict=True)
        ):
            positions = source_positions.to(device=hidden_by_row.device)
            length = int(item.input_ids.numel())
            outputs[index] = replace(
                output,
                target_logprobs=gather(output.target_logprobs),
                logits=gather(output.logits),
                hidden_states=gather(output.hidden_states),
                top_k=(
                    TopK(
                        cast(torch.Tensor, gather(output.top_k.logprobs)),
                        cast(torch.Tensor, gather(output.top_k.tokens)),
                    )
                    if output.top_k is not None
                    else None
                ),
            )
        return outputs

    def _decoder_hidden(
        self,
        prepared: _PreparedPackedForward,
    ) -> torch.Tensor:
        from art.megatron.train import _placeholder_attention_mask

        handler = self.runtime.model_support_handler
        model = _language_model(self.runtime.model[0])
        attention_mask = _placeholder_attention_mask(self.device)
        forward_kwargs = handler.get_forward_kwargs(
            self.runtime.model[0],
            attention_bias=prepared.attention_state,
        )
        extra_block_kwargs = cast(
            dict[str, object] | None,
            forward_kwargs.pop("extra_block_kwargs", None),
        )
        preprocessed = model._preprocess(
            input_ids=prepared.tokens,
            position_ids=prepared.position_ids,
            packed_seq_params=cast("PackedSeqParams", prepared.packed_seq_params),
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preprocessed[:6]
        rotary_pos_cos_sin = preprocessed[6] if len(preprocessed) == 7 else None
        return cast(
            torch.Tensor,
            model.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                packed_seq_params=prepared.packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
                **(extra_block_kwargs or {}),
            ),
        )

    def _project_head(
        self,
        items: Sequence[_ForwardItem],
        prepared: _PreparedPackedForward,
        hidden_by_row: torch.Tensor,
    ) -> list[AnyForwardOutput]:
        model = _language_model(self.runtime.model[0])
        output_weight = (
            model.shared_embedding_or_output_weight()
            if bool(model.share_embeddings_and_output_weights)
            else None
        )
        device = hidden_by_row.device
        target_logprobs = [None for _ in items]
        logits: list[torch.Tensor | None] = [None for _ in items]
        top_k: list[TopK | None] = [None for _ in items]
        label_rows: list[torch.Tensor | None] = [None for _ in items]
        projected_rows: list[torch.Tensor] = []

        for index, (item, positions_cpu) in enumerate(
            zip(items, prepared.positions_by_item, strict=True)
        ):
            positions = positions_cpu.to(device=device)
            if item.request.logits or item.request.top_k is not None:
                projected_rows.append(positions)
            if item.labels is not None:
                source_positions = prepared.source_positions_by_item[index].to(device)
                labels = item.labels.to(device=device).index_select(0, source_positions)
                label_rows[index] = labels
                target_logprobs[index] = torch.zeros(
                    tuple(labels.shape),
                    device=device,
                    dtype=torch.float32,
                )
                if item.request.top_k is None and not item.request.logits:
                    if int(labels.shape[0]):
                        valid = labels != -100
                        if labels.ndim > 1:
                            valid = valid.reshape(int(labels.shape[0]), -1).any(dim=1)
                        valid_offsets = torch.nonzero(valid, as_tuple=False).reshape(-1)
                        if int(valid_offsets.numel()):
                            projected_rows.append(
                                positions.index_select(0, valid_offsets)
                            )
            if item.request.logits:
                logits[index] = torch.empty(
                    (int(positions.numel()), _padded_vocab_size(model)),
                    device=hidden_by_row.device,
                    dtype=hidden_by_row.dtype,
                )
            if item.request.top_k is not None:
                shape = (int(positions.numel()), item.request.top_k)
                top_k[index] = TopK(
                    logprobs=torch.empty(shape, device=device, dtype=torch.float32),
                    tokens=torch.empty(shape, device=device, dtype=torch.long),
                )

        row_tensor = (
            torch.cat(projected_rows).unique(sorted=True)
            if projected_rows
            else torch.empty(0, dtype=torch.long, device=device)
        )
        if int(row_tensor.numel()):
            rows_cpu = row_tensor.detach().cpu()
            cpu_matches = tuple(
                _row_match(
                    positions.cpu(),
                    rows_cpu,
                    chunk_tokens=_HEAD_CHUNK_TOKENS,
                )
                for positions in prepared.positions_by_item
            )
            local_row_matches = tuple(
                (source.to(device), row.to(device), bounds)
                for source, row, bounds in cpu_matches
            )
            logit_rows_cpu = torch.cat(
                tuple(
                    match[1]
                    for item, match in zip(items, cpu_matches, strict=True)
                    if item.request.logits
                )
                or (torch.empty(0, dtype=torch.long),)
            ).unique(sorted=True)
            self._project_vocab_parallel(
                items,
                hidden_by_row,
                row_tensor,
                row_matches=local_row_matches,
                logit_rows=logit_rows_cpu.to(device),
                logit_bounds=_chunk_boundaries(
                    logit_rows_cpu,
                    end=int(row_tensor.numel()),
                    chunk_tokens=_HEAD_CHUNK_TOKENS,
                ),
                output_weight=output_weight,
                target_logprobs=target_logprobs,
                top_k=top_k,
                logits=logits,
                label_rows=label_rows,
            )

        target_logprobs, top_k = _anchor_disconnected_outputs(
            target_logprobs,
            top_k,
            hidden_by_row,
        )
        return [
            ForwardOutput(
                target_logprobs=target_logprobs[index],
                top_k=top_k[index],
                logits=logits[index],
                hidden_states=(
                    _select_positions(hidden_by_row, positions)
                    if item.request.hidden_states
                    else None
                ),
            )
            for index, (item, positions) in enumerate(
                zip(items, prepared.positions_by_item, strict=True)
            )
        ]

    def _project_vocab_parallel(
        self,
        items: Sequence[_ForwardItem],
        hidden_by_row: torch.Tensor,
        rows: torch.Tensor,
        *,
        row_matches: Sequence[_RowMatch],
        logit_rows: torch.Tensor,
        logit_bounds: tuple[int, ...],
        output_weight: torch.Tensor | None,
        target_logprobs: list[torch.Tensor | None],
        top_k: list[TopK | None],
        logits: list[torch.Tensor | None],
        label_rows: list[torch.Tensor | None],
    ) -> None:
        from torch.utils.checkpoint import checkpoint

        model = _language_model(self.runtime.model[0])
        max_top_k = max((int(item.request.top_k or 0) for item in items), default=0)
        need_log_z = any(
            item.labels is not None or item.request.top_k is not None for item in items
        )
        for chunk_index, start in enumerate(
            range(0, int(rows.numel()), _HEAD_CHUNK_TOKENS)
        ):
            chunk_rows = rows[start : start + _HEAD_CHUNK_TOKENS]
            # Recompute vocabulary-sized intermediates one chunk at a time in
            # backward; chunking alone otherwise retains every chunk's logits.
            local_logits, log_z, local_topk = self._checkpointed_head_stats(
                model,
                _select_positions(hidden_by_row, chunk_rows),
                output_weight=output_weight,
                need_log_z=need_log_z,
                max_top_k=max_top_k,
            )
            logit_start, logit_end = logit_bounds[chunk_index : chunk_index + 2]
            logit_chunk_offsets = logit_rows[logit_start:logit_end] - start
            chunk_logits: torch.Tensor | None = None
            if int(logit_chunk_offsets.numel()):
                chunk_logits = _batch_seq_logits(
                    self._gather_tensor_parallel_logits(
                        local_logits.index_select(0, logit_chunk_offsets).unsqueeze(1)
                    ),
                    seq_len=int(logit_chunk_offsets.numel()),
                ).squeeze(0)

            for index, item in enumerate(items):
                offsets, row_offsets, bounds = row_matches[index]
                begin, finish = bounds[chunk_index : chunk_index + 2]
                offsets = offsets[begin:finish]
                chunk_offsets = row_offsets[begin:finish] - start
                if int(offsets.numel()) == 0:
                    continue
                item_logits = logits[index]
                if item_logits is not None:
                    if chunk_logits is None:
                        raise RuntimeError("logits output requires gathered logits")
                    item_logits[offsets] = chunk_logits.index_select(
                        0,
                        torch.searchsorted(logit_chunk_offsets, chunk_offsets),
                    )
                labels = label_rows[index]
                item_logprobs = target_logprobs[index]
                if item_logprobs is not None and labels is not None:
                    if log_z is None:
                        raise RuntimeError("target logprobs require logsumexp")
                    selected_log_z = log_z.index_select(0, chunk_offsets)
                    item_logprobs[offsets] = _vocab_parallel_target_logprobs(
                        local_logits,
                        labels.index_select(0, offsets),
                        selected_log_z,
                        row_offsets=chunk_offsets,
                    )
                k = item.request.top_k
                if k is not None:
                    if log_z is None:
                        raise RuntimeError("top_k requires logsumexp")
                    selected_log_z = log_z.index_select(0, chunk_offsets)
                    if local_topk is not None:
                        local_values, local_tokens = local_topk
                        selected_values = local_values.index_select(0, chunk_offsets)
                        selected_tokens = local_tokens.index_select(0, chunk_offsets)
                    else:
                        selected_logits = local_logits.index_select(0, chunk_offsets)
                        selected_values, selected_tokens = torch.topk(
                            selected_logits.float(),
                            k=min(k, int(selected_logits.shape[1])),
                            dim=-1,
                        )
                        del selected_logits
                    values = _vocab_parallel_topk_from_local(
                        selected_values,
                        selected_tokens,
                        k=k,
                        log_z=selected_log_z,
                        vocab_start=_vocab_range(local_logits)[0],
                    )
                    current = top_k[index]
                    if current is None:
                        raise RuntimeError("top_k output was not allocated")
                    current.logprobs[offsets] = values.logprobs
                    current.tokens[offsets] = values.tokens
            # Do not retain prior chunk buffers while the next stats RHS runs.
            del local_logits, chunk_logits

    def _checkpointed_head_stats(
        self,
        model: "GPTModel",
        hidden: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
        need_log_z: bool,
        max_top_k: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """One chunk's head statistics under non-reentrant checkpointing.

        The recompute must save what the forward saved, so it replays the
        statistics path the forward chose (``path``) instead of attempting
        the kernels again.
        """
        from torch.utils.checkpoint import checkpoint

        return checkpoint(
            self._local_head_stats,
            model,
            hidden,
            output_weight=output_weight,
            need_log_z=need_log_z,
            max_top_k=max_top_k,
            path=[],
            use_reentrant=False,
        )

    def _local_head_stats(
        self,
        model: "GPTModel",
        hidden: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
        need_log_z: bool,
        max_top_k: int,
        path: list[str] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        local_logits = self._local_logits_from_hidden_rows(
            model,
            hidden,
            output_weight=output_weight,
        )
        log_z: torch.Tensor | None = None
        local_topk: tuple[torch.Tensor, torch.Tensor] | None = None
        if need_log_z:
            # ``path`` holds the forward's statistics path; a checkpoint
            # recompute replays it rather than attempting the kernels again.
            recorded = path[0] if path else None
            topk_stats = (
                _try_triton_local_topk_stats(local_logits, k=max_top_k)
                if recorded in (None, "topk")
                else None
            )
            logsumexp_stats = (
                cast(
                    tuple[torch.Tensor, torch.Tensor] | None,
                    _try_triton_stats("local_logsumexp_stats", local_logits),
                )
                if topk_stats is None and recorded in (None, "logsumexp")
                else None
            )
            shape = (int(local_logits.shape[0]), int(local_logits.shape[1]))
            if recorded is None:
                if topk_stats is not None or logsumexp_stats is not None:
                    # A kernel ran at this chunk shape: admission may price it.
                    getattr(self, "_triton_head_stats_failures", set()).discard(shape)
                elif _triton_stats_enabled(local_logits.is_cuda, shape[0]):
                    # An attempted kernel failed at this chunk shape. Admission
                    # priced the kernel's buffers, so compute the same
                    # statistics eagerly within them; later admissions price
                    # the shape for the eager fallback (_triton_head_stats)
                    # until a kernel succeeds at it again.
                    if not hasattr(self, "_triton_head_stats_failures"):
                        self._triton_head_stats_failures = set()
                    self._triton_head_stats_failures.add(shape)
                    logsumexp_stats = _eager_local_logsumexp_stats(local_logits)
                    recorded = "bounded"
                if path is not None:
                    path.append(
                        recorded
                        or (
                            "topk"
                            if topk_stats is not None
                            else "logsumexp"
                            if logsumexp_stats is not None
                            else "eager"
                        )
                    )
            elif recorded == "bounded":
                logsumexp_stats = _eager_local_logsumexp_stats(local_logits)
            elif recorded in ("topk", "logsumexp") and (
                topk_stats is None and logsumexp_stats is None
            ):
                raise RuntimeError(
                    "the head statistics kernel failed in a checkpoint recompute "
                    "after it succeeded in the forward"
                )
            stats = topk_stats if topk_stats is not None else logsumexp_stats
            if stats is not None:
                local_max, local_sum = stats[:2]
                local_max = local_max.detach()
                global_max = _all_reduce_tensor_parallel_max(local_max)
                global_sum = _all_reduce_tensor_parallel_sum(
                    local_sum * torch.exp(local_max - global_max)
                )
                log_z = global_max + torch.log(global_sum)
            else:
                # No kernel attempted (non-CUDA, disabled, short chunk):
                # admission prices this FP32 fallback's seven buffers.
                log_z = _vocab_parallel_log_z(local_logits)

            if topk_stats is not None:
                _, _, local_values, local_tokens = topk_stats
                local_topk = (local_values, local_tokens)
            elif logsumexp_stats is not None and max_top_k > 0:
                local_k = min(max_top_k, int(local_logits.shape[1]))
                local_values, local_tokens = torch.topk(local_logits, k=local_k, dim=-1)
                local_topk = (local_values.float(), local_tokens)
        return local_logits, log_z, local_topk

    def _local_logits_from_hidden_rows(
        self,
        model: "GPTModel",
        hidden: torch.Tensor,
        *,
        output_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        output_layer = model.output_layer
        sequence_parallel = bool(getattr(output_layer, "sequence_parallel", False))
        if sequence_parallel:
            output_layer.sequence_parallel = False
        try:
            logits, _ = output_layer(
                hidden.unsqueeze(1),
                weight=output_weight,
                runtime_gather_output=None,
            )
        finally:
            if sequence_parallel:
                output_layer.sequence_parallel = True
        return _batch_seq_logits(
            model._scale_logits(logits),
            seq_len=int(hidden.shape[0]),
        ).squeeze(0)

    def _gather_sequence_parallel_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as ps

        if int(ps.get_tensor_model_parallel_world_size()) <= 1:
            return hidden.squeeze(1)
        from megatron.core import tensor_parallel

        gathered = tensor_parallel.gather_from_sequence_parallel_region(
            hidden,
            tensor_parallel_output_grad=False,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        )
        return cast(torch.Tensor, gathered).squeeze(1)

    def _prepare_packed_forward(
        self,
        batch: PrefixTreePack,
    ) -> _PreparedPackedForward:
        topology = self._topology()
        batch = _pad_packed_batch(batch, multiple=int(topology.tp))
        if int(topology.cp) > 1:
            return self._prepare_context_parallel_forward(batch, topology=topology)
        from art.megatron.prefix_tree_state import create_prefix_tree_state
        from art.megatron.training.microbatches import (
            _art_flex_sliding_windows,
            _gdn_planner_config_for_provider,
        )

        handler = self.runtime.model_support_handler
        provider = self.runtime.provider
        return _PreparedPackedForward(
            tokens=batch.tokens.to(self.device),
            token_uids=torch.arange(batch.tokens.numel(), dtype=torch.int64).unsqueeze(
                0
            ),
            position_ids=batch.position_ids.to(self.device),
            attention_state=create_prefix_tree_state(
                group_ids=batch.group_ids,
                parent_ids=batch.parent_ids,
                target_device=self.device,
                input_pos=batch.position_ids,
                sliding_windows=_art_flex_sliding_windows(provider),
                build_gdn_execution_spec=handler.build_gdn_execution_spec,
                model_support_handler=handler,
                attention_head_dim=provider.kv_channels,
                attention_value_head_dim=provider.kv_channels,
                gdn_planner_config=_gdn_planner_config_for_provider(provider, handler),
            ),
            packed_seq_params=None,
            positions_by_item=batch.positions_by_sequence,
            source_positions_by_item=tuple(
                torch.arange(
                    int(positions.numel()),
                    dtype=torch.long,
                    device=positions.device,
                )
                for positions in batch.positions_by_sequence
            ),
        )

    def _configure_hybridep(
        self,
        batches: Sequence[PrefixTreePack],
        *,
        topology: "ParallelTopology",
    ) -> tuple[tuple[int, ...], int] | None:
        from megatron.core import parallel_state as ps

        expert_parallel_size = int(ps.get_expert_model_parallel_world_size())
        if expert_parallel_size <= 1:
            self._hybridep_graph_tracking = False
            return None
        self._validate_hybridep_topology(topology)
        if not batches:
            return None
        from megatron.core.transformer.moe import fused_a2a

        from art.megatron.train import (
            _ensure_hybridep_capacity,
            _hybridep_token_capacity,
        )

        padded = tuple(
            _pad_packed_batch(batch, multiple=int(topology.tp)) for batch in batches
        )
        sequence_length = max(int(batch.tokens.shape[1]) for batch in padded)
        rows = tuple(
            self._max_rank_model_tokens(batch, topology=topology) for batch in padded
        )
        current = fused_a2a._hybrid_ep_buffer
        live = self._has_live_hybridep_graphs()
        # The buffer must hold the busiest rank's planned rows: cost-aware CP
        # plans (few long histories, shared prefixes) can put more than the
        # near-balanced extent on one rank, and every rank sizes the buffer
        # identically because the planned counts are the same on all ranks.
        required_capacity = max(
            _hybridep_token_capacity(sequence_length, int(topology.cp)),
            max(rows),
        )
        if live and (
            current is None
            or id(current) != getattr(self, "_hybridep_buffer_id", None)
            or int(current.configurer.buffer_config.max_num_of_tokens_per_rank)
            < required_capacity
        ):
            raise TrainerRankSlotStateError(
                "Cannot grow or replace the HybridEP buffer while an earlier "
                "TrainerRank forward still has a live backward graph. Finish "
                "backward or release those outputs before forwarding a larger batch."
            )
        _ensure_hybridep_capacity(
            self.runtime,
            packed_sequence_length=sequence_length,
            context_parallel_size=int(topology.cp),
            required_capacity=required_capacity,
        )
        current = fused_a2a._hybrid_ep_buffer
        if current is None:
            raise RuntimeError("HybridEP buffer was not initialized")
        if live:
            high_water = max(*rows, int(getattr(self, "_hybridep_rows_high_water", 0)))
        else:
            high_water = max(rows)
        self._hybridep_buffer_id = id(current)
        self._hybridep_rows_high_water = high_water
        self._hybridep_graph_tracking = True
        return rows, high_water

    def _validate_hybridep_topology(
        self,
        topology: "ParallelTopology | None" = None,
    ) -> None:
        if topology is None:
            configured_ep = int(
                getattr(self.runtime.provider, "expert_model_parallel_size", 1) or 1
            )
            if configured_ep <= 1:
                return
            topology = self._topology()
        if int(topology.dp) > 1:
            raise NotImplementedError(
                "TrainerRank does not support combining data parallelism with "
                "expert parallelism because uneven DP inputs can desynchronize "
                "HybridEP collectives. For MoE models, use DP=1 with CP and EP "
                "set to the world size."
            )

    @staticmethod
    def _set_hybridep_rows(rows: int) -> None:
        from art.megatron.train import _set_hybridep_token_count

        _set_hybridep_token_count(rows)

    def _max_rank_model_tokens(
        self,
        batch: PrefixTreePack,
        *,
        topology: "ParallelTopology",
    ) -> int:
        sequence_length = int(batch.tokens.shape[1])
        if int(topology.cp) <= 1:
            return sequence_length
        if int(topology.cp) > 1:
            from art.megatron.context_parallel.runtime import (
                context_parallel_rank_model_token_counts,
            )
            from art.megatron.training.microbatches import (
                _context_parallel_config_for_provider,
                _gdn_planner_config_for_provider,
            )

            handler = self.runtime.model_support_handler
            return max(
                context_parallel_rank_model_token_counts(
                    group_ids=batch.group_ids,
                    parent_ids=batch.parent_ids,
                    topology=topology,
                    config=_context_parallel_config_for_provider(
                        self.runtime.provider,
                        self.device,
                        handler,
                    ),
                    original_seq_len=sequence_length,
                    build_gdn_execution_spec=handler.build_gdn_execution_spec,
                    gdn_planner_config=_gdn_planner_config_for_provider(
                        self.runtime.provider, handler
                    ),
                )
            )
        raise AssertionError("unreachable")

    def _prepare_context_parallel_forward(
        self,
        batch: PrefixTreePack,
        *,
        topology: "ParallelTopology",
    ) -> _PreparedPackedForward:
        from megatron.core import parallel_state as ps

        from art.megatron.context_parallel.runtime import (
            _dispatch_tensor,
            prepare_cp_micro,
        )
        from art.megatron.training.microbatches import (
            _art_flex_cp_block_mask_variants,
            _context_parallel_config_for_provider,
            _gdn_planner_config_for_provider,
        )
        from art.preprocessing.pack import PackedTensors

        assistant_mask = torch.ones_like(batch.tokens, dtype=torch.bool)
        sparse_micro: PackedTensors = {
            "tokens": batch.tokens,
            "group_ids": batch.group_ids,
            "parent_ids": batch.parent_ids,
            "input_pos": batch.position_ids,
            "assistant_mask": assistant_mask,
            "logprobs": torch.full_like(
                batch.tokens, float("nan"), dtype=torch.float32
            ),
            "advantages": torch.zeros_like(batch.tokens, dtype=torch.float32),
            "weights": assistant_mask.to(dtype=torch.float32),
            "pixel_values": [None],
            "image_grid_thw": [None],
            "moe_routing_replay": None,
        }
        handler = self.runtime.model_support_handler
        provider = self.runtime.provider
        prepared = prepare_cp_micro(
            micro=sparse_micro,
            topology=topology,
            config=_context_parallel_config_for_provider(
                provider,
                self.device,
                handler,
            ),
            cp_group=ps.get_context_parallel_group(check_initialized=False),
            cp_rank=ps.get_context_parallel_rank(),
            build_gdn_execution_spec=handler.build_gdn_execution_spec,
            gdn_planner_config=_gdn_planner_config_for_provider(provider, handler),
            block_mask_variants=_art_flex_cp_block_mask_variants(provider, self.device),
            target_device=self.device,
        )
        if prepared.rank_plan is None:
            raise RuntimeError("CP forward preparation did not return a rank plan")
        local_positions = _dispatch_tensor(
            torch.arange(
                int(batch.tokens.shape[1]),
                dtype=torch.long,
            ).unsqueeze(0),
            rank_plan=prepared.rank_plan,
            pad_value=-1,
            pad_multiple=prepared.pad_multiple,
        )
        local_position_pairs = tuple(
            _local_position_pairs(local_positions, positions)
            for positions in batch.positions_by_sequence
        )
        return _PreparedPackedForward(
            tokens=prepared.tensors.tokens,
            token_uids=local_positions,
            position_ids=prepared.tensors.input_pos,
            attention_state=cast("ArtContextParallelState", prepared.attention_state),
            packed_seq_params=prepared.packed_seq_params,
            positions_by_item=tuple(pair[0] for pair in local_position_pairs),
            source_positions_by_item=tuple(pair[1] for pair in local_position_pairs),
            context_parallel_group=ps.get_context_parallel_group(),
        )

    def _topology(self) -> "ParallelTopology":
        from art.megatron.train import _infer_parallel_topology

        return _infer_parallel_topology(self.runtime.model)

    def _gather_tensor_parallel_logits(self, logits: torch.Tensor) -> torch.Tensor:
        from megatron.core import parallel_state as ps

        if int(ps.get_tensor_model_parallel_world_size()) <= 1:
            return logits
        from megatron.core import tensor_parallel

        return cast(
            torch.Tensor,
            tensor_parallel.gather_from_tensor_model_parallel_region(logits),
        )

    _split_required_memory = staticmethod(_memory._split_required_memory)
    _split_memory_key = staticmethod(_memory._split_memory_key)
    _record_split_memory_floor = _memory._record_split_memory_floor
    _split_plan_memory_check = _memory._split_plan_memory_check
    _head_workspace_bytes = _memory._head_workspace_bytes
    _group_head_workspace_bytes = _memory._group_head_workspace_bytes
    _plan_head_workspace_bytes = _memory._plan_head_workspace_bytes
    _plan_hybridep_growth_bytes = _memory._plan_hybridep_growth_bytes
    _checkpoint_moe_bytes_per_token = _memory._checkpoint_moe_bytes_per_token
    _moe_workspace_bytes = _memory._moe_workspace_bytes
    _checkpoint_memory_floor = _memory._checkpoint_memory_floor
    _dense_mlp_widths = _memory._dense_mlp_widths
    _dense_mixer_widths = _memory._dense_mixer_widths
    _dense_layout_floors = _memory._dense_layout_floors
    _dense_adapter_gradient_extra = _memory._dense_adapter_gradient_extra
    _layout_layer_boundaries = _memory._layout_layer_boundaries
    _layer_gdn_inputs = _memory._layer_gdn_inputs
    _layout_pricing_supported = _memory._layout_pricing_supported
    _minimum_layouts = _memory._minimum_layouts
    _te_workspace_growth_bytes = _memory._te_workspace_growth_bytes
    _gradient_slots = staticmethod(_memory._gradient_slots)
    _pending_adapter_gradient_bytes = _memory._pending_adapter_gradient_bytes
    _checkpoint_gradient_groups = _memory._checkpoint_gradient_groups
    _checkpoint_adapter_gradient_bytes = _memory._checkpoint_adapter_gradient_bytes
    _adapter_gradient_walk = staticmethod(_memory._adapter_gradient_walk)
    _retained_memory_bytes = _memory._retained_memory_bytes
    _estimate_flat_forward = _memory._estimate_flat_forward
    _update_peak_memory_profile = _memory._update_peak_memory_profile
    _estimate_group_request_output_bytes = _memory._estimate_group_request_output_bytes
    _memory_signature_from_requests = _memory._memory_signature_from_requests
    _slot_memory_shapes = _memory._slot_memory_shapes
    _memory_check = _memory._memory_check
    _refresh_memory_check = _memory._refresh_memory_check
    _memory_check_required = _memory._memory_check_required
    _forward_memory_group = staticmethod(_memory._forward_memory_group)
    _estimate_required_memory_bytes_from_values = (
        _memory._estimate_required_memory_bytes_from_values
    )
    _gdn_segment_layer_bytes = _memory._gdn_segment_layer_bytes
    _sequence_parallel_workspace_bytes = _memory._sequence_parallel_workspace_bytes
    _sequence_parallel_lora_floor = _memory._sequence_parallel_lora_floor
    _cold_recompute_transient_bytes = _memory._cold_recompute_transient_bytes
    _checkpoint_input_gradient_bytes = _memory._checkpoint_input_gradient_bytes
    _available_memory_bytes = _memory._available_memory_bytes
    _all_ranks_have_memory_profile = _memory._all_ranks_have_memory_profile
    _update_memory_profile = _memory._update_memory_profile

    _forward_batches = _micro_batch_planner._forward_batches
    _plan_admissible_forward = _micro_batch_planner._plan_admissible_forward
    _find_admissible_forward = _micro_batch_planner._find_admissible_forward
    _admit_split_rung = _micro_batch_planner._admit_split_rung
    _split_rung_check = _micro_batch_planner._split_rung_check
    _split_chunk_lower_cost = _micro_batch_planner._split_chunk_lower_cost
    _plan_group_rows = _micro_batch_planner._plan_group_rows
    _plan_cost = _micro_batch_planner._plan_cost
    _plan_group_layouts = _micro_batch_planner._plan_group_layouts
    _compute_group_layouts = _micro_batch_planner._compute_group_layouts
    _split_request_order = _micro_batch_planner._split_request_order
    _reset_planning_telemetry = _micro_batch_planner._reset_planning_telemetry
    _snapshot_planning_telemetry = _micro_batch_planner._snapshot_planning_telemetry
    _select_next_micro_batch = _micro_batch_planner._select_next_micro_batch
    _search_next_micro_batch = _micro_batch_planner._search_next_micro_batch
    _planner_topology_facts = _micro_batch_planner._planner_topology_facts
    _layout_anchor = _micro_batch_planner._layout_anchor
    _layout_cache_key = _micro_batch_planner._layout_cache_key
    _cached_group_layout = _micro_batch_planner._cached_group_layout
    _compute_group_layout = _micro_batch_planner._compute_group_layout
    _plan_structure = _micro_batch_planner._plan_structure
    _select_group_layout = _micro_batch_planner._select_group_layout
    _speculative_planning_executor = _micro_batch_planner._speculative_planning_executor
    _submit_speculative_wave_planning = (
        _micro_batch_planner._submit_speculative_wave_planning
    )
    _plan_flat_forward = _micro_batch_planner._plan_flat_forward
    _fill_planner_snapshot = _micro_batch_planner._fill_planner_snapshot
    _complete_planner_observation = _micro_batch_planner._complete_planner_observation
    finish_planner_observation = _micro_batch_planner.finish_planner_observation
    report_planner_oom = _micro_batch_planner.report_planner_oom
    discard_planner_observation = _micro_batch_planner.discard_planner_observation
    _admission_outcome = _micro_batch_planner._admission_outcome
    _recover_admission = _micro_batch_planner._recover_admission
    _report_planning_failure = _micro_batch_planner._report_planning_failure
    _recover_admission_impl = _micro_batch_planner._recover_admission_impl
    _plan_retained_tokens = _micro_batch_planner._plan_retained_tokens
    _planning_status = _micro_batch_planner._planning_status

    _resolve_custom_checkpoint = _slots._resolve_custom_checkpoint
    prefetch_checkpoints = _slots.prefetch_checkpoints
    _register_checkpoint_prefetch = _slots._register_checkpoint_prefetch
    _register_checkpoint_source = _slots._register_checkpoint_source
    _checkpoint_slot_write = _slots._checkpoint_slot_write
    _checkpoint_snapshot_state = _slots._checkpoint_snapshot_state
    _trim_checkpoint_snapshots = _slots._trim_checkpoint_snapshots
    _checkpoint_prefetch_waiter = _slots._checkpoint_prefetch_waiter
    _prefetched_checkpoint = _slots._prefetched_checkpoint
    _load_registered_checkpoint = _slots._load_registered_checkpoint
    _ensure_checkpoint_slots = _slots._ensure_checkpoint_slots
    load_checkpoint = _slots.load_checkpoint
    _discard_snapshot_checkpoint = _slots._discard_snapshot_checkpoint
    _push_checkpoint = _slots._push_checkpoint
    _push_checkpoint_sync = _slots._push_checkpoint_sync
    pop_checkpoint = _slots.pop_checkpoint
    _resolve_checkpoint_name = _slots._resolve_checkpoint_name
    _checkpoint_group = _slots._checkpoint_group
    _validate_checkpoint_adapter_config = _slots._validate_checkpoint_adapter_config
    _validate_loaded_checkpoint_config = _slots._validate_loaded_checkpoint_config
    _load_checkpoint_slot = _slots._load_checkpoint_slot
    _iter_slot_parameters = _slots._iter_slot_parameters
    _validate_checkpoint_consistency = _slots._validate_checkpoint_consistency
    _set_default_slot = _slots._set_default_slot
    _resolve_slot_ref = _slots._resolve_slot_ref
    _selected_dynamic_checkpoints = _slots._selected_dynamic_checkpoints
    _checkpoint_grad_flags = _slots._checkpoint_grad_flags
    _ensure_checkpoint_slots_for = _slots._ensure_checkpoint_slots_for
    _track_slot_graph_outputs = _slots._track_slot_graph_outputs
    _slot_graphs = _slots._slot_graphs
    _prune_slot_graphs = _slots._prune_slot_graphs
    _has_live_slot_graph = _slots._has_live_slot_graph
    _guard_slot_can_load = _slots._guard_slot_can_load
    _guard_checkpoint_can_step = _slots._guard_checkpoint_can_step
    _guard_checkpoints_can_step = _slots._guard_checkpoints_can_step

    _extend_dynamic_optimizer = _optimizer._extend_dynamic_optimizer
    optim_step = _optimizer.optim_step
    _guard_optim_step_configuration = _optimizer._guard_optim_step_configuration
    _dynamic_optim_step = _optimizer._dynamic_optim_step
    _dynamic_param_step_flags = _optimizer._dynamic_param_step_flags
    _dynamic_optimizer = _optimizer._dynamic_optimizer
    _new_dynamic_optimizer = _optimizer._new_dynamic_optimizer
    _restore_canonical_optimizer = _optimizer._restore_canonical_optimizer
    _zero_dynamic_optimizer_padding = _optimizer._zero_dynamic_optimizer_padding
    _dynamic_optimizer_padding_masks = _optimizer._dynamic_optimizer_padding_masks


def _validate_top_k(top_k: int, model: object) -> None:
    vocab_size = _padded_vocab_size(model)
    if top_k > vocab_size:
        raise ValueError(f"top_k={top_k} exceeds vocabulary size {vocab_size}")


def _active_logical_tokens(requests: Sequence[AnyForwardInput]) -> int:
    return sum(
        int(request.input_tokens.numel())
        for request in requests
        if request.target_tokens is not None
        or request.logits
        or request.top_k is not None
        or request.hidden_states
    )


# Grad-enabled single-target training measured flat per packed row across
# sharing ratios. Wide labels, dense or top-k outputs and no_grad forwards (whose
# GDN branch states are uncharged) keep the logical/packed ratio extrapolation.
_PACKED_PRICED_MIXES = frozenset({"target:single", "inactive"})
# Packed pricing charges head and caller memory per active logical row: label
# copies, positions, row-match vectors, saved masks and the caller's loss saves
# and backward transients. Each request's buffers are separate allocations
# rounded up to 512 B blocks; on an H200, a fully shared one-token request's
# head peaked at nine blocks plus 8 B (4,616 B). Callers whose head and loss
# memory peaks above this charge per token are unsupported.
_PACKED_PRICED_LOGICAL_ROW_BYTES = 12 * 512
# Shorter single-target requests keep the logical extrapolation, so each
# packed-priced request brings at least 384 KiB for per-request constants.
_PACKED_PRICED_MIN_REQUEST_TOKENS = 64


def _packed_priced(signature: "_MemorySignature", one_layer_recompute: bool) -> bool:
    # Measured only under one-layer full recompute: other recompute modes keep
    # more GDN states and activations live per segment and logical row.
    return (
        one_layer_recompute
        and bool(signature.grad_modes)
        and all(signature.grad_modes)
        and _PACKED_PRICED_MIXES.issuperset(signature.request_mix)
        and not signature.short_requests
    )


def _request_mix_key(request: AnyForwardInput) -> str:
    parts = []
    if request.target_tokens is not None:
        target, inputs = request.target_tokens, request.input_tokens
        # Match _forward_item: trailing target dims follow the input shape or a
        # flattened token axis.
        tail_shape = tuple(
            target.shape[inputs.ndim :]
            if target.shape[: inputs.ndim] == inputs.shape
            else target.shape[1:]
        )
        parts.append(f"target:{tail_shape or 'single'}")
    if request.top_k is not None:
        parts.append(f"topk:{int(request.top_k)}")
    if request.logits:
        parts.append("logits")
    if request.hidden_states:
        parts.append("hidden")
    if request.routed_experts is not None:
        parts.append("routes")
    return "+".join(parts) if parts else "inactive"


def _short_request(request: AnyForwardInput) -> bool:
    """Too short for packed pricing: a duplicate adds no packed row, and caller
    memory per request can outgrow its few logical-row charges."""
    return (
        _request_mix_key(request) == "target:single"
        and int(request.input_tokens.numel()) < _PACKED_PRICED_MIN_REQUEST_TOKENS
    )


def _pad_packed_batch(
    batch: PrefixTreePack,
    *,
    multiple: int,
) -> PrefixTreePack:
    if multiple <= 1:
        return batch
    seq_len = int(batch.tokens.shape[1])
    pad = -seq_len % multiple
    if pad == 0:
        return batch

    device = batch.tokens.device
    next_group = (
        int(batch.group_ids.max().item()) + 1 if int(batch.group_ids.numel()) else 1
    )
    pad_group_ids = torch.arange(
        next_group,
        next_group + pad,
        dtype=batch.group_ids.dtype,
        device=device,
    ).unsqueeze(0)
    return PrefixTreePack(
        tokens=torch.cat((batch.tokens, batch.tokens.new_zeros((1, pad))), dim=1),
        group_ids=torch.cat((batch.group_ids, pad_group_ids), dim=1),
        parent_ids=torch.cat((batch.parent_ids, pad_group_ids), dim=1),
        position_ids=torch.cat(
            (batch.position_ids, batch.position_ids.new_zeros((1, pad))), dim=1
        ),
        positions_by_sequence=batch.positions_by_sequence,
        segments=batch.segments,
    )


def _language_model(model: torch.nn.Module) -> "GPTModel":
    module: object = model
    while hasattr(module, "module"):
        module = getattr(module, "module")
    if hasattr(module, "_preprocess") and hasattr(module, "decoder"):
        return cast("GPTModel", module)
    language_model = getattr(module, "language_model", None)
    if language_model is not None:
        return cast("GPTModel", language_model)
    raise RuntimeError("expected a Megatron GPT model")


def _padded_vocab_size(model: object) -> int:
    vocab_size = getattr(getattr(model, "config", None), "padded_vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("could not determine full padded vocabulary size")
    return int(vocab_size)


def _hidden_size(model: "GPTModel | None", provider: object) -> int:
    for source in (getattr(model, "config", None), model, provider):
        if source is None:
            continue
        hidden_size = getattr(source, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise RuntimeError("could not determine hidden size")


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _distributed_grad_norm(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[torch.Tensor],
) -> float:
    return _distributed_grad_norms([(params, grads)])[0]


def _distributed_grad_norms(
    groups: Sequence[tuple[Sequence[torch.nn.Parameter], Sequence[torch.Tensor]]],
) -> tuple[float, ...]:
    if any(len(params) != len(grads) for params, grads in groups):
        raise ValueError("params and grads must have matching lengths")
    device = next(
        (grad.device for _params, grads in groups for grad in grads),
        torch.device("cpu"),
    )
    squared = torch.zeros(len(groups), device=device, dtype=torch.float32)
    for index, (params, grads) in enumerate(groups):
        for param, grad in zip(params, grads, strict=True):
            if _include_in_distributed_grad_norm(param):
                squared[index].add_(grad.float().square().sum())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return tuple(torch.sqrt(squared).tolist())


def _include_in_distributed_grad_norm(param: torch.nn.Parameter) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return True
    from megatron.core import parallel_state as ps

    replica_group = (
        ps.get_data_parallel_group(with_context_parallel=True)
        if bool(getattr(param, "allreduce", True))
        else ps.get_expert_data_parallel_group()
    )
    if replica_group is not None and replica_group.size() > 1:
        if replica_group.rank() != 0:
            return False
    if bool(getattr(param, "lora_tp_sharded", False)):
        return True
    shard_group = (
        ps.get_tensor_model_parallel_group(check_initialized=False)
        if getattr(param, "lora_shard_domain", "tp") == "tp"
        else ps.get_expert_tensor_parallel_group(check_initialized=False)
    )
    return shard_group is None or shard_group.size() <= 1 or shard_group.rank() == 0


def _gradient_staging_bytes(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(
        param.numel() * param.element_size() * (3 - (param.grad is not None))
        for param in parameters
        if param.requires_grad
    )


def _custom_parameters(custom: _CustomObject) -> Iterator[torch.nn.Parameter]:
    if custom.kind == "module":
        yield from cast(torch.nn.Module, custom.value).parameters()
    elif custom.kind == "parameter":
        yield cast(torch.nn.Parameter, custom.value)


def _walk_objects(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_objects(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_objects(item)
    else:
        yield value


def _tracked_tensor_function(
    func: Callable[..., object],
    types: tuple[type, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    from ._heads import (
        _stage_local_buffers,
        head_call_arguments,
        mutates_tensor,
        readonly_buffer_views,
        tensor_metadata_function,
        tensor_mutation_targets,
    )
    from ._tensors import _map_tensor_arguments

    if (captured := head_call_arguments(args, kwargs)) is not None:
        return func(*captured[0], **captured[1])
    del types
    tracked = tuple(
        value
        for value in _walk_objects((args, kwargs))
        if isinstance(value, _TrackedParameter | _TrackedTensor)
    )
    trackers = {id(value._art_tracker): value._art_tracker for value in tracked}
    for tracker in trackers.values():
        tracker.validate()
    if tensor_metadata_function(func) or getattr(func, "__name__", "") in {
        "__format__",
        "__set__",
        "__hash__",
        "__len__",
        "__repr__",
        "__str__",
    }:
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
    if torch._C._current_graph_task_id() >= 0 and any(
        value._art_tracker.active for value in tracked
    ):
        raise RuntimeError(
            "Live checkpoint tensor used during backward recomputation; capture "
            "parameter.clone() or head.snapshot() before activation checkpointing"
        )

    markers: dict[int, torch.Tensor] = {}
    replacements: dict[int, torch.Tensor] = {}

    mutating = mutates_tensor(func, kwargs)
    mutation_targets = tensor_mutation_targets(func, args, kwargs)
    from ._parameter_hooks import parameter_hook_active

    if parameter_hook_active.get() and any(
        isinstance(value, _TrackedParameter) and id(value) in mutation_targets
        for value in tracked
    ):
        raise RuntimeError("Parameter hooks must not mutate checkpoint parameters")
    if getattr(func, "__name__", "") == "requires_grad_" and any(
        value._art_tracker.active for value in tracked
    ):
        raise RuntimeError("Set checkpoint parameter trainability in its factory")
    copied_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []

    def replace(value: object) -> object:
        if isinstance(value, _TrackedParameter | _TrackedTensor):
            cached = replacements.get(id(value))
            if cached is not None:
                return cached
            tracker = value._art_tracker
            with torch._C.DisableTorchFunctionSubclass():
                if (
                    tracker.active
                    and id(value) not in mutation_targets
                    and isinstance(value, _TrackedParameter)
                    and value.requires_grad
                    and torch.is_grad_enabled()
                ):
                    marker = markers.get(id(tracker))
                    if marker is None:
                        marker = torch.zeros((), dtype=torch.bool, device="cpu")
                        markers[id(tracker)] = marker
                        tracker.record(marker)
                    trainer = tracker.validate()
                    assert tracker.ref.name is not None
                    from ._heads import head_staleness

                    snapshot = trainer._snapshot_parameter(
                        value,
                        trainer._capture_checkpoint_version(tracker.ref.name),
                        head_staleness(trainer),
                    )
                    result = _track_slot_graph_tensor(snapshot, marker)
                elif tracker.active and id(value) not in mutation_targets:
                    # Detached/no-grad reads can still be saved by a later graph.
                    result = value.as_subclass(torch.Tensor).detach().clone()
                    if isinstance(value, _TrackedTensor):
                        copied_buffers.append((value, result))
                else:
                    result = value.as_subclass(torch.Tensor)
            replacements[id(value)] = result
            return result
        return _map_tensor_arguments(replace, value)

    result = func(
        *cast(tuple[object, ...], replace(args)),
        **cast(dict[str, object], replace(kwargs)),
    )
    changed_buffers: set[_CustomTensorTracker] = set()
    staged = _stage_local_buffers(
        {str(index): target for index, (target, _) in enumerate(copied_buffers)},
        {str(index): value for index, (_, value) in enumerate(copied_buffers)},
    )
    with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
        for target, value in staged:
            target.copy_(value)
            changed_buffers.add(cast(_TrackedTensor, target)._art_tracker)
    if mutating:
        changed_buffers.update(
            value._art_tracker
            for value in tracked
            if isinstance(value, _TrackedTensor) and id(value) in mutation_targets
        )
    for tracker in changed_buffers:
        tracker.buffer_revision += 1
    if mutating:
        for value in tracked:
            if result is replacements[id(value)]:
                result = value
                break
    if markers and not any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in _walk_objects(result)
    ):
        for marker in markers.values():
            marker.fill_(True)
    return readonly_buffer_views(result, [value for _, value in copied_buffers])


def _graph_marker_is_live(
    marker_ref: weakref.ReferenceType[torch.Tensor],
) -> bool:
    marker = marker_ref()
    return marker is not None and (marker.numel() == 0 or not bool(marker.item()))


def _track_custom_object(
    custom: _CustomObject,
    tracker: _CustomTensorTracker,
) -> _CustomObject:
    if custom.kind == "parameter":
        source = cast(torch.nn.Parameter, custom.value)
        with torch.no_grad():
            value = _TrackedParameter(
                source.detach().clone(), tracker, source.requires_grad
            )
        value.__dict__.update(
            (key, item)
            for key, item in source.__dict__.items()
            if key != "_art_tracker"
        )
        value._art_tracker = tracker
        return _CustomObject(custom.kind, value, custom.generation)
    if custom.kind == "buffer":
        source = cast(torch.Tensor, custom.value)
        with torch.no_grad():
            value = _TrackedTensor(source.detach().clone(), tracker)
        return _CustomObject(custom.kind, value, custom.generation)

    module = cast(torch.nn.Module, custom.value)
    replacements: dict[tuple[str, int], torch.Tensor] = {}
    for child in module.modules():
        for kind in ("parameter", "buffer"):
            for key, source in getattr(child, f"_{kind}s").items():
                if source is None:
                    continue
                identity = (kind, id(source))
                if identity not in replacements:
                    replacements[identity] = cast(
                        torch.Tensor,
                        _track_custom_object(
                            _CustomObject(kind, source, custom.generation), tracker
                        ).value,
                    )
                getattr(child, f"_{kind}s")[key] = replacements[identity]
    from ._heads import native_module_handle

    tracker.validate()
    return replace(custom, handle=native_module_handle(custom, tracker))


def _custom_layout(
    name: str,
    custom: _CustomObject,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if custom.kind == "parameter":
        parameter = cast(torch.nn.Parameter, custom.value)
        return ((name,),), (), (), (name,) if parameter.requires_grad else ()
    if custom.kind == "buffer":
        return (), ((name,),), (name,), ()

    module = cast(torch.nn.Module, custom.value)
    parameter_groups: dict[int, list[str]] = {}
    trainable: list[str] = []
    for key, parameter in module.named_parameters(remove_duplicate=False):
        full_key = f"{name}.{key}"
        aliases = parameter_groups.setdefault(id(parameter), [])
        if not aliases and parameter.requires_grad:
            trainable.append(full_key)
        aliases.append(full_key)
    buffer_groups: dict[int, list[str]] = {}
    persistent: list[str] = []
    for prefix, child in module.named_modules():
        for key, buffer in child._buffers.items():
            if buffer is None:
                continue
            local_key = f"{prefix}.{key}" if prefix else key
            full_key = f"{name}.{local_key}"
            buffer_groups.setdefault(id(buffer), []).append(full_key)
            if key not in child._non_persistent_buffers_set:
                persistent.append(full_key)

    def normalize(groups: Mapping[int, Sequence[str]]) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(tuple(sorted(group)) for group in groups.values()))

    return (
        normalize(parameter_groups),
        normalize(buffer_groups),
        tuple(sorted(persistent)),
        tuple(sorted(trainable)),
    )


def _validate_custom_schema(
    name: str,
    custom: _CustomObject,
    record: Mapping[str, object],
) -> None:
    actual = (
        tuple(
            tuple(group) for group in cast(list[list[str]], record["parameter_aliases"])
        ),
        tuple(
            tuple(group) for group in cast(list[list[str]], record["buffer_aliases"])
        ),
        tuple(cast(list[str], record["persistent_buffer_keys"])),
        tuple(cast(list[str], record["trainable_keys"])),
    )
    if actual != _custom_layout(name, custom):
        raise TrainerRankSlotStateError(
            f"Custom checkpoint object {name!r} schema differs from its factory"
        )


def _custom_signature(
    name: str,
    custom: _CustomObject,
    values: Mapping[str, torch.Tensor],
) -> tuple[object, ...]:
    state = tuple(
        (key, tuple(value.shape), str(value.dtype)) for key, value in values.items()
    )
    return name, custom.kind, state, *_custom_layout(name, custom)


def _custom_named_parameters(
    name: str, custom: _CustomObject
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    if custom.kind == "module":
        for key, parameter in cast(torch.nn.Module, custom.value).named_parameters():
            yield f"{name}.{key}", parameter
    elif custom.kind == "parameter":
        yield name, cast(torch.nn.Parameter, custom.value)


def _custom_state(custom: _CustomObject) -> dict[str, torch.Tensor]:
    if custom.kind == "module":
        return dict(cast(torch.nn.Module, custom.value).state_dict())
    return {"": cast(torch.Tensor, custom.value)}


def _load_custom_state(
    custom: _CustomObject,
    values: Mapping[str, torch.Tensor],
) -> None:
    if custom.kind == "module":
        module = cast(torch.nn.Module, custom.value)
        expected = module.state_dict()
        if set(values) != set(expected):
            raise TrainerRankSlotStateError(
                "Custom module tensor keys differ from checkpoint: "
                f"missing={sorted(set(expected) - set(values))[:8]} "
                f"unexpected={sorted(set(values) - set(expected))[:8]}"
            )
        for key, tensor in values.items():
            target = expected[key]
            if (
                tuple(tensor.shape) != tuple(target.shape)
                or tensor.dtype != target.dtype
            ):
                raise TrainerRankSlotStateError(
                    f"Custom module tensor {key!r} has shape/dtype "
                    f"{tuple(tensor.shape)}/{tensor.dtype}; expected "
                    f"{tuple(target.shape)}/{target.dtype}"
                )
        module.load_state_dict(values, strict=True)
        return
    if set(values) != {""}:
        raise TrainerRankSlotStateError(
            f"Custom {custom.kind} checkpoint must contain exactly one tensor"
        )
    target = cast(torch.Tensor, custom.value)
    tensor = values[""]
    if tuple(tensor.shape) != tuple(target.shape) or tensor.dtype != target.dtype:
        raise TrainerRankSlotStateError(
            f"Custom {custom.kind} has shape/dtype {tuple(target.shape)}/{target.dtype}; "
            f"checkpoint contains {tuple(tensor.shape)}/{tensor.dtype}"
        )
    with torch.no_grad():
        target.copy_(tensor.to(device=target.device))


def _validate_custom_optimizer_state(
    checkpoint: str,
    key: str,
    parameter: torch.nn.Parameter,
    state: "CustomOptimizerState",
) -> None:
    expected_shape = tuple(parameter.shape)
    tensors = {
        "master": state.master,
        "exp_avg": state.exp_avg,
        "exp_avg_sq": state.exp_avg_sq,
    }
    invalid = [
        name
        for name, tensor in tensors.items()
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32
    ]
    if invalid or state.step < 0 or not state.step.is_integer():
        raise TrainerRankSlotStateError(
            f"Custom optimizer state for {checkpoint!r}/{key!r} is invalid; "
            f"expected FP32 tensors with shape {expected_shape} and a nonnegative "
            f"finite integer step (invalid={invalid}, step={state.step})."
        )


def _vocab_parallel_target_logprobs(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    log_z: torch.Tensor,
    *,
    row_offsets: torch.Tensor,
) -> torch.Tensor:
    start, _ = _vocab_range(local_logits)
    flat_labels = labels.reshape(int(labels.shape[0]), -1)
    local_labels = flat_labels - start
    owns_label = (
        (flat_labels != -100)
        & (local_labels >= 0)
        & (local_labels < int(local_logits.shape[1]))
    )
    rows = row_offsets.reshape(-1, 1).expand_as(flat_labels)
    target_logits = local_logits[
        rows,
        local_labels.clamp(0, int(local_logits.shape[1]) - 1),
    ].float()
    target_logits = target_logits.masked_fill(~owns_label, 0.0).reshape(labels.shape)
    target_logits = _all_reduce_tensor_parallel_sum(target_logits)
    log_z = log_z.reshape(int(log_z.shape[0]), *((1,) * (int(labels.ndim) - 1)))
    return (target_logits.float() - log_z).masked_fill(labels == -100, 0.0)


def _anchor_disconnected_outputs(
    target_logprobs: list[torch.Tensor | None],
    top_k: list[TopK | None],
    hidden_by_row: torch.Tensor,
) -> tuple[list[torch.Tensor | None], list[TopK | None]]:
    if not hidden_by_row.requires_grad:
        return target_logprobs, top_k
    anchor: torch.Tensor | None = None

    def anchor_tensor(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal anchor
        if tensor.requires_grad:
            return tensor
        if anchor is None:
            anchor = hidden_by_row.reshape(-1)[:1].float().sum() * 0.0
        return tensor + anchor

    for index, logprobs in enumerate(target_logprobs):
        if logprobs is not None:
            target_logprobs[index] = anchor_tensor(logprobs)
    for index, item in enumerate(top_k):
        if item is not None:
            top_k[index] = TopK(anchor_tensor(item.logprobs), item.tokens)
    return target_logprobs, top_k


def _try_triton_local_topk_stats(
    local_logits: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if k <= 0 or k > int(
        os.environ.get("ART_TRAINER_RANK_TRITON_FUSED_TOPK_MAX", "10")
    ):
        return None
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        _try_triton_stats(
            "local_topk_stats",
            local_logits,
            k=min(k, int(local_logits.shape[1])),
        ),
    )


def _triton_stats_enabled(cuda: bool, rows: int) -> bool:
    """Whether ``_try_triton_stats`` attempts the kernel for a ``rows`` chunk."""
    return (
        cuda
        and os.environ.get("ART_TRAINER_RANK_TRITON_TOPK", "1").lower()
        not in {"0", "false"}
        and rows >= int(os.environ.get("ART_TRAINER_RANK_TRITON_MIN_ROWS", "64"))
    )


@lru_cache(maxsize=None)
def _triton_stats_importable() -> bool:
    try:
        from art.trainer_rank import topk  # noqa: F401
    except Exception:
        return False
    return True


def _triton_head_stats_available(rank: Any) -> bool:
    """Whether the head statistics kernels can run on this rank at all."""
    return (
        _triton_stats_enabled(rank.device.type == "cuda", 1 << 62)
        and _triton_stats_importable()
    )


def _triton_head_stats(rank: Any, rows: int, vocabulary: int) -> bool:
    """The head statistics path a ``rows`` x ``vocabulary`` chunk takes, for admission.

    Mirrors ``_try_triton_stats``'s attempt predicate plus kernel
    importability. A kernel that imports but fails at run time falls back
    (unless ART_TRAINER_RANK_TRITON_TOPK=strict makes that fatal); the first
    such wave was priced for the kernel, and ``_local_head_stats`` then
    records that chunk shape, priced for the fallback until a kernel succeeds
    at it again.
    """
    return (
        _triton_stats_enabled(rank.device.type == "cuda", rows)
        and (rows, vocabulary) not in getattr(rank, "_triton_head_stats_failures", ())
        and _triton_stats_importable()
    )


def _try_triton_stats(
    name: str,
    local_logits: torch.Tensor,
    **kwargs: object,
) -> object | None:
    if not _triton_stats_enabled(local_logits.is_cuda, int(local_logits.shape[0])):
        return None
    try:
        from art.trainer_rank import topk

        return getattr(topk, name)(local_logits, **kwargs)
    except Exception:
        if os.environ.get("ART_TRAINER_RANK_TRITON_TOPK", "1").lower() == "strict":
            raise
        return None


def _vocab_parallel_topk_from_local(
    local_values: torch.Tensor,
    local_tokens: torch.Tensor,
    *,
    k: int,
    log_z: torch.Tensor,
    vocab_start: int,
) -> TopK:
    local_k = min(k, int(local_values.shape[1]))
    local_values = local_values[:, :local_k]
    local_tokens = local_tokens[:, :local_k] + vocab_start

    from megatron.core import parallel_state as ps

    tp_size = int(ps.get_tensor_model_parallel_world_size())
    if tp_size <= 1:
        return TopK(
            logprobs=local_values - log_z.unsqueeze(1),
            tokens=local_tokens,
        )

    from megatron.core import tensor_parallel

    group = ps.get_tensor_model_parallel_group(check_initialized=False)
    values = cast(
        torch.Tensor,
        tensor_parallel.gather_from_tensor_model_parallel_region(
            local_values,
            group=group,
        ),
    )
    gathered_tokens = [torch.empty_like(local_tokens) for _ in range(tp_size)]
    dist.all_gather(gathered_tokens, local_tokens, group=group)
    tokens = torch.cat(gathered_tokens, dim=1)
    top_values, top_offsets = torch.topk(values, k=k, dim=-1)
    return TopK(
        logprobs=top_values - log_z.unsqueeze(1),
        tokens=tokens.gather(1, top_offsets),
    )


# Row sub-chunks per head chunk for the bounded eager statistics: one FP32
# sub-chunk is 2 * ceil(rows / 64) / rows of a BF16 chunk buffer: 1/32 when the
# rows divide by 64, more otherwise (the estimator prices the ceiling).
_EAGER_STATS_SUBCHUNKS = 64


class _EagerLocalStats(torch.autograd.Function):
    """The statistics kernel's (local max, local sum) contract, computed eagerly.

    The logits are processed in up to ``_EAGER_STATS_SUBCHUNKS`` row sub-chunks
    of ceil(rows / 64) rows each through one owned FP32 work buffer, beside
    the logits in forward and beside the logits and their gradient in
    backward: the kernel path's buffers (topk._LocalStatsFunction) plus that
    buffer, instead of the unchunked fallback's seven. The logits are never
    written. Rows are independent; the caller's rescale of local sums to the
    global maximum equals the unchunked fallback in exact arithmetic, and
    FP32 rounding may differ when ranks' local maxima differ.
    """

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(logits.shape[0])
        step = max(1, -(-rows // _EAGER_STATS_SUBCHUNKS))
        local_max = logits.max(dim=-1).values.float()
        local_sum = torch.empty_like(local_max)
        work = torch.empty(
            (min(step, rows), int(logits.shape[1])),
            dtype=torch.float32,
            device=logits.device,
        )
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            block = work[: stop - start]
            block.copy_(logits[start:stop])  # owned: FP32 logits stay intact
            block.sub_(local_max[start:stop, None]).exp_()
            local_sum[start:stop] = block.sum(dim=-1)
        del work
        ctx.save_for_backward(logits, local_max)
        ctx.step = step
        return local_max, local_sum

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # As the kernel: the local max is a detached shift; only the sum
        # carries a gradient, exp(logits - local max) per row.
        _grad_local_max, grad_local_sum = grad_outputs
        logits, local_max = ctx.saved_tensors
        grad = torch.empty_like(logits)
        if grad_local_sum is None:
            return grad.zero_()
        rows, step = int(logits.shape[0]), int(ctx.step)
        work = torch.empty(
            (min(step, rows), int(logits.shape[1])),
            dtype=torch.float32,
            device=logits.device,
        )
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            block = work[: stop - start]
            block.copy_(logits[start:stop])  # owned: the saved logits stay intact
            block.sub_(local_max[start:stop, None]).exp_()
            block.mul_(grad_local_sum[start:stop, None])
            grad[start:stop] = block
        del work
        return grad


def _eager_local_logsumexp_stats(
    local_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = _EagerLocalStats.apply(local_logits)
    return cast(tuple[torch.Tensor, torch.Tensor], stats)


def _vocab_parallel_log_z(local_logits: torch.Tensor) -> torch.Tensor:
    local_logits = local_logits.float()
    local_max = local_logits.max(dim=-1).values.detach()
    global_max = _all_reduce_tensor_parallel_max(local_max)
    local_sum = _local_vocab_exp_sum(local_logits, global_max)
    global_sum = _all_reduce_tensor_parallel_sum(local_sum)
    return global_max + torch.log(global_sum)


def _local_vocab_exp_sum(
    local_logits: torch.Tensor,
    global_max: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(local_logits.float() - global_max.unsqueeze(1)).sum(dim=-1)


def _vocab_range(local_logits: torch.Tensor) -> tuple[int, int]:
    from megatron.core import parallel_state as ps

    local_size = int(local_logits.shape[1])
    rank = int(ps.get_tensor_model_parallel_rank())
    start = rank * local_size
    return start, start + local_size


def _all_reduce_tensor_parallel_sum(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as ps

    if int(ps.get_tensor_model_parallel_world_size()) <= 1:
        return tensor
    from megatron.core import tensor_parallel

    return cast(
        torch.Tensor,
        tensor_parallel.reduce_from_tensor_model_parallel_region(
            tensor,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        ),
    )


def _all_reduce_tensor_parallel_max(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as ps

    if int(ps.get_tensor_model_parallel_world_size()) <= 1:
        return tensor
    output = tensor.clone()
    dist.all_reduce(
        output,
        op=dist.ReduceOp.MAX,
        group=ps.get_tensor_model_parallel_group(check_initialized=False),
    )
    return output


def _row_match(
    positions: torch.Tensor,
    rows: torch.Tensor,
    *,
    chunk_tokens: int,
) -> _RowMatch:
    row_offsets = torch.searchsorted(rows, positions)
    in_bounds = row_offsets < int(rows.numel())
    source_offsets = torch.arange(
        int(positions.numel()), device=positions.device, dtype=torch.long
    )[in_bounds]
    row_offsets = row_offsets[in_bounds]
    keep = rows.index_select(0, row_offsets) == positions.index_select(
        0, source_offsets
    )
    source_offsets, row_offsets = source_offsets[keep], row_offsets[keep]
    if int(row_offsets.numel()) > 1:
        order = row_offsets.argsort()
        source_offsets = source_offsets.index_select(0, order)
        row_offsets = row_offsets.index_select(0, order)
    return (
        source_offsets,
        row_offsets,
        _chunk_boundaries(
            row_offsets,
            end=int(rows.numel()),
            chunk_tokens=chunk_tokens,
        ),
    )


def _chunk_boundaries(
    offsets: torch.Tensor,
    *,
    end: int,
    chunk_tokens: int,
) -> tuple[int, ...]:
    edges = torch.arange(0, end, chunk_tokens, dtype=torch.long)
    edges = torch.cat((edges, torch.tensor((end,), dtype=torch.long)))
    return tuple(torch.searchsorted(offsets, edges).tolist())


def _select_positions(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if int(positions.numel()) == 0:
        return values[:0].clone()
    return values.index_select(0, positions.to(device=values.device))


@lru_cache(maxsize=256)
def _resolved_request_policy(options: ForwardOptions | None) -> ResolvedForwardOptions:
    return resolve_forward_options(input=options)


def _correction_state_bytes(
    group: _ForwardGroupPlan, options: ResolvedForwardOptions
) -> int:
    if not group.grad_enabled or not options.stale_gradient_corrections:
        return 0
    topk = sum(
        item.input_ids.numel() * (item.request.top_k or 0) for item in group.items
    )
    logprobs = 4 * (
        topk
        + sum(item.labels.numel() for item in group.items if item.labels is not None)
    )
    # Explicit always also stages corrected cotangents.
    return topk * 8 + logprobs * (
        2
        if any(c.policy == "always" for c in options.stale_gradient_corrections)
        else 1
    )


def _snapshot_tensor_bytes(value: object) -> int:
    """Graph replay snapshots copy each tensor occurrence in the captured plan."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _snapshot_tensor_bytes(getattr(value, f.name))
            for f in fields(value)
            if f.init
        )
    if isinstance(value, dict):
        return sum(_snapshot_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_snapshot_tensor_bytes(item) for item in value)
    return 0


def _batch_seq_logits(logits: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    if int(logits.ndim) != 3:
        raise RuntimeError(
            f"expected logits with shape [B, S, V] or [S, B, V], got {tuple(logits.shape)}"
        )
    if int(logits.shape[0]) == 1 and int(logits.shape[1]) == seq_len:
        return logits
    if int(logits.shape[0]) == seq_len and int(logits.shape[1]) == 1:
        return logits.transpose(0, 1).contiguous()
    raise RuntimeError(
        f"logits do not match sequence length {seq_len}: {tuple(logits.shape)}"
    )


def _materialize(inputs: ForwardInputs) -> ForwardInputs:
    if isinstance(inputs, ForwardInput):
        return inputs
    return _rebuild_forward_tree(
        inputs, [_materialize(item) for item in _nested_forward_children(inputs)]
    )


def _rebuild_forward_tree(template: Any, children: list[Any]) -> Any:
    if isinstance(template, tuple):
        return (
            type(template)(*children)
            if hasattr(template, "_fields")
            else tuple(children)
        )
    return children


def _is_forward_input(inputs: ForwardInputs) -> TypeIs[AnyForwardInput]:
    return isinstance(inputs, ForwardInput)


def _flatten(inputs: ForwardInputs) -> Iterator[AnyForwardInput]:
    if _is_forward_input(inputs):
        yield inputs
        return
    for item in _nested_forward_children(inputs):
        yield from _flatten(item)


def _unflatten(
    template: ForwardInputs, outputs: Iterator[AnyForwardOutput]
) -> ForwardOutputs:
    if isinstance(template, ForwardInput):
        return next(outputs)
    return _rebuild_forward_tree(
        template,
        [_unflatten(item, outputs) for item in _nested_forward_children(template)],
    )


def _nested_forward_children(inputs: ForwardInputs) -> Iterator[ForwardInputs]:
    if isinstance(inputs, Mapping):
        raise TypeError(
            "dict was passed directly to TrainerRank; gather or materialize the "
            "values into a list/tuple so nested forward output ordering is explicit"
        )
    if isinstance(inputs, str | bytes):
        raise TypeError(
            "TrainerRank forward inputs must be ForwardInput objects or nested "
            "iterables of ForwardInput objects, not strings"
        )
    try:
        return iter(cast(Iterable[ForwardInputs], inputs))
    except TypeError as exc:
        raise TypeError(
            "TrainerRank forward inputs must be ForwardInput objects or nested "
            "iterables of ForwardInput objects"
        ) from exc


__all__ = [
    "AdamParams",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "MicroBatchStats",
    "TopK",
    "TrainerRank",
    "TrainerRankMemoryError",
    "TrainerRankSlotStateError",
]
